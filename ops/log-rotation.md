# Rotating append-only runtime logs

**Published in full.** The stanza pattern is in
[`examples/logrotate-runtime.conf`](examples/logrotate-runtime.conf) and the two code snippets
below are complete.

An agent deployment accumulates append-only runtime logs fast: one status record per monitor tick,
one object per sampler cycle, an engine scheduler log that is never rotated at all. `[measured]`
One status ledger reached **401 MB** before anything rotated it, at ~30 KB/record × 720
records/day ≈ **20 MB/day**; one engine log reached **648 MB / 4.56M lines** across every boot.

Rotation is not the interesting part. The two ways rotation **silently breaks something else**
are.

---

## 1. `copytruncate` is mandatory, not stylistic

The default rotation scheme is rename-then-recreate. That works only if the writer reopens the
path. **A long-lived process holding an `O_APPEND` fd never does** — it keeps writing into the
now-unlinked inode, forever. The symptom is silent log loss *with no size relief*: the new file
stays empty, the old inode keeps growing invisibly, and `du` on the directory does not account for
it.

Two shapes of writer where this is guaranteed, both present here:

- **fd 1+2 of a long-lived process.** A service whose start script does
  `exec python … >>"$LOG" 2>&1` holds that fd for the life of the process.
- **systemd's own `StandardOutput=append:`.** systemd opens the file and holds the fd for the
  life of the unit. This one is easy to miss precisely because the unit file never mentions a
  redirect.

So: `copytruncate` on both. And then `copytruncate` **everywhere in the same config, even where it
is not required** — paths whose writer reopens per record would survive either scheme, but a
policy with one rule is auditable and a policy with two invites someone to apply the wrong one to
the next path they add.

**What `copytruncate` costs**, stated so the trade is explicit: records written between the copy
and the truncate are lost. For a 20 MB/day metrics ledger that is sub-millisecond and acceptable.
For anything where a lost record is a correctness problem, the answer is not a different rotation
mode — it is a writer that reopens, or a writer that rolls its own files per day (§3).

---

## 2. Compressing rotated files silently breaks gz-blind readers

`compress` is the obvious next flag — `[measured]` gzip -6 gives ~**8.8×** on JSONL here, turning
14 daily generations of a 20 MB/day ledger into ~55 MB of steady state. It is also the step that
breaks every analysis script globbing `*.jsonl`, because `name.1.gz` does not match the bare glob.

**And it breaks them by returning less data, not by erroring.** A report over "the last 24 h"
simply comes up short right after a rotation. Nothing raises, no row says "missing", and the
numbers look plausible. This is the same error class as
[client rule 15](../clients/README.md#15-a-proxy-stops-being-a-proxy-when-the-thing-it-proxied-for-moves):
a correct consumer, invalidated by a change underneath it, failing by silently returning a smaller
number.

**The rule: make every reader gz-aware *before* enabling compression, in the same change.** On
this deployment that ordering was held — the readers were made gz-aware while the keep-window was
still 7 days and no day file had yet been compressed, so the broken state never existed. That is
the cheap way to do it; the expensive way is to notice a report is short and work backwards.

The two functions, in full. Note that the docstring states the failure mode, which is what stops
the next person from "simplifying" the glob:

```python
def day_files(pattern: str) -> list[str]:
    """Day files matching `pattern`, including ones rotation has gzipped.

    Compression renames a day file to `<name>.gz`, which does NOT match the bare
    glob — without this, the stale days would silently vanish from every report
    instead of erroring."""
    return sorted(set(glob.glob(pattern)) | set(glob.glob(pattern + ".gz")),
                  key=lambda p: p[:-3] if p.endswith(".gz") else p)


def open_day(path: str):
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), errors="replace")
    return open(path, errors="replace")
```

Two details that matter:

- **The sort key strips `.gz`.** Otherwise `ticks-20260929.jsonl.gz` sorts after
  `ticks-20260930.jsonl` and a chronological reader processes days out of order.
- **`errors="replace"` on both branches**, for the same reason as
  [`observability.md §3a`](observability.md#3-two-traps-in-reading-an-engine-log): a runtime log
  can contain bytes that are not valid UTF-8, and a decode error halfway through a day file
  truncates the window as silently as a missing file.

### Provide a window reader, not just a file opener

"Last 24 h" must span live **and** rotated files, or it is short by exactly the amount that
rotated. One function, which every consumer uses instead of opening anything itself:

```python
def load(pattern: str, since: float) -> list[dict]:
    out = []
    for p in day_files(pattern):            # live + .gz, chronological
        with open_day(p) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue                # a partial last line is normal on a live file
                if float(r.get("ts") or r.get("computed_at") or 0) >= since:
                    out.append(r)
    return out
```

The `try/except continue` is not sloppiness: reading a file that is being appended to will
eventually catch a half-written final line, and a window reader that raises on it is a window
reader that fails once a day at random.

---

## 3. Know which paths you must *not* rotate

A rotation config is an inventory, and the entries it **excludes** are the load-bearing ones.
Two classes, both of which produce a fight if you get them wrong:

- **A writer that rotates itself.** An agent framework capping its own logs at 5 MB with `.1/.2/.3`
  generations does not want logrotate as well; adding it means two rotators racing on the same
  path.
- **A writer that rolls per calendar day.** `ticks-YYYYMMDD.jsonl` is never appended to once
  stale, so logrotate has nothing useful to do with it — but **nothing compresses it either**, so
  it accumulates forever. That needs a separate step, not a logrotate stanza:

```bash
# gzip day files older than KEEP_DAYS; the writer rolls them, we only compress.
cutoff="$(date -u -d "${KEEP_DAYS} days ago" +%Y%m%d)"
for f in "$DIR"/ticks-[0-9]*.jsonl "$DIR"/decisions-[0-9]*.jsonl; do
  [[ -f "$f" ]] || continue
  day="${f##*-}"; day="${day%.jsonl}"
  [[ "$day" =~ ^[0-9]{8}$ ]] || continue
  [[ "$day" < "$cutoff" ]] || continue
  # -N keeps the original name+mtime inside the archive; gzip removes the source only
  # after a successful write, so a crash mid-compress leaves the plain file intact.
  gzip -6 -N "$f"
done
```

**Set `KEEP_DAYS` from the widest window any consumer reads, with margin.** Here the throughput
and attribution analyses read 24 h, so 7 days is a wide margin and plain `grep`/`jq` keeps working
on anything recent. The gz-aware reader means a *correct* consumer is unaffected either way; the
margin is for the ad-hoc one-liner a human types during an incident.

**⛔ Never delete.** logrotate prunes only its own compressed generations past `rotate N` (≥ 14
days for every stanza here); the day-file step only gzips. A rotation job that deletes is a
rotation job somebody disables.

---

## 4. Document the writer and the readers, per path

Every stanza in this deployment's config carries three comment lines — **writer**, **readers**,
**growth** — and they are the reason the policy is auditable at all:

```
# ── monitor: status.jsonl ──
# Writer:  <script> line 63 -> one status record per tick (240 s)
#                   line 145 -> one note record per tick
#          Both open(path,"a") then close: no persistent fd.
# Readers: none in code. latest_status.json (rewritten every tick) is the programmatic
#          consumer; status.jsonl is ad-hoc/interactive history only.
# Growth:  ~30 KB/record, 720 records/day => ~20 MB/day. Was 401 MB unrotated.
# History: 14 daily generations, gzip ~8.8x => steady state ~55 MB total.
```

That header answers, without reading any code, the two questions that decide the stanza:

- **Does the writer hold a persistent fd?** → whether `copytruncate` is mandatory (§1).
- **Does anything read it programmatically?** → whether `compress` needs a reader change first
  (§2). "Readers: none in code" is a real and useful answer, and it is the one that makes
  compression free.

Write the inventory when you write the stanza. Working it out afterwards means reading every
consumer in the repository, which is how a path ends up rotated with the wrong mode.

---

## Driving it

A tiny wrapper on an hourly timer, so the config decides what has earned a rotation rather than
the schedule:

```ini
[Timer]
OnBootSec=10min
OnUnitActiveSec=1h
AccuracySec=5min
Persistent=true
```

```ini
[Service]
Type=oneshot
Nice=10
IOSchedulingClass=idle
ExecStart=<wrapper>
```

`IOSchedulingClass=idle` is deliberate: compressing a few hundred megabytes must not compete for
I/O with the thing being logged. The wrapper takes `--force` (rotate now, ignore intervals) and
`--debug` (dry run, print what it *would* do, change nothing), and keeps its own logrotate state
file rather than sharing the system one — a user-unit rotation job writing system logrotate state
is a surprise waiting for whoever next runs the system job.
