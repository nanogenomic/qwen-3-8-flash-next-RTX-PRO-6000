# The control plane

The layer between the inference engine and the agent framework. Neither repository half
contains it: the [engine patches](../patches/) make one card hold more and go faster, the
[client contract](../clients/README.md) says what an orchestrator must do with it, and in
between sits a set of small, boring processes that decide **whether the backend is up at
all**, how several agent sessions divide it, and whether you can find out why it died.

Every item here exists because something went down. The code is the least interesting part —
most of it is forty lines of Python or a six-line systemd drop-in. The valuable part is the
**failure each one prevents**, because every failure below is silent: no error, no alert, no
non-zero exit, and in four cases a green status readout over a dead lane.

Contents:

| | what it is | where |
|---|---|---|
| **1** | **Cross-session KV pool broker** — one allowance document several agent sessions size themselves from, retunable on a live fleet in ≤ 5 s | [`pool-broker.md`](pool-broker.md) |
| **2** | **Reboot survival** — what has to be recreated after a restart, and the eight systemd/tmux traps that each cost a silent outage | [`reboot-survival.md`](reboot-survival.md) |
| **3** | **Observability and incident capture** — a VRAM/KV sample ring, and one artifact per lane death | [`observability.md`](observability.md) |
| **4** | **Log rotation for append-only runtime logs** — why `copytruncate` is mandatory and compression is a reader bug | [`log-rotation.md`](log-rotation.md) |
| **5** | **Cutover discipline** — how a serving config change lands without restarting the thing it configures | [`cutover.md`](cutover.md) |

Copyable artifacts: [`examples/`](examples/).

---

## Read this first: the eleven lessons

Each is stated as a rule, with the evidence behind it in the linked document. Every figure is
labelled the way the rest of this repository labels them — `[measured]` on the reference
deployment unless it says otherwise.

### Supervision

**1. `StartLimitBurst × RestartSec < StartLimitIntervalSec` latches a unit `failed` with
nothing left to retry.** A crash-loop that trips the start limiter leaves the service
**permanently down**, which is strictly worse than any crash. `[measured]` The serving lane
ran at `StartLimitIntervalSec=600`, `StartLimitBurst=10`, `RestartSec=30` — ten restarts span
300 s, comfortably *inside* the 600 s window — and crash-looped **nine** times on one bad
boot. It survived on a margin of one. The drop-in that made it "more generous" is what
created the trap: it raised the burst 5 → 10 *and* `RestartSec` 5 s → 30 s, and the second
change is what pulled all ten starts inside the window. The preceding 5 × 5 s = 25 s in a
300 s window was safe. → [reboot-survival §1](reboot-survival.md#1-the-start-limiter-is-how-a-must-never-stop-lane-stops-forever)

**2. A directive in the wrong `[section]` is silently ignored.** `StartLimitIntervalSec` and
`StartLimitBurst` belong in `[Unit]`; `RefuseManualStop` belongs in `[Unit]`. Put either under
`[Service]` and systemd loads the unit, reports the sibling settings you *did* place
correctly, and drops yours. `[measured]` `systemd-analyze --user verify` on a unit with
`StartLimitIntervalSec` under `[Service]`:

```
zz-probe.service:6: Unknown key name 'StartLimitIntervalSec' in section 'Service', ignoring.
```

and it still exits 0 — the message is the whole signal. The first version of this deployment's
resilience drop-in put `RefuseManualStop` under `[Service]`, so the unit reported
`Restart=always` while remaining stoppable by anyone. → [cutover §2](cutover.md#2-verify-the-unit-and-every-live-drop-in-before-applying-anything)

**3. A `ConditionPathExists` skip is silent.** Not a failure: the unit records
`inactive (dead)`, logs one debug line, and nothing alerts. `systemctl is-failed` stays clean
and no incident artifact is written, **because the process never started and therefore never
crashed**. A drain flag with no age ceiling, left behind by an interrupted drain, took a GPU
lane down through every subsequent boot that way. → [reboot-survival §2](reboot-survival.md#2-a-conditionpathexists-skip-is-the-quietest-outage-you-can-have)

**4. Retire a stale flag by renaming it, never by deleting it.** A guard that deletes
operator state is a guard nobody will leave enabled. Renaming to `<name>.expired-<utc>` keeps
the evidence and makes the action reversible with one `mv`. And prefer a **wall-clock age
ceiling** over "was it set before this boot": a deliberate pre-reboot drain is armed *seconds*
before the reboot precisely to inhibit the next boot, so a boot-relative test defeats the one
case the flag is for. → [reboot-survival §2](reboot-survival.md#2-a-conditionpathexists-skip-is-the-quietest-outage-you-can-have)

**5. An ordering cycle is resolved by systemd dropping an edge of its own choosing.**
`A After=B` plus `B After=A` does not fail; it produces non-deterministic boot order, or a unit
that silently does not start. `systemd-analyze --user verify` catches it by name
(`Found ordering cycle on …/start … Transaction order is cyclic`). The cycle here was created
by naming a unit in an `After=` when *its own drop-in* already ordered itself after you. →
[cutover §2](cutover.md#2-verify-the-unit-and-every-live-drop-in-before-applying-anything)

### tmux, for anything that restores a session

**6. `TMUX_TMPDIR` must not include the `tmux-$UID` component** — tmux appends it. The live
socket is `$TMUX_TMPDIR/tmux-$UID/default`, so pointing `TMUX_TMPDIR` at the directory that
*contains* the socket addresses `…/tmux-$UID/tmux-$UID/default`: a different, empty server.
`[measured]` With the wrong value every `has-session` check reported absent while both lanes
were live, so a boot restore would have started a second server and launched a **duplicate**
agent process per session. Assert the socket, do not assume it. →
[reboot-survival §3](reboot-survival.md#3-four-tmux-facts-that-each-turn-a-restore-into-a-duplicator)

**7. `tmux start-server` does not leave you a server.** A sessionless tmux server exits
immediately, so pre-creating one is a no-op that then makes every subsequent check fail.
`[measured]` The first version of that helper reported "cannot reach or start a tmux server"
on a clean box and restored nothing — precisely the boot it exists for. The only way to raise a
server is to create a **session**. → [reboot-survival §3](reboot-survival.md#3-four-tmux-facts-that-each-turn-a-restore-into-a-duplicator)

**8. A tmux server inherits a non-close-on-exec flock fd and holds it for the server's
lifetime.** A bash `exec 9>lock` fd is not `O_CLOEXEC`, so the long-lived server born under
your lock keeps it forever. `[measured]` After the first run created the server, every later
timer run exited "another instance is running" — which silently converted a 10-minute retry
timer into a single boot-time attempt, removing the one property the timer existed to provide.
Close it at exec (`9>&-`) **and** cross-check the pidfile against a live process before
believing the lock. → [reboot-survival §3](reboot-survival.md#3-four-tmux-facts-that-each-turn-a-restore-into-a-duplicator)

**9. `IFS=$'\t' read` collapses runs of tabs.** Tab is IFS whitespace, so one empty field
shifts every later column into the wrong variable. `[measured]` An empty `paused_reason` put a
turn counter (`"0/400"`) into the variable a safety gate read as the pause reason. The gate
still denied correctly — by luck, because its patterns do not match a turn count — and a
safety gate must not depend on that. Use US (`0x1f`), which is not IFS whitespace, so empty
fields survive. → [reboot-survival §3](reboot-survival.md#3-four-tmux-facts-that-each-turn-a-restore-into-a-duplicator)

### Reading what the engine left behind

**10. Engine logs contain embedded NUL bytes, and `grep -n` stops listing on them.** It prints
`binary file matches` and says nothing further, so a grep for the exception class **hides every
later match**. `[measured]` A plain grep for `torch.OutOfMemoryError` over one 648 MB
scheduler log surfaced the two oldest deaths and silently hid the two recent ones that were
actually being investigated. Pipe through `tr -d '\000'` for an ad-hoc grep; in code, open
binary and decode with `errors="replace"`. →
[observability §3](observability.md#3-two-traps-in-reading-an-engine-log)

**11. The outermost exception is usually not the cause. Record a cause *chain*.**
`[measured]` A death reported as `torch.distributed.DistBackendError: NCCL error … unhandled
cuda error` had `Failed to CUDA calloc 536870912 bytes` beneath it, and beneath *that* a
warning naming the actual trigger: a Triton kernel device-loaded after serving started with
`free device mem: 0.54 GiB`. A second death — a
`info.status != cudaStreamCaptureStatusInvalidated` assert — had its deepest traceback frame
in a BF16 skinny-GEMM's output allocation, a layer with nothing to do with the fault: the CUDA
graph capture had been poisoned **asynchronously, by another thread**, so the frame that
noticed is unrelated to the frame that caused it. A traceback localises the *detection*. →
[observability §3](observability.md#3-two-traps-in-reading-an-engine-log)

---

## Two patterns that recur in all five

**Fail open, and say so in the output.** Every component here returns "no signal" rather than
a guess, and every consumer treats "no signal" as *the behaviour it had before this component
existed*. The pool broker's allowance goes stale and callers fall back to their configured
defaults; the VRAM sample older than 90 s is discarded rather than trusted; the boot guard
exits 0 on any internal error specifically so a bug in the guard can never be the reason
serving did not start. A control-plane component that can fail the thing it supervises is a
net loss.

**An alert is not a control, and a silent backstop is indistinguishable from a broken one.**
`[measured]` On the night the lane died of a VRAM exhaustion, a 20-second-cadence monitor had
flagged critical headroom on every sample that day, sent a low-VRAM alert 16 h 15 min before
the crash and a queue-depth alert 53 s before it. All of it correct; none of it reached
anything that could defer work (that is [client rule 9](../clients/README.md#9-throttle-fan-out-on-vram-headroom-not-only-on-pool-utilisation),
and [rule 15](../clients/README.md#15-a-proxy-stops-being-a-proxy-when-the-thing-it-proxied-for-moves)
is what replaced the first attempt at acting on it). The converse failure is just as common: a
sampler that had been **silently dead for 24 h** still read `active (elapsed)`. So every
backstop here leaves a timestamp a human can check in one command, and liveness is systemd-native
(`Type=notify` + `WatchdogSec`) rather than hoped for.

---

## Scope, stated honestly

**Much of this layer is published as design and interface rather than as source.** The
[client patches](../clients/hermes/README.md#the-patch-series) were scrubbed line by line and
shipped ([`clients/hermes/NOTICE`](../clients/hermes/NOTICE) records exactly what was
substituted); these scripts resist the same treatment, because they are dense with one fleet's
absolute paths, host names, addresses, unit names, state-file locations and ssh key paths, and
those paths are not decoration — they are most of what each file says. As source they would be
a liability to read and useless to run. Where a file is genuinely portable it is in [`examples/`](examples/) in full. Where it is
only meaningful with its paths, its design and its knobs are here and the file is not. Each
document says which it is, at the top.

Not published anywhere in this repository: credentials, addresses or host names, model weights,
traffic-derived artifacts, or the fleet's lane launcher, capacity governor, admission guard and
messaging bridges — see [`clients/hermes/README.md` § What is *not* here](../clients/hermes/README.md#what-is-not-here-and-why).

In the examples and the documents, these are placeholders:

| placeholder | means |
|---|---|
| `serving@gpu0.service` | the systemd **user** unit running the engine on one card |
| `<stack>/` | the serving host's stack root (logs, state flags, ops scripts) |
| `$HERMES_HOME` | the agent framework's configurable home ([client rule 7](../clients/README.md#7-isolate-test-state-from-live-state)) |
| `http://127.0.0.1:30000` | the engine's HTTP endpoint (SGLang's own default bind) |
| "the serving host" / "the client host" | two machines; the agent sessions do not run on the card |

Everything here runs as **systemd user units** (`systemctl --user`), not system units. That is
not incidental: the agent sessions, the broker, the monitor and the engine all run as one
unprivileged user, which is what lets a drop-in be installed and a timer re-armed without root.
`RequiresMountsFor=` does work in a user unit — the user manager proxies system mount units.
