# Reboot survival

**Published as patterns plus one portable artifact.** The drop-in in
[`examples/never-give-up.conf`](examples/never-give-up.conf) is complete and copyable. The two
boot guards and the session-restore script are published as design, interface and the traps they
encode — they are only meaningful with one fleet's unit names, state-file paths and lane ids,
and as source they would be a liability to read and useless to run.

A restart is the one event that exercises every assumption in a serving deployment at once, and
**every failure in this document is silent**: no error, no alert, no non-zero exit, and in three
cases a green status readout over a dead lane. They are grouped by the thing that produces them.

Contents:

1. [The start limiter is how a "must never stop" lane stops forever](#1-the-start-limiter-is-how-a-must-never-stop-lane-stops-forever)
2. [A `ConditionPathExists` skip is the quietest outage you can have](#2-a-conditionpathexists-skip-is-the-quietest-outage-you-can-have)
3. [Four tmux facts that each turn a restore into a duplicator](#3-four-tmux-facts-that-each-turn-a-restore-into-a-duplicator)
4. [Never two writers: a single-writer guard that waits](#4-never-two-writers-a-single-writer-guard-that-waits)
5. [Classifying a boot-crash recurrence by the *effective* environment](#5-classifying-a-boot-crash-recurrence-by-the-effective-environment)

---

## 1. The start limiter is how a "must never stop" lane stops forever

**The rule.** If `StartLimitBurst × RestartSec` is **less than** `StartLimitIntervalSec`, a
fast-failing unit burns its whole restart budget inside the window, systemd logs *"start request
repeated too quickly"*, and the unit latches **`failed`** with nothing left to retry. For a lane
whose stated requirement is "must never stop", that is the worst available outcome: it is not a
crash, it is a crash *plus* a permanent refusal to try again, and it is silent.

`[measured on the reference deployment]` Effective values on the serving unit before the fix:

```
StartLimitIntervalSec=600   StartLimitBurst=10   RestartSec=30   Restart=always
```

Ten restarts 30 s apart span **300 s**, which is inside the 600 s window. So a fast-failing boot
— a bad import, an `ExecStartPre` refusal, a CUDA init error — exhausts the budget in five
minutes and the card goes down indefinitely, carrying every agent session on it, with no retry
and no human necessarily watching.

**This is not hypothetical, and the near-miss is recorded in a sibling drop-in's own words:** the
lane *"crash-looped 9x on `ModuleNotFoundError` and survived only because the start-limit drop-in
happened to allow 10 starts over 600s — luck, not design."* Nine of ten. A margin of one.

**⛔ The trap was created by making the limiter "more generous".** Two drop-ins worked against
each other. The earlier one set `StartLimitBurst=5` with `RestartSec=5` — 25 s of restarts in a
300 s window, safe. A later one raised the burst 5 → 10 **and** raised `RestartSec` 5 s → 30 s.
Raising the retry interval is what pulled all ten starts inside the window. Changing two coupled
numbers in the direction that each *sounds* safer made the pair unsafe, and nothing in either
file says they interact.

### The fix

`StartLimitIntervalSec=0` **disables rate limiting entirely**, which with `Restart=always` means
the unit retries forever. For a lane that must never stop, that is correct and intended:

```ini
[Unit]
StartLimitIntervalSec=0
```

Forever-retrying is **loud** — every restart edge fires the incident capture in
[`observability.md`](observability.md), which writes an artifact and alerts — and it is
**self-healing** the moment the underlying fault clears. A latched `failed` state is silent and
permanent. Pick the loud one.

Three practical notes:

- **The filename must sort last.** Drop-ins apply in lexical order and the last setter of a
  scalar wins. If an existing drop-in sets the limiter, yours must sort after it —
  `99-zzzz-*.conf` sorts after `99-zzz-*.conf`. Verify the *effect*, never the file:

  ```sh
  systemctl --user show serving@gpu0.service -p StartLimitIntervalUSec -p StartLimitBurst
  ```

  Expect `StartLimitIntervalUSec=0` (or `infinity`, depending on systemd version). On the
  reference deployment this reads `StartLimitIntervalUSec=0` with `StartLimitBurst=10` still set
  — the burst is now irrelevant, which is the intended end state.

- **It changes nothing about the engine.** No model, flag, card, memory budget or `ExecStart` is
  touched. It changes only whether systemd is allowed to give up — which is why it can be
  installed on a live lane with a `daemon-reload` and no restart (see
  [`cutover.md`](cutover.md)).

- **⛔ `StartLimitIntervalSec` and `StartLimitBurst` belong in `[Unit]`.** Under `[Service]` they
  are silently ignored — see [§2 of `cutover.md`](cutover.md#2-verify-the-unit-and-every-live-drop-in-before-applying-anything)
  for the verifier output that catches it.

**Where the start limiter is *right*** is the inverse case, and the same deployment has one: a
CPU observer unit with `StartLimitBurst=3` over 15 s burned all three starts inside 31 s during a
cold boot waiting for a filesystem, was dead 5 m 48 s before the mount arrived, and stayed
`failed` until started by hand. The lesson there is not "disable the limiter" — it is that a
limiter plus a dependency that can be late equals an indefinite outage. Fix the dependency
(`RequiresMountsFor=`, which does work in a user unit: the user manager proxies system mount
units) **and** make the retry unbounded for anything load-bearing.

---

## 2. A `ConditionPathExists` skip is the quietest outage you can have

A drain flag is the right way to keep a lane down deliberately:

```ini
[Unit]
ConditionPathExists=!<stack>/state/NO_AUTO_SERVE
ConditionPathExists=!<stack>/state/SCIENCE_MODE
```

**But a failed condition is not a failure.** systemd records the unit as `inactive (dead)`, logs
one debug line, and moves on. Nothing alerts. `systemctl --user is-failed` is **clean**. No
incident artifact is written, because the engine never started and therefore never crashed. The
card is simply not serving, and every indicator is green.

`[audited on the reference deployment]` Nothing cleared that flag at boot, and no age ceiling or
staleness check existed anywhere in the stack. Three scripts set it; **one sets it deliberately
across a reboot** (a graceful-shutdown script, under the banner "inhibit auto-serve so nothing
races back up on boot"); only two cleared it, both interactive. So a flag abandoned by an
interrupted drain took the lane down through **every subsequent boot**, silently.

### The guard

A `Type=oneshot` unit, ordered **before** every condition-gated lane, that:

1. **always alerts when a flag is present**, cleared or not. Silence was the actual defect. *An
   inhibited lane the operator knows about is a decision; an inhibited lane nobody knows about is
   an outage.*
2. **retires a flag past a wall-clock age ceiling** (default **12 h**) — and
3. **retires it by renaming**, to `<name>.expired-<utc>`, never by deleting. The evidence
   survives and the action is reversible with one `mv`, which is what makes it acceptable to let
   a script touch operator state at all. The alert it sends includes the exact `mv` to undo it.
4. **reports, without changing, every other way the lane can be condition-skipped** — a
   deliberate mode flag, and a hand-editable serving-mode file whose stale value resolves to
   "disable this lane". Announcing is cheap; auto-correcting an operator's deliberate mode is not.

```
Env: NO_AUTO_SERVE_MAX_AGE_H   (default 12)
     SERVING_FLAG_GUARD_DRY_RUN=1   decide and log, change nothing
Exit: always 0
```

**⛔ Why a wall-clock age ceiling and not "clear anything set before this boot".** The
boot-relative test is the obvious design and it is wrong. The graceful-shutdown path arms the
flag **seconds** before the reboot, precisely to inhibit the *next* boot — so clearing anything
set in a previous boot defeats the one case the flag exists for. A wall-clock ceiling separates
the two correctly: a deliberate pre-reboot arm is minutes old and is honoured; a flag abandoned
by an interrupted drain is hours or days old and is retired.

### Three ordering properties, each of which is a bug if you miss it

- **`Before=` every condition-gated lane.** `ConditionPathExists` is evaluated when the unit's
  *job starts*, so retiring the flag afterwards is useless. Also `After=` whatever already
  reconciles boot state, so two things do not race to decide.
- **Nothing `Requires=` the guard, and it exits 0 on any internal error.** A bug in the guard
  must never be the reason serving did not start. The exception handler is explicit about it:

  ```python
  except Exception as e:
      say(f"unexpected error: {e} — exiting 0 so the boot is unaffected")
      sys.exit(0)
  ```

- **`RequiresMountsFor=` the filesystem the state directory lives on**, and
  `ConditionPathExists=` the guard's own script. A guard that runs before its state directory is
  mounted reads "no flag" and reports an all-clear — the exact failure it exists to prevent,
  inverted.

### ⛔ The symlink that hides a mount dependency

`[measured]` On one cold boot, `fsck` ran 6 m 38 s on a large array, `fstab` had `nofail`, so
`local-fs.target` was reached and the user manager launched every unit into a host where that
array was **not yet mounted**. The dependency was invisible in the unit text: a path under the
stack root — which reads as being on the *mounted* filesystem — was a **symlink** into the array
that was not, and the lane additionally reached it through an `EnvironmentFile=` that set
`PYTHONPATH`. Grep the unit for paths and you would conclude it needed nothing.
`RequiresMountsFor=` must name every filesystem the unit reaches **after symlink resolution**,
including through `EnvironmentFile=` and `PYTHONPATH`.

---

## 3. Four tmux facts that each turn a restore into a duplicator

tmux does not survive a reboot, so anything whose interactive sessions are load-bearing needs
something to recreate them. The component here recreates agent lanes from a two-line spec and
re-arms their goals. Its interface:

```
# lanes.conf — pipe-separated; '#' comments and blank lines ignored
#   tmux_session_name | agent_session_id | owner_label
<name> | <session-id> | <label>
```

Adding a lane needs no code change. Env seams: `HERMES_RESTORE_LANES`,
`HERMES_RESTORE_WAIT_SECS` (backend budget, default 1200), `HERMES_RESTORE_DRY_RUN=1`,
`HERMES_RESTORE_ADOPT=1` (also launch into an existing session whose lane process is dead and
whose pane is a bare shell — **off by default, because that pane may be a human's**),
`HERMES_RESTORE_NO_GOAL=1`. Plus test-only seams — a scratch tmux socket name and a substitute
launch command — so the production path is byte-identical when they are unset.

Its three safety properties are worth stating because they are what the traps below attack:

1. **It never touches a live pane.** A lane whose tmux session already exists is skipped
   outright: no `send-keys`, no respawn, no attach. Session names are matched with tmux's
   **exact `=` target form** — without it tmux falls back from an exact match to pattern and
   prefix matching, so a short session name can resolve to a *different* live session — and
   `new-session`
   is used **without `-A`** so it *fails* rather than attaching if the name appeared in the
   meantime. Running it twice with the lanes live is a no-op.
2. **It waits for a genuinely ready backend, not an open port** — the router's own health
   verdict for the pinned model, because the port answers long before the weights are in. A cold
   boot of this engine is ~285 s and a bad boot can crash-loop for several ~5-minute attempts, so
   the budget is generous **and** the unit is on a timer: giving up is always temporary.
3. **It never auto-resumes a human pause.** A goal paused by a real `Ctrl+C` is reported and left
   alone. That distinction is [client rule 14](../clients/README.md#14-classify-an-interrupt-by-provenance-or-your-autonomous-loop-will-stop-on-its-own)
   and it is the reason this script reads a goal's stored provenance rather than its status alone.

### (a) `TMUX_TMPDIR` must not include the `tmux-$UID` component

tmux appends it. The live socket is `$TMUX_TMPDIR/tmux-$UID/default`, so pointing `TMUX_TMPDIR`
at the directory that *contains* the socket — which reads correctly, and is the obvious mistake —
addresses `…/tmux-$UID/tmux-$UID/default`: a **different, empty** server.

`[measured]` With the wrong value, every `has-session` check reported "absent" while both lanes
were live. A boot run would have started a second tmux server and launched a **duplicate** agent
process for each session id. That is safety property 1 defeated silently, which is why the socket
is **asserted**, not assumed:

```bash
export TMUX_TMPDIR=${TMUX_TMPDIR:-/tmp}
readonly TMUX_SOCK="$TMUX_TMPDIR/tmux-$(id -u)/default"
```

and the unit sets it explicitly rather than inheriting whatever the boot environment has.

### (b) `tmux start-server` does not leave you a server

A tmux server with no sessions exits immediately, so pre-creating one is a no-op that then makes
every subsequent check fail. `[measured]` The first version of that helper reported *"cannot
reach or start a tmux server"* on a clean box and restored nothing — precisely the boot it exists
to handle. The only way to bring a server up is to create a **session**, so that is what happens:
the first `new-session` creates the server.

Liveness is therefore a question you ask the server, not the filesystem:

```bash
tmux_server_alive() { tmux list-sessions >/dev/null 2>&1; }
```

**And refuse only the genuinely ambiguous case.** A socket that exists but will not answer
`list-sessions` means the lanes may be live on a server you cannot see, and creating sessions
would duplicate them. That is the one state where the right action is to change nothing, alert,
and exit non-zero. No socket at all is *not* ambiguous — the first lane will create one.

### (c) The server inherits your flock fd and holds it forever

A bash `exec 9>lock` fd is **not** close-on-exec. The long-lived tmux **server** born under that
lock inherits fd 9 and holds the lock for the server's entire lifetime.

`[measured]` After the first run created the server, every later timer run exited *"another
instance is running"*. That silently converted a 10-minute retry timer into a **single
boot-time attempt**, removing the retry-until-the-backend-is-up property the timer existed to
provide. The symptom is indistinguishable from the lock working correctly.

Two halves to the fix, and you need both:

```bash
# close fd 9 across the exec that creates the long-lived server
tmux new-session -d -s "$name" -c "$HOME" 9>&-

# and cross-check the lock against a live process, rather than trusting it
exec 9>"$LOCK"
if ! flock -n 9; then
  if lock_held_by_live_restore; then       # pidfile -> /proc check
    log "another instance is running — exiting"; exit 0
  fi
  log "lock held, but not by a live instance — a tmux server that inherited fd 9."
  log "Continuing WITHOUT exclusivity; systemd will not start a second instance anyway."
fi
```

The second half matters because the first half only helps servers created *after* you deploy it.
A server started by the older build keeps the lock until it dies, so the recovery path has to
exist. Note what it leans on: systemd already guarantees one instance of the unit, so the flock
is defence in depth and degrading it is safe. Say that in the log line, or the next reader will
"fix" it back.

### (d) `IFS=$'\t' read` collapses runs of tabs

Tab is IFS whitespace, so a run of tabs is one separator and **an empty field shifts every later
column into the wrong variable**.

`[measured]` A four-field row with an empty `paused_reason` put the turn counter (`"0/400"`)
into the variable a safety gate read as the pause reason, and the verdict into the turns
variable. The gate still denied correctly — by luck, because its patterns do not match a turn
count — and **a safety gate must not depend on that.**

Use US (`0x1f`), which is not IFS whitespace, so empty fields are preserved:

```python
print("\x1f".join([status, paused_reason, f"{turns_used}/{max_turns}", verdict]))
```
```bash
IFS=$'\x1f' read -r status reason turns verdict < <(goal_row "$sid")
```

### The unit, and why each directive is there

```ini
[Unit]
# ⛔ DO NOT order After= the units that consume the sessions this creates. Their own
# drop-ins order themselves AFTER this unit, so naming them here makes an ORDERING
# CYCLE — systemd then breaks it by dropping an edge of its own choosing, so boot
# order is non-deterministic or a consumer silently fails to start.
# `systemd-analyze --user verify` names it:
#   "Found ordering cycle on <unit>/start ... Transaction order is cyclic"
After=network-online.target
Wants=network-online.target
ConditionPathExists=<lane spec>
ConditionPathExists=!<emergency stop flag>
StartLimitIntervalSec=0          # unattended recovery must never be given up on (§1)

[Service]
# ⛔ Type=simple, NOT oneshot. A oneshot that waits up to 20 min for the backend holds
# default.target "starting" for that whole window, and other units are ordered after it.
Type=simple
Environment=TMUX_TMPDIR=/tmp     # ⛔ not /tmp/tmux-$UID — see (a)
ExecStart=<restore script>
# ⛔ KillMode=process: belt-and-braces so `systemctl --user restart` of THIS unit can
# never reap the panes. (The server is also started in a transient scope — below.)
KillMode=process
Restart=no                       # the timer owns retry
```

```ini
[Timer]
# ⛔ THE TIMER IS WHY GIVING UP IS SAFE. A single boot-time attempt has to pick a
# backend-wait budget, and the lane can crash-loop for longer than any sane budget.
# Re-running makes every give-up temporary — and because the script is idempotent
# (a live lane is never touched) a steady-state pass is a no-op costing one
# has-session and one pgrep per lane. It also gives mid-day lane death a recovery
# path it did not have.
OnBootSec=4min
OnUnitActiveSec=10min
AccuracySec=30s
```

**⛔ The tmux *server* must escape the unit's cgroup.** Modern tmux is systemd-aware and puts
each **pane** in its own transient scope, so panes already escape. The **server** would not:
started from inside the service it lands in the service cgroup and dies with it under the default
`KillMode=control-group`, taking every pane's parent with it. So the first session is created
inside a transient scope:

```bash
systemd-run --user --scope --quiet --collect --unit="tmux-server-$(date -u +%Y%m%d%H%M%S)" \
  -- tmux new-session -d -s "$name" -c "$HOME" 9>&- \
  || tmux new-session -d -s "$name" -c "$HOME" 9>&-
```

**⛔ And re-arm the work, not just the pane.** `[measured]` After a reboot both lanes' goals read
`status=active, paused_reason=None` — a reboot does not change that row. So the goal reads
*active* while nothing at all is driving it, and the supervisor that sweeps for *paused* goals
correctly declines to touch it. Restoring the pane without re-arming the continuation leaves the
goal permanently dead with every indicator green. That is the same shape as
[client rule 14's defect 3 and rule 16's defect 3](../clients/README.md#16-delegate-the-wait-never-park-the-goal-on-it):
**a state transition that does not arm whatever is supposed to carry it forward.** A reboot is
just another such transition.

---

## 4. Never two writers: a single-writer guard that waits

A loop that appends to a JSONL ledger and rewrites a `latest.json` in place cannot have two
copies: the append interleaves and the in-place rewrite tears. The problem when you
*systemd-ify* something that has been hand-run in a terminal for months is that the hand-started
copy is live **right now** and must not be disturbed.

So the `ExecStartPre` **blocks** while a copy is running outside the unit's own cgroup:

```bash
while :; do
  foreign=""
  for p in $(pgrep -f 'monitor_loop' 2>/dev/null); do
    [ "$p" = "$$" ] && continue
    grep -q '<unit>\.service' "/proc/$p/cgroup" 2>/dev/null || foreign="$foreign $p"
  done
  [ -z "$foreign" ] && exit 0
  # announce once, then stay quiet
  sleep "${GUARD_INTERVAL:-30}"
done
```

Four properties:

- **It waits rather than failing.** Failing would mean a restart loop writing a refusal into the
  journal every `RestartSec`, indefinitely, for as long as that terminal lives. Waiting means one
  quiet `activating` state that resolves by itself the moment the hand-started copy exits — **and
  after a reboot there is no hand-started copy, so it returns immediately and the unit starts
  first time.** That is the whole trick: the guard makes "enable it now, it takes over later"
  safe.
- **`TimeoutStartSec=0` is required by it.** `ExecStartPre` is subject to the start timeout, and
  the 90 s default would kill the wait.
- **Cgroup membership is the ownership test**, not a pidfile and not a process name. "Is this
  process mine?" has exactly one reliable answer under systemd, and it is `/proc/<pid>/cgroup`.
- **⛔ It lives in a script, not inline in the unit.** systemd reads `%` as a specifier and `\.`
  as an escape, so the inline form failed verification with `Invalid slot` and refused to start
  the unit at all. Anything with a regex in it goes in a file.

---

## 5. Classifying a boot-crash recurrence by the *effective* environment

A root cause you have paid for is worth protecting. Once a boot failure has been diagnosed and
mitigated by an environment change, the **same symptom now has two very different meanings**, and
an alert that conflates them wastes the diagnosis:

| effective env | meaning | how loud |
|---|---|---|
| mitigation **not** in force | the known race. Expected until the real fix ships. | informational: *the lane retries by itself* |
| mitigation **in force** | **new and unexplained** — either the mitigation did not take on this start, or there is a second cause | ⛔ loud: *do not assume the known root cause* |

The watcher is a `Type=oneshot` on a timer, reading the engine log forward from a persisted byte
offset, and it is a **pure observer**: it starts nothing, stops nothing, sets no condition, and
touches no serving unit.

### ⛔ Read the effective value from `systemctl show`, never from a drop-in file

Drop-ins apply in lexical order and the last setter wins, so parsing one file can disagree with
what the next start will actually use:

```python
out = subprocess.run(["systemctl", "--user", "show", UNIT, "-p", "Environment", "--value"],
                     capture_output=True, text=True, timeout=20).stdout
value = None
for tok in out.split():
    if tok.startswith("MY_FLAG="):
        value = tok.split("=", 1)[1]        # last wins, matching systemd
return value if value is not None else "unset"
```

Report **`unset` distinctly from `0`**. An absent variable means the feature is off by default;
that is a different fact from somebody having explicitly disabled it, and only the second one
tells you the mitigation was applied.

### ⛔ The first run baselines at EOF. It does not scan history

`[measured on the staging dry-run]` Scanning from offset 0 found the **historical** assert — the
one that was already root-caused — and classified it against **today's** environment, reporting
an explained failure as "NEW, UNEXPLAINED" because the mitigation is in force now and was not
then. That is the one alert this watcher must never cry wolf on, because its whole value is that
the loud case is rare.

Baselining also removes a comparison you would otherwise have to get right: there is no need to
check a log timestamp against when the mitigation landed, because **every hit is by construction
from after the baseline**. The state file records `baselined_at`, `baseline_offset`, `offset`,
`last_scan`, `last_hit`, `total_hits`, and is written atomically (`tmp` + `os.replace`).

Handle a shrinking log — `if offset > size: offset = 0` — or a rotation makes the watcher
permanently blind. And exit 0 on any unexpected error: *a watcher must never be able to fail
anything else.*

### The timer must use a calendar anchor

```ini
[Timer]
OnCalendar=*:0/2
AccuracySec=15s
Persistent=true
```

**⛔ Not `OnBootSec=` + `OnUnitActiveSec=`.** A monotonic timer of that shape, once (re)started
long after boot, has `OnBootSec` already in the past and no last-activation anchor for
`OnUnitActiveSec`, so systemd computes **no next elapse and it never fires again**. `[measured]`
A sampler on this deployment sat at `active (elapsed)` with `Trigger: n/a`, having last written
a sample **24 h earlier**, and nobody noticed. A calendar anchor cannot do that. For anything
long-running, prefer a service with `Type=notify` + `WatchdogSec` over a timer altogether —
[`observability.md §1`](observability.md#1-a-sampler-that-cannot-die-quietly) says why.

---

## Installing a staged set

Everything above installs as three independent pieces with an `INSTALL.sh` that takes a subset,
prints the effective values afterwards, and ends by printing its own rollback. That pattern, and
the proof-of-no-op it uses, is [`cutover.md`](cutover.md).
