# Cutover discipline

The engine side of this repository assumes a serving config change can be landed safely and does
not say how. This is the how. All of it is `systemctl --user` on a unit whose stated requirement
is *must never stop*, so every step is chosen to make the change inspectable **before** it takes
effect and reversible **after**.

Six rules. The last two are the ones most often skipped.

1. [Stage the change, with its own apply and rollback in its header](#1-stage-the-change-with-its-own-apply-and-rollback-in-its-header)
2. [Verify the unit and every live drop-in before applying anything](#2-verify-the-unit-and-every-live-drop-in-before-applying-anything)
3. [Drop-ins: last setter wins, so verify the effect and never the file](#3-drop-ins-last-setter-wins-so-verify-the-effect-and-never-the-file)
4. [Prove a config install changed nothing](#4-prove-a-config-install-changed-nothing)
5. [Read what is actually running from `/proc`, not from the unit](#5-read-what-is-actually-running-from-proc-not-from-the-unit)
6. [Drain with a flag the restart policy honours — never `stop`](#6-drain-with-a-flag-the-restart-policy-honours--never-stop)

---

## 1. Stage the change, with its own apply and rollback in its header

A serving drop-in lands as a **staged file that is not installed**, carrying in its own comment
header: why, the measured evidence, the exact apply commands, and the exact rollback. Not in a
wiki, not in a ticket — in the file, because the file is what the next person finds.

```ini
# ⛔⛔ STAGED, NOT INSTALLED. To apply:
#   cp <staged>/99-zzz-example.conf ~/.config/systemd/user/serving@gpu0.service.d/
#   systemctl --user daemon-reload
#   touch <stack>/state/NO_AUTO_SERVE            # so the kill does not auto-restart
#   systemctl --user kill -s SIGTERM serving@gpu0.service
#   rm -f <stack>/state/NO_AUTO_SERVE
#   systemctl --user start serving@gpu0.service
# ROLLBACK: mv this file to *.disabled && systemctl --user daemon-reload && same dance.
#   The previously-effective drop-in then takes effect again unchanged; nothing in it
#   is edited here.
```

Three properties of that header worth copying:

- **The rollback names what becomes effective again**, not just what is removed. "Remove this
  file" is not a rollback if you cannot say which of five drop-ins then wins the key (§3).
- **It states what the change does *not* touch.** The staged sets here open with an explicit
  negative list — no model, no flag, no card, no memory budget, no `ExecStart`, and in one case
  *"it never touches `<the drop-in owned elsewhere>`"*. A reviewer's first question is always
  blast radius, and answering it in the file is cheaper than answering it in person.
- **The evidence is in the header too.** Every drop-in in this deployment that changes a
  resilience parameter carries the measured incident that motivated it, with dates and numbers.
  Six months later that is the only thing standing between the change and someone "simplifying"
  it back.

### The installer

A staged set installs with a script that takes a **subset**, not an all-or-nothing:

```sh
INSTALL.sh          # everything
INSTALL.sh A        # just the start-limiter latch fix
INSTALL.sh B        # just the boot guard
```

with `set -euo pipefail`, and these properties:

- **Each piece is independent and labelled by what it changes**, e.g.
  *A — changes whether systemd may give up*, *C — pure observer, zero serving effect*. If a
  reviewer can only approve one, they can install one.
- **It prints the effective values immediately after installing**, so the install and its
  verification are one command (§3).
- **It dry-runs a new guard before enabling it** — `GUARD_DRY_RUN=1 … ; then one real run` — so
  today's state is announced once and the operator sees what the guard will say before it says it
  to a chat room at 04:00.
- **It baselines a new watcher *before* enabling its timer.** `[measured]` A watcher started with
  its timer first would have scanned log history and classified a historical, already-root-caused
  failure against *today's* environment, reporting it as new and unexplained — see
  [`reboot-survival.md §5`](reboot-survival.md#5-classifying-a-boot-crash-recurrence-by-the-effective-environment).
  One manual `start`, check the state file, *then* `enable --now` the timer.
- **It ends by printing its own rollback**, per piece, with the line *"None of these restarts the
  engine."* The rollback is most needed by whoever did not run the install.

---

## 2. Verify the unit and every live drop-in before applying anything

```sh
systemd-analyze --user verify ~/.config/systemd/user/serving@gpu0.service
```

`systemd-analyze verify` loads the unit **and its drop-ins** the way the manager would, so it
catches two whole classes that `systemctl status` never will. Both are silent in production.

**(a) A directive in the wrong `[section]`.** `[measured]` A probe unit with
`StartLimitIntervalSec` under `[Service]`:

```
zz-probe.service:6: Unknown key name 'StartLimitIntervalSec' in section 'Service', ignoring.
```

— and `verify` still **exits 0**, so the message is the entire signal; check the output, not the
status code. `StartLimitIntervalSec` and `StartLimitBurst` belong in `[Unit]`. So does
`RefuseManualStop`: the first version of this deployment's resilience drop-in put it under
`[Service]`, where it was *"silently accepted and does nothing"*, so the unit reported
`Restart=always` while remaining stoppable by anybody. The drop-in that fixed it records that in
its own header, which is why it is quotable.

**(b) An ordering cycle.** `[measured]` A restore unit named a consumer in its `After=`, while
that consumer's own drop-in already ordered itself `After=` the restore unit:

```
Found ordering cycle on <unit>.service/start
... Transaction order is cyclic
```

`Requires=`/`After=` cycles do not fail. **systemd breaks the cycle by dropping an edge of its own
choosing**, so the production result is non-deterministic boot ordering, or a unit that silently
does not start. The trap is that neither half of the cycle is wrong on its own, and the half you
are editing looks obviously correct — so this is only findable by loading the whole graph, which
is what `verify` does.

Run it against the **live** path with every live drop-in in place, not against the staged file in
isolation. A staged file can be flawless and the merged result cyclic.

---

## 3. Drop-ins: last setter wins, so verify the effect and never the file

Drop-ins apply in **lexical filename order** and the last setter of a scalar wins. A deployment
accumulates them — this lane carries ten:

```
05-require-mounts.conf  10-…  20-vram-gate.conf  80-resilient-gpu0.conf  85-…
98-…  99-oom-startlimit.conf  99-zz-…  99-zzz-…  99-zzzz-never-give-up.conf
```

Two consequences:

- **To override a key, your filename must sort after the current setter's.** `99-zzzz-` after
  `99-zzz-`. State in the header *which* file you are overriding and that the name is deliberate,
  or the next person renames it.
- **Never conclude anything from reading one file.** Ask the manager:

```sh
systemctl --user show serving@gpu0.service \
  -p DropInPaths -p StartLimitIntervalUSec -p StartLimitBurst \
  -p Restart -p RestartUSec -p RefuseManualStop
```

`DropInPaths` prints them in the order they were applied, which is the only authoritative answer
to "who set this". The same applies to the **environment**: read `-p Environment` and take the
last assignment to a variable, never parse a drop-in — a watcher that classifies a failure by a
flag's value gets this wrong in exactly the situation where it matters
([`reboot-survival.md §5`](reboot-survival.md#5-classifying-a-boot-crash-recurrence-by-the-effective-environment)).

---

## 4. Prove a config install changed nothing

`daemon-reload` restarts nothing, and a drop-in that touches only supervision parameters takes
effect at the unit's **next** start, whenever that happens to be. That is the property that makes
a resilience fix installable on a live serving lane — and it is a claim, so prove it rather than
asserting it:

```sh
# before
systemctl --user show serving@gpu0.service -p ExecMainStartTimestamp -p NRestarts
# ... install, daemon-reload ...
# after — both MUST be identical
systemctl --user show serving@gpu0.service -p ExecMainStartTimestamp -p NRestarts
```

`ExecMainStartTimestamp` unchanged means the process was never replaced; `NRestarts` unchanged
means systemd did not restart it either. Together they are a two-field proof that reads in one
line, and they belong in the installer's own output so the evidence is produced whether or not
anyone remembers to look.

`ActiveState` and `SubState` alongside them catch the third case — a unit that went
`active → activating → active` fast enough to look unchanged.

---

## 5. Read what is actually running from `/proc`, not from the unit

The unit text plus its drop-ins describe what the **next** start will use.
`/proc/<pid>/cmdline` describes what the running process was actually given. After a staged
install, a rollback, a restart from a different build, or an `ExecStartPre` that rewrites a
generated argument file, those differ — **and when they differ is exactly when you are reading
them.** So the incident artifact in [`observability.md`](observability.md) records *server flags
in force* from `/proc`, and any question of the form "was the mitigation actually on for that
boot?" is answered there, not from the config.

If the engine exposes its own argument parser as a library, parsing the argv you assembled through
it before installing is strictly better — it catches a typo'd or removed flag at install time
rather than at the next restart. The reference deployment does **not** do that; it reads `/proc`
after the fact, which catches the same class of error one restart later. Noted as the gap it is.

---

## 6. Drain with a flag the restart policy honours — never `stop`

A lane configured the way a must-never-stop lane should be —

```
Restart=always   RestartUSec=30s   RefuseManualStop=yes   StartLimitIntervalUSec=0
ConditionPathExists=!<stack>/state/NO_AUTO_SERVE
```

— **cannot be stopped.** `systemctl --user stop` is refused by `RefuseManualStop`, and anything
that kills the process is undone by `Restart=always` 30 s later. Which is correct, and it means
the only safe drain is the condition flag:

```sh
touch <stack>/state/NO_AUTO_SERVE                        # arm the inhibit
systemctl --user kill -s SIGTERM serving@gpu0.service    # the restart sees the condition and declines
# ... do the work ...
rm -f <stack>/state/NO_AUTO_SERVE                        # disarm
systemctl --user start serving@gpu0.service
```

Order matters: arm **before** the kill, or the automatic restart wins the race.

**⛔ And a drain flag must have an owner and an age ceiling, or it becomes a permanent outage.**
An interrupted drain — the operator's terminal closes between the kill and the `rm` — leaves the
flag armed, and `ConditionPathExists` then skips the lane on every subsequent boot **silently**:
`inactive (dead)`, one debug line, `is-failed` clean, no incident artifact, because the engine
never started and therefore never crashed. That is the failure the boot guard in
[`reboot-survival.md §2`](reboot-survival.md#2-a-conditionpathexists-skip-is-the-quietest-outage-you-can-have)
exists for, and it is the single most expensive item in this directory: a flag retires by
**renaming** past a wall-clock ceiling, and a present flag **always** alerts, cleared or not.

### Two more things a restart gate should and should not do

**Refuse to start into a card someone else is holding.** `[measured]` One lane burned two starts
and produced two multi-megabyte OOM tracebacks because the card was already full when it launched:
it asked CUDA for 47.69 GiB against 12.30 GiB free and died 23 s in, then did the same thing
again. Neither failure said **who** had the card — that took resolving three pids and checking
their cgroups. An `ExecStartPre` that checks free VRAM against the lane's known resident footprint
turns 23 s of weight loading and a 2 MB traceback into an immediate refusal that names the holder.

**⛔ But it must never reap.** The first diagnosis of that incident was *"orphaned scheduler
holding 77 GiB"* — the holder was in fact a healthy, **active** peer unit, and an auto-reap keyed
on "big process while we are failed" would have killed it. Distinguish **stranded** memory (owning
unit inactive, safe to reclaim) from a live **tenant** (another healthy unit), and have the start
gate only ever report and refuse. *A start gate that silently kills whatever is in its way is a
worse bug than the one it fixes.* Reaping, where it is genuinely needed, belongs in
`ExecStopPost` and must be **cgroup-scoped to the unit's own children** — which is also the only
form that is safe when, as happened on three deaths here, systemd could not kill the unit's cgroup
at all (`Failed to kill control group … Invalid argument`) and a scheduler child outlived the unit
still holding the card.

---

## The checklist

```
□ staged, not installed; header carries why + evidence + apply + rollback + what it does NOT touch
□ systemd-analyze --user verify  on the LIVE unit with every live drop-in   (read the output, not $?)
□ filename sorts after the current setter of every key you override
□ systemctl show -p DropInPaths -p <each key>        — the effect, never the file
□ ExecMainStartTimestamp + NRestarts identical before and after a config-only install
□ drain with the condition flag, armed BEFORE the kill, disarmed after
□ the flag has an age ceiling and an owner, or the next interrupted drain is a silent outage
□ /proc, not the unit text, for what the dead process was actually running
```
