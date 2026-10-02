#!/usr/bin/env bash
# Copyright © 2025 Ligandal, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# gpu_hwstate.sh -- capture the GPU hardware state that every benchmark number
# on this card is conditional on. Emits JSON on stdout. Run it before a
# benchmark and store it alongside the result.
#
# This is NOT part of either dashboard; it is the PCI-level companion to them,
# and the one place in this tree that goes below nvidia-smi to the device
# itself. Two settings change under a reboot and BOTH move throughput numbers:
#
#   * Resizable BAR (256 MB window vs full VRAM). Affects host->device transfer
#     and therefore model LOAD TIME off NVMe. There is NO --query-gpu field for
#     BAR1; it has to be parsed out of `nvidia-smi -q -d MEMORY`, and the
#     CAPABILITY itself only exists in lspci -- nvidia-smi shows the window it
#     got, not whether a bigger one was available.
#   * ECC. Reduces USABLE VRAM and costs a little memory bandwidth. nvidia-smi
#     exposes no ECC-overhead field either; it shows up as the gap between
#     memory.total and the driver-reserved figure, so BOTH are recorded and the
#     overhead is MEASURED rather than assumed to be "about 6%".
#
#   gpu_hwstate.sh [gpu-index]        # default: 0
#
# lspci needs no root for `-s`, but the Resizable BAR capability block is only
# printed with sufficient privilege; without it those two fields come back
# empty, which is reported as empty rather than guessed.
set -uo pipefail

idx="${1:-0}"
q() { nvidia-smi --id="$idx" --query-gpu="$1" --format=csv,noheader"${2:-}" 2>/dev/null | head -1; }

pci=$(q pci.bus_id | tr -d ' ')
name=$(q name)
drv=$(q driver_version)
tot=$(q memory.total ,nounits)
used=$(q memory.used ,nounits)
free=$(q memory.free ,nounits)
ecc_cur=$(q ecc.mode.current)
ecc_pend=$(q ecc.mode.pending)

memq=$(nvidia-smi --id="$idx" -q -d MEMORY 2>/dev/null)
reserved=$(echo "$memq" | grep -iE "^[[:space:]]*Reserved" | head -1 | grep -oE "[0-9]+" | head -1)
bar1_total=$(echo "$memq" | awk '/BAR1 Memory Usage/{f=1} f&&/Total/{print $3; exit}')

# lspci wants a domain-less selector (bb:dd.f); nvidia-smi gives 00000000:bb:dd.f
sel="${pci#0000????:}"; sel="${sel#0000:}"
lspci_bar1=$(lspci -vv -s "$sel" 2>/dev/null | awk '/Resizable BAR/{f=1} f&&/BAR 1:/{print; exit}')
lspci_region1=$(lspci -vv -s "$sel" 2>/dev/null | awk '/Region 1:/{print; exit}')
lspci_dev=$(lspci -s "$sel" 2>/dev/null | head -1)

python3 - "$name" "$drv" "$pci" "$tot" "$used" "$free" "$ecc_cur" "$ecc_pend" \
          "${reserved:-}" "${bar1_total:-}" "$lspci_bar1" "$lspci_region1" \
          "$lspci_dev" "$idx" <<'PY'
import json, sys, datetime
k = ["gpu_name","driver","pci_bus_id","memory_total_mib","memory_used_mib",
     "memory_free_mib","ecc_mode_current","ecc_mode_pending",
     "driver_reserved_mib","bar1_total_mib","lspci_resizable_bar1",
     "lspci_region1","lspci_device","gpu_index"]
d = dict(zip(k, [x.strip() for x in sys.argv[1:]]))
for f in ("memory_total_mib","memory_used_mib","memory_free_mib",
          "driver_reserved_mib","bar1_total_mib","gpu_index"):
    try: d[f] = int(d[f])
    except Exception: pass
d["captured_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
# usable VRAM is what every downstream budget is sized off
try:
    d["usable_vram_mib"] = d["memory_total_mib"] - (d.get("driver_reserved_mib") or 0)
except Exception:
    pass
d["_note"] = ("BAR1 and ECC overhead have no dedicated --query-gpu field; BAR1 is "
              "parsed from `nvidia-smi -q -d MEMORY` and cross-checked against "
              "lspci, ECC overhead is inferred from total vs driver-reserved. "
              "Empty lspci_* fields mean lspci could not read the capability "
              "block, not that the capability is absent.")
print(json.dumps(d, indent=2))
PY
