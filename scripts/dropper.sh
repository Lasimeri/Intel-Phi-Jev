#!/usr/bin/env bash
# dropper.sh: run xks with the Phi backend (the payload, libggml_phi.so from
# the sibling repository) installed into it, so the subject's matrix
# multiplies are shared between this host and the cards.
#
#   scripts/dropper.sh [--no-offload] [--card N] [--verbose] <xks args...>
#
# By default the cards compute every row they hold (PHI_GGML_OFFLOAD=1: no
# judge keeping a tensor on the host, the rows paged out of the host after
# upload), so as much of the subject as fits on the cards runs on their
# VPUs. --no-offload restores the payload's default, where it times the
# host against the cards per tensor and keeps what the host does faster.
# --card N    one card only (default: every card that is up)
#
# The payload is delivered by the sibling's scripts/phi-ggml.sh, which
# starts each card's worker when it is not polling and names the library
# to ggml through GGML_BACKEND_PATH; xks loads it at open. One process at a
# time may use the cards: a second one frees the first's rows under it, so
# this refuses to start while another holder is running. See dropper.md.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
sibling=${PHI_AVX512_ROOT:-$HOME/Intel Phi AVX-512}
[ -x "$sibling/scripts/phi-ggml.sh" ] || {
    echo "$0: $sibling/scripts/phi-ggml.sh not found (set PHI_AVX512_ROOT)" >&2
    exit 1
}
pass=()
offload=1
while [ $# -gt 0 ]; do
    case "$1" in
        --offload) shift ;;
        --no-offload) offload=0; shift ;;
        --card|-c) pass+=(--card "$2"); shift 2 ;;
        --verbose|-v) pass+=(--verbose); shift ;;
        --) shift; break ;;
        *) break ;;
    esac
done
bin=${XKS_BIN:-$root/target/release/xks}
[ -x "$bin" ] || { echo "$0: $bin missing; build it: cargo build --release" >&2; exit 1; }
# Another holder of the cards: a llama.cpp program or xks already running
# under the payload. Matched on the environment, not the command line.
for pid in $(pgrep -u "$(id -u)" -x 'llama-server|llama-bench|llama-completion|xks' || true); do
    if tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -q '^GGML_BACKEND_PATH=.*libggml_phi'; then
        echo "$0: pid $pid ($(cat /proc/$pid/comm)) holds the cards; stop it first" >&2
        exit 1
    fi
done
[ "$offload" = 1 ] && export PHI_GGML_OFFLOAD=1
exec "$sibling/scripts/phi-ggml.sh" "${pass[@]}" "$bin" "$@"
