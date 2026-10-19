#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

exec python -m gr00t.eval.run_gr00t_server \
    --embodiment-tag NEW_EMBODIMENT "$@"
