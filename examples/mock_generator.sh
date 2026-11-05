#!/usr/bin/env bash
# 一个不花钱的假生成器，用来把 product-lock 整条链路跑通。
# 它只答应契约要求的那几个动词，不调用任何外部服务。
#
#   export GENERATOR_CLI_REAL="$PWD/examples/mock_generator.sh"
set -euo pipefail

verb="${1:-}"; shift || true

case "$verb" in
  image|video)
    echo "mock: $verb accepted $(date -u +%H:%M:%S)" >&2
    echo '{"thread_id": "mock-thread-1", "run_id": "mock-run-1"}'
    ;;
  fetch)
    outdir="."
    while [ $# -gt 0 ]; do
      case "$1" in
        --outdir) outdir="${2:-.}"; shift 2 ;;
        *) shift ;;
      esac
    done
    mkdir -p "$outdir"
    echo "mock: writing placeholder into $outdir" >&2
    printf 'mock result\n' > "$outdir/mock_result.txt"
    echo '{"completed": true}'
    ;;
  balance)
    echo '{"credits": 9999}'
    ;;
  *)
    echo "mock: unknown verb '$verb'" >&2
    exit 2
    ;;
esac
