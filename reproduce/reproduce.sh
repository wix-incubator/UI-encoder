#!/usr/bin/env bash
# Score the six released encoders on the test split, from the released dataset and weights.
#
#   bash reproduce/reproduce.sh <work-dir> [gpu-list]
#
# gpu-list defaults to 0,1,2,3. The reported runs embedded pages on 4 GPUs, and the probe's seed
# subsample depends on the page order that 4-way sharding produces, so always pass FOUR entries.
# With fewer GPUs, repeat ids: 0,0,0,0 (one GPU) or 0,1,0,1 (two GPUs).
#
# Environment overrides: DATASET (HF id or local parquet dir), WEIGHTS (local dir with one sub-dir
# per encoder, instead of downloading), BATCH_SIZE, WORKERS, MODELS_PER_PASS, BLAS_THREADS.
set -euo pipefail

# The probe's logistic-regression fit runs on CPU BLAS, whose sums depend on the thread count.
# A fixed count makes repeated runs on a machine give identical scores.
BLAS_THREADS=${BLAS_THREADS:-8}
export OPENBLAS_NUM_THREADS=$BLAS_THREADS MKL_NUM_THREADS=$BLAS_THREADS

WORK=${1:?usage: reproduce.sh <work-dir> [gpu-list]}
GPUS=${2:-0,1,2,3}
DATASET=${DATASET:-Wix/ui-components}
BATCH_SIZE=${BATCH_SIZE:-64}
WORKERS=${WORKERS:-32}
MODELS_PER_PASS=${MODELS_PER_PASS:-13}
REPO=$(cd "$(dirname "$0")/.." && pwd)
if [[ -d "$DATASET" ]]; then
  DATASET=$(cd -- "$DATASET" && pwd)
fi
if [[ -n "${WEIGHTS:-}" ]]; then
  WEIGHTS=$(cd -- "$WEIGHTS" && pwd)
fi
WORK=$(mkdir -p "$WORK" && cd "$WORK" && pwd)
ENCODERS=(random random-plus random-plus-bal ours ours-bal)

if [ "$(echo "$GPUS" | tr ',' '\n' | grep -c .)" != 4 ]; then
  echo "gpu-list must have 4 entries (repeat ids on smaller machines, e.g. 0,0,0,0)" >&2
  exit 2
fi

# 1. weights: download once, so the GPU workers load from local disk. The SigLIP2 base is fetched
#    even with WEIGHTS set: it is the frozen baseline and supplies the image processor.
if [ -z "${WEIGHTS:-}" ]; then
  WEIGHTS="$WORK/models"
  python - "$WEIGHTS" "${ENCODERS[@]}" <<'EOF'
import sys
from huggingface_hub import snapshot_download
for name in sys.argv[2:]:
    snapshot_download(f'Wix/ui-encoder-{name}', local_dir=f'{sys.argv[1]}/{name}')
EOF
fi
python -c "from huggingface_hub import snapshot_download; snapshot_download('google/siglip2-so400m-patch16-naflex')"
MODELS=(siglip2_base=google/siglip2-so400m-patch16-naflex)
for name in "${ENCODERS[@]}"; do MODELS+=("$name=$WEIGHTS/$name"); done

# 2. the probe: reads the released seed + test splits, embeds every page with every encoder,
#    fits on seed, scores test (components_masking/probes/probe.py)
cd "$REPO/components_masking"
python -u -m probes.probe \
  --dataset "$DATASET" --root "$WORK" \
  --models "${MODELS[@]}" \
  --gpus "$GPUS" --batch-size "$BATCH_SIZE" --workers "$WORKERS" \
  --models-per-pass "$MODELS_PER_PASS" \
  2>&1 | tee "$WORK/probe.log"

# 3. print the results as the README tables
python "$REPO/reproduce/show_results.py" --root "$WORK"
