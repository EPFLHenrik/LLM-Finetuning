#!/bin/bash
# Evaluate all models only on the provided validation samples.
# Uses two chat templates: v2 first, then base.
#
# Folder layout:
#   eval/v1/base_v2/
#   eval/v1/base_base/
#   eval/v11/distilled_only_v2/
#   eval/v11/distilled_only_base/
#   etc.

set -euo pipefail

BASE=/scratch/workspace/emma/standard-project-m2-mystery-machine
CKPTS=$BASE/multilingual/results/checkpoints
EVAL=$BASE/multilingual/results/eval
ASSETS=$BASE/multilingual/assets

TMPL_V2=$ASSETS/chat_template_multilingual-v2.jinja
TMPL_BASE=$ASSETS/chat_template.jinja

# ---------------------------------------------------------------------------
# Model registry: checkpoint_dir | version_folder | subversion_folder
# ---------------------------------------------------------------------------
MODELS=(
    "$CKPTS/debug_v0/best_adapter|v0|debug"
    "$CKPTS/sft_multilingual_v1/best_adapter|v1|base"
    "$CKPTS/sft_multilingual_v10_distilled/best_adapter|v10|distilled"
    "$CKPTS/sft_multilingual_v10_distilled_short/best_adapter|v10|distilled_short"
    "$CKPTS/sft_multilingual_v11_distilled_only/best_adapter|v11|distilled_only"
    "$CKPTS/sft_multilingual_v11_distilled_only_warmup0p1/best_adapter|v11|distilled_only_warmup0p1"
    "$CKPTS/sft_multilingual_v11_distilled_only_short/best_adapter|v11|distilled_only_short"
    "$CKPTS/sft_multilingual_v12_distilled/best_adapter|v12|distilled"
    "$CKPTS/sft_multilingual_v12_distilled_only_1e-4_warmup0p1/best_adapter|v12|distilled_only_1e-4_warmup0p1"
    "$CKPTS/sft_multilingual_v12_distilled_only_short/best_adapter|v12|distilled_only_short"
    "$CKPTS/sft_multilingual_v13_distilled_1e-4_warmup0p1/individual/best_adapter|v13|individual_distilled_1e-4_warmup0p1"
    "$CKPTS/sft_multilingual_v14_distilled_1e-4_warmup0p1/individual/best_adapter|v14|individual_distilled_1e-4_warmup0p1"
    "$CKPTS/sft_multilingual_v14_distilled_2e-4_warmup0p05_short/individual/best_adapter|v14|individual_distilled_2e-4_warmup0p05_short"
    "$CKPTS/sft_multilingual_v15_distilled_5e-5_warmup0p1/individual/best_adapter|v15|individual_distilled_5e-5_warmup0p1"
)

# ---------------------------------------------------------------------------
# Helper: run validation-only eval
# ---------------------------------------------------------------------------
run_val_eval() {
    local adapter=$1
    local out_dir=$2
    local tmpl=$3

    mkdir -p "$out_dir"

    if [[ ! -f "$adapter/adapter_config.json" && ! -f "$adapter/config.json" ]]; then
        echo "[SKIP] Model not found: $adapter"
        return
    fi

    python -m multilingual.eval.run_eval \
        --model_name "$adapter" \
        --chat_template "$tmpl" \
        --output_dir "$out_dir" \
        --provided_only \
        --overwrite
}

# ---------------------------------------------------------------------------
# Main loop: v2 template first, then base template
# ---------------------------------------------------------------------------
for TMPL_NAME in v2 base; do
    case $TMPL_NAME in
        v2)   TMPL=$TMPL_V2 ;;
        base) TMPL=$TMPL_BASE ;;
    esac

    echo ""
    echo "================================================================"
    echo "=== Template: $TMPL_NAME ==="
    echo "================================================================"

    for ENTRY in "${MODELS[@]}"; do
        IFS='|' read -r ADAPTER VERSION SUBVERSION <<< "$ENTRY"

        SUBDIR="$EVAL/$VERSION/${SUBVERSION}_${TMPL_NAME}"

        echo ""
        echo "--- $VERSION / $SUBVERSION ($TMPL_NAME) ---"
        echo "    Adapter: $ADAPTER"
        echo "    Validation samples only..."

        run_val_eval "$ADAPTER" "$SUBDIR" "$TMPL"
    done
done

echo ""
echo "=== All validation-sample evaluations complete ==="
echo "Results saved under: $EVAL"