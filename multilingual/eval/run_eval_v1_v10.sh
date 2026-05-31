BASE=/scratch/workspace/emma/standard-project-m2-mystery-machine
CKPTS=$BASE/multilingual/results/checkpoints
EVAL=$BASE/multilingual/results/eval

declare -A ADAPTERS=(
    [v1]="$CKPTS/sft_multilingual_v1/best_adapter"
    [v5]="$CKPTS/sft_multilingual_v5-less-data/best_adapter"
    [v6]="$CKPTS/sft_multilingual_v6-ru-zh-heavy/best_adapter"
    [v8]="$CKPTS/sft_multilingual_v8_distilled/best_adapter"
    [v9]="$CKPTS/sft_multilingual_v9_distilled"
    [v10]="$CKPTS/sft_multilingual_v10_distilled_shorter/best_adapter"
)

for VERSION in v1 v5 v6 v8 v9 v10; do
    ADAPTER=${ADAPTERS[$VERSION]}

    for TEMPLATE in base v1 v2; do
        case $TEMPLATE in
            base) TMPL_PATH=multilingual/assets/chat_template.jinja ;;
            v1)   TMPL_PATH=multilingual/assets/chat_template_multilingual-v1.jinja ;;
            v2)   TMPL_PATH=multilingual/assets/chat_template_multilingual-v2.jinja ;;
        esac

        python -m multilingual.eval.run_eval \
            --model_name "$ADAPTER" \
            --provided_only \
            --output_dir "$EVAL/$VERSION/$TEMPLATE" \
            --chat_template "$TMPL_PATH"
    done
done