BASE=/scratch/workspace/emma/standard-project-m2-mystery-machine
CKPTS=$BASE/multilingual/results/checkpoints
EVAL=$BASE/multilingual/results/eval

for ADAPTER_NAME in \
    sft_multilingual_v11_distilled_only \
    sft_multilingual_v11_distilled_only_short \
    sft_multilingual_v11_distilled_only_warmup0p1; do

    ADAPTER=$CKPTS/$ADAPTER_NAME/best_adapter
    VERSION=v11
    MODELNAME=${ADAPTER_NAME#sft_multilingual_v11_}

    for TEMPLATE in base v1 v2; do
        case $TEMPLATE in
            base) TMPL_PATH=multilingual/assets/chat_template.jinja ;;
            v1)   TMPL_PATH=multilingual/assets/chat_template_multilingual-v1.jinja ;;
            v2)   TMPL_PATH=multilingual/assets/chat_template_multilingual-v2.jinja ;;
        esac

        python -m multilingual.eval.run_eval \
            --model_name $ADAPTER \
            --provided_only \
            --output_dir $EVAL/$VERSION/$MODELNAME/$TEMPLATE \
            --chat_template $TMPL_PATH
    done
done

for ADAPTER_NAME in \
    sft_multilingual_v12_distilled \
    sft_multilingual_v12_distilled_only_1e-4_warmup0p1 \
    sft_multilingual_v12_distilled_only_short; do

    ADAPTER=$CKPTS/$ADAPTER_NAME/best_adapter
    VERSION=v12
    MODELNAME=${ADAPTER_NAME#sft_multilingual_v12_}

    for TEMPLATE in base v1 v2; do
        case $TEMPLATE in
            base) TMPL_PATH=multilingual/assets/chat_template.jinja ;;
            v1)   TMPL_PATH=multilingual/assets/chat_template_multilingual-v1.jinja ;;
            v2)   TMPL_PATH=multilingual/assets/chat_template_multilingual-v2.jinja ;;
        esac

        python -m multilingual.eval.run_eval \
            --model_name $ADAPTER \
            --provided_only \
            --output_dir $EVAL/$VERSION/$MODELNAME/$TEMPLATE \
            --chat_template $TMPL_PATH
    done
done

ADAPTER=/scratch/workspace/emma/standard-project-m2-mystery-machine/multilingual/results/checkpoints/sft_multilingual_v14_distilled_2e-4_warmup0p05_short/individual/best_adapter
python -m multilingual.eval.run_eval \
    --model_name $ADAPTER \
    --provided_only \
    --output_dir multilingual/results/eval/v14/best_adapter \
    --chat_template multilingual/assets/chat_template_multilingual-v2.jinja
