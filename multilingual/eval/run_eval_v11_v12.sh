BASE=/scratch/workspace/emma/standard-project-m2-mystery-machine
CKPTS=$BASE/multilingual/results/checkpoints
EVAL=$BASE/multilingual/results/eval
ADAPTER=$CKPTS/sft_multilingual_v12_distilled_only_1e-4_warmup0p1/best_adapter

python -m multilingual.eval.run_eval \
    --model_name $ADAPTER \
    --provided_only \
    --output_dir $EVAL/v12/distilled_only_1e-4_warmup0p1-base \
    --chat_template multilingual/assets/chat_template.jinja

python -m multilingual.eval.run_eval \
    --model_name $ADAPTER \
    --provided_only \
    --output_dir $EVAL/v12/distilled_only_1e-4_warmup0p1-v1 \
    --chat_template multilingual/assets/chat_template_multilingual-v1.jinja

python -m multilingual.eval.run_eval \
    --model_name $ADAPTER \
    --provided_only \
    --output_dir $EVAL/v12/distilled_only_1e-4_warmup0p1-v2 \
    --chat_template multilingual/assets/chat_template_multilingual-v2.jinja