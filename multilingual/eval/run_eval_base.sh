python -m multilingual.eval.run_eval \
    --model_name Qwen/Qwen3-1.7B \
    --provided_only \
    --output_dir $EVAL/base/base \
    --chat_template multilingual/assets/chat_template.jinja

python -m multilingual.eval.run_eval \
    --model_name Qwen/Qwen3-1.7B \
    --provided_only \
    --output_dir $EVAL/base/v1 \
    --chat_template multilingual/assets/chat_template_multilingual-v1.jinja

python -m multilingual.eval.run_eval \
    --model_name Qwen/Qwen3-1.7B \
    --provided_only \
    --output_dir $EVAL/base/v2 \
    --chat_template multilingual/assets/chat_template_multilingual-v2.jinja