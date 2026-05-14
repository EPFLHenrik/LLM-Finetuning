# debug
python -m multilingual.train.sft \
    --run_name debug_sft \
    --max_steps 20 \
    --eval_steps 10 \
    --save_steps 10 \
    --log_steps 1