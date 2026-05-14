# debug
python -m multilingual.train.sft \
    --run_name debug_sft \
    --max_steps 20 \
    --eval_steps 10 \
    --save_steps 10 \
    --log_steps 1


# full run
python -m multilingual.train.sft \
    --run_name sft_v1 \
    --eval_steps 250 \
    --save_steps 250 \
    --log_steps 10