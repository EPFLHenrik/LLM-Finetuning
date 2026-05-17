# debug
python -m multilingual.train.sft \
    --run_name debug_sft \
    --max_steps 20 \
    --eval_steps 10 \
    --save_steps 10 \
    --log_steps 1


# trial run, 10k-15k data train
python -m multilingual.train.sft \
  --run_name sft_multilingual_v1 \
  --epochs 1 \
  --batch_size 1 \
  --grad_accum 16 \
  --lr 2e-4 \
  --eval_steps 250 \
  --save_steps 250 \
  --log_steps 10

# longer, 30k train
python -m multilingual.train.sft \
  --run_name sft_multilingual_v2 \
  --epochs 3 \
  --batch_size 1 \
  --grad_accum 16 \
  --lr 2e-4 \
  --warmup_ratio 0.05 \
  --eval_steps 250 \
  --save_steps 250 \
  --log_steps 10