# dry run, 20 examples per dataset, all datasets
python -m multilingual.distillation.distill_fallback \
  --datasets ceval milu mera_mamuramu global_mmlu_regional include cultsportqa head_qa_spanish head_qa_italian \
  --processed_dir multilingual/data/processed \
  --output_dir multilingual/data/distilled/fallback/dryrun_v2 \
  --dry_run \
  --debug_n 10 \
  --concurrency 8 \
  --log_every 5 \
  --save_every 5 \
  --sample_temperature 0.25 \
  --top_p 0.95

# dry run
python -m multilingual.distillation.distill_fallback \
  --processed_dir multilingual/data/processed \
  --output_dir multilingual/data/distilled/fallback_dryrun_v3 \
  --dry_run \
  --debug_n 10 \
  --concurrency 8 \
  --log_every 5 \
  --save_every 5 \
  --sample_temperature 0.30 \
  --top_p 0.95


# full run
mkdir -p multilingual/data/distilled/v2_fallback
distill_datetime=$(date +"%Y%m%d_%H%M%S")
nohup python -m multilingual.distillation.distill_fallback \
  --processed_dir multilingual/data/processed \
  --output_dir multilingual/data/distilled/v2_fallback \
  --max_valid_per_file 4000 \
  --concurrency 40 \
  --log_every 40 \
  --save_every 40 \
  --sample_temperature 0.35 \
  --top_p 0.95 \
  > multilingual/data/distilled/v2_fallback/distill_${distill_datetime}.log 2>&1 &

tail -f "$(ls -t multilingual/data/distilled/v2_fallback/distill_*.log | head -n 1)" # monitor it

# specific datasets
distill_datetime_2=$(date +"%Y%m%d_%H%M%S")
nohup python -m multilingual.distillation.distill_fallback \
  --datasets global_mmlu_region_agnostic \
  --processed_dir multilingual/data/processed \
  --output_dir multilingual/data/distilled/v2_fallback \
  --max_valid_per_file 3000 \
  --concurrency 10 \
  --log_every 20 \
  --save_every 20 \
  --sample_temperature 0.35 \
  --top_p 0.95 \
  > multilingual/data/distilled/v2_fallback/distill_${distill_datetime_2}.log 2>&1 &
tail -f "$(ls -t multilingual/data/distilled/v2_fallback/distill_*.log | head -n 1)" # monitor it

# bash
bash python -m multilingual.distillation.distill_fallback \
  --datasets exams \
  --processed_dir multilingual/data/processed \
  --output_dir multilingual/data/distilled/v2_fallback \
  --max_valid_per_file 3000 \
  --concurrency 5 \
  --log_every 20 \
  --save_every 20 \
  --sample_temperature 0.35 \
  --top_p 0.95 \
  > multilingual/data/distilled/v2_fallback/distill_${distill_datetime_2}.log 2>&1 &
