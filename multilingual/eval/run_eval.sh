# run a test evaluation
python -m multilingual.eval.run_eval \
    --model_name Qwen/Qwen3-1.7B \
    --n_per_language 5 \
    --output_dir multilingual/results/test \
    --save_official_format \
    --overwrite

# score all with a loop
for f in multilingual/results/test/official/*.jsonl; do
    echo "=== $f ==="
    python -m evaluate.score --generations "$f" --benchmark multilingual
done