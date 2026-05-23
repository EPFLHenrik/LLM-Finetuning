python training/generate_answers.py data/preprocessing/mmlu_aux/train.jsonl --model-dir models/qwen3_thinking --output data/preprocessing/mmlu_aux/train_student.jsonl --hf-token <HF_TOKEN_REDACTED> --max-new-tokens 4096 --student

python training/generate_answers.py data/preprocessing/mmlu_aux/train.jsonl --model-dir models/teacher --output data/preprocessing/mmlu_aux/train_teacher.jsonl --hf-token <HF_TOKEN_REDACTED> --max-new-tokens 4096

python training/resample_wrong_answers.py data/preprocessing/ai2_arc_challenge/train_teacher.jsonl --model-dir models/teacher --output data/preprocessing/ai2_arc_challenge/teacher_corrected.jsonl --hf-token <HF_TOKEN_REDACTED> 

python training/resample_wrong_answers.py data/preprocessing/commonsense_qa/train_student.jsonl --model-dir models/qwen3_thinking --output data/preprocessing/commonsense_qa/student_corrected.jsonl --hf-token <HF_TOKEN_REDACTED> --student

python training/lora_post_train.py --dataset-file data/preprocessing/ai2_arc_challenge/merged_student_teacher.jsonl --model-folder models/qwen3_thinking --output-dir models/commonsense --run-name teacher_commonsense --wandb-key '<WANDB_API_KEY_REDACTED>' --merge-and-save --overwrite-output

python eval/validate_general_knowledge_vllm.py --model-dir models/commonsense --output results/val_samples_commonsense.jsonl

python scripts/evaluate_training.py --checkpoints-dir models/acr2 --validation-file data/preprocessing/ai2_arc_challenge/validation.jsonl --output-dir results/acr2_results --wandb-key '<WANDB_API_KEY_REDACTED>'

python scripts/download_hf.py --hf-token '<HF_TOKEN_REDACTED>' --dataset-path cais/mmlu --dataset-config auxiliary_train --output-dir data/datasets/train_datasets/cais__mmlu__auxiliary_train

python training/generate_answers.py data/preprocessing/mmlu_aux/train.jsonl --model-dir models/teacher --output data/preprocessing/mmlu_aux/train_teacher.jsonl --hf-token <HF_TOKEN_REDACTED> --max-new-tokens 4096