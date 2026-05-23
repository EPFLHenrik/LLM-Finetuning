python training/generate_answers.py data/preprocessing/ai2_arc_easy/train.jsonl --model-dir models/qwen3_thinking --output data/preprocessing/ai2_arc_easy/train_student.jsonl --hf-token <HF_TOKEN_REDACTED> --max-new-tokens 4096 --student

python training/generate_answers.py data/preprocessing/ai2_arc_easy/train.jsonl --model-dir models/teacher --output data/preprocessing/ai2_arc_easy/train_teacher.jsonl --hf-token <HF_TOKEN_REDACTED> --max-new-tokens 4096

python training/resample_wrong_answers.py data/preprocessing/ai2_arc_easy/train_teacher.jsonl --model-dir models/teacher --output data/preprocessing/ai2_arc_easy/teacher_corrected.jsonl --hf-token <HF_TOKEN_REDACTED> 

python training/resample_wrong_answers.py data/preprocessing/ai2_arc_easy/train_student.jsonl --model-dir models/qwen3_thinking --output data/preprocessing/ai2_arc_easy/student_corrected.jsonl --hf-token <HF_TOKEN_REDACTED> --student

python training/merge_reasoning_from_student_teacher.py data/preprocessing/ai2_arc_easy/student_corrected.jsonl data/preprocessing/ai2_arc_easy/teacher_corrected.jsonl --output data/preprocessing/ai2_arc_easy/merged_student_teacher.jsonl

python training/lora_post_train.py --dataset-file data/preprocessing/merged_distillation.jsonl --model-folder models/qwen3_thinking --output-dir models/merged --run-name training_merged --wandb-key '<WANDB_API_KEY_REDACTED>' --merge-and-save --overwrite-output

python eval/validate_general_knowledge_vllm.py --model-dir models/commonsense --output results/val_samples_commonsense.jsonl

python scripts/evaluate_training.py --checkpoints-dir models/merged --validation-file validation_samples/general_knowledge.jsonl --output-dir results/merged_model/gen_val --wandb-key '<WANDB_API_KEY_REDACTED>'

python scripts/download_hf.py --hf-token '<HF_TOKEN_REDACTED>' --dataset-path cais/mmlu --dataset-config auxiliary_train --output-dir data/datasets/train_datasets/cais__mmlu__auxiliary_train

python training/generate_answers.py data/preprocessing/mmlu_aux/train.jsonl --model-dir models/teacher --output data/preprocessing/mmlu_aux/train_teacher.jsonl --hf-token <HF_TOKEN_REDACTED> --max-new-tokens 4096

python scripts/push_model.py /scratch/workspace/henrik/standard-project-m2-mystery-machine/models/merged