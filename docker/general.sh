python training/generate_answers.py data/preprocessing/mmlu_aux_filtered/train.jsonl --model-dir models/merged_science_finance --output data/preprocessing/mmlu_aux_filtered/train_student.jsonl --hf-token <HF_TOKEN_REDACTED> --max-new-tokens 4096 --student

python training/generate_answers.py data/preprocessing/mmlu_aux_filtered/train.jsonl --model-dir models/teacher --output data/preprocessing/mmlu_aux_filtered/train_teacher.jsonl --hf-token <HF_TOKEN_REDACTED> --max-new-tokens 4096

python training/resample_wrong_answers.py data/preprocessing/mmlu_aux_filtered/train_teacher.jsonl --model-dir models/teacher --output data/preprocessing/mmlu_aux_filtered/teacher_corrected.jsonl --hf-token <HF_TOKEN_REDACTED> 

python training/resample_wrong_answers.py data/preprocessing/mmlu_aux_filtered/train_student.jsonl --model-dir models/merged_science_finance --output data/preprocessing/mmlu_aux_filtered/student_corrected.jsonl --hf-token <HF_TOKEN_REDACTED> --student

python training/merge_reasoning_from_student_teacher.py data/preprocessing/science_qa/student_corrected.jsonl data/preprocessing/science_qa/teacher_corrected.jsonl --output data/preprocessing/science_qa/merged_student_teacher.jsonl

python training/lora_post_train.py --dataset-file data/preprocessing/mmlu_aux_filtered/merged_student_teacher.jsonl --model-folder models/merged_science_finance --output-dir models/merged_mmlu_aux --run-name training_mmlu_aux --wandb-key '<WANDB_API_KEY_REDACTED>' --merge-and-save --overwrite-output

python scripts/evaluate_training.py --checkpoints-dir models/merged_science_finance --validation-file data/preprocessing/ai2_arc_challenge/validation.jsonl --output-dir results/merged_science_finance_model/ai2_arc_challenge --wandb-key '<WANDB_API_KEY_REDACTED>'

python scripts/evaluate_training.py --checkpoints-dir models/merged_mmlu_aux --validation-file validation_samples/general_knowledge.jsonl --output-dir results/merged_mmlu_aux_model/gen_val --wandb-key '<WANDB_API_KEY_REDACTED>'

#python scripts/download_hf.py --hf-token '<HF_TOKEN_REDACTED>' --dataset-path cais/mmlu --dataset-config auxiliary_train --output-dir data/datasets/train_datasets/cais__mmlu__auxiliary_train

#python scripts/push_model.py /scratch/workspace/henrik/standard-project-m2-mystery-machine/models/merged

python training/grpo_train.py \
  --model-folder /scratch/workspace/henrik/standard-project-m2-mystery-machine/models/merged_mmlu_aux \
  --dataset-file /scratch/workspace/henrik/standard-project-m2-mystery-machine/data/preprocessing/merged_distillation.jsonl \
  --output-dir /scratch/workspace/henrik/standard-project-m2-mystery-machine/models/grpo \
  --use-lora \
  --overwrite-output \