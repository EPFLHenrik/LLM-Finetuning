from constants.global_variables import DatasetFieldNames, DistillationFieldNames
import datasets
import shutil


RATIONALES_DS_PATH = "data/processed/sft/safetybench/rationales"
RATIONALES_DS_PATH_TEMP = "data/processed/sft/safetybench/rationales_temp"
COT_DS_PATH = "data/processed/sft/safetybench/CoT"
COT_DS_PATH_TEMP = "data/processed/sft/safetybench/CoT_temp"

rationales_ds = datasets.load_from_disk(RATIONALES_DS_PATH)
cot_ds = datasets.load_from_disk(COT_DS_PATH)


# change "prompt" to DatasetFieldNames.SFT_INPUT, "target" to DatasetFieldNames.SFT_TARGET, "expected_answer" to DatasetFieldNames.SFT_ANSWER, and "teacher_model" to DistillationFieldNames.TEACHER_MODEL
def rename_fields(example):
    return {
        DatasetFieldNames.SFT_INPUT: example["prompt"],
        DatasetFieldNames.SFT_TARGET: example["target"],
        DatasetFieldNames.SFT_ANSWER: example["expected_answer"],
        DistillationFieldNames.TEACHER_MODEL: example["teacher_model"]
    }

try:
    rationales_ds_new = rationales_ds.map(rename_fields, remove_columns=rationales_ds.column_names)
    rationales_ds_new.save_to_disk(RATIONALES_DS_PATH_TEMP)

    del rationales_ds
    datasets.load_from_disk(RATIONALES_DS_PATH_TEMP).save_to_disk(RATIONALES_DS_PATH)

    shutil.rmtree(RATIONALES_DS_PATH_TEMP)


except Exception as exc:
    print("Rationales Dataset already conform.")

try:
    cot_ds_new = cot_ds.map(rename_fields, remove_columns=cot_ds.column_names)
    cot_ds_new.save_to_disk(COT_DS_PATH_TEMP)

    del cot_ds
    datasets.load_from_disk(COT_DS_PATH_TEMP).save_to_disk(COT_DS_PATH)

    shutil.rmtree(COT_DS_PATH_TEMP)
except Exception as exc:
    print("CoT Dataset already conform.")


