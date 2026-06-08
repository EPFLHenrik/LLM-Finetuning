"""Preprocess CEval into clean multilingual MCQA JSONL files."""
from pathlib import Path
import json
import pandas as pd
from datasets import get_dataset_config_names, load_dataset
from multilinguality_model.paths import get_processed_data_dir, get_data_dir

DATASET_NAME = "ceval/ceval-exam"
OUTPUT_DIR = get_processed_data_dir() / "ceval"
SPLITS_TO_LOAD = ["dev", "val", "test"]
VALID_ANSWERS = {"A", "B", "C", "D"}
SUBJECT_TO_CATEGORY = {
    "computer_network": "STEM",
    "operating_system": "STEM",
    "computer_architecture": "STEM",
    "college_programming": "STEM",
    "college_physics": "STEM",
    "college_chemistry": "STEM",
    "advanced_mathematics": "STEM",
    "probability_and_statistics": "STEM",
    "discrete_mathematics": "STEM",
    "electrical_engineer": "STEM",
    "metrology_engineer": "STEM",
    "high_school_mathematics": "STEM",
    "high_school_physics": "STEM",
    "high_school_chemistry": "STEM",
    "high_school_biology": "STEM",
    "middle_school_mathematics": "STEM",
    "middle_school_biology": "STEM",
    "middle_school_physics": "STEM",
    "middle_school_chemistry": "STEM",
    "plant_protection": "STEM",
    "environmental_impact_assessment_engineer": "STEM",
    "chinese_language_and_literature": "Humanities",
    "high_school_chinese": "Humanities",
    "modern_chinese_history": "Humanities",
    "high_school_history": "Humanities",
    "middle_school_history": "Humanities",
    "art_studies": "Humanities",
    "logic": "Humanities",
    "college_economics": "Social Science",
    "marxism": "Social Science",
    "mao_zedong_thought": "Social Science",
    "education_science": "Social Science",
    "teacher_qualification": "Social Science",
    "high_school_politics": "Social Science",
    "middle_school_politics": "Social Science",
    "high_school_geography": "Social Science",
    "middle_school_geography": "Social Science",
    "ideological_and_moral_cultivation": "Social Science",
    "civil_servant": "Social Science",
    "law": "Law & Professional",
    "legal_professional": "Law & Professional",
    "business_administration": "Law & Professional",
    "accountant": "Law & Professional",
    "tax_accountant": "Law & Professional",
    "basic_medicine": "Law & Professional",
    "clinical_medicine": "Law & Professional",
    "physician": "Law & Professional",
    "veterinary_medicine": "Law & Professional",
    "urban_and_rural_planner": "Law & Professional",
    "fire_engineer": "Law & Professional",
    "professional_tour_guide": "Other",
    "sports_science": "Other",
}

SUBJECT_TO_REGIONALITY = {
    "law": "regional",
    "legal_professional": "regional",
    "mao_zedong_thought": "regional",
    "marxism": "regional",
    "modern_chinese_history": "regional",
    "high_school_history": "regional",
    "middle_school_history": "regional",
    "high_school_politics": "regional",
    "middle_school_politics": "regional",
    "ideological_and_moral_cultivation": "regional",
    "civil_servant": "regional",
    "high_school_geography": "regional",
    "middle_school_geography": "regional",
    "professional_tour_guide": "regional",
    "accountant": "regional_implicit",
    "tax_accountant": "regional_implicit",
    "teacher_qualification": "regional_implicit",
    "education_science": "regional_implicit",
    "business_administration": "regional_implicit",
    "college_economics": "regional_implicit",
    "basic_medicine": "regional_implicit",
    "clinical_medicine": "regional_implicit",
    "physician": "regional_implicit",
    "veterinary_medicine": "regional_implicit",
    "urban_and_rural_planner": "regional_implicit",
    "fire_engineer": "regional_implicit",
    "environmental_impact_assessment_engineer": "regional_implicit",
    "plant_protection": "regional_implicit",
    "chinese_language_and_literature": "regional_implicit",
    "high_school_chinese": "regional_implicit",
    "art_studies": "regional_implicit",
    "sports_science": "regional_implicit",
    "advanced_mathematics": "region_agnostic",
    "probability_and_statistics": "region_agnostic",
    "discrete_mathematics": "region_agnostic",
    "logic": "region_agnostic",
    "college_physics": "region_agnostic",
    "high_school_physics": "region_agnostic",
    "middle_school_physics": "region_agnostic",
    "college_chemistry": "region_agnostic",
    "high_school_chemistry": "region_agnostic",
    "middle_school_chemistry": "region_agnostic",
    "high_school_biology": "region_agnostic",
    "middle_school_biology": "region_agnostic",
    "college_programming": "region_agnostic",
    "computer_network": "region_agnostic",
    "operating_system": "region_agnostic",
    "computer_architecture": "region_agnostic",
    "electrical_engineer": "region_agnostic",
    "metrology_engineer": "region_agnostic",
    "middle_school_mathematics": "region_agnostic",
    "high_school_mathematics": "region_agnostic",
}


def make_prompt(row):
    return (
        f"{row['question']}\n\n"
        f"A) {row['A']}\n"
        f"B) {row['B']}\n"
        f"C) {row['C']}\n"
        f"D) {row['D']}"
    )


def safe_id_part(text):
    return (
        str(text)
        .strip()
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .replace(":", "_")
    )


def make_record(row):
    subject = safe_id_part(row["subject"])
    source_split = safe_id_part(row["source_split"])
    source_idx = int(row["source_idx"])

    return {
        "id": f"ceval_chinese_all_{subject}_{source_split}_{source_idx}",
        "dataset": "ceval",
        "language": "chinese",
        "split": "all",
        "task_type": "mcq",
        "prompt": make_prompt(row),
        "answer": row["answer"],
        "metadata": {
            "subject": row["subject"],
            "category": row["category"],
            "regionality": row["regionality"],
            "source_split": row["source_split"],
            "source_idx": source_idx,
            "explanation": row.get("explanation", ""),
        },
    }


def validate_record(record):
    assert set(record) == {
        "id",
        "dataset",
        "language",
        "split",
        "task_type",
        "prompt",
        "answer",
        "metadata",
    }
    assert record["id"].startswith("ceval_chinese_all_")
    assert record["dataset"] == "ceval"
    assert record["language"] == "chinese"
    assert record["split"] == "all"
    assert record["task_type"] == "mcq"
    assert record["answer"] in VALID_ANSWERS
    assert "\n\nA) " in record["prompt"]
    assert "\nB) " in record["prompt"]
    assert "\nC) " in record["prompt"]
    assert "\nD) " in record["prompt"]

    metadata = record["metadata"]
    assert set(metadata) == {
        "subject",
        "category",
        "regionality",
        "source_split",
        "source_idx",
        "explanation",
    }
    assert metadata["category"] in {
        "STEM",
        "Humanities",
        "Social Science",
        "Law & Professional",
        "Other",
    }
    assert metadata["regionality"] in {
        "regional",
        "regional_implicit",
        "region_agnostic",
    }
    assert metadata["source_split"] in set(SPLITS_TO_LOAD)
    assert isinstance(metadata["source_idx"], int)


def write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_raw_rows():
    subjects = sorted(get_dataset_config_names(DATASET_NAME))
    missing_categories = sorted(set(subjects) - set(SUBJECT_TO_CATEGORY))
    missing_regionalities = sorted(set(subjects) - set(SUBJECT_TO_REGIONALITY))

    assert not missing_categories, f"Missing CEval categories: {missing_categories}"
    assert not missing_regionalities, f"Missing CEval regionality labels: {missing_regionalities}"

    rows = []
    failed_loads = []

    for subject in subjects:
        for source_split in SPLITS_TO_LOAD:
            try:
                ds = load_dataset(DATASET_NAME, subject, split=source_split)
            except Exception as exc:
                failed_loads.append((subject, source_split, repr(exc)))
                continue

            n_examples = len(ds)
            for idx, ex in enumerate(ds):
                rows.append(
                    {
                        "question": ex.get("question", ""),
                        "A": ex.get("A", ""),
                        "B": ex.get("B", ""),
                        "C": ex.get("C", ""),
                        "D": ex.get("D", ""),
                        "answer": ex.get("answer", ""),
                        "explanation": ex.get("explanation", ""),
                        "subject": subject,
                        "category": SUBJECT_TO_CATEGORY[subject],
                        "regionality": SUBJECT_TO_REGIONALITY[subject],
                        "source_split": source_split,
                        "source_idx": idx,
                        "n_examples_in_subject_split": n_examples,
                    }
                )

    if failed_loads:
        print(f"Warning: failed CEval loads: {len(failed_loads)}")

    return rows


def clean_dataframe(df):
    text_columns = [
        "question",
        "A",
        "B",
        "C",
        "D",
        "answer",
        "explanation",
        "subject",
        "category",
        "regionality",
        "source_split",
    ]

    for col in text_columns:
        df[col] = df[col].fillna("").astype(str).str.strip()

    df["answer"] = df["answer"].str.upper()

    df = df[
        (df["question"] != "")
        & (df["A"] != "")
        & (df["B"] != "")
        & (df["C"] != "")
        & (df["D"] != "")
        & (df["answer"].isin(VALID_ANSWERS))
    ].copy()

    dedup_columns = ["subject", "question", "A", "B", "C", "D"]
    df = df.drop_duplicates(subset=dedup_columns, keep="first").reset_index(drop=True)

    assert df.duplicated(subset=dedup_columns).sum() == 0
    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_rows = load_raw_rows()
    df = clean_dataframe(pd.DataFrame(raw_rows))
    records = [make_record(row) for _, row in df.iterrows()]

    ids = [record["id"] for record in records]
    assert len(ids) == len(set(ids)), "Duplicate CEval ids found."

    for record in records:
        validate_record(record)

    paths = {
        "all": OUTPUT_DIR / "ceval_chinese_all.jsonl",
        "regional": OUTPUT_DIR / "ceval_chinese_regional.jsonl",
        "regional_implicit": OUTPUT_DIR / "ceval_chinese_regional_implicit.jsonl",
        "region_agnostic": OUTPUT_DIR / "ceval_chinese_region_agnostic.jsonl",
    }

    write_jsonl(records, paths["all"])

    for regionality in ["regional", "regional_implicit", "region_agnostic"]:
        subset = [
            record
            for record in records
            if record["metadata"]["regionality"] == regionality
        ]
        write_jsonl(subset, paths[regionality])

    print(f"CEval: wrote {len(records)} records to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
