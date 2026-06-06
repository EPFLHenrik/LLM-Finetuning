# Distilled Data

This folder contains teacher-distilled MCQA examples used for supervised fine-tuning experiments.

Each file contains valid distilled examples for one dataset, language, or subset. The files follow the naming pattern:

```
<dataset>_<language_or_subset>_distilled.jsonl
```

## Distilled datasets

### Chinese

- ceval_chinese_regional_distilled.jsonl
- ceval_chinese_regional_implicit_distilled.jsonl
- cultsportqa_chinese_distilled.jsonl
- global_mmlu_chinese_regional_distilled.jsonl
- global_mmlu_chinese_region_agnostic_distilled.jsonl
- global_piqa_chinese_distilled.jsonl
- include_chinese_all_distilled.jsonl

### Hindi

- cultsportqa_hindi_distilled.jsonl
- global_mmlu_hindi_regional_distilled.jsonl
- global_mmlu_hindi_region_agnostic_distilled.jsonl
- global_piqa_hindi_distilled.jsonl
- include_hindi_all_distilled.jsonl
- milu_hindi_regional_distilled.jsonl
- milu_hindi_regional_implicit_distilled.jsonl

### Italian

- exams_italian_distilled.jsonl
- global_mmlu_italian_regional_distilled.jsonl
- global_mmlu_italian_region_agnostic_distilled.jsonl
- global_piqa_italian_distilled.jsonl
- head_qa_italian_distilled.jsonl
- include_italian_all_distilled.jsonl
- italian_civil_code_distilled.jsonl
- polizia_di_stato_distilled.jsonl
- quaderno_prefettizia_distilled.jsonl
- quaderno_prefettizia_economia_politica_distilled.jsonl

### Russian

- global_mmlu_russian_regional_distilled.jsonl
- global_mmlu_russian_region_agnostic_distilled.jsonl
- global_piqa_russian_distilled.jsonl
- include_russian_all_distilled.jsonl
- mera_russian_mamuramu_distilled.jsonl
- mera_russian_ruopenbookqa_distilled.jsonl
- mera_russian_ruworldtree_distilled.jsonl

### Spanish

- exams_spanish_distilled.jsonl
- global_mmlu_spanish_regional_distilled.jsonl
- global_mmlu_spanish_region_agnostic_distilled.jsonl
- global_piqa_spanish_distilled.jsonl
- head_qa_spanish_distilled.jsonl
- include_spanish_all_distilled.jsonl
- spain_boe_legislation_distilled.jsonl

## Notes

These files are distilled training data, not benchmark data. They should not be mixed with held-out benchmark examples.

The original processed datasets are stored separately under processed_data/ on Hugging Face and under multilingual/data/processed/ locally.