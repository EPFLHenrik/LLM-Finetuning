# Mystery Machine — Multilingual SFT Dataset

This repository contains the processed source pools and generated SFT mixtures used to train the **Mystery Machine** multilingual model (CS-552, EPFL, Spring 2026).

The training data targets five languages — **Spanish, Italian, Russian, Hindi, Chinese** — with an emphasis on regional, professional, and culturally grounded knowledge.

---

## Repository structure

The `processed/` directory contains the processed source pools used by the SFT data builders.

```bash
processed/
├── ccse/                         # Spanish citizenship/culture MCQ dataset
├── ceval/                        # Chinese exam-style benchmark data
├── cmmlu/                        # Chinese MMLU-style benchmark data
├── cultsportqa/                  # Chinese/Hindi culture and sports QA
├── demre/                        # Spanish/Chilean DEMRE-style exam data
├── global_mmlu/                  # Multilingual MMLU-derived general/regional data
├── head_qa/                      # Spanish/Italian healthcare and exam QA
├── include/                      # Multilingual regional knowledge benchmark data
├── italian_civil_code/           # Italian Civil Code generated MCQs
├── mera/                         # Russian MERA benchmark data
├── milu/                         # Hindi MILU benchmark data
├── mmlu_prox/                    # Multilingual MMLU-ProX data
│   └── subsets/                  # 5k-per-language sampled train subsets
├── polizia_di_stato/             # Italian Polizia di Stato exam questions
├── quaderno_prefettizia/         # Italian prefettizia exam questions
├── rubq/                         # Russian RuBQ-derived regional QA
└── spain_boe_legislation/        # Spanish BOE legislation-derived MCQs
```

The `processed/` pools are the stable layer of this repository. Model-specific SFT train/validation mixtures are generated deterministically from them using the project builder scripts and JSON config files, making every training run fully reproducible without uploading each sampled mixture separately.

---

## Processed data sources

All datasets in `processed/` went through a shared cleaning pipeline before being used:

- **Deduplication** by source ID and prompt hash, across splits and datasets.
- **Corrupt input removal**: questions with missing answer keys, unparseable option lists, image/table references without context, or malformed Unicode were discarded.
- **Regional relevance filtering** (select datasets): some source pools contain a mix of globally generic and locally specific questions. For these, a teacher model was used to score and retain only questions that genuinely test regional, cultural, or country-specific knowledge, discarding questions that could be answered from general English-language world knowledge alone. This was applied to Global MMLU and INCLUDE subsets.
- **Format normalization**: all examples are stored as `{"question", "options": [...], "answer": "A"/"B"/..., "language", "dataset", "source_id", ...}`.

---

## Dataset descriptions

| Dataset | Language(s) | Domain | Source |
|---|---|---|---|
| `ccse` | Spanish | Citizenship & culture (CCSE exam) | [TODO: add source link] |
| `ceval` | Chinese | Academic exams (C-Eval benchmark) | [Huang et al., 2023](https://arxiv.org/abs/2305.08322) |
| `cmmlu` | Chinese | Academic & professional (CMMLU benchmark) | [Li et al., 2023](https://arxiv.org/abs/2306.09212) |
| `cultsportqa` | Chinese, Hindi | Culture & sports QA | [TODO: add source link] |
| `demre` | Spanish | Chilean university entrance exam (DEMRE/PSU/PDT) | [TODO: add source link] |
| `global_mmlu` | Russian, Chinese, Hindi, Italian, Spanish | Multilingual MMLU — general & regional subsets | [Singh et al., 2024](https://arxiv.org/abs/2412.03304) |
| `head_qa` | Spanish, Italian | Healthcare & professional licensing exams | [Vilares & Gómez-Rodríguez, 2019](https://arxiv.org/abs/1906.04701) |
| `include` | Russian, Chinese, Hindi, Italian, Spanish | Regional academic & professional exams | [Romanou et al., 2024](https://arxiv.org/abs/2411.14343) |
| `italian_civil_code` | Italian | Italian Civil Code MCQs (teacher-generated) | [TODO: add source link] |
| `mera` | Russian | Russian MERA benchmark (multi-subject) | [Fenogenova et al., 2024](https://arxiv.org/abs/2401.04531) |
| `milu` | Hindi | Hindi MILU benchmark (academic & professional) | [Ahuja et al., 2024](https://arxiv.org/abs/2404.16143) |
| `mmlu_prox` | Russian, Chinese, Hindi, Italian, Spanish | Multilingual MMLU-ProX (variable option counts) | [TODO: add source link] |
| `polizia_di_stato` | Italian | Italian Polizia di Stato entrance exam | [TODO: add source link] |
| `quaderno_prefettizia` | Italian | Italian prefettizia administrative exam | [TODO: add source link] |
| `rubq` | Russian | Russian RuBQ geography & culture QA | [Rybin et al., 2021](https://arxiv.org/abs/2005.10659) |
| `spain_boe_legislation` | Spanish | Spanish BOE legislation MCQs | [TODO: add source link] |


> **TODO:** add a `License` column to the table above for each dataset.
> **TODO:** mention which datasets were used for training, only for course purposes, but do not allow to be redistributed/trained on normally, hence cannot be uploaded here.
> **TODO:** add the configs used for training some of the models
> **TODO:** possibly add the distilled data, even tho not used.

---

### Notes per dataset

**`ccse`** — Questions from the CCSE (Conocimientos Constitucionales y Socioculturales de España), the official Spanish citizenship exam administered by the Ministry of Justice. Small dataset (~300 questions); used at full size given its high regional specificity. 

> **TODO:** add also the pdf that was used as the source of the dataset, used for parsing.

**`ceval` / `cmmlu`** — Two complementary Chinese academic benchmark pools covering elementary through university level subjects, professional certifications, and standardized exams. Both are filtered to remove questions redundant with each other and deduplicated by prompt hash.

**`cultsportqa`** — A culture and sports QA dataset covering Chinese and Hindi. Retained for its native regional signal on topics underrepresented in translated benchmarks.

**`demre`** — Questions derived from the Chilean DEMRE/PSU/PDT university entrance exam series. Small but high-quality regional source (~45 questions after cleaning); used at full size.

> **TODO:** add also the pdf that was used as the source of the dataset, used for parsing.

**`global_mmlu`** — The multilingual release of MMLU covering 42 languages. We use only the five target languages and apply regional relevance filtering to upweight the culturally sensitive subset, which better reflects genuine regional knowledge rather than translated generic academics.

**`head_qa`** — Spanish and Italian medical and professional licensing exam questions (MIR, pharmacy, nursing, psychology). Covers highly specialized domain vocabulary relevant to the Italian and Spanish professional licensing tracks.

**`include`** — The INCLUDE benchmark (Romanou et al., 2024), the largest multilingual exam collection to date: 197K QA pairs from 1,926 exams across 44 languages, 15 scripts, and 58 knowledge domains, covering academic and professional examinations. Native-speaker verified. We use a recarved split (eval-only) to avoid training on benchmark examples.

**`italian_civil_code`** — MCQs generated from the Italian Civil Code using a teacher model. Questions test knowledge of specific articles and legal provisions; generation was followed by manual spot-checking and deduplication.

**`mera`** — The Russian MERA benchmark, a multi-subject evaluation suite for Russian. We use a stratified subset across subjects (mamuramu, ruopenbookqa, ruworldtree, mathlogicqa), dropping non-MCQ subtasks (parus, rcb, rwsd, use).

**`milu`** — The Hindi MILU (Massive Indian Language Understanding) benchmark, covering academic and professional subjects in Hindi. Used as the primary Hindi training source.

**`mmlu_prox`** — A multilingual MMLU-ProX variant with variable option counts (up to 10 per question). Covers all five target languages. The `subsets/` subfolder contains 5k-per-language sampled train subsets used in training runs with per-source caps.

**`polizia_di_stato` / `quaderno_prefettizia`** — Questions from Italian public-sector entrance exams (Polizia di Stato police exam; prefettizia administrative career exam). Both cover Italian law, public administration, and civic knowledge with high regional specificity.

> **TODO:** add also the pdfs that were used as the sources of the dataset, used for parsing.

**`rubq`** — Russian geography and culture QA derived from the RuBQ dataset. Covers Russian history, geography, culture, and general knowledge in a short-answer / MCQ format.

**`spain_boe_legislation`** — MCQs derived from official Spanish legislation published in the Boletín Oficial del Estado (BOE). Covers constitutional law, administrative procedure, and public-sector regulations.

> **TODO:** add also the pdfs that were used as the sources of the dataset, used for parsing.

<!-- ---

## SFT mixture generation

The `sft/` directory contains the final train and validation JSONL files consumed by the training scripts. These are not stored permanently — they can always be regenerated from `processed/` using:

```bash
python -m multilingual.train.build_sft_data \
  --config configs/sft/v19.json \
  --overwrite

# With distractor augmentation:
python -m multilingual.train.build_sft_augmented_mcq \
  --config configs/sft/v19.json \
  --overwrite
```

Each config file in `configs/sft/` fully specifies the per-source caps, language weights, train/val split ratios, assistant format, and augmentation parameters for that training run. -->

---

## Citation

If you use any of the upstream datasets, please cite the original works listed in the table above. This repository is created for academic course purposes (CS-552, EPFL).