# Data Pipeline

This folder contains scripts used to create and prepare multilingual training data before SFT.  
It covers three stages: parsing source files, generating MCQs with teacher models, and distilling rationales.

```
data_pipeline/
├── distillation/
│   ├── old/
│   │   └── distill_2stage.py                     # Deprecated two-stage distillation script.
│   ├── distill_1stage.py                         # One-stage teacher distillation with primary/fallback teacher.
│   ├── distill_fallback.py                       # Main fallback distillation pipeline for multilingual MCQ reasoning data.
│   └── distill_fallback.sh                       # Example commands for running fallback distillation.
├── generation/
│   ├── generate_italian_civil_code_mcq.py        # Generates and verifies MCQs from Italian Civil Code articles.
│   ├── generate_spain_boe_mcq.py                 # Generates and verifies MCQs from Spanish BOE legal articles.
│   └── run_mcq_generation.sh                     # Example dry-run and full-run commands for MCQ generation.
└── parsing/
    ├── parse_polizia_di_stato.py                 # Parses Italian Polizia di Stato MCQ PDF into processed JSONL.
    ├── parse_quaderno_prefettizia.py             # Parses Italian Quaderno Prefettizia MCQ PDF into processed JSONL.
    ├── parse_spain_boe_constitucion_espanola.py  # Parses the Spanish Constitution BOE PDF into article-level JSONL.
    ├── parse_spanish_regional_statutes.py        # Parses Spanish regional autonomy statute PDFs into article-level JSONL.
    ├── run_boe_parsing.sh                        # Example commands for parsing Spanish BOE legal PDFs.
    └── run_italian_mcq_parsing.sh                # Example commands for parsing Italian MCQ PDFs.
```

Pipeline overview:

1. Parsing
   Raw PDFs are converted into JSONL files. Legal PDFs are parsed article by article.
   MCQ PDFs are parsed into the standard processed MCQ format.

2. Generation
   Teacher models generate new MCQs from article-level legal sources.
   Generated questions are verified before being saved.

3. Distillation
   Teacher models generate reasoning traces for processed MCQ datasets.
   Valid examples are saved for downstream supervised fine-tuning.