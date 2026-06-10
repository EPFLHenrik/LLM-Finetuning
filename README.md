# CS-552 Final Project Code - Mystery Machine

Before proceeding further, ensure that all libraries present in [requirements.txt](requirements.txt) are installed on your machine.

## Repository structure

| Folder | Description |
|---|---|
| [code/safety_model/](code/safety_model/) | Fine-tuning pipeline for the safety specialist model |
| [code/multilinguality_model/](code/multilinguality_model/) | Fine-tuning pipeline for the multilingual specialist model |
| [code/math_model/](code/math_model/) | Fine-tuning pipeline for the math specialist model |
| [code/general_knowledge_model/](code/general_knowledge_model/) | Fine-tuning pipeline for the general knowledge specialist model |
| [code/group_model/](code/group_model/) | Routing and ensemble logic that combines the four specialist models |
| [data/](data/) | Raw and processed datasets for each specialist (`general_knowledge/`, `math/`, `multilinguality/`, `safety/`) |
| [checkpoints/](checkpoints/) | Saved model checkpoints produced by each specialist pipeline, organised by model name |
| [runs/](runs/) | Benchmark evaluation outputs |
| [final_report/](final_report/) | Project report |
| [M2-archive/](M2-archive/) | Archive of the M2 milestone codebase |

## Order of execution

**The four specialist models must be trained before running the group model.** Each specialist has its own self-contained pipeline; run them independently in any order. Once all four final checkpoints are available, follow the instructions in [code/group_model/README.md](code/group_model/README.md) to set up and run the group model.

## Specific READMEs
- Safety Model: [README.md](code/safety_model/README.md)
- Multilinguality Model: [README.md](code/multilinguality_model/README.md)
- Math Model: [README.md](code/math_model/README.md)
- General Knowledge Model: [README.md](code/general_knowledge_model/README.md)
- Group Model: [README.md](code/group_model/README.md)
