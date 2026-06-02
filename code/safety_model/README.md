## 1. Datasets Downloading
`python -m code.safety_model.preprocessing.download_raw_datasets`

## 2. Datasets Preprocessing
#### Extraction and MCQ formatting
**Training Datasets**
`python -m code.safety_model.preprocessing.extraction.wildguardmix.preprocess`
`python -m code.safety_model.preprocessing.extraction.safetybench.preprocess`

**Benchmarks**
`python -m code.safety_model.preprocessing.extraction.advbench.benchmark` 
`python -m code.safety_model.preprocessing.extraction.jbb-behaviors.benchmark`
`python -m code.safety_model.preprocessing.extraction.sgxstest.benchmark`
`python -m code.safety_model.preprocessing.extraction.wildjailbreak.preprocess` // Training split not used

#### Distillation
``` 
python -m code.safety_model.preprocessing.distillation.reasoning_generation \
    --input data/safety/processed/sft/wildguardmix/binary_choice \
    --output data/safety/processed/sft/distillation/wildguardmix/binary_choice \
    --split train --from_id 0 --to_id 3000 && \

python -m code.safety_model.preprocessing.distillation.merge \ 
    data/safety/processed/sft/distillation/wildguardmix/binary_choice && \

cp -r \
    data/safety/processed/sft/wildguardmix/binary_choice/test \
    data/safety/processed/sft/distillation/wildguardmix/binary_choice/
```

``` 
python -m code.safety_model.preprocessing.distillation.reasoning_generation \
    --input data/safety/processed/sft/wildguardmix/subcategory_classification \
    --output data/safety/processed/sft/distillation/wildguardmix/subcategory_classification \
    --split train --from_id 0 --to_id 5000 && \
python -m code.safety_model.preprocessing.distillation.merge \
    data/safety/processed/sft/distillation/wildguardmix/subcategory_classification && \
cp -r \
    data/safety/processed/sft/wildguardmix/subcategory_classification/test \
    data/safety/processed/sft/distillation/wildguardmix/subcategory_classification/
```

``` 
python -m code.safety_model.preprocessing.distillation.reasoning_generation \
    --input data/safety/processed/sft/wildguardmix/safety_combo \
    --output data/safety/processed/sft/distillation/wildguardmix/safety_combo \
    --split train --from_id 0 --to_id 5000 && \
python -m code.safety_model.preprocessing.distillation.merge \
    data/safety/processed/sft/distillation/wildguardmix/safety_combo && \
cp -r \
    data/safety/processed/sft/wildguardmix/safety_combo/test \
    data/safety/processed/sft/distillation/wildguardmix/safety_combo/
```

``` 
python -m code.safety_model.preprocessing.distillation.reasoning_generation \
    --input data/safety/processed/sft/safetybench \
    --output data/safety/processed/sft/distillation/safetybench \
    --split train --from_id 0 --to_id 5000 && \
python -m code.safety_model.preprocessing.distillation.merge \
    data/safety/processed/sft/distillation/safetybench && \
cp -r \
    data/safety/processed/sft/safetybench/test \
    data/safety/processed/sft/distillation/safetybench/
```

## 3. Downloading Baseline Model (Qwen3-1.7B)
`python -m code.safety_model.huggingface.base_model_checkpoint_creation`

## 4. SFT
```
python -m code.safety_model.sft \
    --run_name safety_model/sft \
    --datasets \
        distillation/wildguardmix/binary_choice \
        distillation/wildguardmix/subcategory_classification \
        distillation/wildguardmix/safety_combo \
        distillation/safetybench \
    --train-sizes 3000 5000 5000 5000 \
    --epochs 5
```

## 5. DPO Pair Creation
```
python -m code.safety_model.preprocessing.dpo.pair_creation \
    --checkpoint checkpoints/safety_model/sft/hf_model \
    --dataset_path data/safety/processed/sft/wildguardmix/binary_choice \
    --output_path data/safety/processed/dpo/wildguardmix/binary_choice \
    --start_index 3000 \
    --end_index 6000
```

```
python -m code.safety_model.preprocessing.dpo.pair_creation \
    --checkpoint checkpoints/safety_model/sft/hf_model \
    --dataset_path data/safety/processed/sft/wildguardmix/subcategory_classification \
    --output_path data/safety/processed/dpo/wildguardmix/subcategory_classification \
    --start_index 5000 \
    --end_index 10000
```

```
python -m code.safety_model.preprocessing.dpo.pair_creation \
    --checkpoint checkpoints/safety_model/sft/hf_model \
    --dataset_path data/safety/processed/sft/wildguardmix/safety_combo \
    --output_path data/safety/processed/dpo/wildguardmix/safety_combo \
    --start_index 5000 \
    --end_index 10000
```

```
python -m code.safety_model.preprocessing.dpo.pair_creation \
    --checkpoint checkpoints/safety_model/sft/hf_model \
    --dataset_path data/safety/processed/sft/safetybench \
    --output_path data/safety/processed/dpo/safetybench \
    --start_index 0 \
    --end_index 10000
```

## 6. DPO training
```
python -m code.safety_model.dpo \
    --adapter-path checkpoints/safety_model/sft/LoRA \
    --base-model checkpoints/Qwen3-1.7B-baseline \
    --datasets \
        safetybench \
        wildguardmix/binary_choice \
        wildguardmix/subcategory_classification \ 
        wildguardmix/safety_combo \
    --best-checkpoint-dir checkpoints/safety_model/dpo \ 
    --drop-missing-chosen \
    --epochs 3 \
    --run_name dpo
```




