# Adapter configs

- `adapter_baseline.json`: default first-run LoRA config shared by all experts.
- `1_adapter_underfitting.json`: use if both training and validation performance are weak; increases LoRA capacity.
- `2_adapter_overfitting.json`: use if training improves but validation/test performance stagnates or drops; adds regularization.
- `3_adapter_merge_interference.json`: use if individual experts work well but the merged model shows strong interference or forgetting.