import json
from transformers import AutoTokenizer
from trl.chat_template_utils import get_training_chat_template

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
tok.chat_template = get_training_chat_template(tok)

with open("multilingual/data/sft/v7/mixed/val_mixed.jsonl") as f:
    for i, line in enumerate(f):
        ex = json.loads(line)
        out = tok.apply_chat_template(
            ex["messages"],
            tokenize=True,
            return_assistant_tokens_mask=True,
            return_dict=True,
            add_generation_prompt=False,
        )
        n_train_tokens = sum(out["assistant_masks"])
        if n_train_tokens == 0:
            print(f"Example {i} has 0 assistant tokens! dataset={ex['dataset']} lang={ex['language']}")
            print(ex["messages"][2]["content"])
            break