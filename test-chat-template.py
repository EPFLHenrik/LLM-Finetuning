from transformers import AutoTokenizer
from trl.chat_template_utils import get_training_chat_template

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
tokenizer.chat_template = open("multilingual/assets/chat_template.jinja").read()

# This is exactly what SFTTrainer calls — if it doesn't raise, you're good
template = get_training_chat_template(tokenizer)
print("Template OK:", template[:100])

# Also verify the mask is correct
sample = {
    "messages": [
        {"role": "system", "content": "test"},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "\\boxed{4}"},
    ]
}
out = tokenizer.apply_chat_template(
    sample["messages"],
    tokenize=True,
    return_assistant_tokens_mask=True,
    return_dict=True,
    add_generation_prompt=False,
)
print("Mask:", out["assistant_masks"])
print("Tokens:", tokenizer.convert_ids_to_tokens(out["input_ids"]))