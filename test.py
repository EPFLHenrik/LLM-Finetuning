from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
tok.chat_template = open("multilingual/assets/chat_template_multilingual-nonthinking.jinja").read()

messages = [
    {"role": "user", "content": "What is the capital of France?\nA. Berlin\nB. Paris\nC. Rome\nD. Madrid"}
]

rendered = tok.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
print(rendered)