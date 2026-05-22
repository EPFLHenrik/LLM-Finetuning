from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-1.7B"

SYSTEM = "You are a multilingual reasoning assistant. Always enclose the final answer in \\boxed{}."
USER = "Qual è la capitale d'Italia?\nA) Milano\nB) Roma\nC) Napoli\nD) Torino"

ANSWER_ONLY = "\\boxed{B}"

COT_ANSWER = (
    "<think>\n"
    "La domanda chiede la capitale d'Italia. La capitale d'Italia è Roma, quindi l'opzione corretta è B.\n"
    "</think>\n\n"
    "\\boxed{B}"
)

EMPTY_THINK_ANSWER = "<think>\n\n</think>\n\n\\boxed{B}"


def show_case(tokenizer, name, messages, add_generation_prompt, enable_thinking_marker):
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }

    if enable_thinking_marker != "unset":
        kwargs["enable_thinking"] = enable_thinking_marker

    print("=" * 100)
    print(f"CASE: {name}")
    print(f"add_generation_prompt={add_generation_prompt}, enable_thinking={enable_thinking_marker}")
    print("-" * 100)

    text = tokenizer.apply_chat_template(messages, **kwargs)
    print(repr(text))
    print()
    print(text)
    print()


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    single_turn_answer_only = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER},
        {"role": "assistant", "content": ANSWER_ONLY},
    ]

    single_turn_cot = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER},
        {"role": "assistant", "content": COT_ANSWER},
    ]

    single_turn_empty = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER},
        {"role": "assistant", "content": EMPTY_THINK_ANSWER},
    ]

    inference_prompt = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER},
    ]

    multi_turn = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER},
        {"role": "assistant", "content": ANSWER_ONLY},
        {"role": "user", "content": "E qual è la capitale della Francia?\nA) Lyon\nB) Paris\nC) Marseille\nD) Nice"},
        {"role": "assistant", "content": "\\boxed{B}"},
    ]

    for enable_thinking in [True, False, "unset"]:
        show_case(
            tokenizer,
            "SFT single-turn answer-only",
            single_turn_answer_only,
            add_generation_prompt=False,
            enable_thinking_marker=enable_thinking,
        )

        show_case(
            tokenizer,
            "SFT single-turn CoT",
            single_turn_cot,
            add_generation_prompt=False,
            enable_thinking_marker=enable_thinking,
        )

        show_case(
            tokenizer,
            "SFT single-turn explicit empty think",
            single_turn_empty,
            add_generation_prompt=False,
            enable_thinking_marker=enable_thinking,
        )

        show_case(
            tokenizer,
            "Inference prompt only",
            inference_prompt,
            add_generation_prompt=True,
            enable_thinking_marker=enable_thinking,
        )

        show_case(
            tokenizer,
            "SFT multi-turn answer-only",
            multi_turn,
            add_generation_prompt=False,
            enable_thinking_marker=enable_thinking,
        )


if __name__ == "__main__":
    main()