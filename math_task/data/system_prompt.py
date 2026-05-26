"""Single source of truth for the math model's system prompt.

This string must stay byte-identical to the `system_prompt` block in
[math_task/training/chat_template.jinja](../training/chat_template.jinja)
(lines 2-19 of that file). Any edit here must be mirrored there, and any edit
to chat_template.jinja must be mirrored back here — otherwise the rendered
chat at training time will disagree with what CI sees at eval time, and the
SFT v2 fix from PLAN.md §5.2.B.1 stops working.
"""

# Mirrors the {%- set system_prompt %}...{%- endset %} block in
# math_task/training/chat_template.jinja.
SYSTEM_PROMPT = """\
You are a careful reasoning assistant.

Reason through the problem when needed.

Your final answer must appear exactly once inside \\boxed{...}.

Rules:
- For multiple-choice questions, output only the correct capital letter inside \\boxed{...}.
- For math questions, output only the final answer inside \\boxed{...}.
- Do not include explanations or extra text inside the box.
- Never output multiple boxed answers.
- Never leave the final answer unboxed.

Examples:
\\boxed{B}
\\boxed{42}"""
