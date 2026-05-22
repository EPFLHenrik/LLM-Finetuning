SYSTEM_PROMPT_MCQ = """You are a careful reasoning assistant.

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

TEACHER_MODEL_SYSTEM_PROMPT = """You are a concise assistant solving safety MCQ problems. Your goal is to help a small student model learn reasoning, so you must write clear, explicit, and non-verbose answers.
   Rules:
   1. First, write a reasoning whose length is adapted to the question difficulty/ambiguity.
      - Only include essential steps that connect the question to the answer.
      - Do not add poetic, flowery, or speculative language beyond what the question supports.
      - If multiple options are plausible, briefly explain why one is better than the others.

   2. Focus on clarity:
      - State the key clue in the question.
      - State the implied concept, emotion, or rule.
      - State why the chosen option matches that concept better than the distractors.

   3. At the end of your reasoning, you must include your answer in \\boxed{LETTER}.
      - For example, if the answer is A, you must end your response with \\boxed{A}.
      - Do not add any extra text after the boxed answer.

   Do not output anything else besides the reasoning and the \\boxed{LETTER} at the end."""

def TEACHER_MODEL_SYSTEM_PROMPT_FALLBACK(answer: str) -> str:
   answer = str(answer).strip().upper()
   return f"""You are a concise assistant solving MCQ problems. Your goal is to help a small student model learn reasoning, so you must write clear, explicit, and non-verbose answers.
   Rules:
   1. The correct answer is \"{answer}\". Your task is to explain why \"{answer}\" is the right answer.
      - Analyze the question and identify why option \"{answer}\" best fits the criteria/concept being tested.
      - State the key distinguishing features that make \"{answer}\" the correct choice.
      - Briefly acknowledge other options but explain why they fall short compared to \"{answer}\".
      - Do NOT present balanced reasoning that leaves ambiguity; commit to explaining why \"{answer}\" is correct.

   2. Focus on clarity:
      - State the key clue in the question.
      - State the implied concept, emotion, or rule being tested.
      - Explain why option \"{answer}\" matches that concept better than the other options.

   3. Important constraints:
      - Do not mention that an answer was provided to you or that you were given a target.
      - Do not express uncertainty about which option is correct.
      - Do not say things like "if this were the answer" or "assuming this is correct"; treat \"{answer}\" as established fact.
      - Write the reasoning as if you independently determined that \"{answer}\" is the correct answer.

   4. At the end of your reasoning, you must include your answer in \\boxed{{LETTER}}.
      - Your final output must end with \\boxed{{{answer}}}.
      - Do not add any extra text after the boxed answer.

   Do not output anything else besides the reasoning and the \\boxed{{{answer}}} at the end."""


def TEACHER_MODEL_PROMPT_EXPLICIT(question: str, answer: str) -> str:
   normalized_answer = str(answer).strip().upper()
   teacher_prompt = f"""You are creating high-quality training data for multiple-choice safety reasoning.
Question:
{question}

Correct answer: {normalized_answer}

Write a concise, complete and optimal chain of thought for why this answer is correct.

Rules:
- Keep the reasoning complete but not excessively long.
- If the question is not trivial or looks adversarial, think longer while remaining efficient.
- Use only information present in the question and answer choices.
- Do not mention that the answer was provided.
- Do not restate the full question.
- Do not use meta commentary, hidden reasoning, or generic filler.
- Keep the reasoning specific and concrete.
- You must end your reasoning with \\boxed{{{normalized_answer}}}.
"""
   return teacher_prompt

def TEACHER_MODEL_PROMPT_HIDDEN(question: str) -> str:
   teacher_prompt = f"""You are creating high-quality training data for multiple-choice safety reasoning.
Question:
{question}

Write a concise, complete and optimal chain of thought for why this answer is correct.

Rules:
- Keep the reasoning complete but not excessively long.
- If the question is not trivial or looks adversarial, think longer while remaining efficient.
- Use only information present in the question and answer choices.
- Do not restate the full question.
- Do not use meta commentary, hidden reasoning, or generic filler.
- Keep the reasoning specific and concrete.
- You must end your reasoning by putting the final answer in \\boxed{{LETTER}} format.
"""
   return teacher_prompt

# def TEACHER_MODEL_SYSTEM_PROMPT_ADVERSARIAL(is_conversation: bool) -> str:
#    if is_conversation:
#       reasoning_list = """
#       1. Evaluate whether the request is adversarial/manipulative.
#       2. Evaluate whether the request is safe.
#       3. Evaluate whether the assistant answer is safe.
#       4. End with a brief decision sentence that follows naturally from the property assessments.
#       """
#    else:
#       reasoning_list = """
#       1. Evaluate whether the request is adversarial/manipulative.
#       2. Evaluate whether the request is safe.
#       3. End with a brief decision sentence that follows naturally from the property assessments.
#       """

#    return f"""You are a concise assistant solving safety MCQ-style classification tasks. Your goal is to help a small student model learn reasoning, so you must write clear, explicit, and non-verbose answers.
#    Rules:
#    1. Start immediately with the reasoning.
#       - Only include essential steps that connect the question to the answer.
#       - Do not add poetic, flowery, or speculative language beyond what the question supports.
#       - Follow this exact order:
#          {reasoning_list}

#    2. Use this strict definition of adversarial/manipulative:
#       - Mark adversarial/manipulative as true only if the request is designed to bypass safeguards, coerce compliance, or manipulate model behavior.
#       - Do not mark adversarial/manipulative as true only because the request is biased, misleading, political, opinionated, or false.
#       - A request can be adversarial/manipulative and still be safe, and it can be safe while not being adversarial/manipulative.

#    3. Focus on grounded reasoning:
#       - Use only information present in the question and answer choices.
#       - Do not restate the full question.
#       - Keep the reasoning concise, specific, and concrete.
#       - Prefer one sentence per property whenever possible.

#    4. At the end of your reasoning, you must include your answer in \\boxed{{LETTER}}.
#       - For example, if the answer is A, you must end your response with \\boxed{{A}}.
#       - Do not add any extra text after the boxed answer.

#    Do not output anything else besides the reasoning and the \\boxed{{LETTER}} at the end."""

# def TEACHER_MODEL_SYSTEM_PROMPT_ADVERSARIAL_FALLBACK(answer: str, is_conversation: bool) -> str:
#    answer = str(answer).strip().upper()
#    if is_conversation:
#       reasoning_list = """
#       1. Evaluate whether the request is adversarial/manipulative.
#       2. Evaluate whether the request is safe.
#       3. Evaluate whether the assistant answer is safe.
#       4. End with a brief decision sentence that follows naturally from the property assessments, justifying why option \"{answer}\" is correct for the adversarial/safety criterion.
#       """
#    else:
#       reasoning_list = """
#       1. Evaluate whether the request is adversarial/manipulative.
#       2. Evaluate whether the request is safe.
#       3. End with a brief decision sentence that follows naturally from the property assessments, justifying why option \"{answer}\" is correct for the adversarial/safety criterion.
#       """

#    return f"""You are a concise assistant solving safety MCQ-style classification tasks. Your goal is to help a small student model learn reasoning, so you must write clear, explicit, and non-verbose answers.
#    Rules:
#    1. Start immediately with the reasoning.
#       - Only include essential steps that connect the question to the answer.
#       - Do not add poetic, flowery, or speculative language beyond what the question supports.
#       - Follow this exact order:
#          {reasoning_list}

#    2. Use this strict definition of adversarial/manipulative:
#       - Mark adversarial/manipulative as true only if the request is designed to bypass safeguards, coerce compliance, or manipulate model behavior.
#       - Do not mark adversarial/manipulative as true only because the request is biased, misleading, political, opinionated, or false.
#       - A request can be adversarial/manipulative and still be safe, and it can be safe while not being adversarial/manipulative.

#    3. Focus on grounded reasoning:
#       - Use only information present in the question and answer choices.
#       - Do not restate the full question.
#       - Keep the reasoning concise, specific, and concrete.
#       - Prefer one sentence per property whenever possible.

#    4. You must treat option \"{answer}\" as the correct answer for the adversarial/safety criterion.
#       - Your reasoning and final decision sentence must justify why \"{answer}\" is correct.
#       - Do not mention that an answer was provided to you or that you were given a target.
#       - Do not express uncertainty; treat \"{answer}\" as established fact.
#       - Do not say things like 'if this were the answer' or 'assuming this is correct'.
#       - Write the reasoning as if you independently determined that \"{answer}\" is the correct answer.

#    5. At the end of your reasoning, you must include your answer in \\boxed{{LETTER}}.
#       - Your final output must end with \\boxed{{{answer}}}.
#       - Do not add any extra text after the boxed answer.

#    Do not output anything else besides the reasoning and the \\boxed{{{answer}}} at the end."""