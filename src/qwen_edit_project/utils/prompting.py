from __future__ import annotations


def polish_prompt(prompt: str, use_prompt_polish: bool = False, image_context: object | None = None) -> str:
    """Keep polishing optional and off by default for benchmark reproducibility."""
    if not use_prompt_polish:
        return prompt
    prompt = prompt.strip()
    if image_context is None:
        return prompt
    return prompt

