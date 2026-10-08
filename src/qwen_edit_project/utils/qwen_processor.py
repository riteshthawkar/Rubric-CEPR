"""Load the Qwen editor's declared processor components from one revision."""
from __future__ import annotations

from typing import Any


def load_qwen_edit_processor(
    model_id: str,
    *,
    subfolder: str = "processor",
    revision: str | None = None,
    local_files_only: bool = False,
    **kwargs: Any,
) -> Any:
    if subfolder != "processor":
        raise ValueError("The Qwen component loader requires subfolder='processor'")
    from transformers import (
        Qwen2Tokenizer, Qwen2VLImageProcessor, Qwen2VLProcessor, Qwen2VLVideoProcessor,
    )
    options = {"subfolder": subfolder, "revision": revision,
               "local_files_only": local_files_only, **kwargs}
    tokenizer = Qwen2Tokenizer.from_pretrained(model_id, **options)
    return Qwen2VLProcessor(
        tokenizer=tokenizer,
        image_processor=Qwen2VLImageProcessor.from_pretrained(model_id, **options),
        video_processor=Qwen2VLVideoProcessor.from_pretrained(model_id, **options),
        chat_template=tokenizer.chat_template,
    )
