"""Paper-v1 loader compatibility entry point for the frozen LoRA trainer.

The Qwen-Image-Edit-2509 snapshot stores its Qwen2-VL tokenizer, image
processor, and video processor together in ``processor/`` without a model
``config.json``.  Transformers 5.8.1's generic ``ProcessorMixin`` loader asks
``AutoTokenizer`` for that absent model config.  Load the three declared
components directly instead, construct the same concrete processor, and then
delegate all training behavior to the hash-pinned trainer.
"""

from __future__ import annotations

from typing import Any

from transformers import (
    Qwen2Tokenizer,
    Qwen2VLImageProcessor,
    Qwen2VLProcessor,
    Qwen2VLVideoProcessor,
)

from qwen_edit_project.train import diffusers_qwen_edit_lora as frozen_trainer


class ProcessorFolderCompatibilityLoader:
    """Construct ``Qwen2VLProcessor`` from the official processor folder."""

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        subfolder: str,
        revision: str | None = None,
        local_files_only: bool = False,
        **kwargs: Any,
    ) -> Qwen2VLProcessor:
        if subfolder != "processor":
            raise ValueError(
                "The paper-v1 compatibility loader is restricted to subfolder='processor'"
            )
        component_kwargs = {
            "subfolder": subfolder,
            "revision": revision,
            "local_files_only": local_files_only,
            **kwargs,
        }
        tokenizer = Qwen2Tokenizer.from_pretrained(
            pretrained_model_name_or_path,
            **component_kwargs,
        )
        image_processor = Qwen2VLImageProcessor.from_pretrained(
            pretrained_model_name_or_path,
            **component_kwargs,
        )
        video_processor = Qwen2VLVideoProcessor.from_pretrained(
            pretrained_model_name_or_path,
            **component_kwargs,
        )
        return Qwen2VLProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=tokenizer.chat_template,
        )


def main() -> None:
    """Apply the loader-only compatibility shim and run the frozen trainer."""

    frozen_trainer.Qwen2VLProcessor = ProcessorFolderCompatibilityLoader
    frozen_trainer.main()


if __name__ == "__main__":
    main()
