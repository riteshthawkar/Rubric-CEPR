"""Processor compatibility entry point for the Qwen LoRA trainer.

The Qwen-Image-Edit-2509 snapshot stores its Qwen2-VL tokenizer, image
processor, and video processor together in ``processor/`` without a model
``config.json``.  Transformers 5.8.1's generic ``ProcessorMixin`` loader asks
``AutoTokenizer`` for that absent model config.  Load the three declared
components directly instead, construct the same concrete processor, and then
delegate all training behavior to the LoRA trainer.
"""

from __future__ import annotations

from typing import Any

from qwen_edit_project.train import diffusers_qwen_edit_lora as trainer
from qwen_edit_project.utils.qwen_processor import load_qwen_edit_processor


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
    ) -> Any:
        return load_qwen_edit_processor(
            pretrained_model_name_or_path, subfolder=subfolder,
            revision=revision, local_files_only=local_files_only, **kwargs,
        )


def main() -> None:
    """Apply the loader-only compatibility shim and run the LoRA trainer."""

    trainer.Qwen2VLProcessor = ProcessorFolderCompatibilityLoader
    trainer.main()


if __name__ == "__main__":
    main()
