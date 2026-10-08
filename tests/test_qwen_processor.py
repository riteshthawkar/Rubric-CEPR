"""Check processor assembly without importing models or downloading weights."""
import sys
from types import SimpleNamespace

import pytest

from qwen_edit_project.utils.qwen_processor import load_qwen_edit_processor


def test_all_processor_components_use_the_same_revision_and_chat_template(monkeypatch):
    calls = []
    tokenizer = SimpleNamespace(chat_template="official-template")
    def loader(name, value):
        return SimpleNamespace(from_pretrained=lambda model, **kwargs: (
            calls.append((name, model, kwargs)) or value
        ))
    fake = SimpleNamespace(
        Qwen2Tokenizer=loader("tokenizer", tokenizer),
        Qwen2VLImageProcessor=loader("image", "image-processor"),
        Qwen2VLVideoProcessor=loader("video", "video-processor"),
        Qwen2VLProcessor=lambda **kwargs: kwargs,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake)
    processor = load_qwen_edit_processor("Qwen/Qwen-Image-Edit-2509", revision="fixed-revision", local_files_only=True)
    assert len(calls) == 3
    for _, model, kwargs in calls:
        assert model == "Qwen/Qwen-Image-Edit-2509"
        assert kwargs == {"subfolder": "processor", "revision": "fixed-revision", "local_files_only": True}
    assert processor["tokenizer"] is tokenizer
    assert processor["chat_template"] == "official-template"


def test_wrong_subfolder_fails_before_any_model_import(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ValueError, match="subfolder"):
        load_qwen_edit_processor("Qwen/Qwen-Image-Edit-2509", subfolder="text_encoder")
