from __future__ import annotations

import argparse
import base64
import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

from qwen_edit_project.eval.imgedit_scores import extract_imgedit_average
from qwen_edit_project.eval.evaluation_contract import (
    OFFICIAL_OPENAI_BASE_URL,
    enforce_official_openai_endpoint,
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return data


def atomic_save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4, ensure_ascii=True)
    tmp_path.replace(path)


def image_to_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def extract_average(
    entry: object,
    *,
    edit_type: str | None = None,
    strict: bool = False,
) -> float | None:
    return extract_imgedit_average(entry, edit_type=edit_type, strict=strict)


def exception_payload(exc: BaseException) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "error_type": exc.__class__.__name__,
        "error": str(exc),
        "repr": repr(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }
    for attr in ("status_code", "request_id", "code", "param", "type"):
        value = getattr(exc, attr, None)
        if value is not None:
            payload[attr] = value
    response = getattr(exc, "response", None)
    if response is not None:
        payload["response_status_code"] = getattr(response, "status_code", None)
        try:
            payload["response_text"] = response.text
        except Exception:
            pass
    if exc.__cause__ is not None:
        payload["cause_type"] = exc.__cause__.__class__.__name__
        payload["cause"] = repr(exc.__cause__)
    if exc.__context__ is not None:
        payload["context_type"] = exc.__context__.__class__.__name__
        payload["context"] = repr(exc.__context__)
    return payload


def call_gpt(
    *,
    original_image_path: Path,
    result_image_path: Path,
    edit_prompt: str,
    edit_type: str,
    prompts: dict[str, str],
    model: str,
    base_url: str,
    temperature: float | None,
    timeout: float,
    max_retries: int,
) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for ImgEdit scoring")

    original_image_base64 = image_to_base64(original_image_path)
    result_image_base64 = image_to_base64(result_image_path)
    full_prompt = prompts[edit_type].replace("<edit_prompt>", edit_prompt)
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
    )
    request: dict[str, Any] = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": full_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{original_image_base64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{result_image_base64}"}},
                ],
            }
        ],
    }
    if temperature is not None:
        request["temperature"] = temperature
    response = client.chat.completions.create(
        **request,
    )
    return response.choices[0].message.content or ""


def process_one(
    *,
    key: str,
    item: dict[str, Any],
    result_img_folder: Path,
    origin_img_root: Path,
    prompts: dict[str, str],
    model: str,
    base_url: str,
    temperature: float | None,
    timeout: float,
    max_retries: int,
) -> tuple[str, str | dict[str, Any]]:
    try:
        result = call_gpt(
            original_image_path=origin_img_root / item["id"],
            result_image_path=result_img_folder / f"{key}.png",
            edit_prompt=item["prompt"],
            edit_type=item["edit_type"],
            prompts=prompts,
            model=model,
            base_url=base_url,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
        )
        return key, result
    except Exception as exc:
        payload = exception_payload(exc)
        print(
            "Error processing key "
            f"{key}: {payload.get('error_type')}: {payload.get('repr')} "
            f"cause={payload.get('cause')} context={payload.get('context')}",
            flush=True,
        )
        return key, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ImgEdit Basic-Bench with resumable OpenAI calls.")
    parser.add_argument("--result_img_folder", required=True)
    parser.add_argument("--edit_json", required=True)
    parser.add_argument("--origin_img_root", required=True)
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--prompts_json", required=True)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--base-url", default=OFFICIAL_OPENAI_BASE_URL)
    parser.add_argument("--require-official-endpoint", action="store_true")
    parser.add_argument(
        "--result-json",
        default=None,
        help="Optional raw-response destination; hardened wrappers use a fresh score staging path.",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--strict-response-parser",
        action="store_true",
        help="Only resume responses with the exact three official labels and integer scores in [1, 5].",
    )
    args = parser.parse_args()

    if args.require_official_endpoint:
        enforce_official_openai_endpoint(
            {"scoring": {"openai_base_url": args.base_url}},
        )

    result_img_folder = Path(args.result_img_folder)
    edit_infos = load_json(Path(args.edit_json))
    prompts = load_json(Path(args.prompts_json))
    origin_img_root = Path(args.origin_img_root)
    result_json_path = (
        Path(args.result_json) if args.result_json is not None else result_img_folder / "result.json"
    )
    results = load_json(result_json_path)

    pending = []
    for key, item in edit_infos.items():
        if not args.force and extract_average(
            results.get(key),
            edit_type=str(item.get("edit_type", "")),
            strict=args.strict_response_parser,
        ) is not None:
            continue
        pending.append((str(key), item))

    if not pending:
        print(f"All {len(edit_infos)} ImgEdit key(s) already have parseable scores.", flush=True)
        atomic_save_json(results, result_json_path)
        return

    print(
        f"Scoring {len(pending)} ImgEdit key(s) with {max(1, args.num_processes)} worker(s); "
        f"existing parseable={len(edit_infos) - len(pending)}.",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=max(1, args.num_processes)) as executor:
        futures = {
            executor.submit(
                process_one,
                key=key,
                item=item,
                result_img_folder=result_img_folder,
                origin_img_root=origin_img_root,
                prompts=prompts,
                model=args.model,
                base_url=args.base_url,
                temperature=args.temperature,
                timeout=args.timeout,
                max_retries=args.max_retries,
            ): key
            for key, item in pending
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing edits"):
            key, result = future.result()
            results[key] = result
            atomic_save_json(results, result_json_path)


if __name__ == "__main__":
    main()
