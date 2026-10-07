from __future__ import annotations

import json
import logging
import math
import random
import re
import statistics
import gc
from typing import Any

from PIL import Image, ImageEnhance, ImageOps

from qwen_edit_project.self_evolve.edit_schema import (
    extract_json_object,
    normalize_structured_edit,
    proposal_definition_from_structured_edit,
    structured_edit_prompt,
)
from qwen_edit_project.self_evolve.image_metrics import (
    box_mask_from_boxes,
    changed_fraction,
    diff_mask,
    diff_region_statistics,
    edge_preservation_score,
    luminance_mean,
    luminance_std,
    masked_target_structure_preservation,
    masked_region_statistics,
    mean_absolute_difference,
    saturation_mean,
    warmth_score,
)
from qwen_edit_project.self_evolve.proposal_bank import PROPOSAL_BANK, available_proposals
from qwen_edit_project.self_evolve.types import EditProposal, ProposalDefinition, SolverResult, UnlabeledImageRecord
from qwen_edit_project.utils.prompting import polish_prompt
from qwen_edit_project.utils.qwen_pipeline import (
    extract_qwen_edit_understanding_features,
    extract_qwen_text_features,
    extract_qwen_vae_latents,
    load_qwen_edit_pipeline,
    render_edit,
)


LOGGER = logging.getLogger(__name__)

PROPOSAL_BY_ID = {proposal.operation_id: proposal for proposal in PROPOSAL_BANK}
INVERSE_OPERATION_MAP = {
    "brightness_up": "brightness_down",
    "brightness_down": "brightness_up",
    "saturation_up": "saturation_down",
    "saturation_down": "saturation_up",
    "contrast_up": "contrast_down",
    "contrast_down": "contrast_up",
    "warm_tone": "cool_tone",
    "cool_tone": "warm_tone",
}
INTERNAL_ONLY_METRICS = {"internal_prompt_gain", "semantic"}


DEFAULT_TEMPLATE_TARGET_BANK = {
    "replacement_objects": [
        "red ceramic cup",
        "blue canvas tote bag",
        "yellow tennis ball",
        "black camera",
        "silver flashlight",
        "orange traffic cone",
        "white flower vase",
        "striped scarf",
        "brown leather wallet",
        "small wooden box",
        "green glass bottle",
        "purple notebook",
        "gray headphones",
        "pink umbrella",
        "metal water bottle",
        "woven basket",
        "black baseball cap",
        "clear drinking glass",
        "white coffee mug",
        "blue toy car",
        "red apple",
        "yellow rubber duck",
        "small plant pot",
        "folded newspaper",
    ],
    "addition_objects": [
        "small red apple",
        "blue notebook",
        "white ceramic cup",
        "yellow tennis ball",
        "black camera",
        "silver keychain",
        "green glass bottle",
        "orange traffic cone",
        "folded newspaper",
        "small wooden box",
        "purple flower pot",
        "brown paper bag",
        "gray headphones",
        "striped scarf",
        "clear drinking glass",
        "metal water bottle",
        "small toy car",
        "pink umbrella",
        "woven basket",
        "black baseball cap",
    ],
    "colors": ["deep blue", "warm yellow", "matte black", "soft green", "bright red", "clean white", "muted purple", "burnt orange", "silver gray"],
    "materials": ["brushed metal", "polished wood", "matte ceramic", "dark leather", "woven fabric", "clear glass", "brushed steel", "smooth marble"],
    "attributes": [
        "a subtle striped pattern",
        "a glossy finish",
        "a matte finish",
        "a cleaner newer appearance",
        "a slightly brighter highlight",
        "a fine dotted pattern",
        "a soft fabric texture",
    ],
    "style_targets": ["watercolor painting", "cinematic film still", "soft pencil sketch", "vintage photo", "clean product photo", "comic book illustration"],
    "background_targets": ["soft garden background", "plain studio backdrop", "sunny outdoor background", "neutral indoor wall", "clean kitchen background"],
}


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _mean_changed_fraction_score(value: float, expected_range: tuple[float, float]) -> float:
    low, high = expected_range
    if low <= value <= high:
        return 1.0
    if value < low:
        return _clamp(1.0 - (low - value) / max(low, 1e-6))
    return _clamp(1.0 - (value - high) / max(1.0 - high, 1e-6))


def _apply_pillow_operation(image: Image.Image, operation_id: str) -> Image.Image:
    if operation_id == "brightness_up":
        return ImageEnhance.Brightness(image).enhance(1.25)
    if operation_id == "brightness_down":
        return ImageEnhance.Brightness(image).enhance(0.75)
    if operation_id == "saturation_up":
        return ImageEnhance.Color(image).enhance(1.35)
    if operation_id == "saturation_down":
        return ImageEnhance.Color(image).enhance(0.60)
    if operation_id == "contrast_up":
        return ImageEnhance.Contrast(image).enhance(1.35)
    if operation_id == "contrast_down":
        return ImageEnhance.Contrast(image).enhance(0.70)
    if operation_id == "warm_tone":
        red, green, blue = image.split()
        red = red.point(lambda value: min(255, int(value * 1.10 + 8)))
        blue = blue.point(lambda value: max(0, int(value * 0.92)))
        return Image.merge("RGB", (red, green, blue))
    if operation_id == "cool_tone":
        red, green, blue = image.split()
        red = red.point(lambda value: max(0, int(value * 0.92)))
        blue = blue.point(lambda value: min(255, int(value * 1.10 + 8)))
        return Image.merge("RGB", (red, green, blue))
    if operation_id == "grayscale":
        return ImageOps.grayscale(image).convert("RGB")
    raise ValueError(f"Unsupported pillow prototype operation: {operation_id}")


def _reverse_proposal(proposal: EditProposal) -> EditProposal | None:
    inverse_operation_id = proposal.definition.inverse_operation_id or INVERSE_OPERATION_MAP.get(
        proposal.definition.operation_id
    )
    if inverse_operation_id is None:
        return None
    inverse_definition = PROPOSAL_BY_ID[inverse_operation_id]
    return EditProposal(
        record_key=proposal.record_key,
        round_index=proposal.round_index,
        proposal_index=proposal.proposal_index,
        definition=inverse_definition,
        difficulty_level=proposal.difficulty_level,
        instruction=inverse_definition.instruction,
    )


class DifficultyController:
    def __init__(self, initial_level: int, min_level: int, max_level: int, promote_at: float, demote_at: float):
        self.level = initial_level
        self.min_level = min_level
        self.max_level = max_level
        self.promote_at = promote_at
        self.demote_at = demote_at

    def update(self, acceptance_rate: float) -> int:
        if acceptance_rate >= self.promote_at and self.level < self.max_level:
            self.level += 1
        elif acceptance_rate <= self.demote_at and self.level > self.min_level:
            self.level -= 1
        return self.level


class ScriptedProposer:
    def __init__(self, config: dict[str, Any]):
        self.families = config.get("families")
        self.operation_ids = config.get("operation_ids")
        self.family_policy = config.get("family_policy", "random")

    @staticmethod
    def _metadata_family_preferences(record: UnlabeledImageRecord) -> tuple[list[str], list[str]]:
        raw_families = record.metadata.get("edit_families") or record.metadata.get("families") or []
        if isinstance(raw_families, str):
            raw_families = [part.strip() for part in raw_families.split(",")]
        primary_family = record.metadata.get("primary_family")
        primary = []
        if isinstance(primary_family, str) and primary_family:
            primary.append(primary_family)
        families = [*primary]
        if isinstance(raw_families, list):
            families.extend(str(item) for item in raw_families if item)
        seen = set()
        ordered = []
        for family in families:
            if family not in seen:
                seen.add(family)
                ordered.append(family)
        return primary, ordered

    def propose(
        self,
        record: UnlabeledImageRecord,
        round_index: int,
        difficulty_level: int,
        proposals_per_image: int,
        seed: int,
    ) -> list[EditProposal]:
        candidates = available_proposals(difficulty_level, families=self.families)
        if self.operation_ids:
            operation_filter = set(self.operation_ids)
            candidates = [candidate for candidate in candidates if candidate.operation_id in operation_filter]
        if not candidates:
            return []
        stable_key_seed = sum(ord(char) for char in record.key)
        rng = random.Random(seed + round_index * 100_003 + stable_key_seed)
        primary_families, metadata_families = self._metadata_family_preferences(record)
        metadata_family_set = set(metadata_families)
        if self.family_policy == "metadata_preferred" and metadata_family_set:
            primary = [candidate for candidate in candidates if candidate.family in set(primary_families)]
            preferred = [
                candidate
                for candidate in candidates
                if candidate.family in metadata_family_set and candidate.family not in set(primary_families)
            ]
            fallback = [candidate for candidate in candidates if candidate.family not in metadata_family_set]
            rng.shuffle(primary)
            rng.shuffle(preferred)
            rng.shuffle(fallback)
            selected = (primary + preferred + fallback)[:proposals_per_image]
        elif proposals_per_image >= len(candidates):
            selected = candidates[:]
            rng.shuffle(selected)
        else:
            selected = rng.sample(candidates, proposals_per_image)
        return [
            EditProposal(
                record_key=record.key,
                round_index=round_index,
                proposal_index=index,
                definition=proposal,
                difficulty_level=difficulty_level,
                instruction=proposal.instruction,
                structured_edit=normalize_structured_edit(
                    {},
                    instruction=proposal.instruction,
                    family=proposal.family,
                ),
            )
            for index, proposal in enumerate(selected)
        ]


class InternalQwenProposer:
    def propose(
        self,
        record: UnlabeledImageRecord,
        round_index: int,
        difficulty_level: int,
        proposals_per_image: int,
        seed: int,
    ) -> list[EditProposal]:
        raise NotImplementedError(
            "internal_qwen proposer is not available yet. "
            "The control loop is implemented, but the public Qwen edit pipeline does not expose "
            "the internal understanding branch as a standalone proposer API."
        )


class TrainableQwenVLProposer:
    """Round-updatable Qwen-Image-Edit VLM proposer.

    The proposer emits a structured edit JSON object using the autoregressive VLM/text-encoder
    component from the edit checkpoint. The diffusion editor itself cannot emit text; using the
    edit model's VLM component keeps the proposer tied to the same model family while training a
    separate proposer LoRA that is never used for final editor evaluation.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.model_name_or_path = str(config.get("model_name_or_path", "Qwen/Qwen-Image-Edit-2509"))
        self.model_subfolder = config.get("model_subfolder", "text_encoder")
        self.processor_subfolder = config.get("processor_subfolder", "processor")
        self.model_class = str(config.get("model_class", "qwen2_5_vl"))
        self.checkpoint_path = config.get("checkpoint_path")
        self.device = config.get("device", "auto")
        self.torch_dtype = config.get("torch_dtype", "auto")
        self.local_files_only = bool(config.get("local_files_only", False))
        self.generation = dict(config.get("generation", {}))
        self.proposal_mode = str(config.get("proposal_mode", "hybrid")).strip().lower()
        self.vlm_only = self.proposal_mode in {
            "vlm_only",
            "qwen_vlm_only",
            "learned_only",
            "trainable_vlm_only",
        }
        self.fallback_on_error = False if self.vlm_only else bool(config.get("fallback_on_error", True))
        self.scripted_probability = 0.0 if self.vlm_only else float(config.get("scripted_probability", 0.25))
        self.allowed_edit_types = self._string_set(config.get("allowed_edit_types") or config.get("focus_edit_types"))
        self.disallowed_edit_types = self._string_set(
            config.get("disallowed_edit_types") or config.get("avoid_edit_types")
        )
        self.enforce_record_target_edit_type = bool(config.get("enforce_record_target_edit_type", False))
        self.template_fallback_on_target_miss = (
            False if self.vlm_only else bool(config.get("template_fallback_on_target_miss", False))
        )
        self.force_template_from_record = False if self.vlm_only else bool(config.get("force_template_from_record", False))
        self.template_bootstrap_rounds = 0 if self.vlm_only else max(0, int(config.get("template_bootstrap_rounds", 0)))
        self.strict_edit_type_filter = bool(config.get("strict_edit_type_filter", False))
        self.strict_structured_proposal_filter = bool(
            config.get("strict_structured_proposal_filter", self.vlm_only)
        )
        self.require_explicit_state_transitions = bool(
            config.get("require_explicit_state_transitions", False)
        )
        self.local_target_disallowed_heads = self._string_set(
            config.get(
                "local_target_disallowed_heads",
                [
                    "background",
                    "foreground",
                    "sky",
                    "wall",
                    "floor",
                    "ceiling",
                    "ground",
                    "road",
                    "street",
                    "sidewalk",
                    "pavement",
                    "grass",
                    "lawn",
                    "water",
                    "ocean",
                    "sea",
                    "room",
                    "scene",
                    "image",
                    "photo",
                    "picture",
                    "landscape",
                    "porch",
                    "deck",
                    "patio",
                    "driveway",
                    "path",
                    "hallway",
                    "corridor",
                    "facade",
                    "exterior",
                    "interior",
                ],
            )
        )
        self.max_generation_attempts = max(1, int(config.get("max_generation_attempts", 1)))
        self.template_target_bank = {key: list(value) for key, value in DEFAULT_TEMPLATE_TARGET_BANK.items()}
        for key, value in dict(config.get("template_target_bank", {})).items():
            if isinstance(value, str):
                values = [part.strip() for part in value.split(",") if part.strip()]
            else:
                values = [str(item).strip() for item in value if str(item).strip()]
            if values:
                self.template_target_bank[str(key)] = values
        fallback_config = dict(config.get("scripted_fallback", {}))
        fallback_config.setdefault("families", config.get("families"))
        fallback_config.setdefault("operation_ids", config.get("operation_ids"))
        fallback_config.setdefault("family_policy", config.get("family_policy", "metadata_preferred"))
        self.scripted = ScriptedProposer(fallback_config)
        self.processor = None
        self.model = None
        self.last_proposal_diagnostics: list[dict[str, Any]] = []

    def set_checkpoint_path(self, checkpoint_path: str | None) -> None:
        if checkpoint_path == self.checkpoint_path:
            return
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.processor = None

    def model_state(self) -> dict[str, Any]:
        return {
            "backend": self.config.get("backend", "trainable_qwen_image_edit"),
            "model_name_or_path": self.model_name_or_path,
            "model_subfolder": self.model_subfolder,
            "processor_subfolder": self.processor_subfolder,
            "proposal_mode": self.proposal_mode,
            "checkpoint_path": self.checkpoint_path,
        }

    def proposal_diagnostics(self) -> list[dict[str, Any]]:
        return list(self.last_proposal_diagnostics)

    def _retry_feedback_hint(self, *, record_key: str, target_edit_type: str | None) -> str:
        reasons = [
            str(item.get("reason") or "")
            for item in self.last_proposal_diagnostics
            if item.get("record_key") == record_key and item.get("stage") == "filter"
        ]
        if not reasons:
            return ""
        hint = ""
        if target_edit_type == "object_replacement" and any(
            reason in {"proposal_replacement_overlaps_source", "proposal_replacement_same_as_source"}
            for reason in reasons
        ):
            hint += (
                "\nSpecific retry correction for object_replacement: the previous proposal was rejected "
                "because target_object/replacement was the same object class or shared a meaningful noun "
                "with source_object. Choose a genuinely different small physical object category with no "
                "shared head noun with source_object. Bad examples: bench->park bench, cup->mug, "
                "bow tie->tie, apple->green apple. Good examples: bench->traffic cone, cup->tennis ball, "
                "hat->camera, bottle->small plant pot. If no different object category is visually safe, "
                "return {\"proposals\": []}."
            )
        if target_edit_type == "object_replacement" and any(
            reason == "proposal_replacement_targets_background" for reason in reasons
        ):
            hint += (
                "\nSpecific retry correction for object_replacement: replace one bounded physical "
                "foreground object. Do not treat background, backdrop, scenery, surroundings, or "
                "environmental foliage as an object."
            )
        if target_edit_type == "object_addition" and any(
            reason in {
                "proposal_target_object_missing",
                "proposal_addition_forbidden_missing_target_object",
                "proposal_addition_forbidden_not_failure_state",
            }
            for reason in reasons
        ):
            hint += (
                "\nSpecific retry correction for object_addition: set target_object to the exact "
                "new physical object named in the instruction. forbidden_after must name that same "
                "target object and state the positive failure condition that it is absent from the "
                "requested region. Do not describe an existing object's preservation as the failure."
            )
        if any(reason == "proposal_source_object_is_text_like" for reason in reasons):
            hint += (
                "\nSpecific retry correction: the previous source_object was text-like. Do not edit "
                "printed text, signs, labels, logos, book covers, stickers, or captions. Choose a physical "
                "non-text object or return {\"proposals\": []}."
            )
        if any(reason == "proposal_contains_quoted_visible_text" for reason in reasons):
            hint += (
                "\nSpecific retry correction: do not target or transcribe visible words, letters, "
                "logos, labels, signs, or other text. Choose a non-text visual attribute or object."
            )
        if any(reason == "proposal_attribute_change_is_color_edit" for reason in reasons):
            hint += (
                "\nSpecific retry correction for attribute_change: change a non-color property such "
                "as shape, size, openness, cleanliness, expression, or physical state. Do not change "
                "hue, color, material, or texture."
            )
        if target_edit_type == "attribute_change" and any(
            reason
            in {
                "proposal_attribute_source_state_missing",
                "proposal_attribute_target_state_missing",
                "proposal_attribute_target_state_vague",
                "proposal_attribute_state_unchanged",
            }
            for reason in reasons
        ):
            hint += (
                "\nSpecific retry correction for attribute_change: name source_attribute and "
                "target_attribute as two concrete, visibly distinct states. Do not use placeholders "
                "such as 'different', 'another', 'new', or 'changed'. For example, use "
                "source_attribute='closed' and target_attribute='open', or "
                "source_attribute='circular' and target_attribute='star-shaped'."
            )
        if target_edit_type == "color_change" and any(
            reason
            in {
                "proposal_color_source_state_missing",
                "proposal_color_target_state_missing",
                "proposal_color_state_unchanged",
            }
            for reason in reasons
        ):
            hint += (
                "\nSpecific retry correction for color_change: provide the currently visible color "
                "in source_attribute and a different explicit color in target_attribute. Both colors "
                "must also be stated in the visual contract."
            )
        if target_edit_type == "material_change" and any(
            reason
            in {
                "proposal_material_source_state_missing",
                "proposal_material_target_state_missing",
                "proposal_material_state_unchanged",
            }
            for reason in reasons
        ):
            hint += (
                "\nSpecific retry correction for material_change: provide the source object's "
                "currently visible physical material in source_material and a genuinely different "
                "physical material in target_material. Do not propose an intrinsic or equivalent "
                "material, such as changing a rubber object to rubber."
            )
        if target_edit_type == "object_replacement" and any(
            reason == "proposal_replacement_region_uses_source_as_anchor" for reason in reasons
        ):
            hint += (
                "\nSpecific retry correction for object_replacement: target_region must describe the "
                "source object's location using another stable visible anchor. It cannot place the new "
                "object on or inside the object that is being replaced."
            )
        if any(reason == "proposal_action_source_object_missing" for reason in reasons):
            hint += (
                "\nSpecific retry correction for action_change: source_object must explicitly name "
                "the visible person, animal, or articulated subject whose pose changes."
            )
        if target_edit_type == "spatial_move" and any(
            reason == "proposal_spatial_target_fragile_or_main_subject" for reason in reasons
        ):
            hint += (
                "\nSpecific retry correction for spatial_move: the moved source_object must be a small, "
                "bounded, non-living object. Toy/model objects are acceptable, but real people, animals, "
                "faces, bodies, and full-size vehicles are not. Put the movable object in source_object, "
                "and use target_region only as the visible spatial anchor."
            )
        if target_edit_type == "background_change" and any(
            reason == "proposal_background_change_is_object_edit" for reason in reasons
        ):
            hint += (
                "\nSpecific retry correction for background_change: do not replace a foreground object, "
                "floor patch, rug, furniture, wall detail, fireplace, or local surface. The edited noun "
                "in the instruction must be background/backdrop/scene/environment, not a concrete object "
                "or room element. Use wording like 'Change the background behind the person and dog to a "
                "serene forest scene' or 'Replace the backdrop behind the subject with a snowy landscape'. "
                "Set target_region to 'background behind <visible subject>' and preserve the foreground "
                "subjects."
            )
        if any("broad_head" in reason or reason == "proposal_target_region_generic" for reason in reasons):
            hint += (
                "\nSpecific retry correction: the previous proposal used a broad scene target or generic "
                "region. Choose a bounded small/medium object and anchor target_region to visible context."
            )
        return hint

    def _record_proposal_diagnostic(
        self,
        *,
        record_key: str,
        target_edit_type: str | None,
        stage: str,
        reason: str,
        attempt_index: int | None = None,
        payload: dict[str, Any] | None = None,
        structured_edit: dict[str, Any] | None = None,
        decoded: str | None = None,
        payload_count: int | None = None,
    ) -> None:
        item: dict[str, Any] = {
            "record_key": record_key,
            "target_edit_type": target_edit_type,
            "stage": stage,
            "reason": reason,
        }
        if attempt_index is not None:
            item["attempt_index"] = attempt_index
        if payload_count is not None:
            item["payload_count"] = payload_count
        if decoded is not None:
            item["decoded_preview"] = re.sub(r"\s+", " ", decoded.strip())[:1500]
        if payload is not None:
            item["payload_preview"] = json.dumps(payload, ensure_ascii=False, default=str)[:1200]
            item["payload_edit_type"] = payload.get("edit_type")
        if structured_edit is not None:
            item["structured_edit_type"] = structured_edit.get("edit_type")
            item["structured_instruction"] = str(structured_edit.get("instruction") or "")[:300]
        self.last_proposal_diagnostics.append(item)

    def release_memory(self) -> None:
        if self.model is not None and hasattr(self.model, "to"):
            try:
                self.model.to("cpu")
            except Exception:
                pass
        self.model = None
        self.processor = None
        try:
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            return

    @staticmethod
    def _resolve_device(device: str):
        import torch

        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @staticmethod
    def _string_set(value: Any) -> set[str]:
        if value is None:
            return set()
        if isinstance(value, str):
            raw_items = [part.strip() for part in value.split(",")]
        else:
            raw_items = [str(item).strip() for item in value]
        return {item for item in raw_items if item}

    def _template_bank_values(self, key: str) -> list[str]:
        values = self.template_target_bank.get(key) or DEFAULT_TEMPLATE_TARGET_BANK.get(key) or []
        return [str(value).strip() for value in values if str(value).strip()]

    @staticmethod
    def _template_choice(values: list[str], stable_choice: int, salt: str) -> str:
        if not values:
            raise ValueError("Template target bank is empty")
        salted = stable_choice + sum(ord(char) for char in salt)
        return values[salted % len(values)]

    @staticmethod
    def _indefinite_phrase(noun: str) -> str:
        stripped = noun.strip()
        if not stripped:
            return stripped
        if stripped.lower().startswith(("a ", "an ", "the ")):
            return stripped
        article = "an" if stripped[0].lower() in {"a", "e", "i", "o", "u"} else "a"
        return f"{article} {stripped}"

    @staticmethod
    def _replacement_overlaps_source(replacement: str, source_object: str | None) -> bool:
        if not source_object:
            return False
        replacement_terms = {term for term in re.findall(r"[a-z0-9]+", replacement.lower()) if len(term) > 2}
        source_terms = {term for term in re.findall(r"[a-z0-9]+", source_object.lower()) if len(term) > 2}
        if not replacement_terms or not source_terms:
            return False
        return bool(replacement_terms & source_terms) or replacement.lower() in source_object.lower() or source_object.lower() in replacement.lower()

    @staticmethod
    def _canonical_state_terms(value: Any) -> set[str]:
        aliases = {
            "glassy": "glass",
            "metallic": "metal",
            "rubbery": "rubber",
            "wooden": "wood",
        }
        stopwords = {
            "a",
            "an",
            "and",
            "appearance",
            "current",
            "existing",
            "material",
            "original",
            "state",
            "the",
            "texture",
        }
        return {
            aliases.get(token, token)
            for token in re.findall(r"[a-z0-9]+", str(value or "").lower())
            if token not in stopwords
        }

    @classmethod
    def _state_transition_is_noop(cls, source: Any, target: Any) -> bool:
        source_terms = cls._canonical_state_terms(source)
        target_terms = cls._canonical_state_terms(target)
        return bool(source_terms and target_terms and source_terms == target_terms)

    @staticmethod
    def _state_descriptor_is_vague(value: Any) -> bool:
        lowered = re.sub(r"\s+", " ", str(value or "").strip().lower())
        if not lowered:
            return True
        if lowered.startswith(("another ", "different ", "new ", "other ")):
            return True
        return lowered in {
            "altered",
            "alternative",
            "changed",
            "different",
            "enhanced",
            "improved",
            "new",
            "other",
            "something else",
        }

    @staticmethod
    def _replacement_region_uses_source_as_anchor(source_object: str, target_region: str) -> bool:
        source_tokens = [
            token
            for token in re.findall(r"[a-z0-9]+", source_object.lower())
            if token
            not in {
                "a",
                "an",
                "the",
                "small",
                "large",
                "big",
                "tiny",
                "white",
                "black",
                "red",
                "green",
                "blue",
                "round",
                "square",
            }
        ]
        if not source_tokens:
            return False
        source_head = source_tokens[-1]
        region = re.sub(r"\s+", " ", target_region.strip().lower())
        return bool(
            re.search(
                rf"\b(?:on|inside|in|under|beneath|above|below)\s+(?:the\s+|a\s+|an\s+)?{re.escape(source_head)}\b",
                region,
            )
        )

    def _local_target_broad_head(self, phrase: str) -> str | None:
        tokens = [
            token
            for token in re.findall(r"[a-z0-9]+", str(phrase or "").lower())
            if token
            not in {
                "a",
                "an",
                "the",
                "this",
                "that",
                "these",
                "those",
                "small",
                "large",
                "big",
                "tiny",
                "main",
                "visible",
                "wooden",
                "metal",
                "stone",
                "glass",
                "left",
                "right",
                "upper",
                "lower",
                "top",
                "bottom",
                "center",
                "centre",
                "middle",
            }
        ]
        if not tokens:
            return None
        head = tokens[-1]
        return head if head in self.local_target_disallowed_heads else None

    @staticmethod
    def _spatial_source_is_fragile(source_object: str) -> bool:
        tokens = set(re.findall(r"[a-z0-9]+", source_object.lower()))
        human_terms = {
            "person",
            "people",
            "human",
            "man",
            "woman",
            "girl",
            "boy",
            "child",
            "children",
            "baby",
            "face",
            "head",
            "body",
            "skier",
            "rider",
            "cyclist",
            "driver",
            "player",
            "worker",
            "athlete",
            "surfer",
            "dancer",
        }
        animal_terms = {
            "dog",
            "cat",
            "animal",
            "bird",
            "horse",
            "cow",
            "sheep",
            "rabbit",
            "fish",
        }
        vehicle_terms = {
            "vehicle",
            "car",
            "truck",
            "bus",
            "motorcycle",
            "bicycle",
            "bike",
        }
        toy_like_terms = {
            "toy",
            "model",
            "miniature",
            "figurine",
            "plush",
            "stuffed",
            "doll",
        }
        if tokens & human_terms:
            return True
        if tokens & (animal_terms | vehicle_terms):
            return not bool(tokens & toy_like_terms)
        return False

    @staticmethod
    def _source_object_is_text_like(source_object: str) -> bool:
        tokens = set(re.findall(r"[a-z0-9]+", source_object.lower()))
        text_like_tokens = {
            "text",
            "texts",
            "word",
            "words",
            "letter",
            "letters",
            "writing",
            "caption",
            "captions",
            "character",
            "characters",
            "label",
            "labels",
            "tag",
            "tags",
            "sticker",
            "stickers",
            "barcode",
            "barcodes",
            "logo",
            "logos",
            "poster",
            "posters",
            "sign",
            "signs",
            "cover",
            "covers",
            "book",
            "books",
            "comic",
            "comics",
            "price",
        }
        return bool(tokens & text_like_tokens)

    def _template_target_choice(
        self,
        key: str,
        stable_choice: int,
        salt: str,
        *,
        source_object: str | None = None,
    ) -> str:
        values = self._template_bank_values(key)
        if source_object:
            filtered = [value for value in values if not self._replacement_overlaps_source(value, source_object)]
            if filtered:
                values = filtered
        return self._template_choice(values, stable_choice, salt)

    def _edit_type_allowed(self, structured_edit: dict[str, Any]) -> bool:
        edit_type = str(structured_edit.get("edit_type", ""))
        if self.allowed_edit_types and edit_type not in self.allowed_edit_types:
            return False
        if self.disallowed_edit_types and edit_type in self.disallowed_edit_types:
            return False
        return True

    @staticmethod
    def _contains_quoted_visible_text(value: Any) -> bool:
        if isinstance(value, list):
            return any(TrainableQwenVLProposer._contains_quoted_visible_text(item) for item in value)
        text = str(value or "")
        return bool(
            any(quote in text for quote in ('"', "\u201c", "\u201d", "\u2018", "\u2019"))
            or re.search(r"(^|\s)'|'($|\s)", text)
        )

    @staticmethod
    def _has_spatial_anchor(text: Any) -> bool:
        lowered = f" {str(text or '').strip().lower()} "
        anchor_markers = (
            " on ",
            " above ",
            " below ",
            " under ",
            " beneath ",
            " beside ",
            " near ",
            " next to ",
            " left of ",
            " right of ",
            " in front of ",
            " behind ",
            " between ",
            " attached to ",
            " held by ",
            " worn by ",
            " around ",
            " at ",
            " in ",
        )
        coordinate_markers = (
            "upper ",
            "lower ",
            "top ",
            "bottom ",
            "left ",
            "right ",
            "center",
            "centre",
            "middle",
            "corner",
            "edge",
            "foreground",
            "background",
        )
        return any(marker in lowered for marker in anchor_markers) or any(
            marker in lowered for marker in coordinate_markers
        )

    def _proposal_contract_filter_reason(self, structured_edit: dict[str, Any]) -> str | None:
        if not self.strict_structured_proposal_filter:
            return None
        edit_type = str(structured_edit.get("edit_type") or "")
        instruction = str(structured_edit.get("instruction") or "").strip()
        target_region = str(structured_edit.get("target_region") or "").strip()
        source_object = str(
            structured_edit.get("source_object") or structured_edit.get("target") or ""
        ).strip()
        target_object = str(
            structured_edit.get("target_object")
            or structured_edit.get("replacement")
            or ""
        ).strip()
        source_attribute = str(structured_edit.get("source_attribute") or "").strip()
        target_attribute = str(structured_edit.get("target_attribute") or "").strip()
        source_material = str(structured_edit.get("source_material") or "").strip()
        target_material = str(structured_edit.get("target_material") or "").strip()
        forbidden_after = structured_edit.get("forbidden_after") or []
        if isinstance(forbidden_after, str):
            forbidden_after_items = [forbidden_after]
        elif isinstance(forbidden_after, list):
            forbidden_after_items = [str(item) for item in forbidden_after]
        else:
            forbidden_after_items = [str(forbidden_after)]
        if any(
            re.match(r"^\s*(?:no|not|without)\b", item, flags=re.IGNORECASE)
            for item in forbidden_after_items
        ):
            return "proposal_forbidden_after_negative_sentence"
        fields_to_check = (
            instruction,
            source_object,
            target_object,
            target_region,
            structured_edit.get("required_after"),
            structured_edit.get("forbidden_after"),
        )
        if any(self._contains_quoted_visible_text(value) for value in fields_to_check):
            return "proposal_contains_quoted_visible_text"
        if source_object and self._source_object_is_text_like(source_object):
            return "proposal_source_object_is_text_like"

        if edit_type == "attribute_change":
            if self.require_explicit_state_transitions:
                if not source_attribute:
                    return "proposal_attribute_source_state_missing"
                if not target_attribute:
                    return "proposal_attribute_target_state_missing"
                if self._state_descriptor_is_vague(target_attribute):
                    return "proposal_attribute_target_state_vague"
                if self._state_transition_is_noop(source_attribute, target_attribute):
                    return "proposal_attribute_state_unchanged"
            attribute_text = " ".join(
                str(value or "")
                for value in (
                    (source_attribute, target_attribute)
                    if source_attribute and target_attribute
                    else (instruction, target_attribute)
                )
            ).lower()
            color_terms = {
                "black",
                "blue",
                "brown",
                "color",
                "colour",
                "cyan",
                "gray",
                "green",
                "grey",
                "magenta",
                "orange",
                "pink",
                "purple",
                "red",
                "white",
                "yellow",
            }
            if re.search(
                r"\b(?:"
                + "|".join(
                    re.escape(term)
                    for term in sorted(color_terms, key=len, reverse=True)
                )
                + r")\b",
                attribute_text,
            ):
                return "proposal_attribute_change_is_color_edit"

        if edit_type == "color_change" and self.require_explicit_state_transitions:
            if not source_attribute:
                return "proposal_color_source_state_missing"
            if not target_attribute:
                return "proposal_color_target_state_missing"
            if self._state_transition_is_noop(source_attribute, target_attribute):
                return "proposal_color_state_unchanged"

        local_edit_types = {
            "attribute_change",
            "color_change",
            "material_change",
            "object_replacement",
            "object_removal",
            "object_addition",
            "spatial_move",
            "local_enhancement",
        }
        global_region_terms = {
            "entire image",
            "the entire image",
            "whole image",
            "the whole image",
            "overall image",
            "full image",
            "global image",
        }
        global_adjustment_terms = {
            "overall contrast",
            "image contrast",
            "global contrast",
            "entire image contrast",
            "whole image contrast",
            "overall brightness",
            "image brightness",
            "global brightness",
            "overall exposure",
            "image exposure",
            "global exposure",
            "overall saturation",
            "image saturation",
            "global saturation",
        }
        local_text = f"{instruction} {target_region}".lower()
        if edit_type in local_edit_types and (
            any(term in local_text for term in global_region_terms)
            or any(term in local_text for term in global_adjustment_terms)
        ):
            return "proposal_local_edit_is_global_adjustment"

        if edit_type == "background_change":
            background_text = f"{instruction} {target_region} {structured_edit.get('required_after') or ''}".lower()
            background_terms = {
                "background",
                "backdrop",
                "environment",
                "surroundings",
                "surrounding scene",
                "scene behind",
                "behind the",
            }
            if not any(term in background_text for term in background_terms):
                return "proposal_background_change_missing_background_target"
            if target_region and not any(term in target_region.lower() for term in background_terms):
                return "proposal_background_change_target_region_not_background"
            object_edit_match = re.search(
                r"\b(?:replace|remove|add)\s+(?:the\s+|a\s+|an\s+)?(.+?)(?:\s+with\b|\s+from\b|\s+to\b|[.;,]|$)",
                instruction,
                flags=re.IGNORECASE,
            )
            if object_edit_match:
                edited_phrase = object_edit_match.group(1).strip().lower()
                if not any(term in edited_phrase for term in background_terms):
                    return "proposal_background_change_is_object_edit"
            if not structured_edit.get("preserve"):
                return "proposal_background_change_missing_preserve_list"
            return None

        if edit_type == "style_transfer":
            style_text = f"{instruction} {target_region} {structured_edit.get('required_after') or ''}".lower()
            style_region_terms = set(global_region_terms) | {
                "whole scene",
                "the whole scene",
                "full scene",
                "the full scene",
                "scene layout",
            }
            if target_region and not any(term in target_region.lower() for term in style_region_terms):
                return "proposal_style_transfer_target_region_not_global"
            if re.search(r"\btransfer\s+the\s+.+\s+to\s+(?:the\s+)?\w+", instruction, flags=re.IGNORECASE):
                return "proposal_style_transfer_object_to_object_texture"
            if not any(term in style_text for term in style_region_terms | {"style", "stylized", "filter"}):
                return "proposal_style_transfer_missing_style_contract"
            if not structured_edit.get("preserve"):
                return "proposal_style_transfer_missing_preserve_list"
            return None

        local_bounded_target_edit_types = {
            "attribute_change",
            "color_change",
            "material_change",
            "local_enhancement",
        }
        if edit_type in local_bounded_target_edit_types:
            if not source_object:
                return "proposal_local_target_missing_source_object"
            broad_head = self._local_target_broad_head(source_object)
            if broad_head:
                return f"proposal_local_target_broad_head:{broad_head}"
            if edit_type == "material_change":
                if self.require_explicit_state_transitions:
                    if not source_material:
                        return "proposal_material_source_state_missing"
                    if not target_material:
                        return "proposal_material_target_state_missing"
                    if self._state_transition_is_noop(source_material, target_material):
                        return "proposal_material_state_unchanged"
                material_contract_text = json.dumps(
                    {
                        "instruction": instruction,
                        "required_after": structured_edit.get("required_after"),
                        "target_material": structured_edit.get("target_material"),
                    },
                    ensure_ascii=False,
                ).lower()
                material_terms = {
                    "bronze",
                    "cardboard",
                    "ceramic",
                    "clay",
                    "concrete",
                    "copper",
                    "cotton",
                    "fabric",
                    "fur",
                    "glass",
                    "gold",
                    "iron",
                    "leather",
                    "marble",
                    "metal",
                    "metallic",
                    "paper",
                    "plastic",
                    "porcelain",
                    "rubber",
                    "silk",
                    "silver",
                    "stone",
                    "steel",
                    "wool",
                    "wood",
                    "wooden",
                }
                color_terms = {
                    "black",
                    "blue",
                    "brown",
                    "color",
                    "colour",
                    "cyan",
                    "gold",
                    "gray",
                    "green",
                    "grey",
                    "magenta",
                    "orange",
                    "pink",
                    "purple",
                    "red",
                    "silver",
                    "white",
                    "yellow",
                }
                has_material_term = bool(
                    re.search(
                        r"\b(?:"
                        + "|".join(re.escape(term) for term in sorted(material_terms, key=len, reverse=True))
                        + r")\b",
                        material_contract_text,
                    )
                )
                has_color_term = bool(
                    re.search(
                        r"\b(?:"
                        + "|".join(re.escape(term) for term in sorted(color_terms, key=len, reverse=True))
                        + r")\b",
                        material_contract_text,
                    )
                )
                if has_color_term and not has_material_term:
                    return "proposal_material_change_is_color_edit"
                if not has_material_term:
                    return "proposal_material_change_missing_material"

        if edit_type == "spatial_move":
            if not source_object:
                return "proposal_spatial_source_object_missing"
            broad_head = self._local_target_broad_head(source_object)
            if broad_head:
                return f"proposal_spatial_target_broad_head:{broad_head}"
            if self._spatial_source_is_fragile(source_object):
                return "proposal_spatial_target_fragile_or_main_subject"

        if edit_type == "action_change" and not source_object:
            return "proposal_action_source_object_missing"

        if edit_type not in {"object_removal", "object_replacement", "object_addition"}:
            return None

        if edit_type in {"object_removal", "object_replacement"} and not source_object:
            return "proposal_source_object_missing"
        source_object_can_anchor_region = bool(source_object) and not self._local_target_broad_head(source_object)
        if edit_type == "object_addition":
            if not target_object:
                return "proposal_target_object_missing"
            if re.search(r"\b(next|left|right|above|below|under|beneath|near|beside|behind)$", target_object.lower()):
                return "proposal_target_object_contains_relation_tail"
            forbidden_text = json.dumps(structured_edit.get("forbidden_after") or "", ensure_ascii=False).lower()
            target_tokens = {
                term for term in re.findall(r"[a-z0-9]+", target_object.lower()) if len(term) > 2
            }
            if target_tokens and not any(term in forbidden_text for term in target_tokens):
                return "proposal_addition_forbidden_missing_target_object"
            addition_failure_markers = (
                "absent",
                "missing",
                "not present",
                "not visible",
                "does not appear",
                "fails to appear",
                "duplicat",
                "implausib",
                "occlud",
                "wrong location",
                "wrong region",
            )
            if not any(marker in forbidden_text for marker in addition_failure_markers):
                return "proposal_addition_forbidden_not_failure_state"
        if edit_type == "object_replacement":
            replacement_region_text = f"{source_object} {target_region}".lower()
            if any(
                term in replacement_region_text
                for term in (
                    "background",
                    "backdrop",
                    "environment",
                    "surroundings",
                    "surrounding scene",
                    "scenery",
                )
            ):
                return "proposal_replacement_targets_background"
            if not target_object:
                return "proposal_target_object_missing"
            if target_object.lower() == source_object.lower():
                return "proposal_replacement_same_as_source"
            if self._replacement_overlaps_source(target_object, source_object):
                return "proposal_replacement_overlaps_source"
            if self.require_explicit_state_transitions and self._replacement_region_uses_source_as_anchor(
                source_object,
                target_region,
            ):
                return "proposal_replacement_region_uses_source_as_anchor"
        generic_regions = {
            "",
            "main visible target",
            "original location",
            "the original location",
            "same location",
            "the same location",
            "target region",
            "the target region",
        }
        if target_region.lower() in generic_regions and not source_object_can_anchor_region:
            return "proposal_target_region_generic"
        if not self._has_spatial_anchor(target_region) and not source_object_can_anchor_region:
            return "proposal_target_region_missing_anchor"
        return None

    def proposal_contract_filter_reason(
        self, structured_edit: dict[str, Any]
    ) -> str | None:
        """Return the deterministic eligibility failure for a normalized contract."""
        return self._proposal_contract_filter_reason(structured_edit)

    def _proposal_contract_allowed(self, structured_edit: dict[str, Any]) -> bool:
        return self._proposal_contract_filter_reason(structured_edit) is None

    def _structured_edit_allowed(self, structured_edit: dict[str, Any]) -> bool:
        return self._edit_type_allowed(structured_edit) and self._proposal_contract_allowed(structured_edit)

    def _record_target_edit_type(self, record: UnlabeledImageRecord) -> str | None:
        if not self.enforce_record_target_edit_type:
            return None
        metadata = record.metadata or {}
        value = metadata.get("scheduled_edit_type") or metadata.get("target_edit_type")
        if value is None:
            return None
        edit_type = str(value).strip()
        if not edit_type:
            return None
        if self.allowed_edit_types and edit_type not in self.allowed_edit_types:
            return None
        if self.disallowed_edit_types and edit_type in self.disallowed_edit_types:
            return None
        return edit_type

    @staticmethod
    def _caption_object_candidates(record: UnlabeledImageRecord) -> list[str]:
        metadata = record.metadata or {}
        candidates: list[str] = []
        raw_objects = metadata.get("objects") or metadata.get("object_tags") or []
        if isinstance(raw_objects, str):
            raw_objects = [part.strip() for part in raw_objects.split(",")]
        if isinstance(raw_objects, list):
            candidates.extend(str(item).strip().lower() for item in raw_objects if str(item).strip())

        caption = str(record.caption or "").lower()
        multiword_objects = [
            "party hat",
            "life vest",
            "surfboard",
            "cutting board",
            "sports bag",
            "baseball bat",
            "water bottle",
            "speech bubble",
            "fire hydrant",
            "fighter jet",
            "teddy bear",
            "freestanding bathtub",
            "glass-enclosed shower",
            "double sink vanity",
            "mounted tv",
            "window frame",
            "green beans",
        ]
        for phrase in multiword_objects:
            if phrase in caption:
                candidates.append(phrase)

        object_nouns = {
            "airplane",
            "ball",
            "bag",
            "basket",
            "bat",
            "bathtub",
            "beans",
            "bench",
            "bottle",
            "bowl",
            "car",
            "carrot",
            "cat",
            "celery",
            "chair",
            "chicken",
            "crown",
            "cup",
            "door",
            "dog",
            "dress",
            "floor",
            "frisbee",
            "hat",
            "hydrant",
            "jet",
            "knife",
            "mirror",
            "motorcycle",
            "onion",
            "oven",
            "pizza",
            "rack",
            "shirt",
            "shoes",
            "shower",
            "sink",
            "table",
            "tile",
            "tv",
            "turtle",
            "umbrella",
            "vanity",
            "vegetables",
            "vest",
            "wall",
            "window",
        }
        stopwords = {
            "a",
            "an",
            "and",
            "with",
            "near",
            "next",
            "to",
            "on",
            "in",
            "of",
            "from",
            "including",
            "attached",
            "wearing",
            "holding",
            "sitting",
            "standing",
            "looking",
        }
        tokens = re.findall(r"[a-z0-9]+", caption)
        for index, token in enumerate(tokens):
            singular = token[:-1] if len(token) > 3 and token.endswith("s") else token
            if singular not in object_nouns:
                continue
            start = index
            while start > 0 and index - start < 3 and tokens[start - 1] not in stopwords:
                start -= 1
            phrase = " ".join(tokens[start : index + 1]).strip()
            if phrase:
                candidates.append(phrase)

        seen = set()
        output = []
        for candidate in candidates:
            candidate = re.sub(r"\s+", " ", candidate).strip(" .,;:")
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            output.append(candidate)
        return output

    @staticmethod
    def _secondary_object_preferred(candidate: str) -> bool:
        lowered = candidate.lower()
        preferred_terms = {
            "hat",
            "vest",
            "pizza",
            "surfboard",
            "bag",
            "bat",
            "bottle",
            "flame",
            "umbrella",
            "knife",
            "onion",
            "ball",
            "turtle",
            "bear",
            "frisbee",
            "bubble",
            "hydrant",
            "crown",
            "rack",
            "basket",
            "bathtub",
            "shower",
            "sink",
            "vanity",
            "mirror",
            "tv",
            "window",
            "cutting board",
            "green beans",
            "carrot",
            "celery",
            "chicken",
            "motorcycle",
        }
        dominant_terms = {"person", "man", "woman", "child", "dog", "cat", "elephant", "airplane", "car"}
        return any(term in lowered for term in preferred_terms) and not any(term in lowered for term in dominant_terms)

    @staticmethod
    def _template_target_region(record: UnlabeledImageRecord, source_object: str | None) -> str:
        if not source_object:
            return "near the main subject in the visible scene"
        obj = source_object.lower()
        caption = str(record.caption or "").lower()
        has_person = any(term in caption for term in ("person", "man", "woman", "child", "boy", "girl"))
        has_table = any(term in caption for term in ("table", "counter", "plate", "cutting board"))

        if "hat" in obj or "crown" in obj:
            return "on the person's head" if has_person else "on the main subject's head"
        if "vest" in obj or "shirt" in obj or "dress" in obj or "shoes" in obj:
            return "on the person's body" if has_person else "on the main subject"
        if "bag" in obj or "backpack" in obj:
            return "beside the person" if has_person else "near the main subject"
        if "bat" in obj or "frisbee" in obj or "ball" in obj or "umbrella" in obj:
            return "near the person's hands" if has_person else "near the main subject"
        if "surfboard" in obj:
            return "under or beside the person" if has_person else "in the foreground"
        if "bottle" in obj or "cup" in obj or "mug" in obj or "bowl" in obj:
            return "on the table" if has_table else "near the foreground surface"
        if any(term in obj for term in ("knife", "onion", "carrot", "celery", "beans", "chicken", "pizza")):
            return "on the cutting board or plate" if has_table else "in the food preparation area"
        if any(term in obj for term in ("bathtub", "shower", "sink", "vanity", "mirror", "tv", "window")):
            return "in the bathroom scene"
        if "hydrant" in obj:
            return "beside the street or sidewalk"
        if "turtle" in obj or "bear" in obj:
            return "near the main subject"
        return "near the main subject in the visible scene"

    @staticmethod
    def _template_region_phrase(region: str) -> str:
        lowered = region.lower()
        if lowered.startswith(
            (
                "on ",
                "above ",
                "below ",
                "under ",
                "beneath ",
                "beside ",
                "near ",
                "next to ",
                "left of ",
                "right of ",
                "in front of ",
                "behind ",
                "between ",
                "attached to ",
                "held by ",
                "worn by ",
                "around ",
                "at ",
                "in ",
            )
        ):
            return region
        return f"at {region}"

    def _template_target_payload(
        self,
        record: UnlabeledImageRecord,
        edit_type: str,
        seed: int,
    ) -> dict[str, Any] | None:
        candidates = self._caption_object_candidates(record)
        source_object = None
        for candidate in candidates:
            if self._secondary_object_preferred(candidate):
                source_object = candidate
                break
        if source_object is None and candidates:
            source_object = candidates[0]

        stable_choice = sum(ord(char) for char in record.key) + int(seed)
        replacement = self._template_target_choice(
            "replacement_objects",
            stable_choice,
            f"{record.key}:replacement:{edit_type}",
            source_object=source_object,
        )
        addition_object = self._template_target_choice(
            "addition_objects",
            stable_choice,
            f"{record.key}:addition:{edit_type}",
            source_object=source_object,
        )
        color = self._template_target_choice("colors", stable_choice, f"{record.key}:color:{edit_type}")
        material = self._template_target_choice("materials", stable_choice, f"{record.key}:material:{edit_type}")
        attribute = self._template_target_choice("attributes", stable_choice, f"{record.key}:attribute:{edit_type}")
        style_target = self._template_target_choice("style_targets", stable_choice, f"{record.key}:style:{edit_type}")
        background_target = self._template_target_choice("background_targets", stable_choice, f"{record.key}:background:{edit_type}")
        preserve = ["main subject", "background", "lighting", "camera viewpoint"]
        target_region = self._template_target_region(record, source_object)
        target_region_phrase = self._template_region_phrase(target_region)

        if edit_type == "object_removal" and source_object:
            return {
                "edit_type": "object_removal",
                "instruction": (
                    f"Remove the {source_object} {target_region_phrase}. "
                    "Keep all other content, lighting, and layout unchanged."
                ),
                "source_object": source_object,
                "target_region": target_region,
                "required_after": [f"the area {target_region_phrase} is cleanly filled after removing {source_object}"],
                "forbidden_after": [f"{source_object} remains visible {target_region_phrase}"],
                "preserve": preserve,
            }
        if edit_type == "object_replacement" and source_object:
            if self._replacement_overlaps_source(replacement, source_object):
                replacement = self._template_target_choice(
                    "replacement_objects",
                    stable_choice + 1,
                    f"{record.key}:replacement:fallback:{edit_type}",
                    source_object=source_object,
                )
            replacement_phrase = self._indefinite_phrase(replacement)
            return {
                "edit_type": "object_replacement",
                "instruction": (
                    f"Replace the {source_object} {target_region_phrase} with {replacement_phrase}. "
                    "Keep the same location and approximate size, and keep all other content unchanged."
                ),
                "source_object": source_object,
                "target_object": replacement,
                "replacement": replacement,
                "target_region": target_region,
                "required_after": [f"{replacement} is visible {target_region_phrase}"],
                "forbidden_after": [f"{source_object} remains visible {target_region_phrase}"],
                "preserve": preserve,
            }
        if edit_type == "object_addition":
            added_object = addition_object
            added_object_phrase = self._indefinite_phrase(added_object)
            return {
                "edit_type": "object_addition",
                "instruction": f"Add {added_object_phrase} to a plausible open area of the scene.",
                "target_object": added_object,
                "replacement": added_object,
                "target_region": "a plausible open area of the scene",
                "required_after": [f"{added_object} has been added to a plausible open area of the scene"],
                "forbidden_after": [],
                "preserve": preserve,
            }
        if edit_type == "color_change" and source_object:
            return {
                "edit_type": "color_change",
                "instruction": f"Change the color of the {source_object} to {color} while preserving the rest of the scene.",
                "source_object": source_object,
                "target_attribute": color,
                "target_region": source_object,
                "required_after": [f"the {source_object} is {color}"],
                "forbidden_after": [f"the {source_object} keeps its original color"],
                "preserve": preserve,
            }
        if edit_type == "attribute_change" and source_object:
            return {
                "edit_type": "attribute_change",
                "instruction": f"Give the {source_object} {attribute} while preserving its shape and surroundings.",
                "source_object": source_object,
                "target_attribute": attribute,
                "target_region": source_object,
                "required_after": [f"the {source_object} has {attribute}"],
                "forbidden_after": [f"the {source_object} remains unchanged"],
                "preserve": preserve,
            }
        if edit_type == "material_change" and source_object:
            return {
                "edit_type": "material_change",
                "instruction": f"Change the {source_object} material to {material} while keeping its shape and location.",
                "source_object": source_object,
                "target_material": material,
                "target_region": source_object,
                "required_after": [f"the {source_object} appears made of {material}"],
                "forbidden_after": [f"the {source_object} keeps its original material"],
                "preserve": preserve,
            }
        if edit_type == "spatial_move" and source_object:
            directions = ["slightly to the left", "slightly to the right", "slightly upward"]
            target_location = directions[stable_choice % len(directions)]
            return {
                "edit_type": "spatial_move",
                "instruction": f"Move the {source_object} {target_location} while preserving the rest of the image.",
                "source_object": source_object,
                "source_location": "its original location",
                "target_location": target_location,
                "target_region": source_object,
                "required_after": [f"the {source_object} is moved {target_location}"],
                "forbidden_after": [f"the {source_object} remains in its original location"],
                "preserve": preserve,
            }
        if edit_type == "background_change":
            subject = source_object or "main subject"
            return {
                "edit_type": "background_change",
                "instruction": f"Change the background behind the {subject} to a {background_target} while preserving the {subject}.",
                "source_object": subject,
                "target_attribute": background_target,
                "target_region": "background",
                "required_after": [f"the background is a {background_target}"],
                "forbidden_after": ["the original background remains unchanged"],
                "preserve": [subject, "lighting consistency", "camera viewpoint"],
            }
        if edit_type == "style_transfer":
            return {
                "edit_type": "style_transfer",
                "instruction": f"Convert the image into a {style_target} style while preserving the scene layout and main objects.",
                "target_style": style_target,
                "target_region": "whole image",
                "required_after": [f"the image has a {style_target} style"],
                "forbidden_after": ["the image remains in the original photographic style"],
                "preserve": ["scene layout", "main objects", "camera viewpoint"],
            }
        if edit_type == "local_enhancement" and source_object:
            return {
                "edit_type": "local_enhancement",
                "instruction": f"Enhance the detail and clarity of the {source_object} without changing the rest of the image.",
                "source_object": source_object,
                "target_region": source_object,
                "required_after": [f"the {source_object} has clearer local detail"],
                "forbidden_after": [f"the {source_object} stays blurry or unchanged"],
                "preserve": preserve,
            }
        if edit_type == "subject_extraction" and source_object:
            return {
                "edit_type": "subject_extraction",
                "instruction": (
                    f"Keep only the {source_object}; remove the original surrounding scene and place "
                    f"the {source_object} centered on a clean plain white background while preserving "
                    "its shape, identity, viewpoint, and boundary details."
                ),
                "source_object": source_object,
                "target_region": source_object,
                "required_after": [
                    f"only the {source_object} remains visible",
                    f"the {source_object} is centered on a clean plain white or neutral background",
                ],
                "forbidden_after": [
                    f"the original surrounding scene remains around the {source_object}",
                    f"other original scene objects remain around the {source_object}",
                    f"the {source_object} is deformed, cropped away, or missing boundary parts",
                ],
                "preserve": [
                    f"{source_object} identity",
                    f"{source_object} pose and shape",
                    "subject boundary detail",
                ],
            }
        if edit_type == "action_change":
            subject = source_object or "main subject"
            reference = str((record.metadata or {}).get("reference_instruction") or "").strip()
            action_clauses = [
                "raise one arm",
                "turn the head slightly to the side",
                "take a small step forward",
                "lift one hand",
                "tilt the head slightly",
            ]
            action = action_clauses[stable_choice % len(action_clauses)]
            instruction = reference or (
                f"Make the {subject} {action} while keeping its identity, clothing, and the "
                "background unchanged."
            )
            return {
                "edit_type": "action_change",
                "instruction": instruction,
                "source_object": subject,
                "target_region": subject,
                "required_after": [
                    f"the {subject} is performing the new pose or action described in the instruction"
                ],
                "forbidden_after": [f"the {subject} keeps its original pose"],
                "preserve": preserve,
            }
        if edit_type == "compose":
            reference = str((record.metadata or {}).get("reference_instruction") or "").strip()
            adjust_clauses = [
                "increase the brightness of the background",
                "increase the contrast of the scene",
                "make the background warmer",
                "boost the color saturation of the scene",
            ]
            adjust = adjust_clauses[stable_choice % len(adjust_clauses)]
            instruction = reference or (
                f"Remove the {source_object or 'secondary object'} {target_region_phrase}, and {adjust}."
            )
            removed_match = re.search(
                r"\b(?:remove|delete|erase)\s+(?:the\s+)?(.+?)\s*(?:,?\s+and\b|,|\.|$)",
                instruction,
                flags=re.IGNORECASE,
            )
            removed = (
                source_object
                or (removed_match.group(1).strip() if removed_match else None)
                or "secondary object"
            )
            return {
                "edit_type": "compose",
                "instruction": instruction,
                "source_object": removed,
                "target_region": target_region,
                "required_after": [
                    f"the {removed} has been removed and the area is cleanly filled",
                    "the requested scene adjustment is clearly applied",
                ],
                "forbidden_after": [
                    f"the {removed} remains visible",
                    "the requested scene adjustment is missing",
                ],
                "preserve": preserve,
            }
        return None

    def _ensure_model(self):
        if self.model is not None and self.processor is not None:
            return self.model, self.processor
        import torch

        from qwen_edit_project.utils.device import resolve_torch_dtype

        if self.model_class == "qwen2_5_vl":
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
        else:
            try:
                from transformers import AutoModelForImageTextToText as ModelClass
            except ImportError:
                try:
                    from transformers import AutoModelForVision2Seq as ModelClass
                except ImportError:
                    from transformers import AutoModelForCausalLM as ModelClass
        from transformers import AutoProcessor

        resolved_device = self._resolve_device(self.device)
        dtype = resolve_torch_dtype(torch, self.torch_dtype, resolved_device)
        processor_kwargs = {
            "trust_remote_code": True,
            "local_files_only": self.local_files_only,
        }
        if self.processor_subfolder:
            processor_kwargs["subfolder"] = self.processor_subfolder
        model_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "local_files_only": self.local_files_only,
            "low_cpu_mem_usage": bool(self.config.get("low_cpu_mem_usage", True)),
        }
        if self.model_subfolder:
            model_kwargs["subfolder"] = self.model_subfolder
        if bool(self.config.get("load_direct_to_device", True)):
            if resolved_device.type == "cuda":
                model_kwargs["device_map"] = {"": "cuda:0"}
            else:
                model_kwargs["device_map"] = {"": "cpu"}
        self.processor = AutoProcessor.from_pretrained(self.model_name_or_path, **processor_kwargs)
        self.model = ModelClass.from_pretrained(self.model_name_or_path, **model_kwargs)
        if self.checkpoint_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, self.checkpoint_path)
        if "device_map" not in model_kwargs:
            self.model.to(resolved_device)
        self.model.eval()
        return self.model, self.processor

    def _messages(
        self,
        record: UnlabeledImageRecord,
        difficulty_level: int,
        proposals_per_image: int,
        *,
        attempt_index: int = 0,
    ) -> list[dict[str, Any]]:
        metadata_hint = ""
        if record.caption:
            metadata_hint += f"\nImage caption: {record.caption}"
        if record.metadata:
            keys = [
                "primary_family",
                "edit_families",
                "objects",
                "scene",
                "style",
                "source_instruction",
                "source_edit_type",
                "removable_object_description",
                "removable_object_score",
                "small_object_separability",
                "removal_background_fill_score",
                "experiment_focus",
                "target_edit_types",
                "avoid_edit_types",
                "scheduled_edit_type",
                "reference_instruction",
                "underrepresented_edit_types",
                "scheduled_family",
            ]
            parts = [f"{key}: {record.metadata[key]}" for key in keys if key in record.metadata]
            if parts:
                metadata_hint += "\nMetadata: " + "; ".join(parts)
                metadata_hint += (
                    "\nMetadata hints are optional weak hints from dataset/filtering code. Verify them "
                    "against the image; if metadata conflicts with visible content, trust the image."
                )
        focus_edit_types = self.config.get("focus_edit_types") or self.config.get("target_edit_types")
        avoid_edit_types = self.config.get("avoid_edit_types")
        focus_hint = ""
        if focus_edit_types:
            focus_hint += (
                "\nExperiment focus: propose only edits from these edit_type values when visually feasible: "
                f"{focus_edit_types}."
            )
        if avoid_edit_types:
            focus_hint += (
                "\nAvoid these edit_type values for this run unless there is no feasible alternative: "
                f"{avoid_edit_types}."
            )
        coverage_edit_types = self.config.get("coverage_edit_types") or self.config.get("curriculum_edit_types")
        if coverage_edit_types:
            focus_hint += (
                "\nCoverage objective: across rounds, prefer underrepresented medium-hard local edits from "
                f"{coverage_edit_types}. Do not repeatedly choose only easy additions or global color changes "
                "when removal, replacement, spatial, material, or attribute edits are visually feasible."
            )
        coverage_guidance = str(self.config.get("coverage_guidance", "")).strip()
        if coverage_guidance:
            focus_hint += "\n" + coverage_guidance
        focus_values = {str(item) for item in (focus_edit_types or [])}
        if focus_values & {"object_removal", "object_replacement"}:
            focus_hint += (
                "\nObject-edit acceptance guidance: choose localized, clearly separable targets. "
                "Do not ask to remove or replace the main person, animal, vehicle, large furniture, "
                "or the whole background. Prefer accessories, small foreground/background objects, "
                "cups, hats, bags, toys, tools, or other physical non-text secondary objects. "
                "For subject_extraction, choose one clearly visible foreground subject or bounded object "
                "and isolate it from the background while preserving its identity, pose, and boundaries. "
                "For object_removal, the "
                "instruction must explicitly say to remove the object completely and fill the area "
                "naturally with the surrounding scene. For object_replacement, keep the same "
                "location and approximate size, and name both the old object and concrete new object. "
                "For both removal and replacement, target_region must be spatially grounded against "
                "stable visible context, for example 'above the airplane', 'on the table', or "
                "'left of the person'; avoid generic target regions when a visible anchor exists. "
                "For replacement, the anchor must be a different stable object or scene element: never "
                "place the replacement on or inside the source object that is being replaced. "
                "For text-bearing objects such as signs, posters, labels, or books, name the visible "
                "object generically and do not copy exact written text or quoted strings into any field. "
                "For removal/replacement training, avoid text-bearing or label-like targets such as "
                "price tags, stickers, barcodes, printed labels, logos, posters, signs, books, comic covers, "
                "or visible written text."
            )
        target_edit_type = self._record_target_edit_type(record)
        if target_edit_type:
            focus_hint += (
                "\nMandatory edit-type constraint for this sample: every returned proposal must use "
                f"edit_type={target_edit_type}. If no visually safe and feasible proposal of exactly this "
                "edit_type exists, return {\"proposals\": []}. Do not substitute another edit_type."
            )
            removable_hint = str(record.metadata.get("removable_object_description") or "").strip()
            if removable_hint and target_edit_type in {
                "object_removal",
                "object_replacement",
                "subject_extraction",
                "attribute_change",
                "color_change",
                "material_change",
                "local_enhancement",
                "spatial_move",
            }:
                focus_hint += (
                    "\nPreferred visible target hint from the image-selection audit: "
                    f"{removable_hint}. If this target is visibly present and safe, use it as the "
                    "source_object or bounded edited object and write an anchored target_region around "
                    "it. If it is text-bearing, a main subject, or not actually visible, choose a safer "
                    "secondary visible target of the same mandatory edit_type or return {\"proposals\": []}."
                )
            if target_edit_type == "object_addition":
                focus_hint += (
                    " Add a small concrete object in a plausible empty region while preserving all existing "
                    "objects and the global scene. target_region must describe a visible empty area and "
                    "must not overlap an important subject. target_object is required and must exactly name "
                    "the new object. forbidden_after must name that same target_object and state that it is "
                    "absent from target_region; this is the positive failure state used by the verifier."
                )
            elif target_edit_type == "object_replacement":
                focus_hint += (
                    " Replace a small existing secondary object with a concrete new object at the same "
                    "location and approximate size. Include a spatial target_region anchored to a stable "
                    "visible object. source_object must be visible in the original image, and target_object "
                    "must be a different concrete noun. Do not use object_replacement for color, material, "
                    "texture, style, size, expression, or other attribute changes to the same object; for "
                    "example, 'red bow tie to blue bow tie' is color_change, not object_replacement. If "
                    "the only feasible change is an attribute edit, return {\"proposals\": []}."
                )
            elif target_edit_type == "object_removal":
                focus_hint += (
                    " Remove a small existing secondary object completely and fill the area naturally. "
                    "Include a spatial target_region anchored to a stable visible object. source_object "
                    "must be visible in the original image; do not use the object name alone as the region."
                )
            elif target_edit_type == "subject_extraction":
                focus_hint += (
                    " Perform a strict keep-only cutout edit: keep exactly one visible foreground subject "
                    "or bounded object, remove the original surrounding scene and other objects, and place "
                    "the subject centered on a clean plain white or neutral background. Preserve the "
                    "subject identity, pose, shape, fine boundary details, and viewpoint. Do not use this for object removal, "
                    "background replacement, cropping only, or whole-scene style transfer. source_object "
                    "must be the subject being isolated, and target_region must identify that subject."
                )
            elif target_edit_type == "spatial_move":
                focus_hint += (
                    " Move one small or medium physical non-text object by a small local amount inside "
                    "the same scene. Avoid people, faces, bodies, animals, vehicles, and dominant main "
                    "subjects. Include source_location, target_location, and an anchored target_region."
                )
            elif target_edit_type == "background_change":
                focus_hint += (
                    " Change only the background, backdrop, or surrounding environment behind the main "
                    "subject. Do not replace, remove, or add a foreground object. target_region must "
                    "explicitly be background, backdrop, surroundings, environment, or behind a visible "
                    "subject, and preserve must name the foreground subjects or objects to keep unchanged. "
                    "The instruction must name the edited area as the background/backdrop/scene/environment "
                    "itself. Do not write instructions such as 'replace the floor', 'replace the rug', "
                    "'replace the wall', 'replace the fireplace', 'replace the furniture', or 'replace the "
                    "surrounding area'; those are local object/surface replacements, not background_change."
                )
            elif target_edit_type == "style_transfer":
                focus_hint += (
                    " Apply a global visual style to the whole image or whole scene while preserving "
                    "layout and object identities. Do not transfer texture or style from one local object "
                    "to another object, and do not target a local object part. target_region should be "
                    "whole image, entire image, whole scene, or full scene."
                )
            elif target_edit_type in {
                "attribute_change",
                "color_change",
                "material_change",
                "local_enhancement",
            }:
                focus_hint += (
                    " Choose one bounded visible object or object part as source_object. Do not target "
                    "broad scene surfaces or whole-scene regions such as the background, sky, wall, floor, "
                    "ground, road, room, porch, deck, patio, or landscape. Avoid targets that dominate "
                    "most of the frame; prefer small or medium targets with a clear surrounding context. "
                    "If the only feasible target is a broad scene surface or a full-frame object, return "
                    "{\"proposals\": []}."
                )
                if target_edit_type == "material_change":
                    focus_hint += (
                        " For material_change, change the apparent physical material only, e.g. metal, "
                        "wood, glass, ceramic, leather, fabric, rubber, marble, stone, or plastic. Do not "
                        "return a color-only edit such as changing red to green; use color_change for that. "
                        "Do not turn the source object into a different object, food, animal, person, or "
                        "category; for example, 'chicken pieces to raw fish fillets' is object_replacement, "
                        "not material_change. Fill source_material with the object's currently visible "
                        "physical material and target_material with a genuinely different requested "
                        "material. Do not propose the object's existing or an equivalent material."
                    )
                elif target_edit_type == "color_change":
                    focus_hint += (
                        " For color_change, change only the visible color of the source_object while keeping "
                        "its material, shape, identity, and surrounding context unchanged. Fill "
                        "source_attribute with the currently visible color and target_attribute with a "
                        "different explicit target color."
                    )
                elif target_edit_type == "attribute_change":
                    focus_hint += (
                        " For attribute_change, alter a visible non-color attribute such as shape, size, "
                        "openness, cleanliness, expression, or state; avoid pure material replacement. "
                        "Fill source_attribute and target_attribute with two concrete, visibly distinct "
                        "states. Never use vague targets such as 'different', 'another', 'new', or 'changed'."
                    )
                elif target_edit_type == "local_enhancement":
                    focus_hint += (
                        " For local_enhancement, improve detail, clarity, or texture of the source_object "
                        "without changing its identity, color category, material, location, or size."
                    )
            elif target_edit_type == "action_change":
                focus_hint += (
                    " Change the pose, gesture, or action of one clearly visible person or animal subject "
                    "while preserving its identity, clothing, the background, and the rest of the scene. "
                    "source_object must name that subject, and target_region must anchor to it. Describe one "
                    "concrete new pose or motion, for example raising an arm, turning the head, sitting "
                    "down, or stepping forward. Do not turn the subject into a different person or animal, "
                    "do not swap clothing or background, and do not edit facial identity. required_after "
                    "must state the new visible pose or action, and forbidden_after must state that the "
                    "subject keeps its original pose."
                )
            elif target_edit_type == "compose":
                focus_hint += (
                    " Produce one compound edit that combines a localized object removal with a scene-level "
                    "adjustment, written as a single instruction such as 'Remove the umbrella, and increase "
                    "the brightness of the background.'. Name the removed secondary object (not the main "
                    "subject) and the explicit second change to brightness, contrast, saturation, warmth, "
                    "blur, or sharpness of the background or whole scene. required_after must list both the "
                    "removal check and the adjustment check; forbidden_after must list the removed object "
                    "remaining visible and the adjustment being absent. Preserve the main subject and "
                    "unrelated objects."
                )
        if attempt_index > 0:
            focus_hint += (
                "\nRetry correction: the previous response did not yield a usable proposal. Return strict "
                "JSON only in the exact form {\"proposals\": [...]}. Prefer one simpler, clearly visible, "
                "localized proposal of the mandatory edit_type over an ambitious edit. If and only if the "
                "mandatory edit_type is genuinely unsafe or not visually feasible, return {\"proposals\": []}. "
                "For attribute_change, color_change, material_change, local_enhancement, and spatial_move, "
                "do not omit source_object; it must name the bounded visible object or object part being "
                "edited, and target_region must spatially anchor that same object. Do not relabel an edit "
                "from another family to satisfy the mandatory edit_type."
            )
            focus_hint += self._retry_feedback_hint(record_key=record.key, target_edit_type=target_edit_type)
        prompt = structured_edit_prompt(difficulty_level, proposals_per_image) + metadata_hint + focus_hint
        return [
            {
                "role": "system",
                "content": (
                    "You are the Qwen-Image-Edit internal VLM proposer. Inspect the image and return "
                    "strict JSON only. Every proposal must be directly grounded in visible image content "
                    "and must satisfy the requested edit_type exactly."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(record.image_path)},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

    @staticmethod
    def _to_payloads(decoded: str) -> list[dict[str, Any]]:
        parsed = extract_json_object(decoded)
        if parsed is None:
            return []
        proposals = parsed.get("proposals")
        if isinstance(proposals, list):
            return [item for item in proposals if isinstance(item, dict)]
        return [parsed]

    def _generate_payloads(
        self,
        record: UnlabeledImageRecord,
        difficulty_level: int,
        proposals_per_image: int,
        *,
        attempt_index: int = 0,
    ) -> list[dict[str, Any]]:
        import torch

        model, processor = self._ensure_model()
        messages = self._messages(
            record,
            difficulty_level,
            proposals_per_image,
            attempt_index=attempt_index,
        )
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        with Image.open(record.image_path) as image_handle:
            image = image_handle.convert("RGB")
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        max_new_tokens = int(self.generation.get("max_new_tokens", 512))
        temperature = float(self.generation.get("temperature", 0.7))
        top_p = float(self.generation.get("top_p", 0.9))
        do_sample = bool(self.generation.get("do_sample", temperature > 0.0))
        with torch.no_grad():
            generation_kwargs = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
            }
            if do_sample:
                generation_kwargs["temperature"] = temperature
                generation_kwargs["top_p"] = top_p
            output_ids = model.generate(**generation_kwargs)
        generated = output_ids[:, inputs["input_ids"].shape[1] :]
        decoded = processor.batch_decode(generated, skip_special_tokens=True)[0]
        parsed = extract_json_object(decoded)
        if parsed is None:
            payloads: list[dict[str, Any]] = []
            parse_status = "no_json_object"
        else:
            proposals = parsed.get("proposals")
            if isinstance(proposals, list):
                payloads = [item for item in proposals if isinstance(item, dict)]
                parse_status = "empty_proposals" if not payloads else "proposals_list"
            else:
                payloads = [parsed]
                parse_status = "single_proposal_object"
        self._record_proposal_diagnostic(
            record_key=record.key,
            target_edit_type=self._record_target_edit_type(record),
            attempt_index=attempt_index,
            stage="generation",
            reason=parse_status,
            decoded=decoded,
            payload_count=len(payloads),
        )
        if not payloads:
            LOGGER.info(
                "VLM proposer produced no parseable JSON payloads: record_key=%s target_edit_type=%s decoded=%s",
                record.key,
                self._record_target_edit_type(record),
                re.sub(r"\s+", " ", decoded.strip())[:1000],
            )
        return payloads

    def propose(
        self,
        record: UnlabeledImageRecord,
        round_index: int,
        difficulty_level: int,
        proposals_per_image: int,
        seed: int,
    ) -> list[EditProposal]:
        self.last_proposal_diagnostics = []
        rng = random.Random(seed + round_index * 100_003 + sum(ord(char) for char in record.key))
        target_edit_type = self._record_target_edit_type(record)
        use_template_bootstrap = bool(target_edit_type) and (
            self.force_template_from_record
            or (self.template_bootstrap_rounds > 0 and round_index < self.template_bootstrap_rounds)
        )
        if target_edit_type and use_template_bootstrap:
            payload = self._template_target_payload(record, target_edit_type, seed)
            if payload is None and self.force_template_from_record:
                return []
            if payload is not None:
                structured_edit = normalize_structured_edit(payload, instruction=str(payload["instruction"]))
                if self._structured_edit_allowed(structured_edit) and structured_edit.get("edit_type") == target_edit_type:
                    definition = proposal_definition_from_structured_edit(
                        structured_edit,
                        proposal_index=0,
                        difficulty_level=difficulty_level,
                    )
                    return [
                        EditProposal(
                            record_key=record.key,
                            round_index=round_index,
                            proposal_index=0,
                            definition=definition,
                            difficulty_level=difficulty_level,
                            instruction=definition.instruction,
                            structured_edit=structured_edit,
                        )
                    ]
            if self.force_template_from_record:
                return []
        if self.scripted_probability > 0 and rng.random() < self.scripted_probability:
            scripted = self.scripted.propose(record, round_index, difficulty_level, proposals_per_image, seed)
            scripted = [proposal for proposal in scripted if self._structured_edit_allowed(proposal.structured_edit)]
            if target_edit_type:
                scripted = [
                    proposal for proposal in scripted if proposal.structured_edit.get("edit_type") == target_edit_type
                ]
            if scripted:
                return scripted
            if target_edit_type and self.template_fallback_on_target_miss:
                payload = self._template_target_payload(record, target_edit_type, seed)
                if payload is not None:
                    structured_edit = normalize_structured_edit(payload, instruction=str(payload["instruction"]))
                    if self._structured_edit_allowed(structured_edit) and structured_edit.get("edit_type") == target_edit_type:
                        definition = proposal_definition_from_structured_edit(
                            structured_edit,
                            proposal_index=0,
                            difficulty_level=difficulty_level,
                        )
                        return [
                            EditProposal(
                                record_key=record.key,
                                round_index=round_index,
                                proposal_index=0,
                                definition=definition,
                                difficulty_level=difficulty_level,
                                instruction=definition.instruction,
                                structured_edit=structured_edit,
                            )
                        ]
            return []
        proposals: list[EditProposal] = []
        for attempt_index in range(self.max_generation_attempts):
            try:
                payloads = self._generate_payloads(
                    record,
                    difficulty_level,
                    proposals_per_image,
                    attempt_index=attempt_index,
                )
            except Exception:
                if not self.fallback_on_error:
                    raise
                fallback = self.scripted.propose(record, round_index, difficulty_level, proposals_per_image, seed)
                return [
                    proposal for proposal in fallback if self._structured_edit_allowed(proposal.structured_edit)
                ]

            for payload in payloads:
                instruction = str(payload.get("instruction", "")).strip()
                if not instruction:
                    self._record_proposal_diagnostic(
                        record_key=record.key,
                        target_edit_type=target_edit_type,
                        attempt_index=attempt_index,
                        stage="filter",
                        reason="proposal_instruction_missing",
                        payload=payload,
                    )
                    LOGGER.info(
                        "Filtered VLM proposal: record_key=%s target_edit_type=%s reason=proposal_instruction_missing payload=%s",
                        record.key,
                        target_edit_type,
                        json.dumps(payload, ensure_ascii=False)[:700],
                    )
                    continue
                structured_edit = normalize_structured_edit(payload, instruction=instruction)
                filter_reason = None
                if not self._edit_type_allowed(structured_edit):
                    filter_reason = "proposal_edit_type_not_allowed"
                else:
                    filter_reason = self._proposal_contract_filter_reason(structured_edit)
                if filter_reason is not None:
                    self._record_proposal_diagnostic(
                        record_key=record.key,
                        target_edit_type=target_edit_type,
                        attempt_index=attempt_index,
                        stage="filter",
                        reason=filter_reason,
                        payload=payload,
                        structured_edit=structured_edit,
                    )
                    LOGGER.info(
                        "Filtered VLM proposal: record_key=%s target_edit_type=%s reason=%s payload=%s",
                        record.key,
                        target_edit_type,
                        filter_reason,
                        json.dumps(payload, ensure_ascii=False)[:700],
                    )
                    continue
                if target_edit_type and structured_edit.get("edit_type") != target_edit_type:
                    self._record_proposal_diagnostic(
                        record_key=record.key,
                        target_edit_type=target_edit_type,
                        attempt_index=attempt_index,
                        stage="filter",
                        reason="target_edit_type_mismatch",
                        payload=payload,
                        structured_edit=structured_edit,
                    )
                    LOGGER.info(
                        "Filtered VLM proposal: record_key=%s target_edit_type=%s reason=target_edit_type_mismatch payload=%s",
                        record.key,
                        target_edit_type,
                        json.dumps(payload, ensure_ascii=False)[:700],
                    )
                    continue
                definition = proposal_definition_from_structured_edit(
                    structured_edit,
                    proposal_index=len(proposals),
                    difficulty_level=difficulty_level,
                )
                proposals.append(
                    EditProposal(
                        record_key=record.key,
                        round_index=round_index,
                        proposal_index=len(proposals),
                        definition=definition,
                        difficulty_level=difficulty_level,
                        instruction=definition.instruction,
                        structured_edit=structured_edit,
                    )
                )
                self._record_proposal_diagnostic(
                    record_key=record.key,
                    target_edit_type=target_edit_type,
                    attempt_index=attempt_index,
                    stage="accepted",
                    reason="accepted",
                    payload=payload,
                    structured_edit=structured_edit,
                )
                if len(proposals) >= proposals_per_image:
                    return proposals
            if proposals:
                return proposals
        if proposals:
            return proposals
        LOGGER.info(
            "VLM proposer exhausted attempts without an accepted proposal: record_key=%s target_edit_type=%s attempts=%s",
            record.key,
            target_edit_type,
            self.max_generation_attempts,
        )
        self._record_proposal_diagnostic(
            record_key=record.key,
            target_edit_type=target_edit_type,
            stage="summary",
            reason="exhausted_no_accepted_proposal",
            payload_count=0,
        )
        target_edit_type = self._record_target_edit_type(record)
        if target_edit_type and self.template_fallback_on_target_miss:
            payload = self._template_target_payload(record, target_edit_type, seed)
            if payload is not None:
                structured_edit = normalize_structured_edit(payload, instruction=str(payload["instruction"]))
                if self._structured_edit_allowed(structured_edit) and structured_edit.get("edit_type") == target_edit_type:
                    definition = proposal_definition_from_structured_edit(
                        structured_edit,
                        proposal_index=0,
                        difficulty_level=difficulty_level,
                    )
                    return [
                        EditProposal(
                            record_key=record.key,
                            round_index=round_index,
                            proposal_index=0,
                            definition=definition,
                            difficulty_level=difficulty_level,
                            instruction=definition.instruction,
                            structured_edit=structured_edit,
                        )
                    ]
        if self.strict_edit_type_filter or not self.fallback_on_error:
            return []
        return [
            proposal
            for proposal in self.scripted.propose(record, round_index, difficulty_level, proposals_per_image, seed)
            if self._structured_edit_allowed(proposal.structured_edit)
        ]


class PillowPrototypeEditor:
    def edit_image(self, image: Image.Image, instruction: str, operation_id: str | None = None) -> Image.Image:
        if operation_id is None:
            raise ValueError("PillowPrototypeEditor requires operation_id for deterministic editing.")
        return _apply_pillow_operation(image.convert("RGB"), operation_id)

    def edit(self, record: UnlabeledImageRecord, proposal: EditProposal) -> Image.Image:
        image = Image.open(record.image_path).convert("RGB")
        return self.edit_image(image, proposal.instruction, proposal.definition.operation_id)

    def edit_candidate(self, record: UnlabeledImageRecord, proposal: EditProposal, candidate_index: int, seed: int) -> Image.Image:
        image = Image.open(record.image_path).convert("RGB")
        edited = self.edit_image(image, proposal.instruction, proposal.definition.operation_id)
        if candidate_index == 0:
            return edited
        rng = random.Random(seed + candidate_index * 7919)
        brightness = 0.96 + rng.random() * 0.08
        contrast = 0.96 + rng.random() * 0.08
        edited = ImageEnhance.Brightness(edited).enhance(brightness)
        return ImageEnhance.Contrast(edited).enhance(contrast)


class QwenEditEditor:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.device = config.get("device", "auto")
        self.generation = dict(config.get("generation", {}))
        self.drop_generation_modules_for_scoring = bool(config.get("drop_generation_modules_for_scoring", True))
        self.pipeline = None
        self.current_checkpoint_path = config["model"].get("checkpoint_path")
        self.generation_modules_dropped = False

    def set_checkpoint_path(self, checkpoint_path: str | None) -> None:
        if checkpoint_path == self.current_checkpoint_path:
            return
        self.current_checkpoint_path = checkpoint_path
        self.pipeline = None
        self.generation_modules_dropped = False

    def set_model_checkpoint(
        self,
        checkpoint_path: str | None,
        *,
        model_type: str | None = None,
        backend: str | None = None,
    ) -> None:
        """Switch the editor to a checkpoint with an explicitly compatible loader.

        Self-evolve training currently writes checkpoints through DiffSynth. Reloading
        those weights through the official Diffusers base pipeline is not a valid
        research comparison unless the checkpoint is known to be Diffusers-compatible.
        This helper updates the checkpoint, model type, and backend as one atomic
        editor state transition so later rounds cannot accidentally mix them.
        """
        model_cfg = self.config["model"]
        old_state = (
            self.current_checkpoint_path,
            model_cfg.get("model_type", "base"),
            model_cfg.get("backend", "diffsynth"),
        )
        if model_type is not None:
            model_cfg["model_type"] = model_type
        if backend is not None:
            model_cfg["backend"] = backend
        self.current_checkpoint_path = checkpoint_path
        new_state = (
            self.current_checkpoint_path,
            model_cfg.get("model_type", "base"),
            model_cfg.get("backend", "diffsynth"),
        )
        if new_state != old_state:
            self.pipeline = None
            self.generation_modules_dropped = False

    def model_state(self) -> dict[str, Any]:
        model_cfg = self.config.get("model", {})
        return {
            "backend": model_cfg.get("backend", "diffsynth"),
            "model_type": model_cfg.get("model_type", "base"),
            "checkpoint_path": self.current_checkpoint_path,
            "base_model": model_cfg.get("base_model"),
        }

    def release_memory(self) -> None:
        if self.pipeline is not None:
            try:
                if hasattr(self.pipeline, "to"):
                    self.pipeline.to("cpu")
                for name in ("transformer", "dit", "unet", "text_encoder", "vae"):
                    self._move_module(getattr(self.pipeline, name, None), "cpu")
            except Exception:
                pass
        self.pipeline = None
        self.generation_modules_dropped = False
        self._empty_cuda_cache()

    def _ensure_pipeline(self):
        if self.pipeline is not None:
            return self.pipeline
        model_cfg = self.config["model"]
        self.pipeline = load_qwen_edit_pipeline(
            model_id_with_origin_paths=model_cfg["model_id_with_origin_paths"],
            checkpoint_path=self.current_checkpoint_path,
            model_type=model_cfg.get("model_type", "base"),
            device=self.device,
            processor_model_id=model_cfg.get("processor_model_id", "Qwen/Qwen-Image-Edit"),
            torch_dtype=model_cfg.get("torch_dtype", "auto"),
            backend=model_cfg.get("backend", "diffsynth"),
            base_model=model_cfg.get("base_model"),
            local_files_only=bool(model_cfg.get("local_files_only", False)),
        )
        self.generation_modules_dropped = False
        return self.pipeline

    def _resolved_torch_device(self):
        import torch

        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    @staticmethod
    def _empty_cuda_cache() -> None:
        try:
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            return

    @staticmethod
    def _move_module(module: Any, device: Any) -> None:
        if module is not None and hasattr(module, "to"):
            module.to(device)

    def prepare_for_generation(self) -> None:
        """Ensure heavyweight generation modules are on the target device."""
        if self.generation_modules_dropped:
            self.release_memory()
        pipeline = self._ensure_pipeline()
        device = self._resolved_torch_device()
        if hasattr(pipeline, "to"):
            pipeline.to(device)
        for name in ("transformer", "dit", "unet", "text_encoder", "vae"):
            self._move_module(getattr(pipeline, name, None), device)
        self._empty_cuda_cache()

    def prepare_for_internal_scoring(self, scoring_device: str | None = None) -> None:
        """Free generation-only GPU memory before internal CEPR scoring.

        Qwen-Image-Edit generation fits on a large H200, but it leaves almost no free
        memory for extra internal text/VAE reward passes. CEPR scoring only needs the
        editor's text/understanding path and VAE latents, not the diffusion transformer,
        so we drop generation-only modules and reload them before the next generation.
        """
        pipeline = self._ensure_pipeline()
        if not self.drop_generation_modules_for_scoring:
            device = scoring_device or self._resolved_torch_device()
            for name in ("text_encoder", "vae"):
                self._move_module(getattr(pipeline, name, None), device)
            self._empty_cuda_cache()
            return
        # CEPR scoring only needs the text/understanding path and VAE. Moving the
        # full diffusion transformer to CPU can exceed host RAM on long runs, so
        # drop generation-only modules and reload them before the next generation.
        for name in ("transformer", "dit", "unet"):
            module = getattr(pipeline, name, None)
            if module is not None:
                try:
                    setattr(pipeline, name, None)
                except Exception:
                    self._move_module(module, "cpu")
                else:
                    self.generation_modules_dropped = True
                del module
        self._empty_cuda_cache()
        device = scoring_device or self._resolved_torch_device()
        for name in ("text_encoder", "vae"):
            self._move_module(getattr(pipeline, name, None), device)
        self._empty_cuda_cache()

    def edit_image(self, image: Image.Image, instruction: str, operation_id: str | None = None) -> Image.Image:
        self.prepare_for_generation()
        pipeline = self._ensure_pipeline()
        generation = dict(self.generation)
        if bool(generation.get("preserve_input_resolution", True)):
            generation["width"], generation["height"] = image.size
        else:
            generation.pop("width", None)
            generation.pop("height", None)
        prompt = polish_prompt(instruction, use_prompt_polish=False, image_context=image)
        try:
            output = render_edit(pipeline, prompt, [image.convert("RGB")], generation)
            return output.images[0] if hasattr(output, "images") else output
        finally:
            self._empty_cuda_cache()

    @staticmethod
    def _generation_region_phrase(region: Any) -> str:
        text = str(region or "").strip()
        if not text or text == "main visible target":
            return "from the target region"
        if text.lower().startswith(
            (
                "on ",
                "in ",
                "at ",
                "under ",
                "beneath ",
                "below ",
                "above ",
                "beside ",
                "near ",
                "next to ",
                "left of ",
                "right of ",
                "in front of ",
                "behind ",
                "between ",
                "attached to ",
                "held by ",
                "worn by ",
                "around ",
            )
        ):
            return text
        return f"at {text}"

    def _object_removal_instruction_variant(
        self,
        proposal: EditProposal,
        candidate_index: int,
    ) -> str:
        variant_cfg = self.generation.get("object_removal_prompt_variants", {})
        if isinstance(variant_cfg, dict) and not bool(variant_cfg.get("enabled", True)):
            return proposal.instruction
        spec = normalize_structured_edit(
            proposal.structured_edit,
            instruction=proposal.instruction,
            family=proposal.definition.family,
        )
        if str(spec.get("edit_type", "")) != "object_removal":
            return proposal.instruction
        source_object = str(spec.get("source_object") or spec.get("target") or "").strip()
        if not source_object:
            return proposal.instruction
        region_phrase = self._generation_region_phrase(spec.get("target_region"))
        preserve_items = [
            str(item).strip()
            for item in spec.get("preserve", [])
            if str(item).strip()
        ][:3]
        preserve_text = ", ".join(preserve_items) if preserve_items else "all unrelated content"
        from_region = (
            f"the area {region_phrase}"
            if region_phrase.lower().startswith(
                (
                    "on ",
                    "in ",
                    "at ",
                    "under ",
                    "beneath ",
                    "below ",
                    "above ",
                    "beside ",
                    "near ",
                    "next to ",
                    "left of ",
                    "right of ",
                    "in front of ",
                    "behind ",
                    "between ",
                    "attached to ",
                    "held by ",
                    "worn by ",
                    "around ",
                )
            )
            else region_phrase
        )
        variants = [
            proposal.instruction,
            (
                f"Erase only the {source_object} {region_phrase}. Fill the empty area with natural "
                f"surrounding texture. The {source_object} must be completely absent. Preserve {preserve_text}."
            ),
            (
                f"Inpaint the area where the {source_object} appears {region_phrase}: remove the "
                f"{source_object} entirely and replace it with matching background. Do not leave any visible "
                f"{source_object} remnants. Preserve {preserve_text}."
            ),
            (
                f"Delete the {source_object} {region_phrase}. No part of the {source_object} should remain "
                f"visible after the edit. Keep the scene geometry, lighting, and {preserve_text} unchanged."
            ),
            (
                f"Remove the {source_object} completely from {from_region} and naturally reconstruct the "
                f"occluded background. Avoid duplicated edges, shadows, or fragments of the removed object."
            ),
            (
                f"Make the edited image look as if the {source_object} was never present {region_phrase}. "
                f"Use the surrounding scene to fill the region and preserve {preserve_text}."
            ),
        ]
        return variants[candidate_index % len(variants)]

    def edit(self, record: UnlabeledImageRecord, proposal: EditProposal) -> Image.Image:
        image = Image.open(record.image_path).convert("RGB")
        return self.edit_image(image, proposal.instruction, proposal.definition.operation_id)

    def edit_candidate(self, record: UnlabeledImageRecord, proposal: EditProposal, candidate_index: int, seed: int) -> Image.Image:
        image = Image.open(record.image_path).convert("RGB")
        original_seed = self.generation.get("seed")
        self.generation["seed"] = int(seed + candidate_index * 7919)
        instruction = self._object_removal_instruction_variant(proposal, candidate_index)
        try:
            return self.edit_image(image, instruction, proposal.definition.operation_id)
        finally:
            if original_seed is None:
                self.generation.pop("seed", None)
            else:
                self.generation["seed"] = original_seed


class StatSolver:
    def __init__(self, config: dict[str, Any]):
        self.global_weight = float(config.get("global_weight", 0.7))
        self.local_weight = float(config.get("local_weight", 0.3))
        self.acceptance_threshold = float(config.get("acceptance_threshold", 0.72))

    def _global_score(self, definition: ProposalDefinition, original: Image.Image, edited: Image.Image) -> tuple[float, dict[str, float]]:
        if definition.metric in INTERNAL_ONLY_METRICS or definition.verifier == "internal":
            return 0.5, {
                "global_proxy_supported": 0.0,
                "internal_metric_required": 1.0,
                "luminance_delta": 0.0,
                "contrast_delta": 0.0,
                "saturation_delta": 0.0,
                "warmth_delta": 0.0,
                "saturation_level": 0.0,
            }

        original_luminance = luminance_mean(original)
        edited_luminance = luminance_mean(edited)
        original_contrast = luminance_std(original)
        edited_contrast = luminance_std(edited)
        original_saturation = saturation_mean(original)
        edited_saturation = saturation_mean(edited)
        original_warmth = warmth_score(original)
        edited_warmth = warmth_score(edited)

        deltas = {
            "luminance_delta": edited_luminance - original_luminance,
            "contrast_delta": edited_contrast - original_contrast,
            "saturation_delta": edited_saturation - original_saturation,
            "warmth_delta": edited_warmth - original_warmth,
            "saturation_level": edited_saturation,
        }

        if definition.metric == "luminance":
            value = deltas["luminance_delta"]
        elif definition.metric == "contrast":
            value = deltas["contrast_delta"]
        elif definition.metric == "saturation":
            value = deltas["saturation_delta"]
        elif definition.metric == "warmth":
            value = deltas["warmth_delta"]
        elif definition.metric == "saturation_level":
            value = deltas["saturation_level"]
        else:
            raise ValueError(f"Unsupported solver metric: {definition.metric}")

        if definition.direction == "increase":
            score = _clamp(value / definition.target)
        elif definition.direction == "decrease":
            score = _clamp((-value) / definition.target)
        elif definition.direction == "at_most":
            score = 1.0 if value <= definition.target else _clamp(1.0 - (value - definition.target) / (1.0 - definition.target))
        else:
            raise ValueError(f"Unsupported solver direction: {definition.direction}")

        return score, {
            "global_proxy_supported": 1.0,
            "internal_metric_required": 0.0,
            "luminance_delta": deltas["luminance_delta"],
            "contrast_delta": deltas["contrast_delta"],
            "saturation_delta": deltas["saturation_delta"],
            "warmth_delta": deltas["warmth_delta"],
            "saturation_level": deltas["saturation_level"],
        }

    def _local_score(self, proposal: EditProposal, original: Image.Image, edited: Image.Image) -> tuple[float, dict[str, float]]:
        area_fraction = changed_fraction(original, edited)
        area_score = _mean_changed_fraction_score(area_fraction, proposal.definition.expected_changed_fraction)
        edge_score = edge_preservation_score(original, edited)
        local_score = 0.5 * area_score + 0.5 * edge_score
        return local_score, {
            "changed_fraction": area_fraction,
            "area_score": area_score,
            "edge_preservation_score": edge_score,
        }

    def score(
        self,
        proposal: EditProposal,
        original: Image.Image,
        edited: Image.Image,
        editor: Any | None = None,
    ) -> SolverResult:
        global_score, global_signals = self._global_score(proposal.definition, original, edited)
        local_score, local_signals = self._local_score(proposal, original, edited)
        total_score = self.global_weight * global_score + self.local_weight * local_score
        return SolverResult(
            global_score=global_score,
            local_score=local_score,
            total_score=total_score,
            accepted=total_score >= self.acceptance_threshold,
            component_scores={
                "proxy_global_score": global_score,
                "proxy_local_score": local_score,
                "proxy_total_score": total_score,
            },
            signals={**global_signals, **local_signals},
        )


class MultiSignalSolver(StatSolver):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.proxy_weight = float(config.get("proxy_weight", 1.0))
        self.spatial_weight = float(config.get("spatial_weight", 0.0))
        self.cycle_weight = float(config.get("cycle_weight", 0.0))
        self.internal_weight = float(config.get("internal_weight", 0.0))
        self.cycle_neutral_score = float(config.get("cycle_neutral_score", 0.5))
        self.internal_neutral_score = float(config.get("internal_neutral_score", 0.5))
        self.spatial_inside_scale = float(config.get("spatial_inside_scale", 0.10))
        self.spatial_outside_scale = float(config.get("spatial_outside_scale", 0.05))
        self.cycle_change_scale = float(config.get("cycle_change_scale", 0.20))
        self.cycle_mae_scale = float(config.get("cycle_mae_scale", 0.08))
        self.internal_instruction_weight = float(config.get("internal_instruction_weight", 0.65))
        self.internal_prompt_gain_scale = float(config.get("internal_prompt_gain_scale", 0.08))

    def _spatial_score(self, proposal: EditProposal, original: Image.Image, edited: Image.Image) -> tuple[float, dict[str, float]]:
        stats = diff_region_statistics(original, edited)
        area_score = _mean_changed_fraction_score(stats["changed_fraction"], proposal.definition.expected_changed_fraction)
        inside_support = _clamp(stats["inside_mean_delta"] / max(self.spatial_inside_scale, 1e-6))
        outside_preservation = _clamp(1.0 - stats["outside_mean_delta"] / max(self.spatial_outside_scale, 1e-6))
        compactness = stats["mask_compactness"]
        precision = stats["mask_precision"]
        if proposal.definition.scope == "local":
            spatial_score = 0.25 * area_score + 0.25 * inside_support + 0.30 * outside_preservation + 0.20 * compactness
        else:
            spatial_score = 0.35 * area_score + 0.25 * inside_support + 0.25 * outside_preservation + 0.15 * precision
        return spatial_score, {
            "spatial_area_score": area_score,
            "spatial_inside_support": inside_support,
            "spatial_outside_preservation": outside_preservation,
            "spatial_mask_compactness": compactness,
            "spatial_mask_precision": precision,
            "spatial_changed_fraction": stats["changed_fraction"],
            "spatial_inside_mean_delta": stats["inside_mean_delta"],
            "spatial_outside_mean_delta": stats["outside_mean_delta"],
        }

    def _cycle_score(
        self,
        proposal: EditProposal,
        original: Image.Image,
        edited: Image.Image,
        editor: Any | None,
    ) -> tuple[float, dict[str, float]]:
        reverse_proposal = _reverse_proposal(proposal)
        if reverse_proposal is None or editor is None or not hasattr(editor, "edit_image"):
            return self.cycle_neutral_score, {
                "cycle_supported": 0.0,
                "cycle_reconstruction_changed_fraction": 1.0,
                "cycle_reconstruction_edge_score": 0.0,
                "cycle_reconstruction_mae": 1.0,
            }
        try:
            reconstructed = editor.edit_image(edited, reverse_proposal.instruction, reverse_proposal.definition.operation_id)
        except Exception:
            return self.cycle_neutral_score, {
                "cycle_supported": 0.0,
                "cycle_reconstruction_changed_fraction": 1.0,
                "cycle_reconstruction_edge_score": 0.0,
                "cycle_reconstruction_mae": 1.0,
                "cycle_runtime_error": 1.0,
            }
        reconstruction_change = changed_fraction(original, reconstructed)
        reconstruction_edge = edge_preservation_score(original, reconstructed)
        reconstruction_mae = mean_absolute_difference(original, reconstructed)
        change_recovery = _clamp(1.0 - reconstruction_change / max(self.cycle_change_scale, 1e-6))
        mae_recovery = _clamp(1.0 - reconstruction_mae / max(self.cycle_mae_scale, 1e-6))
        cycle_score = 0.4 * change_recovery + 0.3 * reconstruction_edge + 0.3 * mae_recovery
        return cycle_score, {
            "cycle_supported": 1.0,
            "cycle_change_recovery": change_recovery,
            "cycle_reconstruction_changed_fraction": reconstruction_change,
            "cycle_reconstruction_edge_score": reconstruction_edge,
            "cycle_reconstruction_mae": reconstruction_mae,
            "cycle_mae_recovery": mae_recovery,
        }

    def _internal_feature_score(
        self,
        proposal: EditProposal,
        original: Image.Image,
        edited: Image.Image,
        editor: Any | None,
    ) -> tuple[float, dict[str, float]]:
        if not isinstance(editor, QwenEditEditor):
            return self.internal_neutral_score, {
                "internal_supported": 0.0,
                "internal_original_text_similarity": 0.0,
                "internal_edited_text_similarity": 0.0,
                "internal_blank_preservation": 0.0,
                "internal_prompt_gain": 0.0,
            }
        try:
            import torch
            import torch.nn.functional as F

            pipe = editor._ensure_pipeline()
            prompt = polish_prompt(proposal.instruction, use_prompt_polish=False, image_context=original)
            with torch.no_grad():
                text_feature = extract_qwen_text_features(pipe, prompt)["pooled_embedding"][0].float()
                original_prompt_feature = extract_qwen_edit_understanding_features(pipe, prompt, [original])["pooled_embedding"][0].float()
                edited_prompt_feature = extract_qwen_edit_understanding_features(pipe, prompt, [edited])["pooled_embedding"][0].float()
                original_blank_feature = extract_qwen_edit_understanding_features(pipe, " ", [original])["pooled_embedding"][0].float()
                edited_blank_feature = extract_qwen_edit_understanding_features(pipe, " ", [edited])["pooled_embedding"][0].float()

            original_text_similarity = 0.5 * (
                1.0 + float(F.cosine_similarity(original_prompt_feature.unsqueeze(0), text_feature.unsqueeze(0)).item())
            )
            edited_text_similarity = 0.5 * (
                1.0 + float(F.cosine_similarity(edited_prompt_feature.unsqueeze(0), text_feature.unsqueeze(0)).item())
            )
            blank_preservation = 0.5 * (
                1.0 + float(F.cosine_similarity(original_blank_feature.unsqueeze(0), edited_blank_feature.unsqueeze(0)).item())
            )
            prompt_gain = edited_text_similarity - original_text_similarity
            instruction_score = _clamp(0.5 + 0.5 * prompt_gain / max(self.internal_prompt_gain_scale, 1e-6))
            internal_score = (
                self.internal_instruction_weight * instruction_score
                + (1.0 - self.internal_instruction_weight) * blank_preservation
            )
            return internal_score, {
                "internal_supported": 1.0,
                "internal_original_text_similarity": original_text_similarity,
                "internal_edited_text_similarity": edited_text_similarity,
                "internal_blank_preservation": blank_preservation,
                "internal_prompt_gain": prompt_gain,
                "internal_instruction_score": instruction_score,
            }
        except Exception:
            return self.internal_neutral_score, {
                "internal_supported": 0.0,
                "internal_original_text_similarity": 0.0,
                "internal_edited_text_similarity": 0.0,
                "internal_blank_preservation": 0.0,
                "internal_prompt_gain": 0.0,
                "internal_runtime_error": 1.0,
            }

    def score(
        self,
        proposal: EditProposal,
        original: Image.Image,
        edited: Image.Image,
        editor: Any | None = None,
    ) -> SolverResult:
        proxy_result = super().score(proposal, original, edited, editor=editor)
        weighted_total = self.proxy_weight * proxy_result.total_score
        total_weight = self.proxy_weight
        component_scores = dict(proxy_result.component_scores)
        signals = dict(proxy_result.signals)

        if self.spatial_weight > 0.0:
            spatial_score, spatial_signals = self._spatial_score(proposal, original, edited)
            component_scores["spatial_score"] = spatial_score
            signals.update(spatial_signals)
            weighted_total += self.spatial_weight * spatial_score
            total_weight += self.spatial_weight

        if self.cycle_weight > 0.0:
            cycle_score, cycle_signals = self._cycle_score(proposal, original, edited, editor)
            component_scores["cycle_score"] = cycle_score
            signals.update(cycle_signals)
            weighted_total += self.cycle_weight * cycle_score
            total_weight += self.cycle_weight

        if self.internal_weight > 0.0:
            internal_score, internal_signals = self._internal_feature_score(proposal, original, edited, editor)
            component_scores["internal_qwen_score"] = internal_score
            signals.update(internal_signals)
            weighted_total += self.internal_weight * internal_score
            total_weight += self.internal_weight

        total_score = weighted_total / max(total_weight, 1e-6)
        component_scores["hybrid_total_score"] = total_score
        return SolverResult(
            global_score=proxy_result.global_score,
            local_score=proxy_result.local_score,
            total_score=total_score,
            accepted=total_score >= self.acceptance_threshold,
            component_scores=component_scores,
            signals=signals,
        )


class GenericRelativeSelfRewardEvaluator(MultiSignalSolver):
    """Generic self-evolution baseline without editing-specific delta gates.

    This intentionally mirrors the kind of continuous self-reward transplant that works for
    reasoning-style proposer/solver loops but is under-specified for image editing. It ranks K
    candidates by a scalar self-reward and relative group score, without enforcing preservation,
    counterfactual instruction discrimination, or hard edit-success gates.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.top_m = int(config.get("top_m", 1))
        self.quality_threshold = float(config.get("quality_threshold", self.acceptance_threshold))
        self.rank_self_reward_weight = float(config.get("rank_self_reward_weight", 0.70))
        self.rank_relative_weight = float(config.get("rank_relative_weight", 0.30))

    @staticmethod
    def _weighted_mean(weighted_values: list[tuple[float, float]], default: float = 0.0) -> float:
        total_weight = sum(weight for _, weight in weighted_values if weight > 0.0)
        if total_weight <= 0.0:
            return default
        return sum(value * weight for value, weight in weighted_values if weight > 0.0) / total_weight

    def score_group(
        self,
        proposal: EditProposal,
        original: Image.Image,
        edited_candidates: list[Image.Image],
        editor: Any | None = None,
    ) -> list[SolverResult]:
        if not edited_candidates:
            return []

        rows: list[dict[str, Any]] = []
        for candidate_index, edited in enumerate(edited_candidates):
            result = super().score(proposal, original, edited, editor=editor)
            rows.append(
                {
                    "candidate_index": candidate_index,
                    "result": result,
                    "self_reward_score": result.total_score,
                }
            )

        base_scores = [row["self_reward_score"] for row in rows]
        min_base = min(base_scores)
        max_base = max(base_scores)
        value_range = max(max_base - min_base, 1e-6)
        for row in rows:
            relative_score = (
                (row["self_reward_score"] - min_base) / value_range if max_base > min_base else 0.5
            )
            quality_score = self._weighted_mean(
                [
                    (row["self_reward_score"], self.rank_self_reward_weight),
                    (relative_score, self.rank_relative_weight),
                ],
                default=row["self_reward_score"],
            )
            row["relative_score"] = relative_score
            row["quality_score"] = quality_score

        ranked_rows = sorted(rows, key=lambda row: row["quality_score"], reverse=True)
        accepted_candidate_indices = {
            row["candidate_index"]
            for row in ranked_rows[: self.top_m]
            if row["quality_score"] >= self.quality_threshold
        }
        rank_by_candidate = {row["candidate_index"]: rank + 1 for rank, row in enumerate(ranked_rows)}

        outputs: list[SolverResult] = []
        for row in rows:
            result = row["result"]
            candidate_index = row["candidate_index"]
            accepted = candidate_index in accepted_candidate_indices
            component_scores = dict(result.component_scores)
            component_scores.update(
                {
                    "generic_self_reward_score": row["self_reward_score"],
                    "relative_group_score": row["relative_score"],
                    "relative_quality_score": row["quality_score"],
                }
            )
            signals = dict(result.signals)
            signals.update(
                {
                    "candidate_index": float(candidate_index),
                    "group_size": float(len(rows)),
                    "feasible": 1.0,
                    "accepted_by_generic_ranker": 1.0 if accepted else 0.0,
                    "generic_rank": float(rank_by_candidate.get(candidate_index, 0)),
                    "generic_self_reward_threshold": self.quality_threshold,
                    "delta_instruction_gate_used": 0.0,
                    "delta_preservation_gate_used": 0.0,
                    "counterfactual_gate_used": 0.0,
                }
            )
            outputs.append(
                SolverResult(
                    global_score=result.global_score,
                    local_score=result.local_score,
                    total_score=row["quality_score"],
                    accepted=accepted,
                    component_scores=component_scores,
                    signals=signals,
                )
            )
        return outputs


class HardGatedRelativeEvaluator(MultiSignalSolver):
    """Evaluator for the research-grade delta-grounded candidate-ranking path."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.instruction_threshold = float(config.get("instruction_threshold", 0.62))
        self.preservation_threshold = float(config.get("preservation_threshold", 0.68))
        self.quality_threshold = float(config.get("quality_threshold", 0.50))
        self.max_disagreement = float(config.get("max_disagreement", 0.35))
        self.top_m = int(config.get("top_m", 1))
        self.counterfactual_distractors = int(config.get("counterfactual_distractors", 3))
        self.rank_spatial_weight = float(config.get("rank_spatial_weight", 0.35))
        self.rank_counterfactual_weight = float(config.get("rank_counterfactual_weight", 0.30))
        self.rank_relative_weight = float(config.get("rank_relative_weight", 0.25))
        self.rank_cycle_weight = float(config.get("rank_cycle_weight", 0.05))
        self.rank_internal_weight = float(config.get("rank_internal_weight", 0.05))
        self.counterfactual_backend = config.get("counterfactual_backend", "auto")
        self.counterfactual_prompt_gain_scale = float(config.get("counterfactual_prompt_gain_scale", 0.08))
        self.require_internal_when_weighted = bool(config.get("require_internal_when_weighted", False))

    def _distractor_definitions(self, proposal: EditProposal) -> list[ProposalDefinition]:
        chosen: list[ProposalDefinition] = []
        inverse_id = proposal.definition.inverse_operation_id or INVERSE_OPERATION_MAP.get(
            proposal.definition.operation_id
        )
        if inverse_id is not None:
            chosen.append(PROPOSAL_BY_ID[inverse_id])
        for definition in PROPOSAL_BANK:
            if definition.operation_id == proposal.definition.operation_id:
                continue
            if definition in chosen:
                continue
            if definition.family == proposal.definition.family or definition.metric == proposal.definition.metric:
                chosen.append(definition)
        for definition in PROPOSAL_BANK:
            if definition.operation_id != proposal.definition.operation_id and definition not in chosen:
                chosen.append(definition)
        return chosen[: self.counterfactual_distractors]

    def describe_distractors(self, proposal: EditProposal) -> list[dict[str, Any]]:
        return [
            {
                "operation_id": definition.operation_id,
                "family": definition.family,
                "instruction": definition.instruction,
                "verifier": definition.verifier,
            }
            for definition in self._distractor_definitions(proposal)
        ]

    def _qwen_prompt_gain(
        self,
        instruction: str,
        original: Image.Image,
        edited: Image.Image,
        editor: Any | None,
    ) -> float | None:
        if not isinstance(editor, QwenEditEditor):
            return None
        try:
            import torch
            import torch.nn.functional as F

            pipe = editor._ensure_pipeline()
            prompt = polish_prompt(instruction, use_prompt_polish=False, image_context=original)
            with torch.no_grad():
                text_feature = extract_qwen_text_features(pipe, prompt)["pooled_embedding"][0].float()
                original_feature = extract_qwen_edit_understanding_features(
                    pipe, prompt, [original]
                )["pooled_embedding"][0].float()
                edited_feature = extract_qwen_edit_understanding_features(
                    pipe, prompt, [edited]
                )["pooled_embedding"][0].float()
            original_similarity = float(
                F.cosine_similarity(original_feature.unsqueeze(0), text_feature.unsqueeze(0)).item()
            )
            edited_similarity = float(
                F.cosine_similarity(edited_feature.unsqueeze(0), text_feature.unsqueeze(0)).item()
            )
            return edited_similarity - original_similarity
        except Exception:
            return None

    def _proxy_counterfactual_score(
        self,
        proposal: EditProposal,
        original: Image.Image,
        edited: Image.Image,
    ) -> tuple[float, dict[str, float]]:
        true_score, _ = self._global_score(proposal.definition, original, edited)
        distractor_scores: list[float] = []
        for definition in self._distractor_definitions(proposal):
            score, _ = self._global_score(definition, original, edited)
            distractor_scores.append(score)
        max_distractor = max(distractor_scores) if distractor_scores else 0.0
        margin = true_score - max_distractor
        score = _clamp(0.5 + 0.5 * margin)
        return score, {
            "counterfactual_backend_proxy": 1.0,
            "counterfactual_backend_internal": 0.0,
            "counterfactual_true_score": true_score,
            "counterfactual_max_distractor_score": max_distractor,
            "counterfactual_margin": margin,
        }

    def _internal_counterfactual_score(
        self,
        proposal: EditProposal,
        original: Image.Image,
        edited: Image.Image,
        editor: Any | None,
    ) -> tuple[float, dict[str, float]] | None:
        true_gain = self._qwen_prompt_gain(proposal.instruction, original, edited, editor)
        if true_gain is None:
            return None
        distractor_gains = []
        for definition in self._distractor_definitions(proposal):
            gain = self._qwen_prompt_gain(definition.instruction, original, edited, editor)
            if gain is not None:
                distractor_gains.append(gain)
        max_distractor_gain = max(distractor_gains) if distractor_gains else 0.0
        margin = true_gain - max_distractor_gain
        score = _clamp(0.5 + 0.5 * margin / max(self.counterfactual_prompt_gain_scale, 1e-6))
        return score, {
            "counterfactual_backend_proxy": 0.0,
            "counterfactual_backend_internal": 1.0,
            "counterfactual_true_score": true_gain,
            "counterfactual_max_distractor_score": max_distractor_gain,
            "counterfactual_margin": margin,
            "counterfactual_distractor_count": float(len(distractor_gains)),
        }

    def _counterfactual_score(
        self,
        proposal: EditProposal,
        original: Image.Image,
        edited: Image.Image,
        editor: Any | None = None,
    ) -> tuple[float, dict[str, float]]:
        use_internal = self.counterfactual_backend == "internal" or (
            self.counterfactual_backend == "auto" and proposal.definition.verifier == "internal"
        )
        if use_internal:
            internal_result = self._internal_counterfactual_score(proposal, original, edited, editor)
            if internal_result is not None:
                return internal_result
            if self.counterfactual_backend in {"auto", "internal"}:
                return 0.5, {
                    "counterfactual_backend_proxy": 0.0,
                    "counterfactual_backend_internal": 0.0,
                    "counterfactual_true_score": 0.0,
                    "counterfactual_max_distractor_score": 0.0,
                    "counterfactual_margin": 0.0,
                    "counterfactual_runtime_error": 1.0,
                }
        return self._proxy_counterfactual_score(proposal, original, edited)

    @staticmethod
    def _weighted_mean(weighted_values: list[tuple[float, float]], default: float = 0.0) -> float:
        total_weight = sum(weight for _, weight in weighted_values if weight > 0.0)
        if total_weight <= 0.0:
            return default
        return sum(value * weight for value, weight in weighted_values if weight > 0.0) / total_weight

    @staticmethod
    def _disagreement(values: list[float]) -> float:
        if len(values) <= 1:
            return 0.0
        return float(statistics.pstdev(values))

    def _gate_scores(self, result: SolverResult) -> tuple[float, float]:
        instruction = result.global_score
        if result.signals.get("global_proxy_supported", 1.0) <= 0.0:
            instruction = result.signals.get("internal_instruction_score", result.global_score)
        elif "internal_instruction_score" in result.signals:
            instruction = 0.70 * result.global_score + 0.30 * result.signals["internal_instruction_score"]
        if (
            self.require_internal_when_weighted
            and (self.internal_weight > 0.0 or self.rank_internal_weight > 0.0)
            and result.signals.get("internal_supported", 0.0) <= 0.0
        ):
            instruction = 0.0
        outside_preservation = result.signals.get("spatial_outside_preservation", result.local_score)
        edge_preservation = result.signals.get("edge_preservation_score", result.local_score)
        preservation = 0.55 * edge_preservation + 0.45 * outside_preservation
        return _clamp(instruction), _clamp(preservation)

    def _rank_base_score(self, result: SolverResult, counterfactual_score: float) -> float:
        spatial = result.component_scores.get("spatial_score", result.local_score)
        cycle = result.component_scores.get("cycle_score", self.cycle_neutral_score)
        internal = result.component_scores.get("internal_qwen_score", self.internal_neutral_score)
        return self._weighted_mean(
            [
                (spatial, self.rank_spatial_weight),
                (counterfactual_score, self.rank_counterfactual_weight),
                (cycle, self.rank_cycle_weight),
                (internal, self.rank_internal_weight),
            ],
            default=result.total_score,
        )

    def score_group(
        self,
        proposal: EditProposal,
        original: Image.Image,
        edited_candidates: list[Image.Image],
        editor: Any | None = None,
    ) -> list[SolverResult]:
        if not edited_candidates:
            return []

        rows: list[dict[str, Any]] = []
        for candidate_index, edited in enumerate(edited_candidates):
            result = super().score(proposal, original, edited, editor=editor)
            counterfactual_score, counterfactual_signals = self._counterfactual_score(
                proposal, original, edited, editor=editor
            )
            instruction_score, preservation_score = self._gate_scores(result)
            rank_base_score = self._rank_base_score(result, counterfactual_score)
            feasible = instruction_score >= self.instruction_threshold and preservation_score >= self.preservation_threshold
            rows.append(
                {
                    "candidate_index": candidate_index,
                    "result": result,
                    "counterfactual_score": counterfactual_score,
                    "counterfactual_signals": counterfactual_signals,
                    "instruction_score": instruction_score,
                    "preservation_score": preservation_score,
                    "rank_base_score": rank_base_score,
                    "feasible": feasible,
                }
            )

        base_values = [row["rank_base_score"] for row in rows]
        min_base = min(base_values)
        max_base = max(base_values)
        value_range = max(max_base - min_base, 1e-6)
        for row in rows:
            relative_score = (row["rank_base_score"] - min_base) / value_range if max_base > min_base else 0.5
            result = row["result"]
            spatial = result.component_scores.get("spatial_score", result.local_score)
            cycle = result.component_scores.get("cycle_score", self.cycle_neutral_score)
            internal = result.component_scores.get("internal_qwen_score", self.internal_neutral_score)
            quality_score = self._weighted_mean(
                [
                    (spatial, self.rank_spatial_weight),
                    (row["counterfactual_score"], self.rank_counterfactual_weight),
                    (relative_score, self.rank_relative_weight),
                    (cycle, self.rank_cycle_weight),
                    (internal, self.rank_internal_weight),
                ],
                default=row["rank_base_score"],
            )
            component_values = [
                row["instruction_score"],
                row["preservation_score"],
                spatial,
                row["counterfactual_score"],
                relative_score,
            ]
            if "cycle_score" in result.component_scores:
                component_values.append(cycle)
            if "internal_qwen_score" in result.component_scores:
                component_values.append(internal)
            row["relative_score"] = relative_score
            row["quality_score"] = quality_score
            row["disagreement"] = self._disagreement(component_values)

        feasible_rows = [row for row in rows if row["feasible"]]
        ranked_rows = sorted(feasible_rows, key=lambda row: row["quality_score"], reverse=True)
        accepted_candidate_indices = {
            row["candidate_index"]
            for row in ranked_rows[: self.top_m]
            if row["quality_score"] >= self.quality_threshold and row["disagreement"] <= self.max_disagreement
        }
        rank_by_candidate = {row["candidate_index"]: rank + 1 for rank, row in enumerate(ranked_rows)}

        outputs: list[SolverResult] = []
        for row in rows:
            result = row["result"]
            candidate_index = row["candidate_index"]
            rank = rank_by_candidate.get(candidate_index, 0)
            accepted = candidate_index in accepted_candidate_indices
            component_scores = dict(result.component_scores)
            component_scores.update(
                {
                    "instruction_gate_score": row["instruction_score"],
                    "preservation_gate_score": row["preservation_score"],
                    "counterfactual_score": row["counterfactual_score"],
                    "relative_group_score": row["relative_score"],
                    "rank_base_score": row["rank_base_score"],
                    "relative_quality_score": row["quality_score"],
                    "evaluator_disagreement": row["disagreement"],
                }
            )
            signals = dict(result.signals)
            signals.update(row["counterfactual_signals"])
            signals.update(
                {
                    "candidate_index": float(candidate_index),
                    "group_size": float(len(rows)),
                    "feasible": 1.0 if row["feasible"] else 0.0,
                    "accepted_by_ranker": 1.0 if accepted else 0.0,
                    "feasible_rank": float(rank),
                    "instruction_threshold": self.instruction_threshold,
                    "preservation_threshold": self.preservation_threshold,
                    "quality_threshold": self.quality_threshold,
                    "max_disagreement": self.max_disagreement,
                }
            )
            outputs.append(
                SolverResult(
                    global_score=result.global_score,
                    local_score=result.local_score,
                    total_score=row["quality_score"],
                    accepted=accepted,
                    component_scores=component_scores,
                    signals=signals,
                )
            )
        return outputs


class InternalContrastiveEditPreservationEvaluator(HardGatedRelativeEvaluator):
    """Internal-only CEPR evaluator for the main self-evolving method.

    The reward is intentionally constrained instead of additive:

    R(y | x, c) = sqrt(E(y, x, c) * P(y, x)) when edit, preservation, and
    validity gates pass; otherwise R=0.

    E is an internal contrastive prompt-gain score against counterfactual
    instructions. P is internal source preservation from Qwen semantic features
    and Qwen VAE latent locality. Q is a hard validity gate over edit-region
    plausibility and excessive latent drift. No external VLM, detector, CLIP, or
    OCR model is used.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.edit_threshold = float(config.get("edit_threshold", 0.53))
        self.preservation_threshold = float(config.get("preservation_threshold", 0.60))
        self.validity_threshold = float(config.get("validity_threshold", 0.35))
        self.reward_threshold = float(config.get("reward_threshold", config.get("quality_threshold", 0.52)))
        self.top_m = int(config.get("top_m", 1))
        self.require_internal_components = bool(config.get("require_internal_components", True))
        self.edit_temperature = float(config.get("edit_temperature", 0.04))
        self.true_gain_temperature = float(config.get("true_gain_temperature", 0.04))
        self.min_true_gain = float(config.get("min_true_gain", 0.0))
        self.semantic_preservation_temperature = float(config.get("semantic_preservation_temperature", 0.08))
        self.latent_resolution = int(config.get("latent_resolution", 512))
        self.latent_mask_std_scale = float(config.get("latent_mask_std_scale", 0.75))
        self.latent_mask_min_delta = float(config.get("latent_mask_min_delta", 0.03))
        self.latent_preservation_temperature = float(config.get("latent_preservation_temperature", 0.12))
        self.max_total_latent_delta = float(config.get("max_total_latent_delta", 0.45))
        self.latent_drift_temperature = float(config.get("latent_drift_temperature", 0.15))
        self.taxonomy_enabled = bool(config.get("taxonomy_enabled", True))
        self.taxonomy_required = bool(config.get("taxonomy_required", False))
        self.taxonomy_threshold = float(config.get("taxonomy_threshold", 0.40))
        self.taxonomy_temperature = float(config.get("taxonomy_temperature", 0.06))
        self.taxonomy_source_drop_temperature = float(config.get("taxonomy_source_drop_temperature", 0.06))
        self.taxonomy_distractor_temperature = float(config.get("taxonomy_distractor_temperature", 0.05))
        self.max_dynamic_distractors = int(config.get("max_dynamic_distractors", 4))
        self.empty_cache_per_candidate = bool(config.get("empty_cache_per_candidate", True))
        self.retry_cpu_on_oom = bool(config.get("retry_cpu_on_oom", True))

    @staticmethod
    def _cosine_similarity(value_a: Any, value_b: Any) -> float:
        import torch.nn.functional as F

        return float(F.cosine_similarity(value_a.unsqueeze(0), value_b.unsqueeze(0)).item())

    @staticmethod
    def _is_cuda_oom(exc: Exception) -> bool:
        message = str(exc).lower()
        return "cuda out of memory" in message or exc.__class__.__name__ == "OutOfMemoryError"

    def _get_internal_pipe(self, editor: Any | None) -> Any:
        if not isinstance(editor, QwenEditEditor):
            raise ValueError("internal_cepr requires QwenEditEditor so rewards can use Qwen internal features.")
        return editor._ensure_pipeline()

    def _cached_text_feature(self, pipe: Any, prompt: str, cache: dict[tuple[str, str], Any]):
        key = ("text", prompt)
        if key not in cache:
            try:
                cache[key] = extract_qwen_text_features(pipe, prompt)["pooled_embedding"][0].float()
            except Exception:
                # Some public Qwen wrappers expose only the multimodal encoder path.
                # A neutral image keeps the reward internal while still yielding a prompt-conditioned anchor.
                neutral_image = Image.new("RGB", (self.latent_resolution, self.latent_resolution), (127, 127, 127))
                cache[key] = extract_qwen_edit_understanding_features(pipe, prompt, [neutral_image])[
                    "pooled_embedding"
                ][0].float()
        return cache[key]

    def _cached_understanding_feature(
        self,
        pipe: Any,
        prompt: str,
        image_label: str,
        image: Image.Image,
        cache: dict[tuple[str, str], Any],
    ):
        key = ("understanding", f"{image_label}|{prompt}")
        if key not in cache:
            cache[key] = extract_qwen_edit_understanding_features(pipe, prompt, [image])["pooled_embedding"][0].float()
        return cache[key]

    def _cached_vae_latents(
        self,
        pipe: Any,
        image_label: str,
        image: Image.Image,
        cache: dict[tuple[str, str], Any],
    ):
        key = ("vae_latents", image_label)
        if key not in cache:
            cache[key] = extract_qwen_vae_latents(pipe, image, size=self.latent_resolution)
        return cache[key]

    def _prompt_gain_cached(
        self,
        pipe: Any,
        instruction: str,
        original: Image.Image,
        edited: Image.Image,
        candidate_index: int,
        cache: dict[tuple[str, str], Any],
    ) -> float:
        prompt = polish_prompt(instruction, use_prompt_polish=False, image_context=original)
        text_feature = self._cached_text_feature(pipe, prompt, cache)
        original_feature = self._cached_understanding_feature(pipe, prompt, "original", original, cache)
        edited_feature = self._cached_understanding_feature(pipe, prompt, f"candidate:{candidate_index}", edited, cache)
        original_similarity = self._cosine_similarity(original_feature, text_feature)
        edited_similarity = self._cosine_similarity(edited_feature, text_feature)
        return edited_similarity - original_similarity

    @staticmethod
    def _join_terms(*values: Any) -> str:
        terms = [str(value).strip() for value in values if value not in (None, "", [])]
        return " ".join(terms).strip()

    def _taxonomy_prompt_bundle(self, proposal: EditProposal) -> dict[str, Any]:
        spec = normalize_structured_edit(
            proposal.structured_edit,
            instruction=proposal.instruction,
            family=proposal.definition.family,
        )
        if not proposal.structured_edit:
            return {"supported": False, "spec": spec}

        edit_type = str(spec.get("edit_type", "local_enhancement"))
        source_object = spec.get("source_object")
        target_object = spec.get("target_object")
        source_attribute = spec.get("source_attribute") or spec.get("source_material") or spec.get("source_style")
        target_attribute = spec.get("target_attribute") or spec.get("target_material") or spec.get("target_style")
        target_location = spec.get("target_location")
        target_region = spec.get("target_region", "target region")

        main_prompts = [spec.get("instruction", proposal.instruction), proposal.instruction]
        target_prompts: list[str] = []
        source_drop_prompts: list[str] = []
        wrong_prompts: list[str] = []

        if edit_type == "object_replacement":
            if target_object:
                target_prompts.append(f"{target_region} contains {target_object}")
                main_prompts.append(f"{source_object or 'the original object'} is replaced by {target_object}")
            if source_object:
                source_drop_prompts.append(f"{target_region} still contains {source_object}")
                wrong_prompts.append(f"{source_object} is still present and unchanged")
            for wrong in ("a dog", "a chair", "a car", "a plant"):
                if target_object and wrong.lower() not in str(target_object).lower():
                    wrong_prompts.append(f"{source_object or 'the object'} is replaced by {wrong}")
        elif edit_type == "object_removal":
            if source_object:
                source_drop_prompts.append(f"{target_region} contains {source_object}")
                main_prompts.append(f"{source_object} has been removed from {target_region}")
            target_prompts.append(f"{target_region} is cleanly filled after object removal")
        elif edit_type == "object_addition":
            if target_object:
                target_prompts.append(f"{target_object} has been added to {target_region}")
                main_prompts.append(f"{target_region} now contains {target_object}")
            wrong_prompts.append(f"nothing new has been added to {target_region}")
        elif edit_type == "spatial_move":
            moved = source_object or "the target object"
            if target_location:
                target_prompts.append(f"{moved} is {target_location}")
                main_prompts.append(f"{moved} has moved to {target_location}")
            if spec.get("source_location"):
                source_drop_prompts.append(f"{moved} remains {spec['source_location']}")
            wrong_prompts.extend([f"{moved} stays in its original position", f"{moved} is in the wrong location"])
        elif edit_type in {"attribute_change", "color_change", "material_change", "style_transfer"}:
            target = self._join_terms(target_attribute, target_object or source_object)
            source = self._join_terms(source_attribute, source_object)
            if target:
                target_prompts.append(f"{target_region} has {target}")
                main_prompts.append(f"{source_object or 'the target'} is changed to {target}")
            if source:
                source_drop_prompts.append(f"{target_region} still has {source}")
            if target_attribute:
                wrong_prompts.append(f"{target_region} does not have {target_attribute}")
        elif edit_type == "background_change":
            if target_attribute or target_object:
                target_prompts.append(f"background changed to {self._join_terms(target_attribute, target_object)}")
            source_drop_prompts.append("the background is unchanged")
        else:
            target_prompts.append(proposal.instruction)

        preserve_prompts = [
            f"unrelated content is preserved: {item}"
            for item in spec.get("preserve", [])[: self.max_dynamic_distractors]
        ]
        unique = lambda items: list(dict.fromkeys(item for item in items if item))
        return {
            "supported": True,
            "spec": spec,
            "main_prompts": unique(main_prompts),
            "target_prompts": unique(target_prompts),
            "source_drop_prompts": unique(source_drop_prompts),
            "wrong_prompts": unique(wrong_prompts)[: self.max_dynamic_distractors],
            "preserve_prompts": unique(preserve_prompts),
        }

    @staticmethod
    def _geometric_mean(values: list[float], default: float = 1.0) -> float:
        filtered = [_clamp(value) for value in values if math.isfinite(float(value))]
        if not filtered:
            return default
        product = 1.0
        for value in filtered:
            product *= max(value, 1e-6)
        return _clamp(product ** (1.0 / len(filtered)))

    def _taxonomy_score(
        self,
        pipe: Any,
        proposal: EditProposal,
        original: Image.Image,
        edited: Image.Image,
        candidate_index: int,
        cache: dict[tuple[str, str], Any],
    ) -> tuple[float, dict[str, float]]:
        if not self.taxonomy_enabled:
            return 1.0, {"cepr_taxonomy_supported": 0.0, "cepr_taxonomy_score": 1.0}

        bundle = self._taxonomy_prompt_bundle(proposal)
        if not bundle.get("supported"):
            neutral = 0.0 if self.taxonomy_required else 1.0
            return neutral, {
                "cepr_taxonomy_supported": 0.0,
                "cepr_taxonomy_score": neutral,
                "cepr_taxonomy_required": 1.0 if self.taxonomy_required else 0.0,
            }

        main_gains = [
            self._prompt_gain_cached(pipe, prompt, original, edited, candidate_index, cache)
            for prompt in bundle.get("main_prompts", [])
        ]
        target_gains = [
            self._prompt_gain_cached(pipe, prompt, original, edited, candidate_index, cache)
            for prompt in bundle.get("target_prompts", [])
        ]
        source_drops = [
            -self._prompt_gain_cached(pipe, prompt, original, edited, candidate_index, cache)
            for prompt in bundle.get("source_drop_prompts", [])
        ]
        wrong_gains = [
            self._prompt_gain_cached(pipe, prompt, original, edited, candidate_index, cache)
            for prompt in bundle.get("wrong_prompts", [])
        ]

        main_gain = statistics.mean(main_gains) if main_gains else 0.0
        target_gain = statistics.mean(target_gains) if target_gains else main_gain
        source_drop = statistics.mean(source_drops) if source_drops else 0.0
        max_wrong_gain = max(wrong_gains) if wrong_gains else 0.0
        contrastive_margin = main_gain - max_wrong_gain

        main_score = _sigmoid(main_gain / max(self.taxonomy_temperature, 1e-6))
        target_score = _sigmoid(target_gain / max(self.taxonomy_temperature, 1e-6))
        contrastive_score = (
            _sigmoid(contrastive_margin / max(self.taxonomy_distractor_temperature, 1e-6))
            if wrong_gains
            else 1.0
        )
        components = [main_score, target_score, contrastive_score]
        if source_drops:
            source_drop_score = _sigmoid(source_drop / max(self.taxonomy_source_drop_temperature, 1e-6))
            components.append(source_drop_score)
        else:
            source_drop_score = 1.0

        taxonomy_score = self._geometric_mean(components)
        return taxonomy_score, {
            "cepr_taxonomy_supported": 1.0,
            "cepr_taxonomy_score": taxonomy_score,
            "cepr_taxonomy_required": 1.0 if self.taxonomy_required else 0.0,
            "cepr_taxonomy_main_gain": main_gain,
            "cepr_taxonomy_target_gain": target_gain,
            "cepr_taxonomy_source_drop": source_drop,
            "cepr_taxonomy_max_wrong_gain": max_wrong_gain,
            "cepr_taxonomy_contrastive_margin": contrastive_margin,
            "cepr_taxonomy_main_score": main_score,
            "cepr_taxonomy_target_score": target_score,
            "cepr_taxonomy_source_drop_score": source_drop_score,
            "cepr_taxonomy_contrastive_score": contrastive_score,
        }

    def _edit_specificity(
        self,
        pipe: Any,
        proposal: EditProposal,
        original: Image.Image,
        edited: Image.Image,
        candidate_index: int,
        cache: dict[tuple[str, str], Any],
    ) -> tuple[float, dict[str, float]]:
        true_gain = self._prompt_gain_cached(pipe, proposal.instruction, original, edited, candidate_index, cache)
        distractor_gains = [
            self._prompt_gain_cached(pipe, definition.instruction, original, edited, candidate_index, cache)
            for definition in self._distractor_definitions(proposal)
        ]
        max_distractor_gain = max(distractor_gains) if distractor_gains else 0.0
        contrastive_margin = true_gain - max_distractor_gain
        contrastive_score = _sigmoid(contrastive_margin / max(self.edit_temperature, 1e-6))
        absolute_score = _sigmoid((true_gain - self.min_true_gain) / max(self.true_gain_temperature, 1e-6))
        edit_specificity = math.sqrt(max(contrastive_score * absolute_score, 0.0))
        return edit_specificity, {
            "cepr_true_prompt_gain": true_gain,
            "cepr_max_distractor_gain": max_distractor_gain,
            "cepr_contrastive_margin": contrastive_margin,
            "cepr_contrastive_score": contrastive_score,
            "cepr_absolute_edit_score": absolute_score,
            "cepr_distractor_count": float(len(distractor_gains)),
        }

    def _semantic_preservation(
        self,
        pipe: Any,
        original: Image.Image,
        edited: Image.Image,
        candidate_index: int,
        cache: dict[tuple[str, str], Any],
    ) -> tuple[float, dict[str, float]]:
        original_feature = self._cached_understanding_feature(pipe, " ", "original", original, cache)
        edited_feature = self._cached_understanding_feature(pipe, " ", f"candidate:{candidate_index}", edited, cache)
        cosine = self._cosine_similarity(original_feature, edited_feature)
        cosine_01 = 0.5 * (1.0 + cosine)
        semantic_preservation = math.exp(-(1.0 - cosine_01) / max(self.semantic_preservation_temperature, 1e-6))
        return _clamp(semantic_preservation), {
            "cepr_semantic_preservation_cosine": cosine,
            "cepr_semantic_preservation_score": _clamp(semantic_preservation),
        }

    def _latent_locality(
        self,
        pipe: Any,
        proposal: EditProposal,
        original: Image.Image,
        edited: Image.Image,
        candidate_index: int,
        cache: dict[tuple[str, str], Any],
    ) -> tuple[float, float, dict[str, float]]:
        import torch

        original_latents = self._cached_vae_latents(pipe, "original", original, cache)
        edited_latents = self._cached_vae_latents(pipe, f"candidate:{candidate_index}", edited, cache)
        delta = (original_latents - edited_latents).abs().mean(dim=1).squeeze(0)
        normalizer = 0.5 * (
            original_latents.abs().mean(dim=1).squeeze(0) + edited_latents.abs().mean(dim=1).squeeze(0)
        )
        relative_delta = delta / normalizer.clamp_min(1e-4)
        mean_delta = float(relative_delta.mean().item())
        std_delta = float(relative_delta.std(unbiased=False).item())
        threshold = max(self.latent_mask_min_delta, mean_delta + self.latent_mask_std_scale * std_delta)
        mask = relative_delta > threshold
        changed_fraction_value = float(mask.float().mean().item())
        if mask.any():
            inside_delta = float(relative_delta[mask].mean().item())
        else:
            inside_delta = 0.0
        if (~mask).any():
            outside_delta = float(relative_delta[~mask].mean().item())
        else:
            outside_delta = mean_delta

        outside_preservation = math.exp(-outside_delta / max(self.latent_preservation_temperature, 1e-6))
        region_score = _mean_changed_fraction_score(changed_fraction_value, proposal.definition.expected_changed_fraction)
        excess_drift = max(0.0, mean_delta - self.max_total_latent_delta)
        drift_score = math.exp(-excess_drift / max(self.latent_drift_temperature, 1e-6))
        validity = math.sqrt(max(region_score * drift_score, 0.0))
        return _clamp(outside_preservation), _clamp(validity), {
            "cepr_latent_outside_preservation": _clamp(outside_preservation),
            "cepr_latent_region_score": region_score,
            "cepr_latent_validity_score": _clamp(validity),
            "cepr_latent_changed_fraction": changed_fraction_value,
            "cepr_latent_inside_delta": inside_delta,
            "cepr_latent_outside_delta": outside_delta,
            "cepr_latent_total_delta": mean_delta,
            "cepr_latent_delta_std": std_delta,
            "cepr_latent_mask_threshold": threshold,
            "cepr_latent_drift_score": _clamp(drift_score),
        }

    def _preservation_and_validity(
        self,
        pipe: Any,
        proposal: EditProposal,
        original: Image.Image,
        edited: Image.Image,
        candidate_index: int,
        cache: dict[tuple[str, str], Any],
    ) -> tuple[float, float, dict[str, float]]:
        semantic_preservation, semantic_signals = self._semantic_preservation(
            pipe, original, edited, candidate_index, cache
        )
        latent_preservation, validity, latent_signals = self._latent_locality(
            pipe, proposal, original, edited, candidate_index, cache
        )
        preservation = math.sqrt(max(semantic_preservation * latent_preservation, 0.0))
        signals = {
            **semantic_signals,
            **latent_signals,
            "cepr_preservation_score": _clamp(preservation),
            "cepr_validity_score": _clamp(validity),
        }
        return _clamp(preservation), _clamp(validity), signals

    def _failure_result(self, error_name: str, candidate_index: int, group_size: int) -> SolverResult:
        signals = {
            "candidate_index": float(candidate_index),
            "group_size": float(group_size),
            "feasible": 0.0,
            "accepted_by_ranker": 0.0,
            "feasible_rank": 0.0,
            "cepr_internal_supported": 0.0,
            error_name: 1.0,
            "cepr_edit_threshold": self.edit_threshold,
            "cepr_preservation_threshold": self.preservation_threshold,
            "cepr_validity_threshold": self.validity_threshold,
            "cepr_reward_threshold": self.reward_threshold,
        }
        return SolverResult(
            global_score=0.0,
            local_score=0.0,
            total_score=0.0,
            accepted=False,
            component_scores={
                "cepr_edit_specificity": 0.0,
                "cepr_taxonomy": 0.0,
                "cepr_semantic_edit": 0.0,
                "cepr_preservation": 0.0,
                "cepr_validity": 0.0,
                "cepr_reward": 0.0,
            },
            signals=signals,
        )

    def _candidate_error_row(self, exc: Exception, candidate_index: int, scoring_device: str) -> dict[str, Any]:
        return {
            "candidate_index": candidate_index,
            "edit_specificity": 0.0,
            "taxonomy_score": 0.0,
            "semantic_edit": 0.0,
            "preservation": 0.0,
            "validity": 0.0,
            "reward": 0.0,
            "raw_reward": 0.0,
            "feasible": False,
            "signals": {
                "cepr_internal_supported": 0.0,
                "cepr_candidate_runtime_error": 1.0,
                "cepr_candidate_runtime_error_type": exc.__class__.__name__,
                "cepr_candidate_runtime_error_message": str(exc)[:500],
                "cepr_scoring_device": scoring_device,
            },
        }

    def _score_candidate_row(
        self,
        pipe: Any,
        proposal: EditProposal,
        original: Image.Image,
        edited: Image.Image,
        candidate_index: int,
        scoring_device: str,
    ) -> dict[str, Any]:
        cache: dict[tuple[str, str], Any] = {}
        try:
            edit_specificity, edit_signals = self._edit_specificity(
                pipe, proposal, original, edited, candidate_index, cache
            )
            taxonomy_score, taxonomy_signals = self._taxonomy_score(
                pipe, proposal, original, edited, candidate_index, cache
            )
            preservation, validity, preservation_signals = self._preservation_and_validity(
                pipe, proposal, original, edited, candidate_index, cache
            )
            semantic_edit = (
                math.sqrt(max(edit_specificity * taxonomy_score, 0.0))
                if taxonomy_signals.get("cepr_taxonomy_supported", 0.0) > 0.0 or self.taxonomy_required
                else edit_specificity
            )
            reward = math.sqrt(max(semantic_edit * preservation, 0.0))
            feasible = (
                edit_specificity >= self.edit_threshold
                and (
                    taxonomy_score >= self.taxonomy_threshold
                    or (
                        taxonomy_signals.get("cepr_taxonomy_supported", 0.0) <= 0.0
                        and not self.taxonomy_required
                    )
                )
                and preservation >= self.preservation_threshold
                and validity >= self.validity_threshold
                and reward >= self.reward_threshold
            )
            return {
                "candidate_index": candidate_index,
                "edit_specificity": edit_specificity,
                "taxonomy_score": taxonomy_score,
                "semantic_edit": semantic_edit,
                "preservation": preservation,
                "validity": validity,
                "reward": reward if feasible else 0.0,
                "raw_reward": reward,
                "feasible": feasible,
                "signals": {
                    **edit_signals,
                    **taxonomy_signals,
                    **preservation_signals,
                    "cepr_internal_supported": 1.0,
                    "cepr_scoring_device": scoring_device,
                },
            }
        finally:
            cache.clear()
            if self.empty_cache_per_candidate:
                QwenEditEditor._empty_cuda_cache()

    def score(
        self,
        proposal: EditProposal,
        original: Image.Image,
        edited: Image.Image,
        editor: Any | None = None,
    ) -> SolverResult:
        return self.score_group(proposal, original, [edited], editor=editor)[0]

    def _apply_group_judge(
        self,
        proposal: EditProposal,
        original: Image.Image,
        edited_candidates: list[Image.Image],
        rows: list[dict[str, Any]],
        editor: Any | None,
    ) -> None:
        return None

    def _apply_post_group_hard_gates(self, rows: list[dict[str, Any]]) -> None:
        """Reapply non-compensatory gates after optional group-level judges."""
        return None

    def score_group(
        self,
        proposal: EditProposal,
        original: Image.Image,
        edited_candidates: list[Image.Image],
        editor: Any | None = None,
    ) -> list[SolverResult]:
        if not edited_candidates:
            return []
        rows: list[dict[str, Any]] = []
        for candidate_index, edited in enumerate(edited_candidates):
            try:
                if hasattr(editor, "prepare_for_internal_scoring"):
                    editor.prepare_for_internal_scoring()
                pipe = self._get_internal_pipe(editor)
                rows.append(self._score_candidate_row(pipe, proposal, original, edited, candidate_index, "cuda"))
            except Exception as exc:
                if self.retry_cpu_on_oom and self._is_cuda_oom(exc) and hasattr(editor, "prepare_for_internal_scoring"):
                    try:
                        QwenEditEditor._empty_cuda_cache()
                        editor.prepare_for_internal_scoring(scoring_device="cpu")
                        pipe = self._get_internal_pipe(editor)
                        row = self._score_candidate_row(pipe, proposal, original, edited, candidate_index, "cpu")
                        row["signals"]["cepr_gpu_oom_recovered"] = 1.0
                        rows.append(row)
                        continue
                    except Exception as cpu_exc:
                        row = self._candidate_error_row(cpu_exc, candidate_index, "cpu")
                        row["signals"]["cepr_gpu_oom_before_cpu_retry"] = 1.0
                        rows.append(row)
                        continue
                rows.append(self._candidate_error_row(exc, candidate_index, "cuda"))

        self._apply_group_judge(proposal, original, edited_candidates, rows, editor)
        # A group judge may refine scores or provide narrowly scoped fallbacks,
        # but it must not bypass a subclass's non-compensatory raw constraints.
        self._apply_post_group_hard_gates(rows)
        ranked_rows = sorted(
            [row for row in rows if row["feasible"]],
            key=lambda row: row["reward"],
            reverse=True,
        )
        accepted_candidate_indices = {row["candidate_index"] for row in ranked_rows[: self.top_m]}
        rank_by_candidate = {row["candidate_index"]: rank + 1 for rank, row in enumerate(ranked_rows)}

        outputs: list[SolverResult] = []
        for row in rows:
            candidate_index = row["candidate_index"]
            accepted = candidate_index in accepted_candidate_indices
            rank = rank_by_candidate.get(candidate_index, 0)
            reward = row["reward"]
            signals = dict(row["signals"])
            signals.update(
                {
                    "candidate_index": float(candidate_index),
                    "group_size": float(len(rows)),
                    "feasible": 1.0 if row["feasible"] else 0.0,
                    "accepted_by_ranker": 1.0 if accepted else 0.0,
                    "feasible_rank": float(rank),
                    "cepr_edit_threshold": self.edit_threshold,
                    "cepr_preservation_threshold": self.preservation_threshold,
                    "cepr_validity_threshold": self.validity_threshold,
                    "cepr_reward_threshold": self.reward_threshold,
                }
            )
            component_scores = {
                "cepr_edit_specificity": row["edit_specificity"],
                "cepr_taxonomy": row["taxonomy_score"],
                "cepr_semantic_edit": row["semantic_edit"],
                "cepr_preservation": row["preservation"],
                "cepr_validity": row["validity"],
                "cepr_raw_reward": row["raw_reward"],
                "cepr_reward": reward,
            }
            component_scores.update(dict(row.get("component_scores", {})))
            outputs.append(
                SolverResult(
                    global_score=row["edit_specificity"],
                    local_score=row["preservation"],
                    total_score=reward,
                    accepted=accepted,
                    component_scores=component_scores,
                    signals=signals,
                )
            )
        return outputs


class InternalRubricCEPREvaluator(InternalContrastiveEditPreservationEvaluator):
    """Rubric-grounded internal CEPR evaluator.

    This keeps the CEPR reward internal-only, but scores explicit atomic edit
    criteria from the structured proposal before allowing a candidate into SFT.
    The rubric layer is deliberately decomposed: source grounding, required
    after-state, forbidden old-state removal, preservation, and latent validity
    are gated separately so a high prompt-gain scalar cannot compensate for a
    semantically wrong edit.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.rubric_source_threshold = float(config.get("rubric_source_threshold", 0.45))
        self.rubric_required_threshold = float(config.get("rubric_required_threshold", 0.48))
        self.rubric_forbidden_threshold = float(config.get("rubric_forbidden_threshold", 0.48))
        self.rubric_preservation_threshold = float(config.get("rubric_preservation_threshold", 0.35))
        self.rubric_reward_threshold = float(config.get("rubric_reward_threshold", self.reward_threshold))
        self.rubric_gain_temperature = float(config.get("rubric_gain_temperature", 0.06))
        self.rubric_abs_temperature = float(config.get("rubric_abs_temperature", 0.08))
        self.rubric_preservation_temperature = float(config.get("rubric_preservation_temperature", 0.20))
        self.rubric_support_center = float(config.get("rubric_support_center", 0.50))
        self.rubric_forbidden_absence_center = float(config.get("rubric_forbidden_absence_center", 0.52))
        self.min_required_gain = float(config.get("min_required_gain", 0.0))
        self.min_forbidden_drop = float(config.get("min_forbidden_drop", 0.0))
        # These are strict raw lower bounds, evaluated before shifted-sigmoid
        # rubric scores can compensate for the wrong sign. A zero default means
        # the corresponding raw signal must be genuinely positive.
        self.min_raw_true_prompt_gain = float(
            config.get("min_raw_true_prompt_gain", self.min_true_gain)
        )
        self.min_raw_contrastive_margin = float(
            config.get("min_raw_contrastive_margin", 0.0)
        )
        self.min_raw_required_after_gain = float(
            config.get("min_raw_required_after_gain", self.min_required_gain)
        )
        self.min_raw_old_state_suppression = float(
            config.get("min_raw_old_state_suppression", self.min_forbidden_drop)
        )
        raw_thresholds = {
            "min_raw_true_prompt_gain": self.min_raw_true_prompt_gain,
            "min_raw_contrastive_margin": self.min_raw_contrastive_margin,
            "min_raw_required_after_gain": self.min_raw_required_after_gain,
            "min_raw_old_state_suppression": self.min_raw_old_state_suppression,
        }
        invalid_raw_thresholds = [
            name for name, value in raw_thresholds.items() if not math.isfinite(value) or value < 0.0
        ]
        if invalid_raw_thresholds:
            raise ValueError(
                "Raw CEPR gate thresholds must be finite and non-negative: "
                + ", ".join(invalid_raw_thresholds)
            )
        self.max_rubric_items = int(config.get("max_rubric_items", 3))
        self.combine_cepr_semantics = bool(config.get("combine_cepr_semantics", True))
        self.rubric_soft_forbidden_edit_types = self._string_set(
            config.get("rubric_soft_forbidden_edit_types", [])
        )
        self.rubric_hard_forbidden_edit_types = self._string_set(
            config.get("rubric_hard_forbidden_edit_types", [])
        )
        self.rubric_hard_forbidden_threshold = float(
            config.get("rubric_hard_forbidden_threshold", self.rubric_forbidden_threshold)
        )
        self.rubric_soft_forbidden_required_threshold = float(
            config.get("rubric_soft_forbidden_required_threshold", self.rubric_required_threshold)
        )
        self.rubric_soft_forbidden_reward_threshold = float(
            config.get("rubric_soft_forbidden_reward_threshold", self.rubric_reward_threshold)
        )
        self.rubric_soft_forbidden_preservation_threshold = float(
            config.get("rubric_soft_forbidden_preservation_threshold", self.rubric_preservation_threshold)
        )
        self.object_detector_enabled = bool(config.get("object_detector_enabled", False))
        self.object_detector_edit_types = self._string_set(
            config.get("object_detector_edit_types", ["object_removal", "object_replacement"])
        )
        self.object_detector_box_threshold = float(config.get("object_detector_box_threshold", 0.25))
        self.object_detector_text_threshold = float(config.get("object_detector_text_threshold", 0.20))
        self.object_detector_original_min_score = float(config.get("object_detector_original_min_score", 0.20))
        self.object_detector_edited_absent_max_score = float(config.get("object_detector_edited_absent_max_score", 0.12))
        self.object_detector_absent_ratio = float(config.get("object_detector_absent_ratio", 0.55))
        self.object_detector_target_min_score = float(config.get("object_detector_target_min_score", 0.20))
        self.object_detector_score_threshold = float(config.get("object_detector_score_threshold", 0.50))
        self.object_detector_score_temperature = float(config.get("object_detector_score_temperature", 0.05))
        self.object_detector_require_original_detection = bool(
            config.get("object_detector_require_original_detection", True)
        )
        self.object_detector_fail_open_on_unsuitable_phrase = bool(
            config.get("object_detector_fail_open_on_unsuitable_phrase", True)
        )
        self.object_detector_unsuitable_terms = self._string_set(
            config.get(
                "object_detector_unsuitable_terms",
                [
                    "area",
                    "background",
                    "blur",
                    "cloud",
                    "corner",
                    "detail",
                    "fragment",
                    "highlight",
                    "mark",
                    "patch",
                    "region",
                    "shadow",
                    "sky",
                    "spot",
                    "stain",
                    "texture",
                    "tip",
                ],
            )
        )
        self.object_grounder = str(config.get("object_grounder", "qwen_vl_internal")).strip().lower()
        if self.object_grounder != "qwen_vl_internal":
            raise ValueError(
                "Main supports only editor-side Qwen grounding. "
                "External detector variants are on the groundingdino-extraction branch."
            )
        self.internal_grounder_image_resolution = int(
            config.get("internal_grounder_image_resolution", 768)
        )
        self.internal_grounder_max_new_tokens = int(
            config.get("internal_grounder_max_new_tokens", 96)
        )
        self.conservative_region_reward_enabled = bool(
            config.get("conservative_region_reward_enabled", False)
        )
        self.conservative_region_edit_types = self._string_set(
            config.get(
                "conservative_region_edit_types",
                [
                    "object_removal",
                    "object_replacement",
                    "color_change",
                    "attribute_change",
                    "material_change",
                    "local_enhancement",
                ],
            )
        )
        self.conservative_region_require_mask_edit_types = self._string_set(
            config.get(
                "conservative_region_require_mask_edit_types",
                ["object_removal", "object_replacement"],
            )
        )
        self.conservative_region_use_object_detector = bool(
            config.get("conservative_region_use_object_detector", True)
        )
        self.conservative_region_fallback_to_diff_mask = bool(
            config.get("conservative_region_fallback_to_diff_mask", True)
        )
        self.conservative_region_diff_fallback_allows_gate = bool(
            config.get("conservative_region_diff_fallback_allows_gate", False)
        )
        self.conservative_region_mask_size = int(config.get("conservative_region_mask_size", 128))
        self.conservative_region_mask_padding_fraction = float(
            config.get("conservative_region_mask_padding_fraction", 0.04)
        )
        self.conservative_region_diff_threshold = int(config.get("conservative_region_diff_threshold", 18))
        self.conservative_region_diff_dilation_radius = int(
            config.get("conservative_region_diff_dilation_radius", 1)
        )
        self.conservative_region_min_target_area = float(
            config.get("conservative_region_min_target_area", 0.0025)
        )
        self.conservative_region_max_target_area = float(
            config.get("conservative_region_max_target_area", 0.65)
        )
        self.conservative_region_max_target_area_by_edit_type = self._float_by_edit_type(
            config.get("conservative_region_max_target_area_by_edit_type")
        )
        self.conservative_region_max_detector_boxes = int(
            config.get("conservative_region_max_detector_boxes", 3)
        )
        self.conservative_region_min_target_change = float(
            config.get("conservative_region_min_target_change", 0.025)
        )
        self.conservative_region_min_target_change_score = float(
            config.get("conservative_region_min_target_change_score", 0.40)
        )
        self.conservative_region_target_change_temperature = float(
            config.get("conservative_region_target_change_temperature", 0.025)
        )
        self.conservative_region_max_outside_change = float(
            config.get("conservative_region_max_outside_change", 0.055)
        )
        self.conservative_region_outside_change_temperature = float(
            config.get("conservative_region_outside_change_temperature", 0.030)
        )
        self.conservative_region_max_outside_changed_fraction = float(
            config.get("conservative_region_max_outside_changed_fraction", 0.35)
        )
        self.conservative_region_min_localization_precision = float(
            config.get("conservative_region_min_localization_precision", 0.35)
        )
        self.conservative_region_min_localization_precision_by_edit_type = (
            self._float_by_edit_type(
                config.get("conservative_region_min_localization_precision_by_edit_type")
            )
        )
        self.conservative_region_min_outside_preservation = float(
            config.get("conservative_region_min_outside_preservation", 0.55)
        )
        self.conservative_region_min_reward = float(
            config.get("conservative_region_min_reward", 0.35)
        )
        self.conservative_region_target_structure_edit_types = self._string_set(
            config.get(
                "conservative_region_target_structure_edit_types",
                [
                    "color_change",
                    "attribute_change",
                    "material_change",
                    "local_enhancement",
                ],
            )
        )
        self.conservative_region_min_target_structure = float(
            config.get("conservative_region_min_target_structure", 0.45)
        )
        self.conservative_region_min_target_structure_edge = float(
            config.get("conservative_region_min_target_structure_edge", 0.70)
        )
        self.conservative_region_min_target_structure_by_edit_type = (
            self._float_by_edit_type(
                config.get("conservative_region_min_target_structure_by_edit_type")
            )
        )
        self.conservative_region_min_target_structure_edge_by_edit_type = (
            self._float_by_edit_type(
                config.get("conservative_region_min_target_structure_edge_by_edit_type")
            )
        )
        judge_cfg = config.get("internal_vlm_judge", {})
        self.internal_vlm_judge_cfg = dict(judge_cfg) if isinstance(judge_cfg, dict) else {}
        self.internal_vlm_judge_enabled = bool(self.internal_vlm_judge_cfg.get("enabled", False))
        self.internal_vlm_judge_max_candidates = int(self.internal_vlm_judge_cfg.get("max_candidates", 8))
        # Opt-in: skip the (expensive) judge on candidates already rejected by cheaper gates.
        # The judge with require_for_feasible can only remove feasibility, never grant it, so
        # skipping already-infeasible candidates leaves every accept/reject decision unchanged.
        self.internal_vlm_judge_skip_infeasible = bool(
            self.internal_vlm_judge_cfg.get("skip_infeasible", False)
        )
        self.internal_vlm_judge_image_resolution = int(self.internal_vlm_judge_cfg.get("image_resolution", 384))
        self.internal_vlm_judge_max_new_tokens = int(self.internal_vlm_judge_cfg.get("max_new_tokens", 768))
        self.internal_vlm_judge_temperature = float(self.internal_vlm_judge_cfg.get("temperature", 0.0))
        self.internal_vlm_judge_top_p = float(self.internal_vlm_judge_cfg.get("top_p", 0.9))
        self.internal_vlm_judge_cepr_weight = float(self.internal_vlm_judge_cfg.get("cepr_weight", 0.45))
        self.internal_vlm_judge_weight = float(self.internal_vlm_judge_cfg.get("judge_weight", 0.55))
        self.internal_vlm_judge_min_score = float(self.internal_vlm_judge_cfg.get("min_score_for_feasible", 0.35))
        self.internal_vlm_judge_min_score_by_edit_type = self._float_by_edit_type(
            self.internal_vlm_judge_cfg.get("min_score_for_feasible_by_edit_type")
            or self.internal_vlm_judge_cfg.get("min_score_by_edit_type")
        )
        self.internal_vlm_judge_min_semantic = float(
            self.internal_vlm_judge_cfg.get("min_semantic_for_feasible", self.internal_vlm_judge_min_score)
        )
        self.internal_vlm_judge_min_semantic_by_edit_type = self._float_by_edit_type(
            self.internal_vlm_judge_cfg.get("min_semantic_for_feasible_by_edit_type")
            or self.internal_vlm_judge_cfg.get("min_semantic_by_edit_type")
        )
        self.internal_vlm_judge_min_preservation = float(
            self.internal_vlm_judge_cfg.get("min_preservation_for_feasible", self.internal_vlm_judge_min_score)
        )
        self.internal_vlm_judge_min_preservation_by_edit_type = self._float_by_edit_type(
            self.internal_vlm_judge_cfg.get("min_preservation_for_feasible_by_edit_type")
            or self.internal_vlm_judge_cfg.get("min_preservation_by_edit_type")
        )
        self.internal_vlm_judge_min_artifact_free = float(
            self.internal_vlm_judge_cfg.get("min_artifact_free_for_feasible", self.internal_vlm_judge_min_score)
        )
        self.internal_vlm_judge_min_artifact_free_by_edit_type = self._float_by_edit_type(
            self.internal_vlm_judge_cfg.get("min_artifact_free_for_feasible_by_edit_type")
            or self.internal_vlm_judge_cfg.get("min_artifact_free_by_edit_type")
        )
        self.internal_vlm_judge_min_target_correctness = float(
            self.internal_vlm_judge_cfg.get("min_target_correctness_for_feasible", 0.0)
        )
        self.internal_vlm_judge_min_target_correctness_by_edit_type = self._float_by_edit_type(
            self.internal_vlm_judge_cfg.get("min_target_correctness_for_feasible_by_edit_type")
            or self.internal_vlm_judge_cfg.get("min_target_correctness_by_edit_type")
        )
        self.internal_vlm_judge_min_confidence = float(self.internal_vlm_judge_cfg.get("min_confidence", 0.25))
        self.internal_vlm_judge_require_for_feasible = bool(
            self.internal_vlm_judge_cfg.get("require_for_feasible", False)
        )
        self.internal_vlm_judge_fail_open = bool(self.internal_vlm_judge_cfg.get("fail_open", True))
        self.internal_vlm_judge_fail_closed_on_low_score = bool(
            self.internal_vlm_judge_cfg.get("fail_closed_on_low_score", True)
        )
        self.internal_vlm_judge_missing_is_failure = bool(
            self.internal_vlm_judge_cfg.get("missing_score_is_failure", not self.internal_vlm_judge_fail_open)
        )
        self.internal_vlm_judge_require_confidence_for_feasible = bool(
            self.internal_vlm_judge_cfg.get("require_confidence_for_feasible", False)
        )
        self.internal_vlm_judge_use_unreliable_scores = bool(
            self.internal_vlm_judge_cfg.get("use_unreliable_scores", True)
        )
        self.internal_vlm_judge_replacement_detector_fallback = bool(
            self.internal_vlm_judge_cfg.get("replacement_detector_fallback", True)
        )
        self.internal_vlm_judge_replacement_fallback_allow_vlm_contract_override = bool(
            self.internal_vlm_judge_cfg.get("replacement_fallback_allow_vlm_contract_override", False)
        )
        replacement_fallback_allowed_reasons = self.internal_vlm_judge_cfg.get(
            "replacement_fallback_allowed_reject_reasons",
            ["object_detector_contract"],
        )
        if isinstance(replacement_fallback_allowed_reasons, str):
            replacement_fallback_allowed_reasons = [
                item.strip()
                for item in replacement_fallback_allowed_reasons.split(",")
                if item.strip()
            ]
        self.internal_vlm_judge_replacement_fallback_allowed_reject_reasons = {
            str(item).strip()
            for item in (replacement_fallback_allowed_reasons or [])
            if str(item).strip()
        }
        self.internal_vlm_judge_replacement_fallback_require_best = bool(
            self.internal_vlm_judge_cfg.get("replacement_fallback_require_best", True)
        )
        self.internal_vlm_judge_replacement_fallback_min_semantic = float(
            self.internal_vlm_judge_cfg.get("replacement_fallback_min_semantic", 0.80)
        )
        self.internal_vlm_judge_replacement_fallback_min_object_absence = float(
            self.internal_vlm_judge_cfg.get("replacement_fallback_min_object_absence", 0.80)
        )
        self.internal_vlm_judge_replacement_fallback_min_target_correctness = float(
            self.internal_vlm_judge_cfg.get("replacement_fallback_min_target_correctness", 0.85)
        )
        self.internal_vlm_judge_replacement_fallback_min_target_object_visible_after = float(
            self.internal_vlm_judge_cfg.get(
                "replacement_fallback_min_target_object_visible_after",
                self.internal_vlm_judge_replacement_fallback_min_target_correctness,
            )
        )
        self.internal_vlm_judge_replacement_fallback_min_preservation = float(
            self.internal_vlm_judge_cfg.get("replacement_fallback_min_preservation", 0.85)
        )
        self.internal_vlm_judge_replacement_fallback_min_artifact_free = float(
            self.internal_vlm_judge_cfg.get("replacement_fallback_min_artifact_free", 0.85)
        )
        self.internal_vlm_judge_replacement_fallback_max_edited_source_score_ratio = float(
            self.internal_vlm_judge_cfg.get("replacement_fallback_max_edited_source_score_ratio", 0.90)
        )
        self.internal_vlm_judge_replacement_fallback_min_source_score_drop = float(
            self.internal_vlm_judge_cfg.get("replacement_fallback_min_source_score_drop", 0.04)
        )
        self.internal_vlm_judge_replacement_fallback_min_localization_precision = float(
            self.internal_vlm_judge_cfg.get("replacement_fallback_min_localization_precision", 0.30)
        )
        self.internal_vlm_judge_replacement_fallback_min_outside_preservation = float(
            self.internal_vlm_judge_cfg.get("replacement_fallback_min_outside_preservation", 0.80)
        )
        self.internal_vlm_judge_removal_detector_fallback = bool(
            self.internal_vlm_judge_cfg.get("removal_detector_fallback", True)
        )
        self.internal_vlm_judge_removal_fallback_require_best = bool(
            self.internal_vlm_judge_cfg.get("removal_fallback_require_best", True)
        )
        self.internal_vlm_judge_removal_judge_fallback = bool(
            self.internal_vlm_judge_cfg.get("removal_judge_fallback", False)
        )
        self.internal_vlm_judge_removal_structured_absence_fallback = bool(
            self.internal_vlm_judge_cfg.get("removal_structured_absence_fallback", False)
        )
        self.internal_vlm_judge_removal_judge_fallback_require_best = bool(
            self.internal_vlm_judge_cfg.get(
                "removal_judge_fallback_require_best",
                self.internal_vlm_judge_cfg.get("removal_fallback_require_best", False),
            )
        )
        self.internal_vlm_judge_removal_judge_fallback_min_outside_preservation = float(
            self.internal_vlm_judge_cfg.get("removal_judge_fallback_min_outside_preservation", 0.70)
        )
        self.internal_vlm_judge_removal_judge_fallback_min_localization = float(
            self.internal_vlm_judge_cfg.get("removal_judge_fallback_min_localization", 0.0)
        )
        self.internal_vlm_judge_removal_judge_fallback_min_outside_changed_fraction = float(
            self.internal_vlm_judge_cfg.get("removal_judge_fallback_min_outside_changed_fraction", 0.90)
        )
        self.internal_vlm_judge_removal_structured_fallback_require_best = bool(
            self.internal_vlm_judge_cfg.get(
                "removal_structured_fallback_require_best",
                self.internal_vlm_judge_cfg.get("removal_fallback_require_best", False),
            )
        )
        self.internal_vlm_judge_removal_structured_fallback_min_raw_reward = float(
            self.internal_vlm_judge_cfg.get("removal_structured_fallback_min_raw_reward", 0.55)
        )
        self.internal_vlm_judge_removal_structured_fallback_min_localization = float(
            self.internal_vlm_judge_cfg.get("removal_structured_fallback_min_localization", 0.55)
        )
        self.internal_vlm_judge_removal_structured_fallback_min_outside_preservation = float(
            self.internal_vlm_judge_cfg.get("removal_structured_fallback_min_outside_preservation", 0.80)
        )
        self.internal_vlm_judge_removal_structured_fallback_min_outside_changed_fraction = float(
            self.internal_vlm_judge_cfg.get("removal_structured_fallback_min_outside_changed_fraction", 0.90)
        )
        self.internal_vlm_judge_removal_structured_fallback_min_fill_naturalness = float(
            self.internal_vlm_judge_cfg.get("removal_structured_fallback_min_fill_naturalness", 0.50)
        )
        self.internal_vlm_judge_removal_fallback_min_semantic = float(
            self.internal_vlm_judge_cfg.get("removal_fallback_min_semantic", 0.80)
        )
        self.internal_vlm_judge_removal_fallback_min_object_absence = float(
            self.internal_vlm_judge_cfg.get("removal_fallback_min_object_absence", 0.85)
        )
        self.internal_vlm_judge_removal_fallback_min_fill_naturalness = float(
            self.internal_vlm_judge_cfg.get("removal_fallback_min_fill_naturalness", 0.80)
        )
        self.internal_vlm_judge_removal_fallback_min_preservation = float(
            self.internal_vlm_judge_cfg.get("removal_fallback_min_preservation", 0.85)
        )
        self.internal_vlm_judge_removal_fallback_min_artifact_free = float(
            self.internal_vlm_judge_cfg.get("removal_fallback_min_artifact_free", 0.85)
        )
        self.internal_vlm_judge_removal_fallback_max_edited_source_score_ratio = float(
            self.internal_vlm_judge_cfg.get("removal_fallback_max_edited_source_score_ratio", 1.15)
        )
        self.internal_vlm_judge_removal_fallback_max_edited_source_score_margin = float(
            self.internal_vlm_judge_cfg.get("removal_fallback_max_edited_source_score_margin", 0.05)
        )
        self.internal_vlm_judge_subject_extraction_fallback = bool(
            self.internal_vlm_judge_cfg.get(
                "subject_extraction_fallback",
                self.internal_vlm_judge_cfg.get("extraction_fallback", False),
            )
        )
        subject_extraction_fallback_allowed_reasons = self.internal_vlm_judge_cfg.get(
            "subject_extraction_fallback_allowed_reject_reasons",
            ["rubric_forbidden_gate", "cepr_validity", "rubric_cepr_reward"],
        )
        if isinstance(subject_extraction_fallback_allowed_reasons, str):
            subject_extraction_fallback_allowed_reasons = [
                item.strip()
                for item in subject_extraction_fallback_allowed_reasons.split(",")
                if item.strip()
            ]
        self.internal_vlm_judge_subject_extraction_fallback_allowed_reject_reasons = {
            str(item).strip()
            for item in (subject_extraction_fallback_allowed_reasons or [])
            if str(item).strip()
        }
        self.internal_vlm_judge_subject_extraction_fallback_require_best = bool(
            self.internal_vlm_judge_cfg.get("subject_extraction_fallback_require_best", False)
        )
        self.internal_vlm_judge_subject_extraction_fallback_min_score = float(
            self.internal_vlm_judge_cfg.get("subject_extraction_fallback_min_score", 0.80)
        )
        self.internal_vlm_judge_subject_extraction_fallback_min_semantic = float(
            self.internal_vlm_judge_cfg.get("subject_extraction_fallback_min_semantic", 0.80)
        )
        self.internal_vlm_judge_subject_extraction_fallback_min_preservation = float(
            self.internal_vlm_judge_cfg.get("subject_extraction_fallback_min_preservation", 0.85)
        )
        self.internal_vlm_judge_subject_extraction_fallback_min_artifact_free = float(
            self.internal_vlm_judge_cfg.get("subject_extraction_fallback_min_artifact_free", 0.85)
        )
        self.internal_vlm_judge_subject_extraction_fallback_min_raw_reward = float(
            self.internal_vlm_judge_cfg.get("subject_extraction_fallback_min_raw_reward", 0.40)
        )
        self.internal_vlm_judge_subject_extraction_fallback_min_required_after = float(
            self.internal_vlm_judge_cfg.get("subject_extraction_fallback_min_required_after", 0.55)
        )
        self.internal_vlm_judge_action_compose_fallback = bool(
            self.internal_vlm_judge_cfg.get("action_compose_fallback", False)
        )
        action_compose_fallback_edit_types = self.internal_vlm_judge_cfg.get(
            "action_compose_fallback_edit_types",
            ["action_change", "compose"],
        )
        if isinstance(action_compose_fallback_edit_types, str):
            action_compose_fallback_edit_types = [
                item.strip()
                for item in action_compose_fallback_edit_types.split(",")
                if item.strip()
            ]
        self.internal_vlm_judge_action_compose_fallback_edit_types = {
            str(item).strip()
            for item in (action_compose_fallback_edit_types or [])
            if str(item).strip()
        }
        action_compose_fallback_allowed_reasons = self.internal_vlm_judge_cfg.get(
            "action_compose_fallback_allowed_reject_reasons",
            ["rubric_forbidden_gate", "cepr_edit_specificity", "cepr_taxonomy"],
        )
        if isinstance(action_compose_fallback_allowed_reasons, str):
            action_compose_fallback_allowed_reasons = [
                item.strip()
                for item in action_compose_fallback_allowed_reasons.split(",")
                if item.strip()
            ]
        self.internal_vlm_judge_action_compose_fallback_allowed_reject_reasons = {
            str(item).strip()
            for item in (action_compose_fallback_allowed_reasons or [])
            if str(item).strip()
        }
        self.internal_vlm_judge_action_compose_fallback_require_best = bool(
            self.internal_vlm_judge_cfg.get("action_compose_fallback_require_best", False)
        )
        self.internal_vlm_judge_action_compose_fallback_min_edit_success = float(
            self.internal_vlm_judge_cfg.get("action_compose_fallback_min_edit_success", 0.30)
        )
        self.internal_vlm_judge_action_compose_fallback_min_instruction_following = float(
            self.internal_vlm_judge_cfg.get("action_compose_fallback_min_instruction_following", 0.20)
        )
        self.internal_vlm_judge_action_compose_fallback_min_overall = float(
            self.internal_vlm_judge_cfg.get("action_compose_fallback_min_overall", 0.30)
        )
        self.internal_vlm_judge_action_compose_fallback_min_preservation = float(
            self.internal_vlm_judge_cfg.get("action_compose_fallback_min_preservation", 0.50)
        )
        self.internal_vlm_judge_action_compose_fallback_min_artifact_free = float(
            self.internal_vlm_judge_cfg.get("action_compose_fallback_min_artifact_free", 0.50)
        )
        self.internal_vlm_judge_action_compose_fallback_min_outside_preservation = float(
            self.internal_vlm_judge_cfg.get("action_compose_fallback_min_outside_preservation", 0.80)
        )
        self.internal_vlm_judge_mode = str(self.internal_vlm_judge_cfg.get("mode", "per_candidate")).strip().lower()
        if self.internal_vlm_judge_mode not in {"per_candidate", "group"}:
            self.internal_vlm_judge_mode = "per_candidate"

    @staticmethod
    def _unique_texts(items: list[Any], max_items: int) -> list[str]:
        output: list[str] = []
        seen = set()
        for item in items:
            text = str(item).strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            output.append(text)
            if len(output) >= max_items:
                break
        return output

    @staticmethod
    def _string_set(value: Any) -> set[str]:
        if value is None:
            return set()
        if isinstance(value, str):
            raw_items = [part.strip() for part in value.split(",")]
        else:
            raw_items = [str(item).strip() for item in value]
        return {item for item in raw_items if item}

    @staticmethod
    def _float_by_edit_type(value: Any) -> dict[str, float]:
        if not isinstance(value, dict):
            return {}
        out: dict[str, float] = {}
        for key, raw in value.items():
            edit_type = str(key).strip()
            if not edit_type:
                continue
            parsed = _finite_float(raw, math.nan)
            if math.isfinite(parsed):
                out[edit_type] = float(parsed)
        return out

    @staticmethod
    def _weighted_geometric_blend(value_a: float, weight_a: float, value_b: float, weight_b: float) -> float:
        value_a = _clamp(value_a)
        value_b = _clamp(value_b)
        weight_a = max(float(weight_a), 0.0)
        weight_b = max(float(weight_b), 0.0)
        total_weight = weight_a + weight_b
        if total_weight <= 0.0:
            return value_a
        eps = 1.0e-6
        blended = math.exp(
            (weight_a * math.log(max(value_a, eps)) + weight_b * math.log(max(value_b, eps)))
            / total_weight
        )
        return _clamp(blended)

    @staticmethod
    def _normalize_judge_score(value: Any, default: float = 0.5) -> float:
        score = _finite_float(value, math.nan)
        if not math.isfinite(score):
            return default
        if score > 10.0:
            return _clamp(score / 100.0)
        if score > 5.0:
            return _clamp(score / 10.0)
        if score > 1.0:
            return _clamp((score - 1.0) / 4.0)
        return _clamp(score)

    @staticmethod
    def _parse_judge_candidate_index(value: Any, default: int = -1) -> int:
        parsed = _finite_float(value, math.nan)
        if math.isfinite(parsed):
            return int(parsed)
        match = re.search(r"\d+", str(value or ""))
        return int(match.group(0)) if match else default

    def _resize_for_internal_vlm_judge(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        max_side = max(64, self.internal_vlm_judge_image_resolution)
        width, height = image.size
        scale = min(1.0, max_side / max(width, height))
        if scale >= 1.0:
            return image.copy()
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        return image.resize(new_size, Image.Resampling.BICUBIC)

    def _internal_vlm_judge_prompt(
        self,
        proposal: EditProposal,
        candidate_indices: list[int],
    ) -> str:
        spec = normalize_structured_edit(
            proposal.structured_edit,
            instruction=proposal.instruction,
            family=proposal.definition.family,
        )
        candidate_lines = "\n".join(
            f"- Candidate {candidate_index}: edited image shown after the original."
            for candidate_index in candidate_indices
        )
        spec_json = json.dumps(spec, ensure_ascii=True, sort_keys=True)
        candidate_index_json = json.dumps(candidate_indices)
        schema_json = json.dumps(
            {
                "candidates": [
                    {
                        "candidate_index": candidate_index,
                        "instruction_following": None,
                        "edit_success": None,
                        "target_correctness": None,
                        "target_object_visible_after": None,
                        "preservation": None,
                        "artifact_free": None,
                        "overall": None,
                        "confidence": None,
                        "source_object_visible_before": None,
                        "object_visible_after": None,
                        "object_absence": None,
                        "fill_naturalness": None,
                        "reason": "short visual reason",
                    }
                    for candidate_index in candidate_indices
                ],
                "best_candidate_index": candidate_indices[0] if candidate_indices else 0,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
        return (
            "You are Qwen-Image-Edit's internal multimodal self-judge. Compare each edited candidate "
            "against the original image and the requested edit. Use only visual evidence in the images. "
            "Do not reward candidates that merely look plausible if they fail the requested edit.\n\n"
            f"Instruction: {proposal.instruction}\n"
            f"Structured edit: {spec_json}\n\n"
            "Each image is introduced by a text label immediately before the image. "
            "The original image is only for comparison; when scoring a candidate, judge the edited "
            "candidate image after applying the instruction, not the original image.\n"
            f"{candidate_lines}\n\n"
            f"Required candidate indices: {candidate_index_json}. You must return exactly one scored JSON object "
            "for every required candidate index, including failed or low-quality candidates. Do not omit any "
            "candidate.\n\n"
            "For each candidate, score these fields from 0.0 to 1.0:\n"
            "- instruction_following: requested edit is actually performed.\n"
            "- edit_success: target object/attribute/location/style after-state is correct.\n"
            "- target_correctness: edited target and target region match the structured edit.\n"
            "- target_object_visible_after: for object_replacement/object_addition, the requested new target object is visible in the edited image; for other edit types, use target_correctness.\n"
            "- preservation: unrelated source content, identity, layout, and viewpoint are preserved.\n"
            "- artifact_free: no obvious artifacts, distortions, duplicated objects, or broken text.\n"
            "- overall: your final visual edit quality score.\n"
            "- confidence: confidence in your judgment.\n\n"
            "For object_removal edits, also score these fields from 0.0 to 1.0:\n"
            "- source_object_visible_before: the requested source object is visible in the original image.\n"
            "- object_visible_after: the requested source object or its remnants are still visible in the edited image; 1.0 means clearly still visible and 0.0 means absent.\n"
            "- object_absence: the requested source object is absent from the edited image; 1.0 means fully removed.\n"
            "- fill_naturalness: the removed region is naturally filled with surrounding image content.\n"
            "For object_removal, a candidate with object_visible_after above 0.2 should receive low "
            "instruction_following, edit_success, target_correctness, object_absence, and overall scores. "
            "If the requested object is still visible after editing, set object_visible_after close to 1.0 "
            "and object_absence close to 0.0. If the object is moved, recolored, transformed into a "
            "similar object, or appears anywhere else in the edited image, treat it as still visible "
            "and give low removal scores. Also penalize viewpoint, crop, layout, or unrelated object "
            "changes even if the target object seems removed.\n\n"
            "For object_replacement edits, source_object_visible_before means the old/source object is visible "
            "in the original image, object_visible_after means that old/source object or its remnants are still "
            "visible in the edited image, and object_absence means the old/source object is absent after the "
            "replacement. target_object_visible_after means the new replacement object is clearly present in "
            "the edited image at the intended region. A correct replacement must have high target_correctness, "
            "high target_object_visible_after, and high object_absence while preserving unrelated content. If "
            "the old object is removed but the new object is missing, vague, malformed, or in the wrong region, "
            "give low edit_success, target_correctness, target_object_visible_after, and overall scores.\n\n"
            "Return JSON only, with this exact shape and the required indices:\n"
            f"{schema_json}\n"
            "Replace every null placeholder with a numeric float from 0.0 to 1.0. Do not return null values."
        )

    @staticmethod
    def _judge_reason_says_object_removal_failed(reason: str) -> bool:
        text = str(reason or "").lower()
        if not text:
            return False
        failure_patterns = [
            r"\bstill\s+(?:clearly\s+)?(?:visible|present|there)\b",
            r"\bremains?\s+(?:clearly\s+)?(?:visible|present|there)?\b",
            r"\bremaining\s+(?:object|part|remnant|piece|portion)\b",
            r"\bnot\s+(?:removed|absent|erased|deleted|fully removed|completely removed)\b",
            r"\bfailed\s+to\s+(?:remove|erase|delete)\b",
            r"\bcontinues?\s+to\s+(?:be\s+)?(?:visible|appear)\b",
        ]
        return any(re.search(pattern, text) for pattern in failure_patterns)

    def _parse_internal_vlm_judge_output(
        self,
        decoded: str,
        candidate_indices: set[int],
    ) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
        parsed = extract_json_object(decoded)
        if parsed is None:
            raise ValueError(f"Qwen-VL judge did not return parseable JSON: {decoded[:500]}")
        raw_candidates = parsed.get("candidates", [])
        if isinstance(raw_candidates, dict):
            indexed_candidates = []
            for raw_index, raw_value in raw_candidates.items():
                if not isinstance(raw_value, dict):
                    continue
                item = dict(raw_value)
                item.setdefault("candidate_index", raw_index)
                indexed_candidates.append(item)
            raw_candidates = indexed_candidates
        if not isinstance(raw_candidates, list):
            raise ValueError("Qwen-VL judge JSON is missing a candidates list.")

        scores_by_index: dict[int, dict[str, Any]] = {}
        for raw_item in raw_candidates:
            if not isinstance(raw_item, dict):
                continue
            candidate_index = self._parse_judge_candidate_index(raw_item.get("candidate_index"))
            if candidate_index not in candidate_indices and len(candidate_indices) == 1:
                candidate_index = next(iter(candidate_indices))
            if candidate_index not in candidate_indices:
                continue
            instruction_following = self._normalize_judge_score(raw_item.get("instruction_following"))
            edit_success = self._normalize_judge_score(raw_item.get("edit_success"))
            target_correctness = self._normalize_judge_score(raw_item.get("target_correctness"), edit_success)
            target_object_visible_after = self._normalize_judge_score(
                raw_item.get("target_object_visible_after"),
                target_correctness,
            )
            preservation = self._normalize_judge_score(raw_item.get("preservation"))
            artifact_free = self._normalize_judge_score(raw_item.get("artifact_free"))
            overall = self._normalize_judge_score(raw_item.get("overall"))
            confidence = self._normalize_judge_score(raw_item.get("confidence"), default=0.0)
            missing_fields = [
                field
                for field in (
                    "instruction_following",
                    "edit_success",
                    "target_correctness",
                    "target_object_visible_after",
                    "preservation",
                    "artifact_free",
                    "overall",
                    "confidence",
                    "source_object_visible_before",
                    "object_visible_after",
                    "object_absence",
                    "fill_naturalness",
                )
                if raw_item.get(field) is None
            ]
            source_object_visible_before = self._normalize_judge_score(
                raw_item.get("source_object_visible_before"),
                default=1.0,
            )
            object_visible_after = self._normalize_judge_score(
                raw_item.get("object_visible_after"),
                default=1.0 - edit_success,
            )
            object_absence = self._normalize_judge_score(
                raw_item.get("object_absence"),
                default=1.0 - object_visible_after,
            )
            fill_naturalness = self._normalize_judge_score(
                raw_item.get("fill_naturalness"),
                default=target_correctness,
            )
            semantic = self._geometric_mean([instruction_following, edit_success, target_correctness])
            score = self._geometric_mean([semantic, preservation, artifact_free, overall])
            scores_by_index[candidate_index] = {
                "instruction_following": instruction_following,
                "edit_success": edit_success,
                "target_correctness": target_correctness,
                "target_object_visible_after": target_object_visible_after,
                "semantic": semantic,
                "preservation": preservation,
                "artifact_free": artifact_free,
                "overall": overall,
                "confidence": confidence,
                "source_object_visible_before": source_object_visible_before,
                "object_visible_after": object_visible_after,
                "object_absence": object_absence,
                "fill_naturalness": fill_naturalness,
                "score": score,
                "reason": str(raw_item.get("reason", ""))[:240],
                "missing_fields": missing_fields,
            }
        if not scores_by_index:
            raise ValueError("Qwen-VL judge JSON contained no candidate scores matching this group.")
        best_candidate_index = parsed.get("best_candidate_index")
        if best_candidate_index is not None:
            best_candidate_index = self._parse_judge_candidate_index(best_candidate_index)
            if best_candidate_index not in candidate_indices:
                best_candidate_index = None
        return scores_by_index, {
            "best_candidate_index": best_candidate_index,
            "raw_text_preview": decoded[:500],
        }

    @staticmethod
    def _missing_internal_vlm_judge_score(candidate_index: int, reason: str) -> dict[str, Any]:
        return {
            "instruction_following": 0.0,
            "edit_success": 0.0,
            "target_correctness": 0.0,
            "target_object_visible_after": 0.0,
            "semantic": 1.0e-6,
            "preservation": 0.0,
            "artifact_free": 0.0,
            "overall": 0.0,
            "confidence": 0.0,
            "source_object_visible_before": 0.0,
            "object_visible_after": 1.0,
            "object_absence": 0.0,
            "fill_naturalness": 0.0,
            "score": 1.0e-6,
            "reason": reason[:240],
            "missing_fallback": True,
            "candidate_index": candidate_index,
        }

    def _generate_internal_vlm_judge_text(
        self,
        pipe: Any,
        proposal: EditProposal,
        original: Image.Image,
        candidate_items: list[tuple[int, Image.Image]],
    ) -> str:
        if getattr(pipe, "processor", None) is None or getattr(pipe, "text_encoder", None) is None:
            raise ValueError("Qwen-VL judge requires a pipeline with processor and text_encoder.")
        model = pipe.text_encoder
        processor = pipe.processor
        if not hasattr(model, "generate") or not hasattr(processor, "apply_chat_template"):
            raise ValueError("Qwen-VL judge requires a generative Qwen text_encoder and chat processor.")

        candidate_indices = [candidate_index for candidate_index, _ in candidate_items]
        prompt = self._internal_vlm_judge_prompt(proposal, candidate_indices)
        content: list[dict[str, Any]] = [
            {"type": "text", "text": "Original image:"},
            {"type": "image", "image": "original"},
        ]
        images = [self._resize_for_internal_vlm_judge(original)]
        for candidate_index, image in candidate_items:
            content.extend(
                [
                    {"type": "text", "text": f"Edited candidate {candidate_index}:"},
                    {"type": "image", "image": f"candidate_{candidate_index}"},
                ]
            )
            images.append(self._resize_for_internal_vlm_judge(image))
        content.append({"type": "text", "text": prompt})
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict image-editing evaluator. Return compact JSON only; "
                    "do not include markdown or prose outside JSON."
                ),
            },
            {"role": "user", "content": content},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        device = getattr(model, "device", None)
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = getattr(pipe, "device", "cpu")
        inputs = processor(text=[text], images=images, padding=True, return_tensors="pt")
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        do_sample = self.internal_vlm_judge_temperature > 0.0
        generation_kwargs = {
            **inputs,
            "max_new_tokens": self.internal_vlm_judge_max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            generation_kwargs["temperature"] = self.internal_vlm_judge_temperature
            generation_kwargs["top_p"] = self.internal_vlm_judge_top_p
        import torch

        with torch.no_grad():
            output_ids = model.generate(**generation_kwargs)
        generated = output_ids[:, inputs["input_ids"].shape[1] :]
        return processor.batch_decode(generated, skip_special_tokens=True)[0]

    def _run_internal_vlm_judge(
        self,
        pipe: Any,
        proposal: EditProposal,
        original: Image.Image,
        candidate_items: list[tuple[int, Image.Image]],
    ) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
        candidate_indices = [candidate_index for candidate_index, _ in candidate_items]
        if self.internal_vlm_judge_mode == "per_candidate" and len(candidate_items) > 1:
            scores_by_index: dict[int, dict[str, Any]] = {}
            fallback_errors: dict[int, str] = {}
            raw_previews: dict[int, str] = {}
            for candidate_index, image in candidate_items:
                try:
                    decoded = self._generate_internal_vlm_judge_text(
                        pipe,
                        proposal,
                        original,
                        [(candidate_index, image)],
                    )
                    raw_previews[candidate_index] = decoded[:240]
                    single_scores, _ = self._parse_internal_vlm_judge_output(decoded, {candidate_index})
                    if candidate_index in single_scores:
                        scores_by_index[candidate_index] = single_scores[candidate_index]
                        continue
                except Exception as exc:
                    fallback_errors[candidate_index] = f"{exc.__class__.__name__}: {str(exc)[:160]}"
                scores_by_index[candidate_index] = self._missing_internal_vlm_judge_score(
                    candidate_index,
                    "Internal VLM judge failed to score this candidate in per-candidate mode; assigning a conservative low-confidence score.",
                )
            best_candidate_index = max(
                scores_by_index,
                key=lambda index: float(scores_by_index[index].get("score", 0.0)),
            ) if scores_by_index else None
            return scores_by_index, {
                "best_candidate_index": best_candidate_index,
                "raw_text_preview": json.dumps(raw_previews, ensure_ascii=True)[:500],
                "judge_mode": "per_candidate",
                "missing_candidate_indices": [
                    index for index, score in scores_by_index.items() if score.get("missing_fallback")
                ],
                "fallback_candidate_indices": [
                    index for index, score in scores_by_index.items() if score.get("missing_fallback")
                ],
                "fallback_errors": fallback_errors,
            }
        decoded = self._generate_internal_vlm_judge_text(pipe, proposal, original, candidate_items)
        scores_by_index, summary = self._parse_internal_vlm_judge_output(decoded, set(candidate_indices))
        missing_indices = [candidate_index for candidate_index in candidate_indices if candidate_index not in scores_by_index]
        fallback_indices: list[int] = []
        fallback_errors: dict[int, str] = {}
        candidate_item_by_index = dict(candidate_items)
        for candidate_index in missing_indices:
            image = candidate_item_by_index.get(candidate_index)
            if image is None:
                continue
            try:
                single_decoded = self._generate_internal_vlm_judge_text(
                    pipe,
                    proposal,
                    original,
                    [(candidate_index, image)],
                )
                single_scores, _ = self._parse_internal_vlm_judge_output(single_decoded, {candidate_index})
                if candidate_index in single_scores:
                    scores_by_index[candidate_index] = single_scores[candidate_index]
                    fallback_indices.append(candidate_index)
                    continue
            except Exception as exc:
                fallback_errors[candidate_index] = f"{exc.__class__.__name__}: {str(exc)[:160]}"
            scores_by_index[candidate_index] = self._missing_internal_vlm_judge_score(
                candidate_index,
                "Internal VLM judge omitted this candidate after fallback; assigning a conservative low-confidence score.",
            )
            fallback_indices.append(candidate_index)
        summary["missing_candidate_indices"] = missing_indices
        summary["fallback_candidate_indices"] = fallback_indices
        summary["fallback_errors"] = fallback_errors
        return scores_by_index, summary

    def _apply_group_judge(
        self,
        proposal: EditProposal,
        original: Image.Image,
        edited_candidates: list[Image.Image],
        rows: list[dict[str, Any]],
        editor: Any | None,
    ) -> None:
        if not self.internal_vlm_judge_enabled or not rows:
            return
        candidate_rows = sorted(
            rows,
            key=lambda row: (
                1.0 if row.get("feasible") else 0.0,
                _finite_float(row.get("raw_reward")),
                _finite_float(row.get("reward")),
            ),
            reverse=True,
        )
        candidate_rows = candidate_rows[: max(1, self.internal_vlm_judge_max_candidates)]
        candidate_items = [
            (int(row["candidate_index"]), edited_candidates[int(row["candidate_index"])])
            for row in candidate_rows
            if 0 <= int(row["candidate_index"]) < len(edited_candidates)
            and (not self.internal_vlm_judge_skip_infeasible or row.get("feasible"))
        ]
        if not candidate_items:
            return
        try:
            if hasattr(editor, "prepare_for_internal_scoring"):
                editor.prepare_for_internal_scoring()
            pipe = self._get_internal_pipe(editor)
            judge_scores, judge_summary = self._run_internal_vlm_judge(
                pipe,
                proposal,
                original,
                candidate_items,
            )
        except Exception as exc:
            for row in rows:
                signals = row.setdefault("signals", {})
                signals["internal_vlm_judge_supported"] = 0.0
                signals["internal_vlm_judge_error"] = 1.0
                signals["internal_vlm_judge_error_type"] = exc.__class__.__name__
                signals["internal_vlm_judge_error_message"] = str(exc)[:500]
                if not self.internal_vlm_judge_fail_open:
                    row["feasible"] = False
                    row["reward"] = 0.0
            return

        best_candidate_index = judge_summary.get("best_candidate_index")
        missing_candidate_indices = set(judge_summary.get("missing_candidate_indices") or [])
        fallback_candidate_indices = set(judge_summary.get("fallback_candidate_indices") or [])
        fallback_errors = judge_summary.get("fallback_errors") or {}
        spec = normalize_structured_edit(
            proposal.structured_edit,
            instruction=proposal.instruction,
            family=proposal.definition.family,
        )
        edit_type = str(spec.get("edit_type", "local_enhancement"))
        for row in rows:
            candidate_index = int(row["candidate_index"])
            if self.internal_vlm_judge_skip_infeasible and not row.get("feasible"):
                # Already rejected by a cheaper gate; the judge was intentionally not run for it.
                # Leave its existing feasibility/reward/reject_reason untouched.
                continue
            signals = row.setdefault("signals", {})
            component_scores = row.setdefault("component_scores", {})
            if candidate_index not in judge_scores:
                signals["internal_vlm_judge_supported"] = 0.0
                signals["internal_vlm_judge_missing_from_group"] = 1.0
                if self.internal_vlm_judge_require_for_feasible or not self.internal_vlm_judge_fail_open:
                    row["feasible"] = False
                    row["reward"] = 0.0
                    signals["rubric_reject_reason"] = "internal_vlm_judge_missing"
                continue
            score = judge_scores[candidate_index]
            missing_fallback = bool(score.get("missing_fallback", False))
            judge_score = float(score["score"])
            judge_semantic = float(score["semantic"])
            judge_preservation = float(score["preservation"])
            judge_artifact_free = float(score["artifact_free"])
            judge_target_correctness = float(score["target_correctness"])
            judge_target_object_visible_after = float(
                score.get("target_object_visible_after", judge_target_correctness)
            )
            judge_confidence = float(score["confidence"])
            judge_source_visible_before = float(score.get("source_object_visible_before", 1.0))
            judge_object_visible_after = float(score.get("object_visible_after", 1.0 - judge_semantic))
            judge_object_absence = float(score.get("object_absence", 1.0 - judge_object_visible_after))
            judge_fill_naturalness = float(score.get("fill_naturalness", score.get("target_correctness", judge_semantic)))
            raw_judge_object_visible_after = judge_object_visible_after
            raw_judge_object_absence = judge_object_absence
            object_removal_reason_failed = False
            object_removal_field_pass = True
            object_removal_source_visible_pass = True
            object_removal_visibility_pass = True
            object_removal_gate_pass = True
            if edit_type == "object_removal":
                object_removal_reason_failed = self._judge_reason_says_object_removal_failed(
                    str(score.get("reason", ""))
                )
                missing_object_fields = {
                    "source_object_visible_before",
                    "object_visible_after",
                    "object_absence",
                    "fill_naturalness",
                }.intersection(set(score.get("missing_fields") or []))
                object_removal_field_pass = not missing_object_fields
                judge_object_visible_after = _clamp(judge_object_visible_after)
                judge_object_absence = _clamp(judge_object_absence)
                judge_fill_naturalness = _clamp(judge_fill_naturalness)
                judge_source_visible_before = _clamp(judge_source_visible_before)
                object_removal_source_visible_pass = judge_source_visible_before >= 0.5
                effective_object_absence = min(judge_object_absence, _clamp(1.0 - judge_object_visible_after))
                object_removal_visibility_pass = (
                    judge_object_visible_after <= 0.2 and effective_object_absence >= 0.8
                )
                object_removal_gate_pass = (
                    object_removal_field_pass
                    and object_removal_source_visible_pass
                    and object_removal_visibility_pass
                )
                object_not_visible_after = _clamp(1.0 - judge_object_visible_after)
                judge_semantic = self._geometric_mean(
                    [
                        judge_semantic,
                        judge_source_visible_before,
                        effective_object_absence,
                        object_not_visible_after,
                        judge_fill_naturalness,
                    ]
                )
                judge_score = self._geometric_mean(
                    [
                        judge_semantic,
                        judge_preservation,
                        judge_artifact_free,
                        float(score["overall"]),
                    ]
                )
            elif edit_type == "object_replacement":
                effective_source_absence = min(
                    _clamp(judge_object_absence),
                    _clamp(1.0 - judge_object_visible_after),
                )
                judge_target_object_visible_after = _clamp(judge_target_object_visible_after)
                judge_semantic = self._geometric_mean(
                    [
                        judge_semantic,
                        effective_source_absence,
                        judge_target_correctness,
                        judge_target_object_visible_after,
                    ]
                )
                judge_score = self._geometric_mean(
                    [
                        judge_semantic,
                        judge_preservation,
                        judge_artifact_free,
                        float(score["overall"]),
                    ]
                )
            pre_judge_reward = _finite_float(row.get("reward"))
            pre_judge_raw_reward = _finite_float(row.get("raw_reward"))
            pre_judge_semantic = _finite_float(row.get("semantic_edit"))
            judge_supported = not missing_fallback
            judge_reliable = (not missing_fallback) and judge_confidence >= self.internal_vlm_judge_min_confidence
            min_judge_score = self.internal_vlm_judge_min_score_by_edit_type.get(
                edit_type, self.internal_vlm_judge_min_score
            )
            min_judge_semantic = self.internal_vlm_judge_min_semantic_by_edit_type.get(
                edit_type, self.internal_vlm_judge_min_semantic
            )
            min_judge_preservation = self.internal_vlm_judge_min_preservation_by_edit_type.get(
                edit_type, self.internal_vlm_judge_min_preservation
            )
            min_judge_artifact_free = self.internal_vlm_judge_min_artifact_free_by_edit_type.get(
                edit_type, self.internal_vlm_judge_min_artifact_free
            )
            min_judge_target_correctness = self.internal_vlm_judge_min_target_correctness_by_edit_type.get(
                edit_type, self.internal_vlm_judge_min_target_correctness
            )
            judge_low_score = judge_supported and (
                judge_score < min_judge_score
                or judge_semantic < min_judge_semantic
                or judge_preservation < min_judge_preservation
                or judge_artifact_free < min_judge_artifact_free
                or judge_target_correctness < min_judge_target_correctness
            )
            if edit_type == "object_removal" and not object_removal_gate_pass:
                judge_low_score = True
            confidence_pass = (
                judge_confidence >= self.internal_vlm_judge_min_confidence
                or not self.internal_vlm_judge_require_confidence_for_feasible
            )
            judge_pass = judge_supported and not judge_low_score and confidence_pass
            use_judge_for_reward = judge_supported and (
                judge_reliable or self.internal_vlm_judge_use_unreliable_scores or judge_low_score
            )
            if use_judge_for_reward:
                combined_raw_reward = self._weighted_geometric_blend(
                    pre_judge_raw_reward,
                    self.internal_vlm_judge_cepr_weight,
                    judge_score,
                    self.internal_vlm_judge_weight,
                )
                combined_semantic = self._weighted_geometric_blend(
                    pre_judge_semantic,
                    self.internal_vlm_judge_cepr_weight,
                    judge_semantic,
                    self.internal_vlm_judge_weight,
                )
            else:
                combined_raw_reward = pre_judge_raw_reward
                combined_semantic = pre_judge_semantic
            judge_hard_fail = (
                (missing_fallback and self.internal_vlm_judge_missing_is_failure)
                or (judge_low_score and self.internal_vlm_judge_fail_closed_on_low_score)
                or (self.internal_vlm_judge_require_for_feasible and not judge_pass)
            )
            combined_reward_pass = combined_raw_reward >= self.reward_threshold
            component_scores.update(
                {
                    "cepr_pre_vlm_reward": pre_judge_reward,
                    "cepr_pre_vlm_raw_reward": pre_judge_raw_reward,
                    "cepr_pre_vlm_semantic_edit": pre_judge_semantic,
                    "internal_vlm_judge_score": judge_score,
                    "internal_vlm_judge_semantic": judge_semantic,
                    "internal_vlm_judge_instruction_following": float(score["instruction_following"]),
                    "internal_vlm_judge_edit_success": float(score["edit_success"]),
                    "internal_vlm_judge_target_correctness": judge_target_correctness,
                    "internal_vlm_judge_target_object_visible_after": judge_target_object_visible_after,
                    "internal_vlm_judge_preservation": judge_preservation,
                    "internal_vlm_judge_artifact_free": judge_artifact_free,
                    "internal_vlm_judge_overall": float(score["overall"]),
                    "internal_vlm_judge_confidence": judge_confidence,
                    "internal_vlm_judge_source_object_visible_before": judge_source_visible_before,
                    "internal_vlm_judge_object_visible_after": judge_object_visible_after,
                    "internal_vlm_judge_object_absence": judge_object_absence,
                    "internal_vlm_judge_fill_naturalness": judge_fill_naturalness,
                    "internal_vlm_judge_raw_object_visible_after": raw_judge_object_visible_after,
                    "internal_vlm_judge_raw_object_absence": raw_judge_object_absence,
                    "internal_vlm_judge_combined_raw_reward": combined_raw_reward,
                }
            )
            signals.update(
                {
                    "internal_vlm_judge_supported": 0.0 if missing_fallback else 1.0,
                    "internal_vlm_judge_pass": 1.0 if judge_pass else 0.0,
                    "internal_vlm_judge_best": 1.0 if candidate_index == best_candidate_index else 0.0,
                    "internal_vlm_judge_missing_from_group": (
                        1.0 if candidate_index in missing_candidate_indices else 0.0
                    ),
                    "internal_vlm_judge_fallback_used": (
                        1.0 if candidate_index in fallback_candidate_indices else 0.0
                    ),
                    "internal_vlm_judge_missing_fallback": 1.0 if missing_fallback else 0.0,
                    "internal_vlm_judge_reliable": 1.0 if judge_reliable else 0.0,
                    "internal_vlm_judge_low_confidence_ignored": (
                        1.0 if judge_supported and not judge_reliable and not use_judge_for_reward else 0.0
                    ),
                    "internal_vlm_judge_used_for_reward": 1.0 if use_judge_for_reward else 0.0,
                    "internal_vlm_judge_low_score": 1.0 if judge_low_score else 0.0,
                    "internal_vlm_judge_hard_fail": 1.0 if judge_hard_fail else 0.0,
                    "internal_vlm_judge_object_removal_specific": 1.0 if edit_type == "object_removal" else 0.0,
                    "internal_vlm_judge_object_removal_gate_pass": (
                        1.0 if object_removal_gate_pass else 0.0
                    ),
                    "internal_vlm_judge_object_removal_source_visible_pass": (
                        1.0 if object_removal_source_visible_pass else 0.0
                    ),
                    "internal_vlm_judge_object_removal_visibility_pass": (
                        1.0 if object_removal_visibility_pass else 0.0
                    ),
                    "internal_vlm_judge_object_removal_reason_failed": (
                        1.0 if object_removal_reason_failed else 0.0
                    ),
                    "internal_vlm_judge_missing_field_count": float(len(score.get("missing_fields") or [])),
                    "internal_vlm_judge_combined_reward_gate_pass": 1.0 if combined_reward_pass else 0.0,
                    "internal_vlm_judge_min_score": min_judge_score,
                    "internal_vlm_judge_min_semantic": min_judge_semantic,
                    "internal_vlm_judge_min_preservation": min_judge_preservation,
                    "internal_vlm_judge_min_artifact_free": min_judge_artifact_free,
                    "internal_vlm_judge_min_target_correctness": min_judge_target_correctness,
                    "internal_vlm_judge_min_confidence": self.internal_vlm_judge_min_confidence,
                    "internal_vlm_judge_require_for_feasible": 1.0 if self.internal_vlm_judge_require_for_feasible else 0.0,
                    "internal_vlm_judge_require_confidence_for_feasible": (
                        1.0 if self.internal_vlm_judge_require_confidence_for_feasible else 0.0
                    ),
                    "internal_vlm_judge_reason": score["reason"],
                }
            )
            if candidate_index in fallback_errors:
                signals["internal_vlm_judge_fallback_error"] = fallback_errors[candidate_index]
            row["semantic_edit"] = combined_semantic
            row["raw_reward"] = combined_raw_reward
            detector_original_source_score = _finite_float(
                component_scores.get("object_detector_original_source_score"),
                0.0,
            )
            detector_edited_source_score = _finite_float(
                component_scores.get("object_detector_edited_source_score"),
                1.0,
            )
            replacement_detector_edited_source_threshold = min(
                detector_original_source_score
                * self.internal_vlm_judge_replacement_fallback_max_edited_source_score_ratio,
                detector_original_source_score
                - self.internal_vlm_judge_replacement_fallback_min_source_score_drop,
            )
            replacement_fallback_detector_score_pass = (
                detector_edited_source_score <= replacement_detector_edited_source_threshold
            )
            replacement_fallback_localization_pass = (
                _finite_float(component_scores.get("conservative_localization_precision"), 0.0)
                >= self.internal_vlm_judge_replacement_fallback_min_localization_precision
            )
            replacement_fallback_outside_preservation_pass = (
                _finite_float(component_scores.get("conservative_outside_preservation"), 0.0)
                >= self.internal_vlm_judge_replacement_fallback_min_outside_preservation
            )
            replacement_reject_reason = str(signals.get("rubric_reject_reason", ""))
            replacement_reason_allowed = (
                not self.internal_vlm_judge_replacement_fallback_allowed_reject_reasons
                or replacement_reject_reason
                in self.internal_vlm_judge_replacement_fallback_allowed_reject_reasons
            )
            replacement_fallback_semantic_contract_pass = (
                judge_reliable
                and judge_pass
                and combined_reward_pass
                and judge_semantic >= self.internal_vlm_judge_replacement_fallback_min_semantic
                and judge_object_absence
                >= self.internal_vlm_judge_replacement_fallback_min_object_absence
                and float(score["target_correctness"])
                >= self.internal_vlm_judge_replacement_fallback_min_target_correctness
                and judge_target_object_visible_after
                >= self.internal_vlm_judge_replacement_fallback_min_target_object_visible_after
                and judge_preservation
                >= self.internal_vlm_judge_replacement_fallback_min_preservation
                and judge_artifact_free
                >= self.internal_vlm_judge_replacement_fallback_min_artifact_free
            )
            replacement_fallback_source_score_override = (
                self.internal_vlm_judge_replacement_fallback_allow_vlm_contract_override
                and replacement_fallback_semantic_contract_pass
                and replacement_fallback_localization_pass
                and replacement_fallback_outside_preservation_pass
                and signals.get("object_detector_source_detected_pass", 0.0) >= 0.5
                and signals.get("object_detector_target_present_pass", 0.0) >= 0.5
                and signals.get("rubric_gate_rubric_required_after_pass", 0.0) >= 0.5
                and signals.get("conservative_region_reward_pass", 0.0) >= 0.5
                and signals.get("conservative_region_outside_preservation_pass", 0.0) >= 0.5
                and signals.get("conservative_region_outside_change_pass", 0.0) >= 0.5
            )
            replacement_fallback_detector_score_effective_pass = (
                replacement_fallback_detector_score_pass
                or replacement_fallback_source_score_override
            )
            replacement_fallback_forbidden_gate_pass = (
                signals.get("rubric_gate_rubric_forbidden_gate_pass", 0.0) >= 0.5
            )
            replacement_fallback_forbidden_gate_override = (
                replacement_fallback_source_score_override
                and replacement_reject_reason == "rubric_forbidden_gate"
            )
            replacement_fallback_forbidden_gate_effective_pass = (
                replacement_fallback_forbidden_gate_pass
                or replacement_fallback_forbidden_gate_override
            )
            replacement_detector_fallback = (
                self.internal_vlm_judge_replacement_detector_fallback
                and edit_type == "object_replacement"
                and not row.get("feasible")
                and replacement_reason_allowed
                and signals.get("object_detector_supported", 0.0) >= 0.5
                and signals.get("object_detector_source_detected_pass", 0.0) >= 0.5
                and signals.get("object_detector_source_absent_pass", 0.0) < 0.5
                and signals.get("object_detector_target_present_pass", 0.0) >= 0.5
                and signals.get("rubric_gate_rubric_required_after_pass", 0.0) >= 0.5
                and replacement_fallback_forbidden_gate_effective_pass
                and signals.get("conservative_region_reward_pass", 0.0) >= 0.5
                and signals.get("conservative_region_outside_preservation_pass", 0.0) >= 0.5
                and signals.get("conservative_region_outside_change_pass", 0.0) >= 0.5
                and replacement_fallback_detector_score_effective_pass
                and replacement_fallback_localization_pass
                and replacement_fallback_outside_preservation_pass
                and replacement_fallback_semantic_contract_pass
                and (
                    not self.internal_vlm_judge_replacement_fallback_require_best
                    or candidate_index == best_candidate_index
                )
            )
            signals["internal_vlm_judge_replacement_detector_fallback"] = (
                1.0 if replacement_detector_fallback else 0.0
            )
            signals["internal_vlm_judge_replacement_fallback_require_best"] = (
                1.0 if self.internal_vlm_judge_replacement_fallback_require_best else 0.0
            )
            signals["internal_vlm_judge_replacement_fallback_detector_score_pass"] = (
                1.0 if replacement_fallback_detector_score_pass else 0.0
            )
            signals["internal_vlm_judge_replacement_fallback_detector_score_effective_pass"] = (
                1.0 if replacement_fallback_detector_score_effective_pass else 0.0
            )
            signals["internal_vlm_judge_replacement_fallback_vlm_contract_override"] = (
                1.0 if replacement_fallback_source_score_override else 0.0
            )
            signals["internal_vlm_judge_replacement_fallback_allow_vlm_contract_override"] = (
                1.0 if self.internal_vlm_judge_replacement_fallback_allow_vlm_contract_override else 0.0
            )
            signals["internal_vlm_judge_replacement_fallback_reason_allowed"] = (
                1.0 if replacement_reason_allowed else 0.0
            )
            signals["internal_vlm_judge_replacement_fallback_forbidden_gate_effective_pass"] = (
                1.0 if replacement_fallback_forbidden_gate_effective_pass else 0.0
            )
            signals["internal_vlm_judge_replacement_fallback_forbidden_gate_override"] = (
                1.0 if replacement_fallback_forbidden_gate_override else 0.0
            )
            signals["internal_vlm_judge_replacement_fallback_allowed_reject_reasons"] = sorted(
                self.internal_vlm_judge_replacement_fallback_allowed_reject_reasons
            )
            signals["internal_vlm_judge_replacement_fallback_detector_edited_source_threshold"] = (
                replacement_detector_edited_source_threshold
            )
            signals["internal_vlm_judge_replacement_fallback_max_edited_source_score_ratio"] = (
                self.internal_vlm_judge_replacement_fallback_max_edited_source_score_ratio
            )
            signals["internal_vlm_judge_replacement_fallback_min_source_score_drop"] = (
                self.internal_vlm_judge_replacement_fallback_min_source_score_drop
            )
            signals["internal_vlm_judge_replacement_fallback_localization_pass"] = (
                1.0 if replacement_fallback_localization_pass else 0.0
            )
            signals["internal_vlm_judge_replacement_fallback_min_localization_precision"] = (
                self.internal_vlm_judge_replacement_fallback_min_localization_precision
            )
            signals["internal_vlm_judge_replacement_fallback_outside_preservation_pass"] = (
                1.0 if replacement_fallback_outside_preservation_pass else 0.0
            )
            signals["internal_vlm_judge_replacement_fallback_min_outside_preservation"] = (
                self.internal_vlm_judge_replacement_fallback_min_outside_preservation
            )
            signals["internal_vlm_judge_replacement_fallback_min_target_object_visible_after"] = (
                self.internal_vlm_judge_replacement_fallback_min_target_object_visible_after
            )
            if replacement_detector_fallback:
                row["feasible"] = True
                row["reward"] = combined_raw_reward
                component_scores["object_detector_contract_effective"] = 1.0
                signals["rubric_reject_reason"] = "accepted_vlm_replacement_detector_fallback"
                signals["rubric_gate_object_detector_contract_pass"] = 1.0
                signals["object_detector_gate_pass_effective"] = 1.0
            detector_edited_source_threshold = max(
                detector_original_source_score
                * self.internal_vlm_judge_removal_fallback_max_edited_source_score_ratio,
                detector_original_source_score
                + self.internal_vlm_judge_removal_fallback_max_edited_source_score_margin,
            )
            removal_fallback_detector_score_pass = (
                detector_edited_source_score <= detector_edited_source_threshold
            )
            removal_detector_fallback = (
                self.internal_vlm_judge_removal_detector_fallback
                and edit_type == "object_removal"
                and not row.get("feasible")
                and str(signals.get("rubric_reject_reason", "")) == "object_detector_contract"
                and signals.get("object_detector_supported", 0.0) >= 0.5
                and signals.get("object_detector_source_detected_pass", 0.0) >= 0.5
                and signals.get("object_detector_source_absent_pass", 0.0) < 0.5
                and signals.get("rubric_gate_rubric_required_after_pass", 0.0) >= 0.5
                and signals.get("rubric_gate_rubric_forbidden_gate_pass", 0.0) >= 0.5
                and signals.get("rubric_gate_conservative_region_pass", 0.0) >= 0.5
                and removal_fallback_detector_score_pass
                and object_removal_gate_pass
                and judge_reliable
                and judge_pass
                and combined_reward_pass
                and judge_semantic >= self.internal_vlm_judge_removal_fallback_min_semantic
                and judge_object_absence >= self.internal_vlm_judge_removal_fallback_min_object_absence
                and judge_fill_naturalness >= self.internal_vlm_judge_removal_fallback_min_fill_naturalness
                and judge_preservation >= self.internal_vlm_judge_removal_fallback_min_preservation
                and judge_artifact_free >= self.internal_vlm_judge_removal_fallback_min_artifact_free
                and (
                    not self.internal_vlm_judge_removal_fallback_require_best
                    or candidate_index == best_candidate_index
                )
            )
            removal_fallback_outside_preservation = _finite_float(
                component_scores.get("conservative_outside_preservation"),
                0.0,
            )
            removal_fallback_outside_changed_fraction = _finite_float(
                component_scores.get("conservative_outside_changed_fraction_score"),
                0.0,
            )
            removal_fallback_localization = _finite_float(
                component_scores.get("conservative_localization_precision"),
                0.0,
            )
            removal_judge_fallback = (
                self.internal_vlm_judge_removal_judge_fallback
                and edit_type == "object_removal"
                and not row.get("feasible")
                and str(signals.get("rubric_reject_reason", ""))
                in {"object_detector_contract", "conservative_region"}
                and signals.get("rubric_gate_rubric_required_after_pass", 0.0) >= 0.5
                and signals.get("rubric_gate_rubric_forbidden_gate_pass", 0.0) >= 0.5
                and removal_fallback_detector_score_pass
                and object_removal_gate_pass
                and judge_reliable
                and judge_pass
                and combined_reward_pass
                and judge_semantic >= self.internal_vlm_judge_removal_fallback_min_semantic
                and judge_object_absence >= self.internal_vlm_judge_removal_fallback_min_object_absence
                and judge_fill_naturalness >= self.internal_vlm_judge_removal_fallback_min_fill_naturalness
                and judge_preservation >= self.internal_vlm_judge_removal_fallback_min_preservation
                and judge_artifact_free >= self.internal_vlm_judge_removal_fallback_min_artifact_free
                and removal_fallback_outside_preservation
                >= self.internal_vlm_judge_removal_judge_fallback_min_outside_preservation
                and removal_fallback_localization
                >= self.internal_vlm_judge_removal_judge_fallback_min_localization
                and removal_fallback_outside_changed_fraction
                >= self.internal_vlm_judge_removal_judge_fallback_min_outside_changed_fraction
                and (
                    not self.internal_vlm_judge_removal_judge_fallback_require_best
                    or candidate_index == best_candidate_index
                )
            )
            removal_structured_absence_fallback = (
                self.internal_vlm_judge_removal_structured_absence_fallback
                and edit_type == "object_removal"
                and not row.get("feasible")
                and str(signals.get("rubric_reject_reason", ""))
                in {"object_detector_contract", "conservative_region", "internal_vlm_judge_hard_fail"}
                and signals.get("rubric_gate_rubric_required_after_pass", 0.0) >= 0.5
                and signals.get("rubric_gate_rubric_forbidden_gate_pass", 0.0) >= 0.5
                and removal_fallback_detector_score_pass
                and object_removal_gate_pass
                and judge_reliable
                and judge_source_visible_before >= 0.5
                and judge_object_visible_after <= 0.2
                and judge_object_absence >= self.internal_vlm_judge_removal_fallback_min_object_absence
                and judge_fill_naturalness
                >= self.internal_vlm_judge_removal_structured_fallback_min_fill_naturalness
                and judge_preservation >= self.internal_vlm_judge_removal_fallback_min_preservation
                and judge_artifact_free >= self.internal_vlm_judge_removal_fallback_min_artifact_free
                and pre_judge_raw_reward
                >= self.internal_vlm_judge_removal_structured_fallback_min_raw_reward
                and combined_reward_pass
                and removal_fallback_localization
                >= self.internal_vlm_judge_removal_structured_fallback_min_localization
                and removal_fallback_outside_preservation
                >= self.internal_vlm_judge_removal_structured_fallback_min_outside_preservation
                and removal_fallback_outside_changed_fraction
                >= self.internal_vlm_judge_removal_structured_fallback_min_outside_changed_fraction
                and (
                    not self.internal_vlm_judge_removal_structured_fallback_require_best
                    or candidate_index == best_candidate_index
                )
            )
            subject_extraction_reject_reason = str(signals.get("rubric_reject_reason", ""))
            subject_extraction_reason_allowed = (
                not self.internal_vlm_judge_subject_extraction_fallback_allowed_reject_reasons
                or subject_extraction_reject_reason
                in self.internal_vlm_judge_subject_extraction_fallback_allowed_reject_reasons
            )
            subject_extraction_required_after = _finite_float(
                component_scores.get("rubric_required_after"),
                0.0,
            )
            subject_extraction_fallback = (
                self.internal_vlm_judge_subject_extraction_fallback
                and edit_type == "subject_extraction"
                and not row.get("feasible")
                and subject_extraction_reason_allowed
                and judge_reliable
                and judge_pass
                and combined_reward_pass
                and pre_judge_raw_reward
                >= self.internal_vlm_judge_subject_extraction_fallback_min_raw_reward
                and judge_score
                >= self.internal_vlm_judge_subject_extraction_fallback_min_score
                and judge_semantic
                >= self.internal_vlm_judge_subject_extraction_fallback_min_semantic
                and judge_preservation
                >= self.internal_vlm_judge_subject_extraction_fallback_min_preservation
                and judge_artifact_free
                >= self.internal_vlm_judge_subject_extraction_fallback_min_artifact_free
                and subject_extraction_required_after
                >= self.internal_vlm_judge_subject_extraction_fallback_min_required_after
                and (
                    not self.internal_vlm_judge_subject_extraction_fallback_require_best
                    or candidate_index == best_candidate_index
                )
            )
            signals["internal_vlm_judge_removal_detector_fallback"] = (
                1.0 if removal_detector_fallback else 0.0
            )
            signals["internal_vlm_judge_removal_judge_fallback"] = (
                1.0 if removal_judge_fallback else 0.0
            )
            signals["internal_vlm_judge_removal_structured_absence_fallback"] = (
                1.0 if removal_structured_absence_fallback else 0.0
            )
            signals["internal_vlm_judge_removal_fallback_detector_score_pass"] = (
                1.0 if removal_fallback_detector_score_pass else 0.0
            )
            signals["internal_vlm_judge_removal_structured_fallback_localization_pass"] = (
                1.0
                if removal_fallback_localization
                >= self.internal_vlm_judge_removal_structured_fallback_min_localization
                else 0.0
            )
            signals["internal_vlm_judge_removal_judge_fallback_outside_preservation_pass"] = (
                1.0
                if removal_fallback_outside_preservation
                >= self.internal_vlm_judge_removal_judge_fallback_min_outside_preservation
                else 0.0
            )
            signals["internal_vlm_judge_removal_judge_fallback_localization_pass"] = (
                1.0
                if removal_fallback_localization
                >= self.internal_vlm_judge_removal_judge_fallback_min_localization
                else 0.0
            )
            signals["internal_vlm_judge_removal_judge_fallback_outside_changed_fraction_pass"] = (
                1.0
                if removal_fallback_outside_changed_fraction
                >= self.internal_vlm_judge_removal_judge_fallback_min_outside_changed_fraction
                else 0.0
            )
            signals["internal_vlm_judge_removal_judge_fallback_require_best"] = (
                1.0 if self.internal_vlm_judge_removal_judge_fallback_require_best else 0.0
            )
            signals["internal_vlm_judge_removal_judge_fallback_min_outside_preservation"] = (
                self.internal_vlm_judge_removal_judge_fallback_min_outside_preservation
            )
            signals["internal_vlm_judge_removal_judge_fallback_min_localization"] = (
                self.internal_vlm_judge_removal_judge_fallback_min_localization
            )
            signals["internal_vlm_judge_removal_judge_fallback_min_outside_changed_fraction"] = (
                self.internal_vlm_judge_removal_judge_fallback_min_outside_changed_fraction
            )
            signals["internal_vlm_judge_removal_structured_fallback_require_best"] = (
                1.0 if self.internal_vlm_judge_removal_structured_fallback_require_best else 0.0
            )
            signals["internal_vlm_judge_removal_structured_fallback_min_raw_reward"] = (
                self.internal_vlm_judge_removal_structured_fallback_min_raw_reward
            )
            signals["internal_vlm_judge_removal_structured_fallback_min_localization"] = (
                self.internal_vlm_judge_removal_structured_fallback_min_localization
            )
            signals["internal_vlm_judge_removal_structured_fallback_min_outside_preservation"] = (
                self.internal_vlm_judge_removal_structured_fallback_min_outside_preservation
            )
            signals["internal_vlm_judge_removal_structured_fallback_min_outside_changed_fraction"] = (
                self.internal_vlm_judge_removal_structured_fallback_min_outside_changed_fraction
            )
            signals["internal_vlm_judge_removal_structured_fallback_min_fill_naturalness"] = (
                self.internal_vlm_judge_removal_structured_fallback_min_fill_naturalness
            )
            signals["internal_vlm_judge_removal_fallback_detector_edited_source_threshold"] = (
                detector_edited_source_threshold
            )
            signals["internal_vlm_judge_removal_fallback_max_edited_source_score_ratio"] = (
                self.internal_vlm_judge_removal_fallback_max_edited_source_score_ratio
            )
            signals["internal_vlm_judge_removal_fallback_max_edited_source_score_margin"] = (
                self.internal_vlm_judge_removal_fallback_max_edited_source_score_margin
            )
            signals["internal_vlm_judge_removal_fallback_require_best"] = (
                1.0 if self.internal_vlm_judge_removal_fallback_require_best else 0.0
            )
            signals["internal_vlm_judge_subject_extraction_fallback"] = (
                1.0 if subject_extraction_fallback else 0.0
            )
            signals["internal_vlm_judge_subject_extraction_fallback_reason_allowed"] = (
                1.0 if subject_extraction_reason_allowed else 0.0
            )
            signals["internal_vlm_judge_subject_extraction_fallback_require_best"] = (
                1.0 if self.internal_vlm_judge_subject_extraction_fallback_require_best else 0.0
            )
            signals["internal_vlm_judge_subject_extraction_fallback_min_score"] = (
                self.internal_vlm_judge_subject_extraction_fallback_min_score
            )
            signals["internal_vlm_judge_subject_extraction_fallback_min_semantic"] = (
                self.internal_vlm_judge_subject_extraction_fallback_min_semantic
            )
            signals["internal_vlm_judge_subject_extraction_fallback_min_preservation"] = (
                self.internal_vlm_judge_subject_extraction_fallback_min_preservation
            )
            signals["internal_vlm_judge_subject_extraction_fallback_min_artifact_free"] = (
                self.internal_vlm_judge_subject_extraction_fallback_min_artifact_free
            )
            signals["internal_vlm_judge_subject_extraction_fallback_min_raw_reward"] = (
                self.internal_vlm_judge_subject_extraction_fallback_min_raw_reward
            )
            signals["internal_vlm_judge_subject_extraction_fallback_min_required_after"] = (
                self.internal_vlm_judge_subject_extraction_fallback_min_required_after
            )
            signals["internal_vlm_judge_subject_extraction_fallback_required_after_pass"] = (
                1.0
                if subject_extraction_required_after
                >= self.internal_vlm_judge_subject_extraction_fallback_min_required_after
                else 0.0
            )
            if removal_detector_fallback:
                row["feasible"] = True
                row["reward"] = combined_raw_reward
                component_scores["object_detector_contract_effective"] = 1.0
                signals["rubric_reject_reason"] = "accepted_vlm_removal_detector_fallback"
                signals["rubric_gate_object_detector_contract_pass"] = 1.0
                signals["object_detector_gate_pass_effective"] = 1.0
            if removal_judge_fallback:
                row["feasible"] = True
                row["reward"] = combined_raw_reward
                component_scores["object_detector_contract_effective"] = 1.0
                component_scores["conservative_region_effective"] = 1.0
                signals["rubric_reject_reason"] = "accepted_vlm_removal_judge_fallback"
                signals["rubric_gate_object_detector_contract_pass"] = 1.0
                signals["rubric_gate_conservative_region_pass"] = 1.0
                signals["object_detector_gate_pass_effective"] = 1.0
                signals["conservative_region_gate_pass_effective"] = 1.0
            if removal_structured_absence_fallback:
                judge_hard_fail = False
                row["feasible"] = True
                row["reward"] = combined_raw_reward
                component_scores["object_detector_contract_effective"] = 1.0
                component_scores["conservative_region_effective"] = 1.0
                signals["rubric_reject_reason"] = "accepted_vlm_removal_structured_absence_fallback"
                signals["rubric_gate_object_detector_contract_pass"] = 1.0
                signals["rubric_gate_conservative_region_pass"] = 1.0
                signals["object_detector_gate_pass_effective"] = 1.0
                signals["conservative_region_gate_pass_effective"] = 1.0
                signals["internal_vlm_judge_hard_fail_effective"] = 0.0
            if subject_extraction_fallback:
                judge_hard_fail = False
                row["feasible"] = True
                row["reward"] = combined_raw_reward
                component_scores["subject_extraction_contract_effective"] = 1.0
                component_scores["subject_extraction_validity_effective"] = max(
                    _finite_float(component_scores.get("cepr_validity"), 0.0),
                    judge_artifact_free,
                )
                component_scores["rubric_forbidden_gate_effective"] = 1.0
                component_scores["cepr_validity_effective"] = component_scores[
                    "subject_extraction_validity_effective"
                ]
                signals["rubric_reject_reason"] = "accepted_vlm_subject_extraction_fallback"
                signals["rubric_gate_rubric_forbidden_gate_pass"] = 1.0
                signals["rubric_gate_cepr_validity_pass"] = 1.0
                signals["rubric_forbidden_gate_effective_pass"] = 1.0
                signals["cepr_validity_effective_pass"] = 1.0
                signals["internal_vlm_judge_hard_fail_effective"] = 0.0
            action_compose_reject_reason = str(signals.get("rubric_reject_reason", ""))
            action_compose_reason_allowed = (
                not self.internal_vlm_judge_action_compose_fallback_allowed_reject_reasons
                or action_compose_reject_reason
                in self.internal_vlm_judge_action_compose_fallback_allowed_reject_reasons
            )
            action_compose_edit_success = _finite_float(score.get("edit_success"), 0.0)
            action_compose_instruction_following = _finite_float(
                score.get("instruction_following"), 0.0
            )
            action_compose_overall = _finite_float(score.get("overall"), 0.0)
            action_compose_outside_preservation = _finite_float(
                component_scores.get("conservative_outside_preservation"), 0.0
            )
            action_compose_preservation_gates_pass = (
                _finite_float(signals.get("rubric_gate_rubric_preservation_pass"), 0.0) >= 0.5
                and _finite_float(signals.get("rubric_gate_cepr_preservation_pass"), 0.0) >= 0.5
                and _finite_float(signals.get("rubric_gate_cepr_validity_pass"), 0.0) >= 0.5
                and _finite_float(signals.get("rubric_gate_rubric_required_after_pass"), 0.0) >= 0.5
                and _finite_float(signals.get("rubric_gate_conservative_region_pass"), 0.0) >= 0.5
                and action_compose_outside_preservation
                >= self.internal_vlm_judge_action_compose_fallback_min_outside_preservation
            )
            action_compose_judge_fallback = (
                self.internal_vlm_judge_action_compose_fallback
                and edit_type in self.internal_vlm_judge_action_compose_fallback_edit_types
                and not row.get("feasible")
                and action_compose_reason_allowed
                and judge_reliable
                and not missing_fallback
                and combined_reward_pass
                and action_compose_edit_success
                >= self.internal_vlm_judge_action_compose_fallback_min_edit_success
                and action_compose_instruction_following
                >= self.internal_vlm_judge_action_compose_fallback_min_instruction_following
                and action_compose_overall
                >= self.internal_vlm_judge_action_compose_fallback_min_overall
                and judge_preservation
                >= self.internal_vlm_judge_action_compose_fallback_min_preservation
                and judge_artifact_free
                >= self.internal_vlm_judge_action_compose_fallback_min_artifact_free
                and action_compose_preservation_gates_pass
                and (
                    not self.internal_vlm_judge_action_compose_fallback_require_best
                    or candidate_index == best_candidate_index
                )
            )
            signals["internal_vlm_judge_action_compose_fallback"] = (
                1.0 if action_compose_judge_fallback else 0.0
            )
            signals["internal_vlm_judge_action_compose_fallback_reason_allowed"] = (
                1.0 if action_compose_reason_allowed else 0.0
            )
            signals["internal_vlm_judge_action_compose_fallback_preservation_pass"] = (
                1.0 if action_compose_preservation_gates_pass else 0.0
            )
            signals["internal_vlm_judge_action_compose_fallback_edit_success"] = (
                action_compose_edit_success
            )
            signals["internal_vlm_judge_action_compose_fallback_overall"] = action_compose_overall
            signals["internal_vlm_judge_action_compose_fallback_require_best"] = (
                1.0 if self.internal_vlm_judge_action_compose_fallback_require_best else 0.0
            )
            signals["internal_vlm_judge_action_compose_fallback_min_edit_success"] = (
                self.internal_vlm_judge_action_compose_fallback_min_edit_success
            )
            signals["internal_vlm_judge_action_compose_fallback_min_overall"] = (
                self.internal_vlm_judge_action_compose_fallback_min_overall
            )
            signals["internal_vlm_judge_action_compose_fallback_min_outside_preservation"] = (
                self.internal_vlm_judge_action_compose_fallback_min_outside_preservation
            )
            if action_compose_judge_fallback:
                judge_hard_fail = False
                row["feasible"] = True
                row["reward"] = combined_raw_reward
                component_scores["rubric_forbidden_gate_effective"] = 1.0
                signals["rubric_reject_reason"] = "accepted_vlm_action_compose_fallback"
                signals["rubric_gate_rubric_forbidden_gate_pass"] = 1.0
                signals["rubric_forbidden_gate_effective_pass"] = 1.0
                signals["internal_vlm_judge_hard_fail_effective"] = 0.0
            if (
                row.get("feasible")
                and combined_reward_pass
                and not judge_hard_fail
            ):
                row["reward"] = combined_raw_reward
            elif row.get("feasible"):
                row["feasible"] = False
                row["reward"] = 0.0
                if judge_hard_fail:
                    signals["rubric_reject_reason"] = "internal_vlm_judge_hard_fail"
                else:
                    signals["rubric_reject_reason"] = "internal_vlm_judge_reward"
                signals["rubric_gate_internal_vlm_judge_pass"] = 0.0
                signals["rubric_gate_internal_vlm_judge_reward_pass"] = 0.0 if not combined_reward_pass else 1.0
            else:
                row["reward"] = 0.0
            if row.get("feasible"):
                signals["rubric_gate_internal_vlm_judge_pass"] = 1.0 if judge_pass else 0.0
                signals["rubric_gate_internal_vlm_judge_reward_pass"] = 1.0

    @staticmethod
    def _scope_rubric_prompt(text: str, target_region: str, mode: str) -> str:
        text = str(text).strip()
        target_region = str(target_region or "").strip()
        if not text or not target_region or target_region == "main visible target":
            return text
        lowered = text.lower()
        region_lowered = target_region.lower()
        if region_lowered in lowered:
            return text

        words = lowered.split()
        relation_terms = {
            "visible",
            "present",
            "contains",
            "contain",
            "added",
            "removed",
            "replaced",
            "changed",
            "background",
            "foreground",
            "left",
            "right",
            "above",
            "below",
            "behind",
            "front",
        }
        has_relation = any(term in lowered for term in relation_terms)
        has_auxiliary = any(token in words for token in {"is", "are", "has", "have", "still", "remains", "remain"})
        if len(words) <= 3 and not has_relation and not has_auxiliary:
            return f"{target_region} is {text}"
        if mode == "forbidden":
            return f"{text} in {target_region}"
        return f"{text} in {target_region}"

    @staticmethod
    def _is_generic_forbidden_prompt(text: str) -> bool:
        lowered = str(text).strip().lower()
        if not lowered:
            return True
        generic_markers = (
            "no other",
            "not other",
            "any other",
            "any object that changes",
            "changes the overall",
            "overall layout",
            "other objects",
            "other background",
            "unrelated objects",
            "extra objects",
            "additional objects",
        )
        return any(marker in lowered for marker in generic_markers)

    @staticmethod
    def _is_placeholder_source_descriptor(text: Any) -> bool:
        lowered = str(text or "").strip().lower()
        if not lowered:
            return True
        placeholders = {
            "old",
            "original",
            "current",
            "previous",
            "existing",
            "source",
            "old color",
            "original color",
            "current color",
            "previous color",
            "existing color",
            "source color",
            "old appearance",
            "original appearance",
            "current appearance",
            "previous appearance",
            "existing appearance",
        }
        return lowered in placeholders

    @staticmethod
    def _rubric_content_terms(text: Any) -> set[str]:
        stopwords = {
            "a",
            "an",
            "the",
            "no",
            "not",
            "without",
            "any",
            "other",
            "than",
            "is",
            "are",
            "has",
            "have",
            "had",
            "be",
            "been",
            "being",
            "in",
            "on",
            "at",
            "to",
            "of",
            "for",
            "with",
            "from",
            "near",
            "visible",
            "remains",
            "remain",
            "still",
            "added",
            "add",
            "new",
            "small",
            "large",
        }
        terms = set()
        for token in re.findall(r"[a-z0-9]+", str(text or "").lower()):
            if token in stopwords:
                continue
            if len(token) > 3 and token.endswith("s"):
                token = token[:-1]
            terms.add(token)
        return terms

    def _is_contradictory_forbidden_prompt(self, text: str, spec: dict[str, Any]) -> bool:
        edit_type = str(spec.get("edit_type", ""))
        if edit_type not in {
            "object_addition",
            "attribute_change",
            "color_change",
            "material_change",
            "style_transfer",
            "background_change",
        }:
            return False
        lowered = str(text or "").lower()
        if not any(marker in lowered for marker in ("no ", "not ", "without ", "absent", "missing")):
            return False
        required_terms: set[str] = set()
        for prompt in spec.get("required_after", []):
            required_terms.update(self._rubric_content_terms(prompt))
        for key in ("target_object", "replacement", "target_attribute", "target_material", "target_style"):
            required_terms.update(self._rubric_content_terms(spec.get(key)))
        forbidden_terms = self._rubric_content_terms(text)
        return bool(required_terms and forbidden_terms and required_terms.intersection(forbidden_terms))

    def _prompt_support_cached(
        self,
        pipe: Any,
        prompt: str,
        image_label: str,
        image: Image.Image,
        cache: dict[tuple[str, str], Any],
    ) -> float:
        prompt = polish_prompt(prompt, use_prompt_polish=False, image_context=image)
        text_feature = self._cached_text_feature(pipe, prompt, cache)
        image_feature = self._cached_understanding_feature(pipe, prompt, image_label, image, cache)
        return _clamp(0.5 * (1.0 + self._cosine_similarity(image_feature, text_feature)))

    def _feature_preservation_score(
        self,
        pipe: Any,
        prompt: str,
        original: Image.Image,
        edited: Image.Image,
        candidate_index: int,
        cache: dict[tuple[str, str], Any],
    ) -> float:
        prompt = polish_prompt(prompt, use_prompt_polish=False, image_context=original)
        original_feature = self._cached_understanding_feature(pipe, prompt, "original", original, cache)
        edited_feature = self._cached_understanding_feature(
            pipe,
            prompt,
            f"candidate:{candidate_index}",
            edited,
            cache,
        )
        cosine_01 = _clamp(0.5 * (1.0 + self._cosine_similarity(original_feature, edited_feature)))
        return _clamp(math.exp(-(1.0 - cosine_01) / max(self.rubric_preservation_temperature, 1e-6)))

    def _rubric_prompts(self, proposal: EditProposal) -> dict[str, Any]:
        spec = normalize_structured_edit(
            proposal.structured_edit,
            instruction=proposal.instruction,
            family=proposal.definition.family,
        )
        edit_type = str(spec.get("edit_type", "local_enhancement"))
        source_object = spec.get("source_object") or spec.get("target")
        target_region = spec.get("target_region", "main visible target")
        source_required_types = {
            "object_replacement",
            "object_removal",
            "spatial_move",
            "attribute_change",
            "color_change",
            "material_change",
        }
        source_prompts = []
        if source_object and edit_type in source_required_types:
            source_prompts.append(f"{source_object} is visible in {target_region}")
        required_prompts = self._unique_texts(
            [
                self._scope_rubric_prompt(prompt, target_region, "required")
                for prompt in spec.get("required_after", [])
            ],
            self.max_rubric_items,
        )
        source_state_fields = {
            "attribute_change": ("source_attribute",),
            "color_change": ("source_attribute",),
            "material_change": ("source_material",),
            "style_transfer": ("source_style",),
        }
        old_state_is_explicit = any(
            spec.get(field) and not self._is_placeholder_source_descriptor(spec.get(field))
            for field in source_state_fields.get(edit_type, ())
        )
        use_forbidden_prompts = edit_type not in source_state_fields or old_state_is_explicit
        if edit_type == "object_addition":
            use_forbidden_prompts = False
        forbidden_prompts = self._unique_texts(
            [
                self._scope_rubric_prompt(prompt, target_region, "forbidden")
                for prompt in (spec.get("forbidden_after", []) if use_forbidden_prompts else [])
                if not self._is_generic_forbidden_prompt(prompt)
                and not self._is_contradictory_forbidden_prompt(prompt, spec)
            ],
            self.max_rubric_items,
        )
        preserve_prompts = self._unique_texts(
            [f"{item} is preserved" for item in spec.get("preserve", [])],
            self.max_rubric_items,
        )
        return {
            "spec": spec,
            "source_prompts": self._unique_texts(source_prompts, self.max_rubric_items),
            "required_prompts": required_prompts,
            "forbidden_prompts": forbidden_prompts,
            "preserve_prompts": preserve_prompts,
        }

    def _source_grounding_score(
        self,
        pipe: Any,
        prompts: list[str],
        original: Image.Image,
        cache: dict[tuple[str, str], Any],
    ) -> tuple[float, dict[str, float]]:
        if not prompts:
            return 1.0, {
                "rubric_source_grounding_score": 1.0,
                "rubric_source_grounding_supported": 0.0,
            }
        scores = []
        supports = []
        for prompt in prompts:
            support = self._prompt_support_cached(pipe, prompt, "original", original, cache)
            supports.append(support)
            scores.append(
                _sigmoid((support - self.rubric_support_center) / max(self.rubric_abs_temperature, 1e-6))
            )
        score = self._geometric_mean(scores)
        return score, {
            "rubric_source_grounding_score": score,
            "rubric_source_grounding_supported": 1.0,
            "rubric_source_grounding_support": statistics.mean(supports) if supports else 0.0,
            "rubric_source_grounding_prompt_count": float(len(prompts)),
        }

    def _required_after_score(
        self,
        pipe: Any,
        prompts: list[str],
        original: Image.Image,
        edited: Image.Image,
        candidate_index: int,
        cache: dict[tuple[str, str], Any],
    ) -> tuple[float, dict[str, float]]:
        if not prompts:
            return 1.0, {
                "rubric_required_after_score": 1.0,
                "rubric_required_after_supported": 0.0,
            }
        scores = []
        gains = []
        supports = []
        for prompt in prompts:
            gain = self._prompt_gain_cached(pipe, prompt, original, edited, candidate_index, cache)
            support = self._prompt_support_cached(pipe, prompt, f"candidate:{candidate_index}", edited, cache)
            gains.append(gain)
            supports.append(support)
            gain_score = _sigmoid((gain - self.min_required_gain) / max(self.rubric_gain_temperature, 1e-6))
            support_score = _sigmoid(
                (support - self.rubric_support_center) / max(self.rubric_abs_temperature, 1e-6)
            )
            scores.append(math.sqrt(max(gain_score * support_score, 0.0)))
        score = self._geometric_mean(scores)
        return score, {
            "rubric_required_after_score": score,
            "rubric_required_after_supported": 1.0,
            "rubric_required_after_gain": statistics.mean(gains) if gains else 0.0,
            "rubric_required_after_support": statistics.mean(supports) if supports else 0.0,
            "rubric_required_after_prompt_count": float(len(prompts)),
        }

    def _forbidden_after_absent_score(
        self,
        pipe: Any,
        prompts: list[str],
        original: Image.Image,
        edited: Image.Image,
        candidate_index: int,
        cache: dict[tuple[str, str], Any],
    ) -> tuple[float, dict[str, float]]:
        if not prompts:
            return 1.0, {
                "rubric_forbidden_after_absent_score": 1.0,
                "rubric_forbidden_after_supported": 0.0,
            }
        scores = []
        drops = []
        edited_supports = []
        for prompt in prompts:
            original_support = self._prompt_support_cached(pipe, prompt, "original", original, cache)
            edited_support = self._prompt_support_cached(pipe, prompt, f"candidate:{candidate_index}", edited, cache)
            gain = self._prompt_gain_cached(pipe, prompt, original, edited, candidate_index, cache)
            drop = max(-gain, original_support - edited_support)
            drops.append(drop)
            edited_supports.append(edited_support)
            drop_score = _sigmoid((drop - self.min_forbidden_drop) / max(self.rubric_gain_temperature, 1e-6))
            absence_score = _sigmoid(
                (self.rubric_forbidden_absence_center - edited_support)
                / max(self.rubric_abs_temperature, 1e-6)
            )
            scores.append(math.sqrt(max(drop_score * absence_score, 0.0)))
        score = self._geometric_mean(scores)
        return score, {
            "rubric_forbidden_after_absent_score": score,
            "rubric_forbidden_after_supported": 1.0,
            "rubric_forbidden_after_drop": statistics.mean(drops) if drops else 0.0,
            "rubric_forbidden_after_edited_support": statistics.mean(edited_supports) if edited_supports else 0.0,
            "rubric_forbidden_after_prompt_count": float(len(prompts)),
        }

    def _rubric_preservation_score(
        self,
        pipe: Any,
        prompts: list[str],
        original: Image.Image,
        edited: Image.Image,
        candidate_index: int,
        cache: dict[tuple[str, str], Any],
    ) -> tuple[float, dict[str, float]]:
        if not prompts:
            return 1.0, {
                "rubric_preservation_score": 1.0,
                "rubric_preservation_supported": 0.0,
            }
        scores = [
            self._feature_preservation_score(pipe, prompt, original, edited, candidate_index, cache)
            for prompt in prompts
        ]
        score = self._geometric_mean(scores)
        return score, {
            "rubric_preservation_score": score,
            "rubric_preservation_supported": 1.0,
            "rubric_preservation_prompt_count": float(len(prompts)),
        }

    def _rubric_score(
        self,
        pipe: Any,
        proposal: EditProposal,
        original: Image.Image,
        edited: Image.Image,
        candidate_index: int,
        validity: float,
        cache: dict[tuple[str, str], Any],
    ) -> tuple[dict[str, float], dict[str, float]]:
        prompts = self._rubric_prompts(proposal)
        source_grounded, source_signals = self._source_grounding_score(
            pipe, prompts["source_prompts"], original, cache
        )
        required_after, required_signals = self._required_after_score(
            pipe, prompts["required_prompts"], original, edited, candidate_index, cache
        )
        forbidden_absent, forbidden_signals = self._forbidden_after_absent_score(
            pipe, prompts["forbidden_prompts"], original, edited, candidate_index, cache
        )
        explicit_preservation, preservation_signals = self._rubric_preservation_score(
            pipe, prompts["preserve_prompts"], original, edited, candidate_index, cache
        )
        edit_components = [required_after]
        if prompts["forbidden_prompts"]:
            edit_components.append(forbidden_absent)
        edit_success = self._geometric_mean(edit_components)
        reward = source_grounded * self._geometric_mean([edit_success, explicit_preservation, validity])
        scores = {
            "rubric_source_grounded": source_grounded,
            "rubric_required_after": required_after,
            "rubric_forbidden_after_absent": forbidden_absent,
            "rubric_edit_success": edit_success,
            "rubric_preservation": explicit_preservation,
            "rubric_validity": validity,
            "rubric_reward": _clamp(reward),
        }
        signals = {
            **source_signals,
            **required_signals,
            **forbidden_signals,
            **preservation_signals,
            "rubric_source_threshold": self.rubric_source_threshold,
            "rubric_required_threshold": self.rubric_required_threshold,
            "rubric_forbidden_threshold": self.rubric_forbidden_threshold,
            "rubric_preservation_threshold": self.rubric_preservation_threshold,
            "rubric_reward_threshold": self.rubric_reward_threshold,
        }
        return scores, signals

    @staticmethod
    def _raw_gate_value(signals: dict[str, Any], name: str) -> tuple[float, bool]:
        value = signals.get(name)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value), True
        return 0.0, False

    def _raw_semantic_gate_status(
        self,
        signals: dict[str, Any],
    ) -> tuple[dict[str, bool], dict[str, Any]]:
        """Evaluate strict-sign gates on unshifted semantic evidence.

        The forbidden/old-state gate is conditional because additions and edits
        without a meaningful old-state rubric have nothing legitimate to
        suppress. The other three signals are required for every rubric CEPR
        candidate; absent or non-finite evidence fails closed.
        """
        true_gain, true_gain_present = self._raw_gate_value(signals, "cepr_true_prompt_gain")
        contrastive_margin, contrastive_margin_present = self._raw_gate_value(
            signals, "cepr_contrastive_margin"
        )
        distractor_count, distractor_count_present = self._raw_gate_value(
            signals, "cepr_distractor_count"
        )
        required_gain, required_gain_present = self._raw_gate_value(
            signals, "rubric_required_after_gain"
        )
        required_supported, required_supported_present = self._raw_gate_value(
            signals, "rubric_required_after_supported"
        )
        old_state_suppression, old_state_suppression_present = self._raw_gate_value(
            signals, "rubric_forbidden_after_drop"
        )
        old_state_supported, old_state_supported_present = self._raw_gate_value(
            signals, "rubric_forbidden_after_supported"
        )

        distractor_evidence_supported = distractor_count_present and distractor_count > 0.0
        required_after_applicable = required_supported_present and required_supported >= 0.5
        old_state_applicability_known = old_state_supported_present
        old_state_applicable = old_state_supported_present and old_state_supported >= 0.5
        status = {
            "raw_true_prompt_gain": (
                true_gain_present and true_gain > self.min_raw_true_prompt_gain
            ),
            "raw_true_over_distractor_margin": (
                distractor_evidence_supported
                and contrastive_margin_present
                and contrastive_margin > self.min_raw_contrastive_margin
            ),
            "raw_required_after_gain": (
                required_after_applicable
                and required_gain_present
                and required_gain > self.min_raw_required_after_gain
            ),
            "raw_old_state_suppression": (
                old_state_applicability_known
                and (
                    not old_state_applicable
                    or (
                        old_state_suppression_present
                        and old_state_suppression > self.min_raw_old_state_suppression
                    )
                )
            ),
        }
        failure_reasons = [name for name, passed in status.items() if not passed]
        diagnostics: dict[str, Any] = {
            "raw_gate_true_prompt_gain_value": true_gain,
            "raw_gate_true_prompt_gain_value_present": 1.0 if true_gain_present else 0.0,
            "raw_gate_true_prompt_gain_threshold": self.min_raw_true_prompt_gain,
            "raw_gate_contrastive_margin_value": contrastive_margin,
            "raw_gate_contrastive_margin_value_present": 1.0 if contrastive_margin_present else 0.0,
            "raw_gate_contrastive_margin_threshold": self.min_raw_contrastive_margin,
            "raw_gate_distractor_count": distractor_count,
            "raw_gate_distractor_evidence_supported": (
                1.0 if distractor_evidence_supported else 0.0
            ),
            "raw_gate_required_after_gain_value": required_gain,
            "raw_gate_required_after_gain_value_present": 1.0 if required_gain_present else 0.0,
            "raw_gate_required_after_gain_threshold": self.min_raw_required_after_gain,
            "raw_gate_required_after_applicable": 1.0 if required_after_applicable else 0.0,
            "raw_gate_old_state_suppression_value": old_state_suppression,
            "raw_gate_old_state_suppression_value_present": (
                1.0 if old_state_suppression_present else 0.0
            ),
            "raw_gate_old_state_suppression_threshold": self.min_raw_old_state_suppression,
            "raw_gate_old_state_applicability_known": (
                1.0 if old_state_applicability_known else 0.0
            ),
            "raw_gate_old_state_suppression_applicable": 1.0 if old_state_applicable else 0.0,
            "raw_gate_all_pass": 1.0 if not failure_reasons else 0.0,
            "raw_gate_failure_reasons": failure_reasons,
            "raw_gate_primary_failure_reason": failure_reasons[0] if failure_reasons else "accepted",
        }
        diagnostics.update(
            {
                f"rubric_gate_{name}_pass": 1.0 if passed else 0.0
                for name, passed in status.items()
            }
        )
        return status, diagnostics

    def _apply_post_group_hard_gates(self, rows: list[dict[str, Any]]) -> None:
        """Prevent rank/judge fallbacks from reviving raw-semantic failures."""
        for row in rows:
            signals = row.setdefault("signals", {})
            status, diagnostics = self._raw_semantic_gate_status(signals)
            signals.update(diagnostics)
            failure_reasons = [name for name, passed in status.items() if not passed]
            if not failure_reasons:
                continue
            was_feasible = bool(row.get("feasible"))
            row["feasible"] = False
            row["reward"] = 0.0
            signals["raw_gate_post_judge_enforced"] = 1.0
            current_reason = str(signals.get("rubric_reject_reason", ""))
            if was_feasible or not current_reason or current_reason.startswith("accepted"):
                signals["rubric_reject_reason"] = failure_reasons[0]




    @staticmethod
    def _object_detector_phrase(text: Any) -> str:
        phrase = str(text or "").strip().lower()
        phrase = re.sub(r"\b(the|a|an)\b", " ", phrase)
        phrase = re.sub(r"\b(original|requested|source|target|same|visible|location|area|object|objects)\b", " ", phrase)
        phrase = re.sub(r"[^a-z0-9 ]+", " ", phrase)
        tokens = [token for token in re.sub(r"\s+", " ", phrase).strip().split(" ") if token]
        relation_tokens = {
            "above",
            "across",
            "around",
            "at",
            "attached",
            "behind",
            "below",
            "beneath",
            "beside",
            "between",
            "by",
            "in",
            "inside",
            "near",
            "next",
            "on",
            "over",
            "under",
            "with",
        }
        for index, token in enumerate(tokens):
            if index > 0 and token in relation_tokens:
                tokens = tokens[:index]
                break
        context_modifiers = {
            "background",
            "ceiling",
            "corner",
            "floor",
            "foreground",
            "ground",
            "hanging",
            "left",
            "lower",
            "mounted",
            "right",
            "road",
            "room",
            "scene",
            "shelf",
            "small",
            "street",
            "table",
            "tiny",
            "upper",
            "wall",
        }
        if len(tokens) > 1:
            filtered = [token for token in tokens if token not in context_modifiers]
            if filtered:
                tokens = filtered
        return " ".join(tokens)

    def _object_detector_phrase_is_suitable(self, phrase: str) -> bool:
        tokens = [token for token in str(phrase or "").split() if token]
        if not tokens:
            return False
        if any(token in self.object_detector_unsuitable_terms for token in tokens):
            return False
        joined = " ".join(tokens)
        if any(term and term in joined for term in self.object_detector_unsuitable_terms):
            return False
        return True

    def _object_detector_skip_result(self, reason: str) -> tuple[dict[str, float], dict[str, float]]:
        return {
            "object_detector_contract": 1.0,
        }, {
            "object_detector_supported": 0.0,
            "object_detector_gate_pass": 1.0,
            "object_detector_skipped": 1.0,
            "object_detector_skip_reason": reason,
        }

    def _detect_object_boxes(
        self,
        image: Image.Image,
        phrase: str,
        cache: dict[tuple[str, str], Any] | None = None,
        pipe: Any = None,
    ) -> list[dict[str, Any]]:
        phrase = self._object_detector_phrase(phrase)
        if not phrase:
            return []
        return self._internal_grounding_boxes(pipe, image, phrase, cache)

    def _detect_object_score(
        self,
        image: Image.Image,
        phrase: str,
        cache: dict[tuple[str, str], Any] | None = None,
        pipe: Any = None,
    ) -> float:
        return self._internal_grounding_presence(
            pipe, image, self._object_detector_phrase(phrase), cache
        )

    # ------------------------------------------------------------------
    # Internal Qwen-VL grounding. When object_grounder == "qwen_vl_internal"
    # the object presence/absence and target-region localization used by the
    # conservative-editing gates are produced by the already-loaded editor
    # text_encoder (Qwen2.5-VL) + processor. No extra weights are loaded; the reward path
    # stays fully internal to Qwen-Image-Edit.
    # ------------------------------------------------------------------
    def _internal_grounder_resize(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        max_side = max(64, self.internal_grounder_image_resolution)
        width, height = image.size
        scale = min(1.0, max_side / max(width, height))
        if scale >= 1.0:
            return image.copy()
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        return image.resize(new_size, Image.Resampling.BICUBIC)

    def _internal_grounder_generate(self, pipe: Any, image: Image.Image, prompt: str) -> str:
        if (
            pipe is None
            or getattr(pipe, "processor", None) is None
            or getattr(pipe, "text_encoder", None) is None
        ):
            raise ValueError("Internal grounder requires a pipeline with processor and text_encoder.")
        model = pipe.text_encoder
        processor = pipe.processor
        if not hasattr(model, "generate") or not hasattr(processor, "apply_chat_template"):
            raise ValueError("Internal grounder requires a generative Qwen text_encoder and chat processor.")
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Qwen-Image-Edit's internal visual grounding module. "
                    "Use only visual evidence in the image and answer in the exact format requested."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        device = getattr(model, "device", None)
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = getattr(pipe, "device", "cpu")
        inputs = processor(
            text=[text],
            images=[self._internal_grounder_resize(image)],
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        import torch

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max(8, self.internal_grounder_max_new_tokens),
                do_sample=False,
            )
        generated = output_ids[:, inputs["input_ids"].shape[1] :]
        return processor.batch_decode(generated, skip_special_tokens=True)[0]

    def _internal_grounding_presence(
        self,
        pipe: Any,
        image: Image.Image,
        phrase: str,
        cache: dict[tuple[str, str], Any] | None = None,
    ) -> float:
        phrase = (phrase or "").strip()
        if not phrase:
            return 0.0
        cache_key = (
            "internal_grounding_presence",
            f"{id(image)}:{image.size[0]}x{image.size[1]}:{phrase}",
        )
        if cache is not None and cache_key in cache:
            return float(cache[cache_key])
        prompt = (
            f"How clearly is a {phrase} visible in this image? "
            "Respond with only a single number between 0.0 and 1.0, where "
            f"1.0 means the {phrase} is clearly and prominently visible and 0.0 means "
            f"no {phrase} is present at all. Output only the number."
        )
        decoded = self._internal_grounder_generate(pipe, image, prompt)
        match = re.search(r"[-+]?\d*\.?\d+", decoded or "")
        value = float(match.group(0)) if match else 0.0
        if value > 1.0:
            value = value / 100.0 if value <= 100.0 else 1.0
        value = max(0.0, min(1.0, value))
        if cache is not None:
            cache[cache_key] = value
        return value

    def _internal_grounding_boxes(
        self,
        pipe: Any,
        image: Image.Image,
        phrase: str,
        cache: dict[tuple[str, str], Any] | None = None,
    ) -> list[dict[str, Any]]:
        phrase = (phrase or "").strip()
        if not phrase:
            return []
        cache_key = (
            "internal_grounding_boxes",
            f"{id(image)}:{image.size[0]}x{image.size[1]}:{phrase}",
        )
        if cache is not None and cache_key in cache:
            return list(cache[cache_key])
        prompt = (
            f"Locate the {phrase} in this image. Respond with only compact JSON of the form "
            '{"present": true_or_false, "box": [x0, y0, x1, y1], "confidence": value}. '
            "Use integer coordinates in a 0-1000 normalized frame (top-left origin, x to the "
            f"right, y downward) relative to image width and height. If no {phrase} is visible, "
            "set present to false and box to an empty list. confidence is between 0.0 and 1.0. "
            "Output only the JSON."
        )
        decoded = self._internal_grounder_generate(pipe, image, prompt)
        parsed = self._extract_first_json_object(decoded)
        result: list[dict[str, Any]] = []
        if (
            isinstance(parsed, dict)
            and parsed.get("present")
            and isinstance(parsed.get("box"), (list, tuple))
        ):
            box = list(parsed.get("box"))
            if len(box) == 4 and all(isinstance(value, (int, float)) for value in box):
                width, height = image.size
                coords = [float(value) for value in box]
                max_coord = max(abs(coord) for coord in coords) if coords else 0.0
                divisor = 1.0 if max_coord <= 1.5 else 1000.0
                x0 = coords[0] / divisor * width
                y0 = coords[1] / divisor * height
                x1 = coords[2] / divisor * width
                y1 = coords[3] / divisor * height
                x0, x1 = sorted((x0, x1))
                y0, y1 = sorted((y0, y1))
                x0 = max(0.0, min(float(width), x0))
                x1 = max(0.0, min(float(width), x1))
                y0 = max(0.0, min(float(height), y0))
                y1 = max(0.0, min(float(height), y1))
                if x1 - x0 >= 1.0 and y1 - y0 >= 1.0:
                    try:
                        confidence = float(parsed.get("confidence", 1.0))
                    except (TypeError, ValueError):
                        confidence = 1.0
                    confidence = max(0.0, min(1.0, confidence))
                    result.append({"score": confidence, "box": (x0, y0, x1, y1), "label": phrase})
        if cache is not None:
            cache[cache_key] = list(result)
        return result

    @staticmethod
    def _extract_first_json_object(text: str) -> Any:
        if not text:
            return None
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        for index in range(start, len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : index + 1])
                    except Exception:
                        return None
        return None

    def _object_detector_contract_score(
        self,
        spec: dict[str, Any],
        original: Image.Image,
        edited: Image.Image,
        cache: dict[tuple[str, str], Any] | None = None,
        pipe: Any = None,
    ) -> tuple[dict[str, float], dict[str, float]]:
        edit_type = str(spec.get("edit_type", ""))
        if not self.object_detector_enabled or edit_type not in self.object_detector_edit_types:
            return {
                "object_detector_contract": 1.0,
            }, {
                "object_detector_supported": 0.0,
                "object_detector_gate_pass": 1.0,
            }

        source_phrase = self._object_detector_phrase(
            spec.get("source_object") or spec.get("target") or spec.get("target_object")
        )
        target_phrase = self._object_detector_phrase(spec.get("target_object") or spec.get("replacement"))
        if not source_phrase:
            return {
                "object_detector_contract": 1.0,
            }, {
                "object_detector_supported": 0.0,
                "object_detector_missing_source_phrase": 1.0,
                "object_detector_gate_pass": 1.0,
            }
        if (
            self.object_detector_fail_open_on_unsuitable_phrase
            and not self._object_detector_phrase_is_suitable(source_phrase)
        ):
            return self._object_detector_skip_result("unsuitable_source_phrase")
        if (
            self.object_detector_fail_open_on_unsuitable_phrase
            and edit_type == "object_replacement"
            and target_phrase
            and not self._object_detector_phrase_is_suitable(target_phrase)
        ):
            return self._object_detector_skip_result("unsuitable_target_phrase")

        try:
            original_source_score = self._detect_object_score(original, source_phrase, cache=cache, pipe=pipe)
            edited_source_score = self._detect_object_score(edited, source_phrase, cache=cache, pipe=pipe)
        except Exception as exc:
            return {
                "object_detector_contract": 0.0,
                "object_detector_original_source_score": 0.0,
                "object_detector_edited_source_score": 1.0,
                "object_detector_edited_target_score": 0.0,
                "object_detector_source_grounding": 0.0,
                "object_detector_source_absence": 0.0,
                "object_detector_target_presence": 0.0,
            }, {
                "object_detector_supported": 1.0,
                "object_detector_error": 1.0,
                "object_detector_error_type": exc.__class__.__name__,
                "object_detector_error_message": str(exc)[:300],
                "object_detector_source_detected_pass": 0.0,
                "object_detector_source_absent_pass": 0.0,
                "object_detector_target_present_pass": 0.0,
                "object_detector_gate_pass": 0.0,
                "object_detector_original_min_score": self.object_detector_original_min_score,
                "object_detector_edited_absent_max_score": self.object_detector_edited_absent_max_score,
                "object_detector_absent_ratio": self.object_detector_absent_ratio,
                "object_detector_target_min_score": self.object_detector_target_min_score,
                "object_detector_score_threshold": self.object_detector_score_threshold,
            }
        source_detected = (
            original_source_score >= self.object_detector_original_min_score
            or not self.object_detector_require_original_detection
        )
        absent_cutoff = max(
            self.object_detector_edited_absent_max_score,
            original_source_score * self.object_detector_absent_ratio,
        )
        source_absent = edited_source_score <= absent_cutoff
        source_grounding_score = _sigmoid(
            (original_source_score - self.object_detector_original_min_score)
            / max(self.object_detector_score_temperature, 1e-6)
        )
        source_absence_score = _sigmoid(
            (absent_cutoff - edited_source_score) / max(self.object_detector_score_temperature, 1e-6)
        )

        target_score = 1.0
        target_present = True
        target_presence_score = 1.0
        if edit_type == "object_replacement":
            if not target_phrase:
                target_score = 0.0
                target_present = False
                target_presence_score = 0.0
            else:
                try:
                    target_score = self._detect_object_score(edited, target_phrase, cache=cache, pipe=pipe)
                except Exception as exc:
                    return {
                        "object_detector_contract": 0.0,
                        "object_detector_original_source_score": original_source_score,
                        "object_detector_edited_source_score": edited_source_score,
                        "object_detector_absent_cutoff": absent_cutoff,
                        "object_detector_edited_target_score": 0.0,
                        "object_detector_source_grounding": source_grounding_score,
                        "object_detector_source_absence": source_absence_score,
                        "object_detector_target_presence": 0.0,
                    }, {
                        "object_detector_supported": 1.0,
                        "object_detector_error": 1.0,
                        "object_detector_error_type": exc.__class__.__name__,
                        "object_detector_error_message": str(exc)[:300],
                        "object_detector_source_detected_pass": 1.0 if source_detected else 0.0,
                        "object_detector_source_absent_pass": 1.0 if source_absent else 0.0,
                        "object_detector_target_present_pass": 0.0,
                        "object_detector_gate_pass": 0.0,
                        "object_detector_original_min_score": self.object_detector_original_min_score,
                        "object_detector_edited_absent_max_score": self.object_detector_edited_absent_max_score,
                        "object_detector_absent_ratio": self.object_detector_absent_ratio,
                        "object_detector_target_min_score": self.object_detector_target_min_score,
                        "object_detector_score_threshold": self.object_detector_score_threshold,
                    }
                target_present = target_score >= self.object_detector_target_min_score
                target_presence_score = _sigmoid(
                    (target_score - self.object_detector_target_min_score)
                    / max(self.object_detector_score_temperature, 1e-6)
                )

        component_values = [source_absence_score, target_presence_score]
        if self.object_detector_require_original_detection:
            component_values.insert(0, source_grounding_score)
        contract_score = self._geometric_mean(component_values)
        gate_pass = (
            source_detected
            and source_absent
            and target_present
            and contract_score >= self.object_detector_score_threshold
        )
        scores = {
            "object_detector_contract": contract_score,
            "object_detector_original_source_score": original_source_score,
            "object_detector_edited_source_score": edited_source_score,
            "object_detector_absent_cutoff": absent_cutoff,
            "object_detector_edited_target_score": target_score,
            "object_detector_source_grounding": source_grounding_score,
            "object_detector_source_absence": source_absence_score,
            "object_detector_target_presence": target_presence_score,
        }
        signals = {
            "object_detector_supported": 1.0,
            "object_detector_source_detected_pass": 1.0 if source_detected else 0.0,
            "object_detector_source_absent_pass": 1.0 if source_absent else 0.0,
            "object_detector_target_present_pass": 1.0 if target_present else 0.0,
            "object_detector_gate_pass": 1.0 if gate_pass else 0.0,
            "object_detector_original_min_score": self.object_detector_original_min_score,
            "object_detector_edited_absent_max_score": self.object_detector_edited_absent_max_score,
            "object_detector_absent_ratio": self.object_detector_absent_ratio,
            "object_detector_target_min_score": self.object_detector_target_min_score,
            "object_detector_score_threshold": self.object_detector_score_threshold,
        }
        return scores, signals

    def _conservative_region_target_phrase(self, spec: dict[str, Any]) -> str:
        edit_type = str(spec.get("edit_type", ""))
        if edit_type == "object_addition":
            return ""
        phrase = (
            spec.get("source_object")
            or spec.get("target")
            or spec.get("target_object")
            or spec.get("target_region")
        )
        phrase = self._object_detector_phrase(phrase)
        generic_phrases = {
            "background",
            "whole image",
            "entire image",
            "main visible target",
            "plausible open area of scene",
            "target region",
        }
        return "" if phrase in generic_phrases else phrase

    def _conservative_region_mask(
        self,
        spec: dict[str, Any],
        original: Image.Image,
        edited: Image.Image,
        cache: dict[tuple[str, str], Any] | None = None,
        pipe: Any = None,
    ) -> tuple[Any | None, dict[str, Any]]:
        mask_size = (
            max(16, self.conservative_region_mask_size),
            max(16, self.conservative_region_mask_size),
        )
        phrase = self._conservative_region_target_phrase(spec)
        signals: dict[str, Any] = {
            "conservative_region_target_phrase": phrase,
            "conservative_region_mask_source": "none",
            "conservative_region_mask_supported": 0.0,
            "conservative_region_detector_score": 0.0,
            "conservative_region_detector_box_count": 0.0,
            "conservative_region_diff_fallback_used": 0.0,
        }
        detector_phrase_suitable = self._object_detector_phrase_is_suitable(phrase) if phrase else False
        if phrase and not detector_phrase_suitable:
            signals["conservative_region_object_detector_skipped"] = 1.0
            signals["conservative_region_object_detector_skip_reason"] = "unsuitable_target_phrase"
        if (
            self.conservative_region_use_object_detector
            and self.object_detector_enabled
            and phrase
            and detector_phrase_suitable
        ):
            try:
                detections = self._detect_object_boxes(original, phrase, cache=cache, pipe=pipe)
            except Exception as exc:
                detections = []
                signals["conservative_region_detector_error"] = 1.0
                signals["conservative_region_detector_error_type"] = exc.__class__.__name__
                signals["conservative_region_detector_error_message"] = str(exc)[:300]
            boxes = [
                tuple(item["box"])
                for item in detections[: max(1, self.conservative_region_max_detector_boxes)]
                if "box" in item
            ]
            if boxes:
                mask = box_mask_from_boxes(
                    original.size,
                    boxes,
                    size=mask_size,
                    padding_fraction=self.conservative_region_mask_padding_fraction,
                )
                if mask.any():
                    signals["conservative_region_mask_source"] = "object_detector"
                    signals["conservative_region_mask_supported"] = 1.0
                    signals["conservative_region_detector_score"] = float(detections[0].get("score", 0.0))
                    signals["conservative_region_detector_box_count"] = float(len(boxes))
                    return mask, signals

        if self.conservative_region_fallback_to_diff_mask:
            mask = diff_mask(
                original,
                edited,
                diff_threshold=self.conservative_region_diff_threshold,
                size=mask_size,
                dilation_radius=self.conservative_region_diff_dilation_radius,
            )
            if mask.any():
                signals["conservative_region_mask_source"] = "diff_fallback"
                signals["conservative_region_diff_fallback_used"] = 1.0
                return mask, signals

        return None, signals

    def _conservative_region_score(
        self,
        spec: dict[str, Any],
        original: Image.Image,
        edited: Image.Image,
        cache: dict[tuple[str, str], Any] | None = None,
        pipe: Any = None,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        edit_type = str(spec.get("edit_type", "local_enhancement"))
        max_target_area = self.conservative_region_max_target_area_by_edit_type.get(
            edit_type,
            self.conservative_region_max_target_area,
        )
        min_localization_precision = (
            self.conservative_region_min_localization_precision_by_edit_type.get(
                edit_type,
                self.conservative_region_min_localization_precision,
            )
        )
        active = (
            self.conservative_region_reward_enabled
            and edit_type in self.conservative_region_edit_types
        )
        base_scores = {
            "conservative_region_reward": 1.0,
            "conservative_region_observed_reward": 1.0,
            "conservative_target_change_score": 1.0,
            "conservative_target_structure_preservation": 1.0,
            "conservative_target_structure_edge_similarity": 1.0,
            "conservative_target_structure_luminance_correlation": 1.0,
            "conservative_outside_preservation": 1.0,
            "conservative_outside_changed_fraction_score": 1.0,
            "conservative_localization_precision": 1.0,
        }
        base_signals: dict[str, Any] = {
            "conservative_region_active": 1.0 if active else 0.0,
            "conservative_region_gate_pass": 1.0,
            "conservative_region_gate_applicable": 0.0,
            "conservative_region_requires_mask": (
                1.0 if edit_type in self.conservative_region_require_mask_edit_types else 0.0
            ),
            "conservative_region_min_reward": self.conservative_region_min_reward,
            "conservative_region_min_outside_preservation": self.conservative_region_min_outside_preservation,
            "conservative_region_max_target_area": max_target_area,
            "conservative_region_max_outside_change": self.conservative_region_max_outside_change,
            "conservative_region_max_outside_changed_fraction": (
                self.conservative_region_max_outside_changed_fraction
            ),
            "conservative_region_min_target_change_score": (
                self.conservative_region_min_target_change_score
            ),
            "conservative_region_min_localization_precision": min_localization_precision,
            "conservative_region_min_target_structure": (
                self.conservative_region_min_target_structure_by_edit_type.get(
                    edit_type, self.conservative_region_min_target_structure
                )
            ),
            "conservative_region_min_target_structure_edge": (
                self.conservative_region_min_target_structure_edge_by_edit_type.get(
                    edit_type, self.conservative_region_min_target_structure_edge
                )
            ),
        }
        if not active:
            return base_scores, base_signals

        requires_mask = edit_type in self.conservative_region_require_mask_edit_types
        mask, mask_signals = self._conservative_region_mask(spec, original, edited, cache=cache, pipe=pipe)
        base_signals.update(mask_signals)
        if mask is None:
            gate_pass = not requires_mask
            base_scores["conservative_region_reward"] = 1.0 if gate_pass else 0.0
            base_scores["conservative_region_observed_reward"] = 0.0
            base_signals.update(
                {
                    "conservative_region_missing_mask": 1.0,
                    "conservative_region_gate_pass": 1.0 if gate_pass else 0.0,
                    "conservative_region_reject_reason": (
                        "missing_target_mask" if not gate_pass else "not_applicable"
                    ),
                }
            )
            return base_scores, base_signals

        stats = masked_region_statistics(
            original,
            edited,
            mask,
            diff_threshold=self.conservative_region_diff_threshold,
            size=(
                max(16, self.conservative_region_mask_size),
                max(16, self.conservative_region_mask_size),
            ),
        )
        target_area = stats["target_area_fraction"]
        area_pass = (
            target_area >= self.conservative_region_min_target_area
            and target_area <= max_target_area
        )
        target_change_score = _sigmoid(
            (stats["target_change"] - self.conservative_region_min_target_change)
            / max(self.conservative_region_target_change_temperature, 1e-6)
        )
        outside_preservation = _sigmoid(
            (self.conservative_region_max_outside_change - stats["outside_change"])
            / max(self.conservative_region_outside_change_temperature, 1e-6)
        )
        outside_fraction_score = _sigmoid(
            (
                self.conservative_region_max_outside_changed_fraction
                - stats["outside_changed_fraction"]
            )
            / max(0.08, self.conservative_region_outside_change_temperature)
        )
        localization_score = _sigmoid(
            (
                stats["localization_precision"]
                - min_localization_precision
            )
            / 0.10
        )
        mask_source = str(base_signals.get("conservative_region_mask_source", "none"))
        structure_gate_applicable = (
            edit_type in self.conservative_region_target_structure_edit_types
            and mask_source == "object_detector"
        )
        structure_stats = (
            masked_target_structure_preservation(
                original,
                edited,
                mask,
                size=(
                    max(16, self.conservative_region_mask_size),
                    max(16, self.conservative_region_mask_size),
                ),
            )
            if structure_gate_applicable
            else {
                "target_structure_preservation": 1.0,
                "target_structure_luminance_correlation": 1.0,
                "target_structure_edge_similarity": 1.0,
                "target_structure_edge_delta": 0.0,
                "target_structure_mean_edge": 0.0,
            }
        )
        target_structure_score = float(structure_stats["target_structure_preservation"])
        observed_reward = self._geometric_mean(
            [
                target_change_score,
                target_structure_score,
                outside_preservation,
                outside_fraction_score,
                localization_score,
            ]
        )
        diff_fallback_gate_allowed = (
            mask_source == "diff_fallback"
            and self.conservative_region_diff_fallback_allows_gate
        )
        gate_applicable = mask_source == "object_detector" or diff_fallback_gate_allowed
        if requires_mask and mask_source != "object_detector" and not diff_fallback_gate_allowed:
            gate_applicable = True
            mask_gate_pass = False
        else:
            mask_gate_pass = True
        target_change_pass = target_change_score >= self.conservative_region_min_target_change_score
        target_structure_pass = (
            not structure_gate_applicable
            or target_structure_score
            >= float(base_signals["conservative_region_min_target_structure"])
        )
        target_structure_edge_score = float(structure_stats["target_structure_edge_similarity"])
        target_structure_edge_pass = (
            not structure_gate_applicable
            or target_structure_edge_score
            >= float(base_signals["conservative_region_min_target_structure_edge"])
        )
        outside_change_pass = stats["outside_change"] <= self.conservative_region_max_outside_change
        outside_fraction_pass = (
            stats["outside_changed_fraction"]
            <= self.conservative_region_max_outside_changed_fraction
        )
        outside_preservation_pass = (
            outside_preservation >= self.conservative_region_min_outside_preservation
        )
        localization_pass = stats["localization_precision"] >= min_localization_precision
        reward_pass = observed_reward >= self.conservative_region_min_reward
        gate_pass = (
            (not gate_applicable)
            or (
                mask_gate_pass
                and area_pass
                and target_change_pass
                and target_structure_pass
                and target_structure_edge_pass
                and localization_pass
                and outside_change_pass
                and outside_fraction_pass
                and outside_preservation_pass
                and reward_pass
            )
        )
        effective_reward = observed_reward if gate_applicable else 1.0
        reject_reason = "accepted"
        if not mask_gate_pass:
            reject_reason = "target_mask_not_supported"
        elif not area_pass:
            reject_reason = "target_mask_area"
        elif not target_change_pass:
            reject_reason = "target_change"
        elif not target_structure_pass:
            reject_reason = "target_structure"
        elif not target_structure_edge_pass:
            reject_reason = "target_structure_edge"
        elif not localization_pass:
            reject_reason = "localization_precision"
        elif not outside_change_pass:
            reject_reason = "outside_change"
        elif not outside_fraction_pass:
            reject_reason = "outside_changed_fraction"
        elif not outside_preservation_pass:
            reject_reason = "outside_preservation"
        elif not reward_pass:
            reject_reason = "region_reward"
        elif not gate_applicable:
            reject_reason = "not_applicable"

        scores = {
            "conservative_region_reward": _clamp(effective_reward),
            "conservative_region_observed_reward": _clamp(observed_reward),
            "conservative_target_change_score": _clamp(target_change_score),
            "conservative_target_structure_preservation": _clamp(target_structure_score),
            "conservative_target_structure_edge_similarity": _clamp(
                float(structure_stats["target_structure_edge_similarity"])
            ),
            "conservative_target_structure_luminance_correlation": _clamp(
                float(structure_stats["target_structure_luminance_correlation"])
            ),
            "conservative_outside_preservation": _clamp(outside_preservation),
            "conservative_outside_changed_fraction_score": _clamp(outside_fraction_score),
            "conservative_localization_precision": _clamp(stats["localization_precision"]),
        }
        signals = {
            **base_signals,
            "conservative_region_gate_applicable": 1.0 if gate_applicable else 0.0,
            "conservative_region_gate_pass": 1.0 if gate_pass else 0.0,
            "conservative_region_reject_reason": reject_reason,
            "conservative_region_area_pass": 1.0 if area_pass else 0.0,
            "conservative_region_mask_gate_pass": 1.0 if mask_gate_pass else 0.0,
            "conservative_region_target_change_pass": 1.0 if target_change_pass else 0.0,
            "conservative_region_target_structure_applicable": (
                1.0 if structure_gate_applicable else 0.0
            ),
            "conservative_region_target_structure_pass": (
                1.0 if target_structure_pass else 0.0
            ),
            "conservative_region_target_structure_edge_pass": (
                1.0 if target_structure_edge_pass else 0.0
            ),
            "conservative_region_localization_pass": 1.0 if localization_pass else 0.0,
            "conservative_region_outside_change_pass": 1.0 if outside_change_pass else 0.0,
            "conservative_region_outside_fraction_pass": 1.0 if outside_fraction_pass else 0.0,
            "conservative_region_outside_preservation_pass": (
                1.0 if outside_preservation_pass else 0.0
            ),
            "conservative_region_reward_pass": 1.0 if reward_pass else 0.0,
            "conservative_target_area_fraction": target_area,
            "conservative_outside_area_fraction": stats["outside_area_fraction"],
            "conservative_changed_fraction": stats["changed_fraction"],
            "conservative_target_change": stats["target_change"],
            "conservative_outside_change": stats["outside_change"],
            "conservative_target_changed_fraction": stats["target_changed_fraction"],
            "conservative_outside_changed_fraction": stats["outside_changed_fraction"],
            "conservative_target_changed_pixel_fraction": stats["target_changed_pixel_fraction"],
            "conservative_outside_changed_pixel_fraction": stats["outside_changed_pixel_fraction"],
            "conservative_target_structure_edge_delta": float(
                structure_stats["target_structure_edge_delta"]
            ),
            "conservative_target_structure_mean_edge": float(
                structure_stats["target_structure_mean_edge"]
            ),
            "conservative_outside_psnr": stats["outside_psnr"],
            "conservative_target_psnr": stats["target_psnr"],
            "conservative_region_min_target_change": self.conservative_region_min_target_change,
            "conservative_region_min_localization_precision": min_localization_precision,
        }
        return scores, signals

    def _score_candidate_row(
        self,
        pipe: Any,
        proposal: EditProposal,
        original: Image.Image,
        edited: Image.Image,
        candidate_index: int,
        scoring_device: str,
    ) -> dict[str, Any]:
        cache: dict[tuple[str, str], Any] = {}
        try:
            edit_specificity, edit_signals = self._edit_specificity(
                pipe, proposal, original, edited, candidate_index, cache
            )
            taxonomy_score, taxonomy_signals = self._taxonomy_score(
                pipe, proposal, original, edited, candidate_index, cache
            )
            preservation, validity, preservation_signals = self._preservation_and_validity(
                pipe, proposal, original, edited, candidate_index, cache
            )
            rubric_scores, rubric_signals = self._rubric_score(
                pipe, proposal, original, edited, candidate_index, validity, cache
            )
            # Lock the sign of the unshifted semantic evidence before any
            # sigmoid/geometric aggregation can make a wrong-direction edit
            # look compensatorily strong.
            raw_gate_status, raw_gate_signals = self._raw_semantic_gate_status(
                {**edit_signals, **rubric_signals}
            )
            spec = normalize_structured_edit(
                proposal.structured_edit,
                instruction=proposal.instruction,
                family=proposal.definition.family,
            )
            edit_type = str(spec.get("edit_type", "local_enhancement"))
            object_detector_scores, object_detector_signals = self._object_detector_contract_score(
                spec,
                original,
                edited,
                cache,
                pipe=pipe,
            )
            conservative_scores, conservative_signals = self._conservative_region_score(
                spec,
                original,
                edited,
                cache,
                pipe=pipe,
            )

            cepr_semantic_edit = (
                math.sqrt(max(edit_specificity * taxonomy_score, 0.0))
                if taxonomy_signals.get("cepr_taxonomy_supported", 0.0) > 0.0 or self.taxonomy_required
                else edit_specificity
            )
            semantic_components = [rubric_scores["rubric_edit_success"]]
            if object_detector_signals.get("object_detector_supported", 0.0) > 0.0:
                semantic_components.append(object_detector_scores["object_detector_contract"])
            if self.combine_cepr_semantics:
                semantic_components.append(edit_specificity)
                if taxonomy_signals.get("cepr_taxonomy_supported", 0.0) > 0.0 or self.taxonomy_required:
                    semantic_components.append(taxonomy_score)
            semantic_edit = self._geometric_mean(semantic_components)
            preservation_score = self._geometric_mean(
                [
                    preservation,
                    rubric_scores["rubric_preservation"],
                    conservative_scores["conservative_region_reward"],
                ]
            )
            reward = self._geometric_mean([semantic_edit, preservation_score])

            forbidden_after_supported = rubric_signals.get("rubric_forbidden_after_supported", 0.0)
            hard_forbidden_required = edit_type in self.rubric_hard_forbidden_edit_types
            if hard_forbidden_required:
                forbidden_gate_strict = (
                    forbidden_after_supported > 0.0
                    and rubric_scores["rubric_forbidden_after_absent"] >= self.rubric_hard_forbidden_threshold
                )
            else:
                forbidden_gate_strict = (
                    rubric_scores["rubric_forbidden_after_absent"] >= self.rubric_forbidden_threshold
                    or forbidden_after_supported <= 0.0
                )
            forbidden_gate_softened = (
                not hard_forbidden_required
                and edit_type in self.rubric_soft_forbidden_edit_types
                and rubric_scores["rubric_required_after"] >= self.rubric_soft_forbidden_required_threshold
                and rubric_scores["rubric_reward"] >= self.rubric_soft_forbidden_reward_threshold
                and rubric_scores["rubric_preservation"] >= self.rubric_soft_forbidden_preservation_threshold
                and edit_specificity >= self.edit_threshold
                and preservation >= self.preservation_threshold
                and validity >= self.validity_threshold
            )
            forbidden_gate = forbidden_gate_strict or forbidden_gate_softened
            taxonomy_gate = (
                taxonomy_score >= self.taxonomy_threshold
                or (
                    taxonomy_signals.get("cepr_taxonomy_supported", 0.0) <= 0.0
                    and not self.taxonomy_required
                )
            )
            gate_status = {
                **raw_gate_status,
                "rubric_source_grounded": rubric_scores["rubric_source_grounded"] >= self.rubric_source_threshold,
                "rubric_required_after": rubric_scores["rubric_required_after"] >= self.rubric_required_threshold,
                "rubric_forbidden_gate": forbidden_gate,
                "object_detector_contract": (
                    object_detector_signals.get("object_detector_gate_pass", 1.0) >= 0.5
                ),
                "conservative_region": (
                    conservative_signals.get("conservative_region_gate_pass", 1.0) >= 0.5
                ),
                "rubric_preservation": rubric_scores["rubric_preservation"] >= self.rubric_preservation_threshold,
                "cepr_edit_specificity": edit_specificity >= self.edit_threshold,
                "cepr_taxonomy": taxonomy_gate,
                "cepr_preservation": preservation >= self.preservation_threshold,
                "cepr_validity": validity >= self.validity_threshold,
                "rubric_cepr_reward": reward >= self.rubric_reward_threshold,
            }
            feasible = all(gate_status.values())
            reject_reason = "accepted"
            if not feasible:
                reject_reason = next(
                    gate_name for gate_name, passed in gate_status.items() if not passed
                )
            return {
                "candidate_index": candidate_index,
                "edit_specificity": edit_specificity,
                "taxonomy_score": taxonomy_score,
                "semantic_edit": semantic_edit,
                "preservation": preservation_score,
                "validity": validity,
                "reward": reward if feasible else 0.0,
                "raw_reward": reward,
                "feasible": feasible,
                "component_scores": {
                    "cepr_embedding_semantic_edit": cepr_semantic_edit,
                    "rubric_source_grounded": rubric_scores["rubric_source_grounded"],
                    "rubric_required_after": rubric_scores["rubric_required_after"],
                    "rubric_forbidden_after_absent": rubric_scores["rubric_forbidden_after_absent"],
                    "rubric_edit_success": rubric_scores["rubric_edit_success"],
                    "rubric_preservation": rubric_scores["rubric_preservation"],
                    "rubric_validity": rubric_scores["rubric_validity"],
                    "rubric_reward": rubric_scores["rubric_reward"],
                    "rubric_cepr_reward": reward if feasible else 0.0,
                    "rubric_cepr_raw_reward": reward,
                    **object_detector_scores,
                    **conservative_scores,
                },
                "signals": {
                    **edit_signals,
                    **taxonomy_signals,
                    **preservation_signals,
                    **rubric_signals,
                    **raw_gate_signals,
                    **object_detector_signals,
                    **conservative_signals,
                    "cepr_internal_supported": 1.0,
                    "cepr_scoring_device": scoring_device,
                    "rubric_forbidden_gate_pass": 1.0 if forbidden_gate else 0.0,
                    "rubric_forbidden_gate_strict_pass": 1.0 if forbidden_gate_strict else 0.0,
                    "rubric_forbidden_gate_softened": 1.0 if forbidden_gate_softened else 0.0,
                    "rubric_hard_forbidden_required": 1.0 if hard_forbidden_required else 0.0,
                    "rubric_hard_forbidden_threshold": self.rubric_hard_forbidden_threshold,
                    "rubric_soft_forbidden_required_threshold": self.rubric_soft_forbidden_required_threshold,
                    "rubric_soft_forbidden_reward_threshold": self.rubric_soft_forbidden_reward_threshold,
                    "rubric_soft_forbidden_preservation_threshold": self.rubric_soft_forbidden_preservation_threshold,
                    "rubric_taxonomy_gate_pass": 1.0 if taxonomy_gate else 0.0,
                    "rubric_reject_reason": reject_reason,
                    **{
                        f"rubric_gate_{gate_name}_pass": 1.0 if passed else 0.0
                        for gate_name, passed in gate_status.items()
                    },
                },
            }
        finally:
            cache.clear()
            if self.empty_cache_per_candidate:
                QwenEditEditor._empty_cuda_cache()


def build_proposer(config: dict[str, Any]):
    backend = config.get("backend", "scripted")
    if backend == "scripted":
        return ScriptedProposer(config)
    if backend == "internal_qwen":
        return InternalQwenProposer()
    if backend in {"trainable_qwen_image_edit", "qwen_image_edit_lora", "trainable_qwen_vl", "qwen_vl_lora"}:
        return TrainableQwenVLProposer(config)
    raise ValueError(f"Unsupported proposer backend: {backend}")


def build_editor(config: dict[str, Any]):
    backend = config.get("backend", "qwen_edit")
    if backend == "pillow_demo":
        return PillowPrototypeEditor()
    if backend == "qwen_edit":
        return QwenEditEditor(config)
    raise ValueError(f"Unsupported editor backend: {backend}")


def build_evaluator(config: dict[str, Any]):
    backend = config.get("backend", "stat")
    if backend == "stat":
        return StatSolver(config)
    if backend == "internal_qwen":
        evaluator_config = dict(config)
        evaluator_config.setdefault("proxy_weight", 0.7)
        evaluator_config.setdefault("internal_weight", 0.3)
        return MultiSignalSolver(evaluator_config)
    if backend == "hybrid":
        return MultiSignalSolver(config)
    if backend in {"generic_relative_self_reward", "evolmm_style"}:
        evaluator_config = dict(config)
        evaluator_config.setdefault("global_weight", 1.0)
        evaluator_config.setdefault("local_weight", 0.0)
        evaluator_config.setdefault("proxy_weight", 1.0)
        evaluator_config.setdefault("spatial_weight", 0.0)
        evaluator_config.setdefault("cycle_weight", 0.0)
        evaluator_config.setdefault("internal_weight", 0.0)
        return GenericRelativeSelfRewardEvaluator(evaluator_config)
    if backend in {"hard_gated_relative", "delta_ranker"}:
        evaluator_config = dict(config)
        evaluator_config.setdefault("spatial_weight", 0.20)
        evaluator_config.setdefault("cycle_weight", 0.0)
        evaluator_config.setdefault("internal_weight", 0.0)
        return HardGatedRelativeEvaluator(evaluator_config)
    if backend in {"internal_cepr", "contrastive_edit_preservation"}:
        evaluator_config = dict(config)
        evaluator_config.setdefault("counterfactual_backend", "internal")
        evaluator_config.setdefault("counterfactual_distractors", 4)
        evaluator_config.setdefault("top_m", 1)
        return InternalContrastiveEditPreservationEvaluator(evaluator_config)
    if backend in {"internal_cepr_rubric", "rubric_cepr", "internal_rubric_cepr"}:
        evaluator_config = dict(config)
        evaluator_config.setdefault("counterfactual_backend", "internal")
        evaluator_config.setdefault("counterfactual_distractors", 4)
        evaluator_config.setdefault("top_m", 1)
        return InternalRubricCEPREvaluator(evaluator_config)
    raise ValueError(f"Unsupported evaluator backend: {backend}")


def build_solver(config: dict[str, Any]):
    """Backward-compatible name for historical configs and scripts."""
    return build_evaluator(config)
