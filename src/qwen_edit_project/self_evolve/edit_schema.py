from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from qwen_edit_project.self_evolve.types import ProposalDefinition


EDIT_TYPES = {
    "attribute_change",
    "color_change",
    "material_change",
    "object_replacement",
    "object_removal",
    "object_addition",
    "spatial_move",
    "style_transfer",
    "background_change",
    "global_adjustment",
    "local_enhancement",
    "subject_extraction",
    "compose",
    "action_change",
}

LOCAL_EDIT_TYPES = {
    "attribute_change",
    "color_change",
    "material_change",
    "object_replacement",
    "object_removal",
    "object_addition",
    "spatial_move",
    "local_enhancement",
    "action_change",
}

EXPECTED_CHANGE_FRACTIONS = {
    "attribute_change": (0.03, 0.45),
    "color_change": (0.03, 0.45),
    "material_change": (0.04, 0.50),
    "object_replacement": (0.05, 0.70),
    "object_removal": (0.04, 0.55),
    "object_addition": (0.04, 0.60),
    "spatial_move": (0.05, 0.70),
    "style_transfer": (0.15, 0.90),
    "background_change": (0.12, 0.85),
    "global_adjustment": (0.25, 1.0),
    "local_enhancement": (0.03, 0.45),
    "subject_extraction": (0.20, 0.95),
    "compose": (0.08, 0.75),
    "action_change": (0.05, 0.55),
}

DEFAULT_PRESERVE = [
    "background",
    "scene layout",
    "camera viewpoint",
    "lighting consistency",
    "unrelated objects",
]

_SPATIAL_RELATION_PATTERN = re.compile(
    r"\s+("
    r"on|in|at|under|beneath|below|above|beside|near|next to|left of|right of|"
    r"in front of|behind|between|attached to|held by|worn by|around"
    r")\s+(.+)$",
    flags=re.IGNORECASE,
)


def _clean_text(value: Any, max_len: int = 160) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in {"none", "null", "n/a", "na", "not applicable", "unknown"}:
        return None
    return re.sub(r"\s+", " ", text)[:max_len]


def _clean_list(value: Any, max_items: int = 8, max_len: int = 160) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = [part.strip() for part in re.split(r"[;\n]", value)]
    elif isinstance(value, list):
        raw_items = value
    else:
        raw_items = [value]
    output = []
    seen = set()
    for item in raw_items:
        cleaned = _clean_text(item, max_len=max_len)
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            output.append(cleaned)
        if len(output) >= max_items:
            break
    return output


def _looks_generic_forbidden(text: str) -> bool:
    lowered = text.lower()
    generic_markers = (
        "no other",
        "not other",
        "any other",
        "other objects",
        "other background",
        "unrelated objects",
        "extra objects",
        "additional objects",
    )
    return any(marker in lowered for marker in generic_markers)


def _is_negative_forbidden_sentence(text: str) -> bool:
    return bool(re.match(r"^\s*(?:no|not|without)\b", text, flags=re.IGNORECASE))


def _is_background_change_forbidden_failure(text: str) -> bool:
    """Keep only old-background failure checks in background-change forbidden prompts."""
    lowered = text.lower()
    if _looks_generic_forbidden(lowered):
        return False
    explicit_old_background = (
        "original background",
        "old background",
        "previous background",
        "source background",
        "unchanged background",
        "original backdrop",
        "old backdrop",
        "previous backdrop",
        "source backdrop",
        "unchanged backdrop",
        "original scene",
        "old scene",
        "previous scene",
        "source scene",
        "original scenery",
        "old scenery",
    )
    if any(marker in lowered for marker in explicit_old_background):
        return True
    if any(marker in lowered for marker in ("background", "backdrop", "scenery", "scene behind")):
        return any(marker in lowered for marker in ("remain", "still", "unchanged", "same", "visible"))
    return False


def _strip_article(text: str | None) -> str | None:
    cleaned = _clean_text(text)
    if cleaned is None:
        return None
    return re.sub(r"^(?:only\s+)?(?:a|an|the|any|new)\s+", "", cleaned, flags=re.IGNORECASE).strip() or None


def _strip_terminal_punctuation(text: str | None) -> str | None:
    cleaned = _clean_text(text)
    if cleaned is None:
        return None
    return re.sub(r"\s*[.;,]\s*$", "", cleaned).strip() or None


def _strip_visible_text_label(text: str | None) -> str | None:
    cleaned = _clean_text(text)
    if cleaned is None:
        return None
    cleaned = re.sub(
        r"\s+(?:reading|labeled|labelled|saying|with\s+text(?:\s+reading)?)\s+['\"].*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+['\"].*$", "", cleaned).strip()
    return cleaned or None


def _trim_removal_object_tail(text: str | None) -> str | None:
    cleaned = _strip_article(text)
    if cleaned is None:
        return None
    cleaned = _strip_terminal_punctuation(cleaned) or cleaned
    cleaned = _strip_visible_text_label(cleaned) or cleaned
    cleanup_patterns = (
        r"\s+(?:completely|fully|entirely)\s+(?:and\s+)?(?:fill|inpaint|replace|remove|erase)\b.*$",
        r"\s+(?:and|then)\s+(?:fill|inpaint|replace|preserve|keep)\b.*$",
        r"\s+(?:while|but|without)\s+.*$",
        r"\s+(?:completely|fully|entirely)$",
        r"\s+(?:from|in)\s+(?:the\s+)?(?:image|scene|photo|picture)$",
    )
    for pattern in cleanup_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned or None


def _split_spatial_relation(text: str | None) -> tuple[str | None, str | None]:
    cleaned = _trim_removal_object_tail(text)
    if cleaned is None:
        return None, None
    match = _SPATIAL_RELATION_PATTERN.search(cleaned)
    if not match:
        return cleaned, None
    object_text = cleaned[: match.start()].strip()
    relation = match.group(1).strip()
    anchor = match.group(2).strip()
    if not object_text or not anchor:
        return cleaned, None
    # Avoid turning descriptors such as "black and white" into fake regions.
    if len(object_text.split()) > 8:
        return cleaned, None
    return object_text, f"{relation} {anchor}"


def _normalize_removal_object_and_region(
    source_object: str | None,
    target_region: str | None,
) -> tuple[str | None, str | None]:
    object_text, inferred_region = _split_spatial_relation(source_object)
    region = _clean_text(target_region) or inferred_region
    if region:
        region = _trim_removal_object_tail(region) or region
    object_text = _trim_removal_object_tail(object_text)
    if object_text:
        object_text = re.sub(
            r"\b(?:cleanly|naturally|visible|unchanged|removed|filled|surrounding)\b",
            " ",
            object_text,
            flags=re.IGNORECASE,
        )
        object_text = re.sub(r"\s+", " ", object_text).strip()
    bad_fragments = ("fill the area", "area naturally", "surrounding", "unchanged")
    if object_text and any(fragment in object_text.lower() for fragment in bad_fragments):
        object_text = None
    return object_text, region


def _clean_inferred_source_object(text: str | None) -> tuple[str | None, str | None]:
    cleaned = _strip_terminal_punctuation(text)
    if cleaned is None:
        return None, None
    cleaned = re.sub(
        r"^(?:the\s+)?(?:color|colour|material|texture|textures|appearance|detail|details|clarity|sharpness)\s+of\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    cleaned = re.sub(
        r"\s+(?:while|without|but|so that|and\s+(?:keep|preserve|maintain))\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    source_object, inferred_region = _split_spatial_relation(cleaned)
    source_object = _strip_article(source_object)
    if source_object is None:
        return None, inferred_region
    source_object = re.sub(
        r"\b(?:slightly|somewhat|more|less|very|extra)\b",
        " ",
        source_object,
        flags=re.IGNORECASE,
    )
    source_object = re.sub(r"\s+", " ", source_object).strip()
    source_object = _strip_terminal_punctuation(source_object)
    if source_object is None:
        return None, inferred_region
    source_tokens = [token for token in re.findall(r"[a-z0-9]+", source_object.lower()) if token]
    generic_sources = {
        "area",
        "background",
        "color",
        "colour",
        "detail",
        "image",
        "material",
        "object",
        "part",
        "region",
        "scene",
        "section",
        "thing",
        "texture",
    }
    if not source_tokens or set(source_tokens).issubset(generic_sources):
        return None, inferred_region
    if len(source_tokens) > 8:
        return None, inferred_region
    return source_object, inferred_region


def _infer_first_object_match(patterns: list[str], instruction: str) -> tuple[str | None, str | None]:
    for pattern in patterns:
        match = re.search(pattern, instruction, flags=re.IGNORECASE)
        if not match:
            continue
        source_object, target_region = _clean_inferred_source_object(match.group(1))
        if source_object:
            return source_object, target_region
    return None, None


def _infer_object_slots_from_instruction(instruction: str, edit_type: str) -> dict[str, str]:
    lowered = instruction.strip()
    slots: dict[str, str] = {}
    clause_boundary = r"(?=(?:[.;]|,\s*(?:while|preserv(?:e|ing)|keep(?:ing)?|without|but)\b)|$)"
    if edit_type == "object_removal":
        match = re.search(
            r"\b(?:remove|delete|erase|get rid of|take out)\s+(.+)$",
            lowered,
            flags=re.IGNORECASE,
        )
        if match:
            removal_text = _strip_terminal_punctuation(match.group(1)) or match.group(1)
            source_text = removal_text
            region_text = None
            from_match = re.search(r"\s+from\s+(.+)$", removal_text, flags=re.IGNORECASE)
            if from_match:
                source_text = removal_text[: from_match.start()].strip()
                region_text = from_match.group(1).strip()
            source_object, target_region = _normalize_removal_object_and_region(
                source_text,
                _clean_text(region_text) if region_text else None,
            )
            if source_object:
                slots["source_object"] = source_object
                slots["target"] = source_object
            if target_region:
                slots["target_region"] = target_region
        return slots

    if edit_type in {"color_change", "material_change", "attribute_change", "local_enhancement"}:
        local_patterns: list[str] = []
        if edit_type == "color_change":
            local_patterns.extend(
                [
                    r"\bchange\s+(?:the\s+)?(?:color|colour)\s+of\s+(.+?)\s+(?:to|into|from)\b",
                    r"\bchange\s+(.+?)\s+(?:to|into)\s+(?:red|blue|green|yellow|orange|purple|pink|brown|black|white|gray|grey|gold|silver|cyan|magenta)\b",
                    r"\bmake\s+(.+?)\s+(?:red|blue|green|yellow|orange|purple|pink|brown|black|white|gray|grey|gold|silver|cyan|magenta)\b",
                ]
            )
        elif edit_type == "material_change":
            local_patterns.extend(
                [
                    r"\bchange\s+(?:the\s+)?(?:material|texture)\s+of\s+(.+?)\s+(?:to|into)\b",
                    r"\bchange\s+(.+?)\s+material\s+(?:to|into)\b",
                    r"\bchange\s+(?:the\s+)?(.+?)\s+to\s+(?:look|appear)\s+(?:like\s+|as\s+|made\s+of\s+|made\s+from\s+)?(?:metal|metallic|wood|wooden|glass|plastic|leather|fabric|stone|marble|ceramic|rubber|paper|cardboard|clay|porcelain|steel|iron|copper|bronze|gold|silver)\b",
                    r"\bchange\s+(?:the\s+)?(.+?)\s+to\s+(?:a\s+|an\s+|the\s+)?(?:shiny\s+|matte\s+|glossy\s+|rough\s+|smooth\s+|polished\s+|brushed\s+|woven\s+|transparent\s+|metallic\s+)*(?:metallic\s+|wooden\s+|glass\s+|plastic\s+|leather\s+|fabric\s+|stone\s+|marble\s+|ceramic\s+|rubber\s+|paper\s+|cardboard\s+|clay\s+|porcelain\s+|steel\s+|iron\s+|copper\s+|bronze\s+|gold\s+|silver\s+)?(?:material|texture|surface|finish)\b",
                    r"\bchange\s+(?:the\s+)?(.+?)\s+to\s+(?:look|appear)\s+(?:like|as)\b",
                    r"\bmake\s+(.+?)\s+(?:look|appear)\s+(?:like\s+)?(?:metal|metallic|wood|wooden|glass|plastic|leather|fabric|stone|marble|ceramic|rubber)\b",
                    r"\bmake\s+(.+?)\s+(?:look|appear)\s+(?:like\s+|as\s+|made\s+of\s+|made\s+from\s+)?(?:paper|cardboard|clay|porcelain|steel|iron|copper|bronze|gold|silver)\b",
                ]
            )
        elif edit_type == "attribute_change":
            local_patterns.extend(
                [
                    r"\bchange\s+(?:the\s+)?(?:appearance|size|shape|expression|attribute|look)\s+of\s+(.+?)\s+(?:to|into|so|while|without)\b",
                    r"\bgive\s+(.+?)\s+.+?(?:[.;,]|$)",
                    r"\bmake\s+(.+?)\s+(?:more|less|larger|smaller|brighter|darker|older|younger|cleaner|dirty|wet|dry|open|closed|smiling|serious)\b",
                    r"\bchange\s+(.+?)\s+(?:to|into)\s+.+?(?:[.;,]|$)",
                ]
            )
        else:
            local_patterns.extend(
                [
                    r"\benhance\s+(?:the\s+)?(?:detail|details|clarity|texture|textures|sharpness|color|colour|appearance)(?:\s+and\s+\w+)*\s+of\s+(.+?)(?:[.;,]|$|\s+while\b|\s+without\b)",
                    r"\bsharpen\s+(?:the\s+)?(.+?)(?:[.;,]|$|\s+while\b|\s+without\b)",
                    r"\benhance\s+(?:the\s+)?(.+?)(?:[.;,]|$|\s+while\b|\s+without\b)",
                ]
            )
        source_object, inferred_region = _infer_first_object_match(local_patterns, lowered)
        if source_object:
            slots["source_object"] = source_object
            slots["target"] = source_object
        if inferred_region:
            slots["target_region"] = inferred_region
        return slots

    if edit_type == "spatial_move":
        source_object, inferred_region = _infer_first_object_match(
            [
                r"\bmove\s+(?:the\s+)?(.+?)\s+(?:slightly|a\s+bit|somewhat|to|toward|towards|left|right|forward|backward|up|down|higher|lower|closer|farther|further|onto|into|on|in|along|from)\b",
                r"\brelocate\s+(?:the\s+)?(.+?)\s+(?:to|toward|towards|onto|into|on|in|near|beside|left|right|above|below)\b",
            ],
            lowered,
        )
        if source_object:
            slots["source_object"] = source_object
            slots["target"] = source_object
        if inferred_region:
            slots["target_region"] = inferred_region
        return slots

    if edit_type == "object_replacement":
        patterns = [
            rf"\breplace\s+(.+?)\s+with\s+(.+?){clause_boundary}",
            rf"\bchange\s+(.+?)\s+(?:into|to)\s+(.+?){clause_boundary}",
            rf"\bturn\s+(.+?)\s+into\s+(.+?){clause_boundary}",
        ]
        for pattern in patterns:
            match = re.search(pattern, lowered, flags=re.IGNORECASE)
            if not match:
                continue
            source_object = _strip_article(match.group(1))
            target_object = _strip_article(match.group(2))
            if source_object:
                slots["source_object"] = source_object
                slots["target"] = source_object
            if target_object:
                slots["target_object"] = target_object
                slots["replacement"] = target_object
            break
        return slots

    if edit_type == "object_addition":
        match = re.search(
            r"\b(?:add|insert|place|put)\s+(.+?)(?:\s+(next to|left of|right of|in front of|"
            rf"behind|above|below|under|beneath|near|beside|onto|on|into|in|to)\s+(.+?))?{clause_boundary}",
            lowered,
            flags=re.IGNORECASE,
        )
        if match:
            target_object = _strip_terminal_punctuation(_strip_article(match.group(1)))
            relation = _clean_text(match.group(2)) if match.group(2) else None
            anchor = _strip_terminal_punctuation(_clean_text(match.group(3))) if match.group(3) else None
            target_region = f"{relation} {anchor}" if relation and anchor else None
            if target_object:
                slots["target_object"] = target_object
                slots["replacement"] = target_object
            if target_region:
                slots["target_region"] = target_region
    if edit_type == "subject_extraction":
        source_object, inferred_region = _infer_first_object_match(
            [
                r"\b(?:extract|isolate|cut out|separate)\s+(?:the\s+)?(.+?)\s+(?:from|and|onto|on|with|while|$)",
                r"\bremove\s+(?:the\s+)?background\s+(?:around|behind)\s+(?:the\s+)?(.+?)(?:[.;,]|$|\s+while\b)",
                r"\bmake\s+(?:the\s+)?(.+?)\s+(?:isolated|stand alone|standalone)(?:[.;,]|$|\s+while\b)",
            ],
            lowered,
        )
        if source_object:
            slots["source_object"] = source_object
            slots["target"] = source_object
        if inferred_region:
            slots["target_region"] = inferred_region
    if edit_type == "action_change":
        action_verbs = (
            r"turn|raise|lift|extend|bend|lean|tilt|wave|jump|run|walk|sit|stand|look|"
            r"reach|kneel|crouch|stretch|nod|point|hold|grab|push|pull|step|dance|throw|"
            r"catch|climb"
        )
        patterns = (
            rf"\b(?:make|have)\s+(?:the\s+)?(.+?)\s+(?:{action_verbs})\b",
            rf"^\s*(?:raise|lift|extend|bend|move|turn|tilt)\s+(?:the\s+)?(.+?)(?:'s|\u2019s)\s+"
            r"(?:(?:left|right)\s+)?(?:arm|hand|leg|head)\b",
        )
        for pattern in patterns:
            match = re.search(pattern, lowered, flags=re.IGNORECASE)
            if not match:
                continue
            subject = _strip_article(match.group(1))
            if subject:
                slots["source_object"] = subject
                slots["target"] = subject
                slots["target_region"] = subject
            break
    if edit_type == "compose":
        match = re.search(
            r"\b(?:remove|delete|erase)\s+(?:the\s+)?(.+?)\s*(?:,?\s+and\b|,|\.|$)",
            lowered,
            flags=re.IGNORECASE,
        )
        if match:
            removed = _strip_article(_strip_terminal_punctuation(match.group(1)))
            if removed:
                slots["source_object"] = removed
                slots["target"] = removed
    return slots


def _region_mentions_source_object(region: str | None, source_object: str | None) -> bool:
    if not region or not source_object:
        return False
    region_terms = set(re.findall(r"[a-z0-9]+", region.lower()))
    source_terms = {
        token
        for token in re.findall(r"[a-z0-9]+", source_object.lower())
        if token not in {"a", "an", "the", "any", "new", "old", "original"}
    }
    return bool(source_terms and source_terms.issubset(region_terms))


def _region_has_spatial_anchor(region: str | None) -> bool:
    if not region:
        return False
    lowered = f" {region.lower()} "
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
        " at the ",
        " in the ",
    )
    return any(marker in lowered for marker in anchor_markers)


def _region_phrase(region: str | None) -> str:
    cleaned = _clean_text(region) or "the target region"
    lowered = cleaned.lower()
    if lowered == "main visible target":
        return "in the target region"
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
        return cleaned
    return f"at {cleaned}"


def normalize_edit_type(value: Any, fallback: str = "local_enhancement") -> str:
    text = _clean_text(value, max_len=64)
    if text is None:
        return fallback
    normalized = text.lower().replace(" ", "_").replace("-", "_")
    alias_map = {
        "replace": "object_replacement",
        "replacement": "object_replacement",
        "remove": "object_removal",
        "delete": "object_removal",
        "add": "object_addition",
        "insert": "object_addition",
        "move": "spatial_move",
        "relocate": "spatial_move",
        "color": "color_change",
        "colour_change": "color_change",
        "material": "material_change",
        "style": "style_transfer",
        "background": "background_change",
        "global": "global_adjustment",
        "local": "local_enhancement",
        "extract": "subject_extraction",
        "extraction": "subject_extraction",
        "isolate": "subject_extraction",
        "isolation": "subject_extraction",
        "cutout": "subject_extraction",
        "cut_out": "subject_extraction",
        "compound": "compose",
        "compound_edit": "compose",
        "multi_edit": "compose",
        "composite": "compose",
        "action": "action_change",
        "pose": "action_change",
        "pose_change": "action_change",
        "motion": "action_change",
    }
    normalized = alias_map.get(normalized, normalized)
    return normalized if normalized in EDIT_TYPES else fallback


def infer_edit_type_from_instruction(instruction: str, family: str | None = None) -> str:
    lowered = instruction.lower()
    family = (family or "").lower()
    if family in {"exposure", "contrast", "color", "tone"}:
        return "global_adjustment"
    if re.search(
        r"\b(?:remove|delete|erase)\b.*\b(?:and|then|,)\b.*\b(?:increase|decrease|adjust|brighten|darken|saturat"
        r"|contrast|warmer|cooler|blur|sharpen|brightness|exposure)\b",
        lowered,
    ):
        return "compose"
    if re.search(
        r"\bmake\s+(?:the\s+)?.+?\b(?:turn|raise|lift|bend|lean|tilt|wave|jump|run|walk|sit|stand|look|reach"
        r"|kneel|crouch|stretch|nod|point|hold|grab|push|pull|step|dance|throw|catch|climb)\b",
        lowered,
    ):
        return "action_change"
    if any(term in lowered for term in ("extract ", "isolate ", "cut out ", "remove the background")):
        return "subject_extraction"
    if any(term in lowered for term in ("replace ", "change the object", "turn the", "make the person into")):
        return "object_replacement"
    if any(term in lowered for term in ("remove ", "delete ", "erase ")):
        return "object_removal"
    if any(term in lowered for term in ("add ", "insert ", "place a new")):
        return "object_addition"
    if any(term in lowered for term in ("move ", "relocate ", "to the left", "to the right", "higher", "lower")):
        return "spatial_move"
    if any(term in lowered for term in ("color", "colour", "red", "blue", "green", "yellow", "black", "white")):
        return "color_change"
    if any(term in lowered for term in ("metal", "wood", "glass", "plastic", "denim", "leather")):
        return "material_change"
    if family == "style":
        return "style_transfer"
    if family == "background":
        return "background_change"
    return "local_enhancement"


def normalize_structured_edit(payload: dict[str, Any] | None, instruction: str, family: str | None = None) -> dict[str, Any]:
    data = dict(payload or {})
    edit_type = normalize_edit_type(
        data.get("edit_type") or data.get("type"),
        fallback=infer_edit_type_from_instruction(instruction, family=family),
    )
    inferred_slots = _infer_object_slots_from_instruction(instruction, edit_type)
    target_alias = _clean_text(data.get("target") or inferred_slots.get("target"))
    replacement_alias = _clean_text(data.get("replacement") or inferred_slots.get("replacement"))
    source_object = _clean_text(
        data.get("source_object")
        or data.get("object")
        or target_alias
        or inferred_slots.get("source_object")
    )
    if source_object:
        source_object = re.sub(r"(?:'s|\u2019s)$", "", source_object).strip() or None
    target_object = _clean_text(
        data.get("target_object")
        or data.get("replacement_object")
        or replacement_alias
        or inferred_slots.get("target_object")
    )
    source_attribute = _clean_text(data.get("source_attribute"))
    target_attribute = _clean_text(data.get("target_attribute") or data.get("attribute"))
    source_material = _clean_text(data.get("source_material"))
    target_material = _clean_text(data.get("target_material"))
    source_style = _clean_text(data.get("source_style"))
    target_style = _clean_text(data.get("target_style") or data.get("style"))
    source_location = _clean_text(data.get("source_location"))
    target_location = _clean_text(data.get("target_location") or data.get("relation"))
    raw_target_region = _clean_text(data.get("target_region") or data.get("region"))
    inferred_target_region = _clean_text(inferred_slots.get("target_region"))
    target_region = raw_target_region or inferred_target_region or "main visible target"
    if edit_type == "object_removal":
        source_object, normalized_region = _normalize_removal_object_and_region(
            source_object or target_alias or inferred_slots.get("source_object"),
            raw_target_region or inferred_target_region,
        )
        if source_object:
            target_alias = source_object
            target_object = None
        if normalized_region:
            target_region = normalized_region
    if edit_type == "subject_extraction":
        source_object = _strip_article(
            source_object or target_alias or inferred_slots.get("source_object")
        )
        target_alias = source_object
        target_object = None
        if not raw_target_region and not inferred_target_region and source_object:
            target_region = source_object
        if source_object:
            instruction = (
                f"Keep only the {source_object}; remove the original surrounding scene and place "
                f"the {source_object} centered on a clean plain white background while preserving "
                "its shape, identity, viewpoint, and boundary details."
            )
            data["instruction"] = instruction
    if edit_type in {"object_replacement", "object_removal"} and _region_mentions_source_object(
        target_region,
        source_object,
    ) and not _region_has_spatial_anchor(target_region):
        target_region = inferred_target_region or (
            "the original location" if edit_type == "object_removal" else "the same location"
        )
    preserve = _clean_list(data.get("preserve") or data.get("preservation_constraints"))
    if not preserve:
        preserve = DEFAULT_PRESERVE[:]
    if edit_type == "subject_extraction":
        preserve = [
            item
            for item in preserve
            if item.lower()
            not in {
                "background",
                "scene layout",
                "unrelated objects",
            }
        ]
        for item in [
            f"{source_object} identity" if source_object else "subject identity",
            f"{source_object} pose and shape" if source_object else "subject pose and shape",
            "subject boundary detail",
        ]:
            if item and item.lower() not in {entry.lower() for entry in preserve}:
                preserve.insert(0, item)
    preserve_defaults = (
        ["foreground subjects", "foreground object geometry", "camera viewpoint"]
        if edit_type == "background_change"
        else DEFAULT_PRESERVE
    )
    for item in preserve_defaults:
        if len(preserve) >= 3:
            break
        if item.lower() not in {entry.lower() for entry in preserve}:
            preserve.append(item)
    required_after = _clean_list(data.get("required_after") or data.get("must_have_after"))
    forbidden_after = _clean_list(data.get("forbidden_after") or data.get("must_not_have_after"))
    forbidden_after = [
        item
        for item in forbidden_after
        if not _is_negative_forbidden_sentence(item)
    ]

    if edit_type == "object_removal" and source_object:
        canonical_required = f"the area {_region_phrase(target_region)} is cleanly filled after removing {source_object}"
        if not any("fill" in item.lower() and source_object.lower() in item.lower() for item in required_after):
            required_after.insert(0, canonical_required)
        forbidden_after = [
            item
            for item in forbidden_after
            if source_object.lower() in item.lower()
            and not _looks_generic_forbidden(item)
            and "any object other than" not in item.lower()
        ]
    elif edit_type == "subject_extraction" and source_object:
        canonical_required = [
            f"only {source_object} remains visible",
            f"{source_object} is centered on a clean plain white or neutral background",
        ]
        for item in reversed(canonical_required):
            if item.lower() not in {entry.lower() for entry in required_after}:
                required_after.insert(0, item)
        canonical_background_failure = f"the original surrounding scene remains around {source_object}"
        cleaned_forbidden = []
        for item in forbidden_after:
            lowered_item = item.lower()
            mentions_subject = source_object.lower() in lowered_item
            if "background" in lowered_item and ("remain" in lowered_item or "remaining" in lowered_item):
                cleaned_forbidden.append(canonical_background_failure)
            elif "original" in lowered_item and mentions_subject:
                cleaned_forbidden.append(item)
            elif mentions_subject and any(
                term in lowered_item
                for term in ("cropped", "deformed", "missing", "boundary", "unrelated object")
            ):
                cleaned_forbidden.append(item)
        forbidden_after = list(dict.fromkeys(cleaned_forbidden))
    elif edit_type == "background_change":
        forbidden_after = [
            item
            for item in forbidden_after
            if _is_background_change_forbidden_failure(item)
        ]

    if not required_after:
        if edit_type == "object_replacement" and target_object:
            required_after.append(f"{target_object} is visible {_region_phrase(target_region)}")
        elif edit_type == "object_addition" and target_object:
            required_after.append(f"{target_object} has been added {_region_phrase(target_region)}")
        elif edit_type == "object_removal" and source_object:
            required_after.append(f"the area {_region_phrase(target_region)} is cleanly filled after removing {source_object}")
        elif edit_type == "subject_extraction" and source_object:
            required_after.append(f"only {source_object} remains visible")
            required_after.append(f"{source_object} is centered on a clean plain white or neutral background")
        elif edit_type == "spatial_move" and (source_object or target_object) and target_location:
            moved_object = source_object or target_object
            required_after.append(f"{moved_object} is {target_location}")
        elif edit_type in {"attribute_change", "color_change", "material_change", "style_transfer"}:
            target_descriptor = (
                target_attribute
                or target_material
                or target_style
                or target_object
            )
            if target_descriptor:
                required_after.append(f"{target_region} has {target_descriptor}")
        elif edit_type == "background_change" and (target_attribute or target_object):
            required_after.append(f"background changed to {_clean_text(' '.join([part for part in [target_attribute, target_object] if part]))}")
        elif edit_type == "global_adjustment":
            lowered_instruction = instruction.lower()
            if "saturation" in lowered_instruction:
                if any(term in lowered_instruction for term in ("reduce", "decrease", "lower", "less")):
                    required_after.append("image has reduced color saturation")
                else:
                    required_after.append("image has stronger color saturation")
            elif "contrast" in lowered_instruction:
                if any(term in lowered_instruction for term in ("soften", "reduce", "decrease", "lower", "less")):
                    required_after.append("image has softer contrast")
                else:
                    required_after.append("image has stronger contrast")
            elif any(term in lowered_instruction for term in ("brighter", "brighten")):
                required_after.append("image is brighter")
            elif any(term in lowered_instruction for term in ("darker", "darken")):
                required_after.append("image is darker")
            elif "warmer" in lowered_instruction:
                required_after.append("image has a warmer tone")
            elif "cooler" in lowered_instruction:
                required_after.append("image has a cooler tone")
        elif edit_type == "style_transfer" and not (target_attribute or target_style):
            required_after.append(_clean_text(instruction, max_len=160))
        elif edit_type == "action_change":
            action_subject = source_object or "the main subject"
            required_after.append(
                f"{action_subject} is performing the new pose or action described in the instruction"
            )
        elif edit_type == "compose":
            cleaned_instruction = _clean_text(instruction, max_len=160)
            if cleaned_instruction:
                required_after.append(f"all requested changes are applied: {cleaned_instruction}")

    if not forbidden_after:
        if edit_type in {"object_replacement", "object_removal"} and source_object:
            forbidden_after.append(f"{source_object} remains visible {_region_phrase(target_region)}")
        elif edit_type == "object_addition" and target_object:
            forbidden_after.append(
                f"{target_object} is absent {_region_phrase(target_region)}"
            )
        elif edit_type == "subject_extraction" and source_object:
            forbidden_after.append(f"the original surrounding scene remains around {source_object}")
            forbidden_after.append(f"other original scene objects remain around {source_object}")
            forbidden_after.append(f"{source_object} is cropped away, deformed, or missing boundary parts")
        elif edit_type == "background_change":
            forbidden_after.append("the original background remains unchanged")
        elif edit_type == "spatial_move" and source_object and source_location:
            forbidden_after.append(f"{source_object} remains {source_location}")
        elif edit_type in {"attribute_change", "color_change", "material_change"}:
            source_descriptor = source_attribute or source_material or source_style
            if source_descriptor and (source_object or target_region):
                forbidden_after.append(f"{source_object or target_region} still has {source_descriptor}")
        elif edit_type == "local_enhancement" and (source_object or target_region):
            local_subject = source_object or target_region
            local_verb = "remain" if str(local_subject).lower().endswith("s") else "remains"
            forbidden_after.append(f"{local_subject} {local_verb} blurry or unchanged")
        elif edit_type == "style_transfer":
            forbidden_after.append("the image remains in the original visual style")
        elif edit_type == "action_change":
            action_subject = source_object or "the main subject"
            action_verb = "remain" if str(action_subject).lower().endswith("s") else "remains"
            forbidden_after.append(
                f"{action_subject} {action_verb} in the original pose with no action change"
            )
        elif edit_type == "compose":
            forbidden_after.append(
                "the image is unchanged or only one of the requested changes is applied"
            )
    elif edit_type in {"object_replacement", "object_removal"} and source_object:
        has_specific_source_forbidden = any(
            source_object.lower() in item.lower() and not _looks_generic_forbidden(item)
            for item in forbidden_after
        )
        if not has_specific_source_forbidden:
            forbidden_after.append(f"{source_object} remains visible {_region_phrase(target_region)}")
    elif edit_type == "object_addition" and target_object:
        target_specific = [
            item for item in forbidden_after if target_object.lower() in item.lower()
        ]
        forbidden_after = target_specific or [
            f"{target_object} is absent {_region_phrase(target_region)}"
        ]
    if edit_type == "object_replacement" and source_object:
        has_separate_visibility_forbidden = any(
            source_object.lower() in item.lower()
            and any(marker in item.lower() for marker in ("separate", "duplicate", "extra", "additional"))
            for item in forbidden_after
        )
        if not has_separate_visibility_forbidden:
            forbidden_after.append(f"a separate {source_object} remains visible {_region_phrase(target_region)}")

    normalized = {
        "edit_type": edit_type,
        "instruction": _clean_text(data.get("instruction") or instruction, max_len=512) or instruction,
        "target": target_alias or source_object,
        "replacement": replacement_alias or target_object,
        "source_object": source_object,
        "target_object": target_object,
        "source_attribute": source_attribute,
        "target_attribute": target_attribute,
        "source_material": source_material,
        "target_material": target_material,
        "source_style": source_style,
        "target_style": target_style,
        "source_location": source_location,
        "target_location": target_location,
        "target_region": target_region,
        "required_after": required_after,
        "forbidden_after": forbidden_after,
        "preserve": preserve,
        "difficulty": data.get("difficulty"),
    }
    return {key: value for key, value in normalized.items() if value not in (None, [], "")}


def proposal_definition_from_structured_edit(
    structured_edit: dict[str, Any],
    *,
    proposal_index: int,
    difficulty_level: int,
) -> ProposalDefinition:
    edit_type = normalize_edit_type(structured_edit.get("edit_type"))
    instruction = str(structured_edit.get("instruction", "")).strip()
    digest = hashlib.sha1(
        json.dumps(structured_edit, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:8]
    scope = "local" if edit_type in LOCAL_EDIT_TYPES else "global"
    expected_range = EXPECTED_CHANGE_FRACTIONS.get(edit_type, (0.04, 0.65))
    return ProposalDefinition(
        operation_id=f"learned_{edit_type}_{proposal_index:02d}_{digest}",
        instruction=instruction,
        family=edit_type,
        difficulty=max(1, int(structured_edit.get("difficulty") or difficulty_level)),
        scope=scope,
        metric="internal_prompt_gain",
        direction="increase",
        target=0.0,
        expected_changed_fraction=expected_range,
        verifier="internal_cepr_plus",
    )


def extract_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    candidate = text[start : end + 1]
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def structured_edit_prompt(difficulty_level: int, proposals_per_image: int) -> str:
    return (
        "Inspect the image and generate image-grounded edit instructions for self-training an image "
        "editing model. Return compact JSON only, with no markdown and no prose outside JSON, in this "
        "exact outer form: {\"proposals\": [ ... ]}. Return exactly "
        f"{proposals_per_image} proposals unless the requested edit type is not visually feasible; in "
        "that case return {\"proposals\": []}. "
        f"Target difficulty level: {difficulty_level}. The edits must be useful but feasible: prefer "
        "localized, checkable edits with one explicit visible target, such as object replacement, object "
        "removal, object addition, object color/material/attribute changes, and spatial moves when "
        "visually plausible. Do not propose trivial brightness-only changes, text editing, face or "
        "identity changes, invisible objects, benchmark-specific references, multiple unrelated edits, "
        "or impossible scene rewrites. Avoid broad whole-background replacement of sky, snow, ground, "
        "walls, or scenery unless the unchanged subject can clearly be preserved. "
        "Each proposal object must use this schema: edit_type, instruction, target_region, preserve, "
        "required_after, forbidden_after, difficulty, plus applicable fields from source_object, "
        "target_object, replacement, source_attribute, target_attribute, source_material, "
        "target_material, source_style, target_style, source_location, target_location. The instruction "
        "must be one concrete sentence grounded in visible image content. target_region must be a "
        "spatial phrase anchored to stable visible context, such as 'on the table', 'left of the "
        "person', 'above the airplane', or 'beside the red chair'; do not use generic regions like "
        "'the original location', 'same place', or 'main visible target' when a visual anchor exists. "
        "preserve must list 3 to 6 specific non-target elements or properties to keep unchanged. "
        "required_after must list visual checks proving the requested edit happened. forbidden_after "
        "must list visual failure checks that should be absent; do not use generic phrases like "
        "'no other changes', and do not phrase forbidden_after items as negative sentences starting "
        "with 'no', 'not', or 'without'. Write the bad visible state directly, such as 'the old cup "
        "remains on the table' or 'the cucumber slice is missing next to the lemon'. "
        "For object_removal: source_object is required and must be a visible small or secondary object, "
        "not the main person, animal, vehicle, large furniture item, or whole background. The instruction "
        "must explicitly remove that object completely and naturally fill/inpaint the surrounding area. "
        "required_after must mention the filled area. forbidden_after must mention the source object, "
        "remnants, shadows, or duplicates remaining visible. If the target is a sign, poster, book, "
        "label, or other text-bearing object, name the object generically, such as 'ceiling sign' or "
        "'poster on the wall'; do not copy exact visible text or quoted strings into source_object, "
        "target_region, required_after, or forbidden_after. Do not propose removing only written text, "
        "words, letters, captions, or labels; remove a physical object only when it is safe to remove "
        "the whole object. "
        "For object_replacement: source_object is the visible old object and target_object/replacement "
        "is the concrete new object from a different object category/head noun. Do not replace an object "
        "with a variant of the same class, such as bench to park bench, cup to mug, bow tie to tie, or "
        "red apple to green apple; those are attribute/color edits, not object replacement. Replace only "
        "one small or secondary object at the same location and approximate scale. required_after must "
        "mention the new object in the anchored region. "
        "forbidden_after must mention the old object remaining or both old and new objects appearing. "
        "For object_addition: target_object/replacement and an anchored plausible empty target_region "
        "are required; do not place the new object on top of an important existing subject. "
        "required_after must mention the new object in the target_region. forbidden_after must "
        "mention the new object missing, implausibly placed, duplicated, or occluding the target; "
        "do not write preserve objects as forbidden successes. "
        "For subject_extraction: source_object is required and must be one visible foreground subject "
        "or bounded object. The instruction must isolate that source_object from the background onto a "
        "clean neutral or transparent-looking backdrop while preserving its identity, pose, shape, and "
        "fine boundary details. required_after must mention the isolated subject and removed background. "
        "forbidden_after must mention the original surrounding scene remaining around the subject, "
        "missing subject parts, deformed boundaries, or extra unrelated objects. Do not use "
        "subject_extraction for object removal, "
        "background replacement, cropping only, or whole-scene style transfer. "
        "For background_change: target_region must be the background/backdrop behind a visible subject, "
        "required_after must mention the new background, and forbidden_after must only mention the "
        "old/original surrounding scene remaining. Put foreground subject preservation requirements only in "
        "preserve, not in forbidden_after; do not write failures such as 'the dog is missing' or "
        "'the person's legs are not visible' as forbidden_after items. "
        "For spatial_move: source_object, source_location, target_location, and target_region are "
        "required; source_object must be the bounded visible object being moved, and target_region "
        "must be anchored to stable visible context. Make a small local move inside the image while "
        "preserving the object identity. "
        "For color_change, material_change, attribute_change, or local_enhancement: source_object is "
        "required and must be the bounded visible object or object part being edited. target_region "
        "is also required and must spatially locate that same source_object against visible context, "
        "for example 'on the dog's face', 'near the plate edge', or 'held by the woman'. The changed "
        "attribute/material/color/detail must be named explicitly. Use concise concrete nouns. "
        "For attribute_change, source_attribute and target_attribute are required and must name two "
        "concrete, visibly distinct non-color states; never use placeholders such as 'different', "
        "'another', 'new', or 'changed'. For color_change, source_attribute and target_attribute are "
        "required and must name the current and requested colors explicitly. For material_change, "
        "source_material and target_material are required and must be visibly distinct physical "
        "materials such as glass, wood, metal, ceramic, leather, fabric, rubber, marble, stone, paper, "
        "cardboard, or plastic. Do not propose changing an object to its existing or equivalent "
        "material, and do not use material_change to turn the source into another object, food, animal, "
        "or category. Do not list the old source state among preserve constraints. "
        "For style_transfer: target_region must be whole image, entire image, whole scene, or full "
        "scene, and preserve must name the major objects/layout to keep unchanged."
    )
