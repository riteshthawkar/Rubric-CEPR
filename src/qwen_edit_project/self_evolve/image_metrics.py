from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageChops, ImageFilter, ImageStat


def prepare_rgb(image: Image.Image, size: tuple[int, int] = (128, 128)) -> Image.Image:
    return image.convert("RGB").resize(size)


def luminance_mean(image: Image.Image) -> float:
    gray = prepare_rgb(image).convert("L")
    return ImageStat.Stat(gray).mean[0] / 255.0


def luminance_std(image: Image.Image) -> float:
    gray = prepare_rgb(image).convert("L")
    return ImageStat.Stat(gray).stddev[0] / 255.0


def saturation_mean(image: Image.Image) -> float:
    hsv = prepare_rgb(image).convert("HSV")
    return ImageStat.Stat(hsv).mean[1] / 255.0


def warmth_score(image: Image.Image) -> float:
    red, _, blue = prepare_rgb(image).split()
    red_mean = ImageStat.Stat(red).mean[0] / 255.0
    blue_mean = ImageStat.Stat(blue).mean[0] / 255.0
    return red_mean - blue_mean


def changed_fraction(
    original: Image.Image,
    edited: Image.Image,
    diff_threshold: int = 18,
) -> float:
    original_rgb = prepare_rgb(original)
    edited_rgb = prepare_rgb(edited)
    diff = ImageChops.difference(original_rgb, edited_rgb).convert("L")
    histogram = diff.histogram()
    total = sum(histogram)
    changed = sum(histogram[diff_threshold + 1 :])
    if total == 0:
        return 0.0
    return changed / total


def edge_preservation_score(original: Image.Image, edited: Image.Image) -> float:
    orig_edges = prepare_rgb(original).convert("L").filter(ImageFilter.FIND_EDGES)
    edit_edges = prepare_rgb(edited).convert("L").filter(ImageFilter.FIND_EDGES)
    diff = ImageChops.difference(orig_edges, edit_edges)
    edge_delta = ImageStat.Stat(diff).mean[0] / 255.0
    return max(0.0, 1.0 - edge_delta * 2.0)


def mean_absolute_difference(original: Image.Image, edited: Image.Image) -> float:
    original_arr = np.asarray(prepare_rgb(original), dtype=np.float32)
    edited_arr = np.asarray(prepare_rgb(edited), dtype=np.float32)
    return float(np.abs(original_arr - edited_arr).mean() / 255.0)


def _resize_bool_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    mask_arr = np.asarray(mask, dtype=np.uint8)
    if mask_arr.ndim != 2:
        raise ValueError("target mask must be a 2D array")
    width, height = size
    if mask_arr.shape == (height, width):
        return mask_arr.astype(bool)
    pil_mask = Image.fromarray(mask_arr * 255, mode="L").resize(size, Image.NEAREST)
    return np.asarray(pil_mask, dtype=np.uint8) > 0


def dilate_bool_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    mask_arr = np.asarray(mask, dtype=np.uint8)
    if mask_arr.ndim != 2:
        raise ValueError("target mask must be a 2D array")
    if radius <= 0 or not mask_arr.any():
        return mask_arr.astype(bool)
    kernel_size = 2 * int(radius) + 1
    pil_mask = Image.fromarray(mask_arr * 255, mode="L").filter(ImageFilter.MaxFilter(kernel_size))
    return np.asarray(pil_mask, dtype=np.uint8) > 0


def box_mask_from_boxes(
    image_size: tuple[int, int],
    boxes: list[tuple[float, float, float, float]] | tuple[tuple[float, float, float, float], ...],
    size: tuple[int, int] = (128, 128),
    padding_fraction: float = 0.03,
) -> np.ndarray:
    """Rasterize xyxy boxes from image coordinates into a low-resolution mask."""

    image_width, image_height = image_size
    metric_width, metric_height = size
    mask = np.zeros((metric_height, metric_width), dtype=bool)
    if image_width <= 0 or image_height <= 0:
        return mask
    pad_x = max(0.0, float(padding_fraction)) * metric_width
    pad_y = max(0.0, float(padding_fraction)) * metric_height
    scale_x = metric_width / float(image_width)
    scale_y = metric_height / float(image_height)
    for raw_box in boxes:
        if raw_box is None or len(raw_box) != 4:
            continue
        x0, y0, x1, y1 = (float(value) for value in raw_box)
        if not all(math.isfinite(value) for value in (x0, y0, x1, y1)):
            continue
        left = min(x0, x1) * scale_x - pad_x
        right = max(x0, x1) * scale_x + pad_x
        top = min(y0, y1) * scale_y - pad_y
        bottom = max(y0, y1) * scale_y + pad_y
        x_start = max(0, int(math.floor(left)))
        x_end = min(metric_width, int(math.ceil(right)))
        y_start = max(0, int(math.floor(top)))
        y_end = min(metric_height, int(math.ceil(bottom)))
        if x_end <= x_start or y_end <= y_start:
            continue
        mask[y_start:y_end, x_start:x_end] = True
    return mask


def diff_mask(
    original: Image.Image,
    edited: Image.Image,
    diff_threshold: int = 18,
    size: tuple[int, int] = (128, 128),
    dilation_radius: int = 1,
) -> np.ndarray:
    original_arr = np.asarray(prepare_rgb(original, size), dtype=np.float32)
    edited_arr = np.asarray(prepare_rgb(edited, size), dtype=np.float32)
    diff = np.abs(original_arr - edited_arr).mean(axis=2)
    mask = diff > float(diff_threshold)
    return dilate_bool_mask(mask, dilation_radius)


def masked_region_statistics(
    original: Image.Image,
    edited: Image.Image,
    target_mask: np.ndarray,
    diff_threshold: int = 18,
    size: tuple[int, int] = (128, 128),
) -> dict[str, float]:
    """Measure target-region edit support and non-target preservation.

    The target mask should represent the *intended* editable region, not the
    observed changed pixels. This makes outside corruption visible instead of
    allowing the metric to explain it away as part of the edit.
    """

    original_arr = np.asarray(prepare_rgb(original, size), dtype=np.float32) / 255.0
    edited_arr = np.asarray(prepare_rgb(edited, size), dtype=np.float32) / 255.0
    target = _resize_bool_mask(target_mask, size)
    outside = ~target
    diff_rgb = np.abs(original_arr - edited_arr)
    diff = diff_rgb.mean(axis=2)
    changed = diff > (float(diff_threshold) / 255.0)
    target_area = float(target.mean())
    outside_area = float(outside.mean())
    if target.any():
        target_change = float(diff[target].mean())
        target_changed_fraction = float(changed[target].mean())
        target_mse = float(np.square(diff_rgb[target]).mean())
    else:
        target_change = 0.0
        target_changed_fraction = 0.0
        target_mse = 0.0
    if outside.any():
        outside_change = float(diff[outside].mean())
        outside_changed_fraction = float(changed[outside].mean())
        outside_mse = float(np.square(diff_rgb[outside]).mean())
    else:
        outside_change = float(diff.mean())
        outside_changed_fraction = float(changed.mean())
        outside_mse = float(np.square(diff_rgb).mean())
    target_changed_pixels = float(np.logical_and(changed, target).sum())
    outside_changed_pixels = float(np.logical_and(changed, outside).sum())
    changed_pixels = target_changed_pixels + outside_changed_pixels
    localization_precision = (
        target_changed_pixels / changed_pixels
        if changed_pixels > 0.0
        else 1.0
    )
    outside_psnr = 60.0 if outside_mse <= 1.0e-12 else min(60.0, -10.0 * math.log10(outside_mse))
    target_psnr = 60.0 if target_mse <= 1.0e-12 else min(60.0, -10.0 * math.log10(target_mse))
    return {
        "target_area_fraction": target_area,
        "outside_area_fraction": outside_area,
        "changed_fraction": float(changed.mean()),
        "target_change": target_change,
        "outside_change": outside_change,
        "target_changed_fraction": target_changed_fraction,
        "outside_changed_fraction": outside_changed_fraction,
        "target_changed_pixel_fraction": target_changed_pixels / max(float(target.size), 1.0),
        "outside_changed_pixel_fraction": outside_changed_pixels / max(float(target.size), 1.0),
        "localization_precision": float(localization_precision),
        "target_mse": target_mse,
        "outside_mse": outside_mse,
        "target_psnr": target_psnr,
        "outside_psnr": outside_psnr,
    }


def masked_target_structure_preservation(
    original: Image.Image,
    edited: Image.Image,
    target_mask: np.ndarray,
    size: tuple[int, int] = (128, 128),
) -> dict[str, float]:
    """Measure whether an attribute edit preserves target-region structure.

    The score is intentionally invariant to global color/brightness shifts by
    combining target luminance correlation with gradient-magnitude similarity.
    It is meant for color/material/local enhancement edits, not for edits that
    legitimately move, remove, or replace objects.
    """

    original_gray = np.asarray(prepare_rgb(original, size).convert("L"), dtype=np.float32) / 255.0
    edited_gray = np.asarray(prepare_rgb(edited, size).convert("L"), dtype=np.float32) / 255.0
    target = _resize_bool_mask(target_mask, size)
    if not target.any():
        return {
            "target_structure_preservation": 1.0,
            "target_structure_luminance_correlation": 1.0,
            "target_structure_edge_similarity": 1.0,
            "target_structure_edge_delta": 0.0,
            "target_structure_mean_edge": 0.0,
        }

    orig_values = original_gray[target]
    edit_values = edited_gray[target]
    orig_std = float(orig_values.std())
    edit_std = float(edit_values.std())
    if orig_std < 1.0e-6 or edit_std < 1.0e-6:
        correlation_score = 1.0
    else:
        corr = float(np.corrcoef(orig_values, edit_values)[0, 1])
        if not math.isfinite(corr):
            corr = 0.0
        correlation_score = max(0.0, min(1.0, 0.5 * (corr + 1.0)))

    orig_grad_y, orig_grad_x = np.gradient(original_gray)
    edit_grad_y, edit_grad_x = np.gradient(edited_gray)
    orig_mag = np.sqrt(orig_grad_x * orig_grad_x + orig_grad_y * orig_grad_y)
    edit_mag = np.sqrt(edit_grad_x * edit_grad_x + edit_grad_y * edit_grad_y)
    edge_delta = float(np.abs(orig_mag - edit_mag)[target].mean())
    mean_edge = float((orig_mag[target].mean() + edit_mag[target].mean()) * 0.5)
    edge_similarity = 1.0 - edge_delta / max(mean_edge + edge_delta, 1.0e-6)
    edge_similarity = max(0.0, min(1.0, edge_similarity))
    structure = math.sqrt(max(correlation_score * edge_similarity, 0.0))
    return {
        "target_structure_preservation": float(max(0.0, min(1.0, structure))),
        "target_structure_luminance_correlation": float(correlation_score),
        "target_structure_edge_similarity": float(edge_similarity),
        "target_structure_edge_delta": edge_delta,
        "target_structure_mean_edge": mean_edge,
    }


def diff_region_statistics(
    original: Image.Image,
    edited: Image.Image,
    diff_threshold: int = 18,
) -> dict[str, float]:
    original_arr = np.asarray(prepare_rgb(original), dtype=np.float32)
    edited_arr = np.asarray(prepare_rgb(edited), dtype=np.float32)
    diff = np.abs(original_arr - edited_arr).mean(axis=2) / 255.0
    threshold = diff_threshold / 255.0
    mask = diff > threshold

    changed_fraction_value = float(mask.mean())
    if mask.any():
        inside_mean = float(diff[mask].mean())
        ys, xs = np.where(mask)
        bbox_area = float((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))
        compactness = float(mask.sum() / max(bbox_area, 1.0))
    else:
        inside_mean = 0.0
        compactness = 0.0

    if (~mask).any():
        outside_mean = float(diff[~mask].mean())
    else:
        outside_mean = inside_mean

    total_mean = float(diff.mean())
    precision = float(inside_mean / max(inside_mean + outside_mean, 1e-6))
    return {
        "changed_fraction": changed_fraction_value,
        "inside_mean_delta": inside_mean,
        "outside_mean_delta": outside_mean,
        "total_mean_delta": total_mean,
        "mask_compactness": compactness,
        "mask_precision": precision,
    }
