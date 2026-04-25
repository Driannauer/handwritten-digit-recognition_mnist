"""
Predict local handwritten digit images with the saved TensorFlow/Keras model.

This file is intentionally independent from train.py:
- train.py trains and saves models/best_digit_model.keras.
- test.py loads that saved model and predicts local handwritten images.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import keras
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from scipy.ndimage import center_of_mass, find_objects, gaussian_filter, label, shift


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_DIR = PROJECT_DIR / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "best_digit_model.keras"
KNOWN_EXPECTED_DEFAULTS = {
    "t1.jpg": 6,
    "t2.jpg": 1,
    "t3.jpg": 8,
    "t4.jpg": 9,
    "t5.jpg": 7,
    "t6.jpg": 2,
    "t7.jpg": 0,
    "t8.jpg": 5,
    "t9.jpg": 3,
    "t10.jpg": 4,
}


@dataclass
class PreprocessCandidate:
    processed: np.ndarray
    rotation_degrees: int
    orientation_score: float
    threshold: int
    quality_score: float
    preprocessing_adjustment: str | None = None


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)


def clear_prediction_outputs() -> None:
    generated_names = {
        "prediction_summary.json",
        "prediction_summary.txt",
    }
    for path in OUTPUT_DIR.iterdir():
        if not path.is_file():
            continue
        if path.name.startswith(("07_", "08_")) or path.name in generated_names:
            path.unlink()


def natural_sort_key(path: Path):
    parts = re.split(r"(\d+)", path.stem.lower())
    key = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part)
    key.append(path.suffix.lower())
    return key


def discover_default_images() -> list[str]:
    candidates = []
    for suffix in ("jpg", "jpeg", "png", "bmp"):
        candidates.extend(PROJECT_DIR.glob(f"t*.{suffix}"))
    unique_paths = sorted({path.resolve() for path in candidates}, key=natural_sort_key)
    return [path.name for path in unique_paths]


def default_expected_items() -> list[str]:
    items = []
    for file_name, label_value in KNOWN_EXPECTED_DEFAULTS.items():
        if (PROJECT_DIR / file_name).exists():
            items.append(f"{file_name}={label_value}")
    return items


def parse_expected_labels(items: list[str]) -> dict[str, int]:
    expected: dict[str, int] = {}
    for item in items:
        if "=" not in item:
            continue
        name, label_text = item.split("=", 1)
        try:
            label_value = int(label_text.strip())
        except ValueError:
            print(f"Ignore invalid expected label setting: {item}")
            continue
        if 0 <= label_value <= 9:
            expected[name.strip()] = label_value
        else:
            print(f"Ignore out-of-range expected label setting: {item}")
    return expected


def resolve_image_paths(image_names: list[str]) -> list[Path]:
    paths: list[Path] = []
    for image_name in image_names:
        path = Path(image_name)
        if not path.is_absolute():
            path = PROJECT_DIR / path
        if path.exists():
            paths.append(path)
        else:
            print(f"Image not found, skipped: {image_name}")
    return sorted(paths, key=natural_sort_key)


def resolve_model_path(model_path: str) -> Path:
    path = Path(model_path)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path


def open_exif_corrected_image(image_path: Path, mode: str = "RGB") -> Image.Image:
    return ImageOps.exif_transpose(Image.open(image_path)).convert(mode)


def score_processed_digit_quality(processed: np.ndarray) -> float:
    mask = processed > 0.05
    pixel_count = int(mask.sum())
    if pixel_count < 8:
        return 0.0

    y_coords, x_coords = np.nonzero(mask)
    height = int(y_coords.max() - y_coords.min() + 1)
    width = int(x_coords.max() - x_coords.min() + 1)
    if height < 4 or width < 2:
        return 0.0

    bbox_area = max(1, height * width)
    fill_ratio = pixel_count / bbox_area

    size_score = min(1.0, max(0.2, bbox_area / 160))
    if fill_ratio < 0.10:
        fill_score = max(0.25, fill_ratio / 0.10)
    elif fill_ratio <= 0.62:
        fill_score = 1.0
    else:
        fill_score = max(0.18, 1.0 - (fill_ratio - 0.62) / 0.30)

    aspect_ratio = max(height, width) / max(1, min(height, width))
    aspect_score = 0.85 + 0.15 * min(1.0, aspect_ratio / 4)
    if height >= width * 1.8:
        aspect_score *= 1.08

    margin = min(
        int(y_coords.min()),
        int(x_coords.min()),
        27 - int(y_coords.max()),
        27 - int(x_coords.max()),
    )
    border_score = 0.75 if margin <= 0 else 1.0

    return float(np.clip(size_score * fill_score * aspect_score * border_score, 0, 1))


def candidate_is_blank(candidate: PreprocessCandidate) -> bool:
    return candidate.quality_score <= 0 or candidate.processed.sum() < 0.5


def empty_preprocess_candidate(rotation_degrees: int, threshold: int) -> PreprocessCandidate:
    return PreprocessCandidate(
        processed=np.zeros((28, 28), dtype=np.float32),
        rotation_degrees=rotation_degrees,
        orientation_score=0.0,
        threshold=threshold,
        quality_score=0.0,
    )


def recenter_processed_digit(processed: np.ndarray) -> np.ndarray:
    mass = center_of_mass(processed)
    if not np.isnan(mass[0]) and not np.isnan(mass[1]):
        processed = shift(
            processed,
            shift=(13.5 - mass[0], 13.5 - mass[1]),
            order=1,
            mode="constant",
            cval=0.0,
        )
    return np.clip(processed, 0, 1)


def digit_shape_features(processed: np.ndarray) -> dict[str, int]:
    mask = processed > 0.05
    y_coords, x_coords = np.nonzero(mask)
    if len(y_coords) == 0 or len(x_coords) == 0:
        return {
            "width": 0,
            "height": 0,
            "top_row": 0,
            "top_width": 0,
            "bottom_width": 0,
            "upper_pixels": 0,
            "lower_pixels": 0,
        }

    y_min = int(y_coords.min())
    y_max = int(y_coords.max())
    x_min = int(x_coords.min())
    x_max = int(x_coords.max())
    top_band = mask[y_min : min(y_min + 5, mask.shape[0])]
    bottom_band = mask[max(0, y_max - 4) : y_max + 1]

    return {
        "width": x_max - x_min + 1,
        "height": y_max - y_min + 1,
        "top_row": y_min,
        "top_width": int(top_band.any(axis=0).sum()),
        "bottom_width": int(bottom_band.any(axis=0).sum()),
        "upper_pixels": int(mask[:14].sum()),
        "lower_pixels": int(mask[14:].sum()),
    }


def looks_like_top_bar_seven(processed: np.ndarray) -> bool:
    return looks_like_top_bar_seven_features(digit_shape_features(processed))


def looks_like_top_bar_seven_features(features: dict[str, int]) -> bool:
    return (
        features["width"] >= 8
        and features["height"] >= 18
        and features["top_width"] >= max(7, int(np.ceil(features["width"] * 0.75)))
        and features["top_width"] >= features["bottom_width"] + 5
        and features["upper_pixels"] >= 0.8 * max(1, features["lower_pixels"])
    )


def enhance_top_bar_seven(processed: np.ndarray) -> tuple[np.ndarray, str | None]:
    features = digit_shape_features(processed)
    if not looks_like_top_bar_seven_features(features):
        return processed, None

    enhanced = processed.copy()
    top = features["top_row"]
    enhanced[top : top + 4] = np.clip(enhanced[top : top + 4] * 1.8, 0, 1)
    return recenter_processed_digit(enhanced), "top_bar_boost"


def preprocess_digit_array(
    rgb: np.ndarray,
    rotation_degrees: int,
    sigma: int,
    threshold: int,
) -> PreprocessCandidate:
    """
    Convert one oriented handwritten photo into a 28x28 MNIST-style image.

    The model was trained on bright digits over a dark background, so this step
    extracts dark handwriting from photos, crops the main digit, and recenters it.
    """
    bright = rgb.max(axis=2)
    background = gaussian_filter(bright, sigma=sigma)
    ink = np.clip(background - bright, 0, None)
    if ink.max() > 0:
        ink = ink / ink.max() * 255

    mask = ink > threshold
    labels, _ = label(mask)
    best_score = None
    best_slice = None

    for component_id, component_slice in enumerate(find_objects(labels), start=1):
        if component_slice is None:
            continue
        ys, xs = component_slice
        component = labels[component_slice] == component_id
        area = int(component.sum())
        if area < 30:
            continue

        height = ys.stop - ys.start
        width = xs.stop - xs.start
        image_height, image_width = rgb.shape[:2]
        aspect_ratio = max(height, width) / max(1, min(height, width))
        if aspect_ratio > 12:
            continue
        if (
            ys.start <= 2
            or xs.start <= 2
            or ys.stop >= image_height - 2
            or xs.stop >= image_width - 2
        ) and aspect_ratio > 4:
            continue

        fill_ratio = area / max(1, height * width)
        stroke_density_bonus = min(1.0, fill_ratio / 0.18)
        if fill_ratio > 0.58:
            stroke_density_bonus *= max(0.2, 1.0 - (fill_ratio - 0.58) / 0.32)

        center_y = (ys.start + ys.stop) / 2
        center_x = (xs.start + xs.stop) / 2
        center_bonus = 1 / (
            1
            + ((center_x - image_width / 2) / (image_width / 2)) ** 2
            + ((center_y - image_height / 2) / (image_height / 2)) ** 2
        )
        height_width_ratio = height / max(1, width)
        if height_width_ratio >= 1:
            shape_bonus = min(1.45, 0.70 + 0.18 * height_width_ratio)
        else:
            shape_bonus = max(0.35, height_width_ratio)

        ink_strength = float(ink[component_slice][component].sum())
        score = ink_strength * stroke_density_bonus * center_bonus * shape_bonus

        if best_score is None or score > best_score:
            best_score = score
            best_slice = component_slice

    if best_slice is None:
        return empty_preprocess_candidate(rotation_degrees, threshold)

    ys, xs = best_slice
    crop = ink[ys, xs]
    crop_mask = crop > threshold
    y_coords, x_coords = np.nonzero(crop_mask)
    if len(y_coords) == 0 or len(x_coords) == 0:
        return empty_preprocess_candidate(rotation_degrees, threshold)

    crop = crop[
        y_coords.min() : y_coords.max() + 1,
        x_coords.min() : x_coords.max() + 1,
    ]
    crop = np.where(crop > threshold, crop, 0).astype(np.float32)

    # Weakly suppress wide, low-contrast rows. These are often notebook guide lines.
    row_mean = crop.mean(axis=1)
    row_std = crop.std(axis=1)
    for idx in range(crop.shape[0]):
        if row_mean[idx] > row_mean.mean() + 0.5 * row_mean.std() and row_std[idx] < 25:
            crop[idx] *= 0.15

    side = max(crop.shape) + 12
    canvas = np.zeros((side, side), dtype=np.float32)
    top = (side - crop.shape[0]) // 2
    left = (side - crop.shape[1]) // 2
    canvas[top : top + crop.shape[0], left : left + crop.shape[1]] = crop

    image = (
        Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
        .filter(ImageFilter.MaxFilter(3))
        .resize((28, 28), Image.Resampling.LANCZOS)
    )
    arr = np.array(image).astype(np.float32) / 255.0

    processed = recenter_processed_digit(arr)
    processed, preprocessing_adjustment = enhance_top_bar_seven(processed)
    return PreprocessCandidate(
        processed=processed,
        rotation_degrees=rotation_degrees,
        orientation_score=float(best_score or 0.0),
        threshold=threshold,
        quality_score=score_processed_digit_quality(processed),
        preprocessing_adjustment=preprocessing_adjustment,
    )


def preprocess_user_digit_candidates(
    image_path: Path,
    sigma: int = 18,
    threshold: int = 30,
) -> list[PreprocessCandidate]:
    base_image = open_exif_corrected_image(image_path, "RGB")
    candidates = []
    for rotation_degrees in (0, 90, 270, 180):
        oriented = base_image.rotate(rotation_degrees, expand=True)
        rgb = np.array(oriented).astype(np.float32)
        candidates.append(
            preprocess_digit_array(
                rgb,
                rotation_degrees=rotation_degrees,
                sigma=sigma,
                threshold=threshold,
            )
        )
    return candidates


def predict_probabilities(model: keras.Model, processed: np.ndarray) -> np.ndarray:
    if len(model.input_shape) == 4:
        return model.predict(processed[None, ..., None], verbose=0)[0]
    return model.predict(processed[None], verbose=0)[0]


def append_preprocessing_adjustment(
    current_adjustment: str | None,
    new_adjustment: str,
) -> str:
    if not current_adjustment:
        return new_adjustment
    if new_adjustment in current_adjustment.split("+"):
        return current_adjustment
    return f"{current_adjustment}+{new_adjustment}"


def predict_probabilities_batch(
    model: keras.Model,
    processed_items: list[np.ndarray],
) -> np.ndarray:
    batch = np.stack(processed_items)
    if len(model.input_shape) == 4:
        return model.predict(batch[..., None], verbose=0)
    return model.predict(batch, verbose=0)


def choose_candidate_for_model(
    model: keras.Model,
    candidates: list[PreprocessCandidate],
) -> PreprocessCandidate:
    usable_candidates = [item for item in candidates if not candidate_is_blank(item)]
    if not usable_candidates:
        return candidates[0]

    max_orientation_score = max(
        (item.orientation_score for item in usable_candidates),
        default=0.0,
    )
    if max_orientation_score <= 0:
        return usable_candidates[0]

    probability_rows = predict_probabilities_batch(
        model,
        [item.processed for item in usable_candidates],
    )

    original_index = next(
        (
            index
            for index, item in enumerate(usable_candidates)
            if item.rotation_degrees == 0 and item.threshold == 30
        ),
        0,
    )
    original_candidate = usable_candidates[original_index]
    original_probabilities = probability_rows[original_index]
    original_confidence = float(np.max(original_probabilities))
    if (
        original_confidence >= 0.85
        and original_candidate.orientation_score >= 0.7 * max_orientation_score
        and original_candidate.quality_score >= 0.45
    ):
        return original_candidate

    best_candidate = usable_candidates[0]
    best_score = -1.0
    best_confidence = 0.0
    for candidate, probabilities in zip(usable_candidates, probability_rows):
        confidence = float(np.max(probabilities))
        orientation_weight = candidate.orientation_score / max_orientation_score
        combined_score = confidence * (
            0.35 + 0.30 * orientation_weight + 0.35 * candidate.quality_score
        )
        if combined_score > best_score:
            best_score = combined_score
            best_confidence = confidence
            best_candidate = candidate

    if (
        best_candidate.rotation_degrees == 180
        and original_candidate.orientation_score >= 0.95 * max_orientation_score
        and original_confidence >= 0.30
        and best_confidence <= 0.75
    ):
        return original_candidate

    return best_candidate


def refine_low_confidence_candidate(
    model: keras.Model,
    image_path: Path,
    candidate: PreprocessCandidate,
) -> PreprocessCandidate:
    probabilities = predict_probabilities(model, candidate.processed)
    predicted_label = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_label])
    if confidence >= 0.60:
        return candidate

    base_image = open_exif_corrected_image(image_path, "RGB")
    oriented = base_image.rotate(candidate.rotation_degrees, expand=True)
    refined = preprocess_digit_array(
        np.array(oriented).astype(np.float32),
        rotation_degrees=candidate.rotation_degrees,
        sigma=30,
        threshold=candidate.threshold,
    )
    refined_probabilities = predict_probabilities(model, refined.processed)
    refined_label = int(np.argmax(refined_probabilities))
    refined_confidence = float(refined_probabilities[refined_label])

    if refined_label == predicted_label and refined_confidence >= confidence + 0.20:
        refined.preprocessing_adjustment = append_preprocessing_adjustment(
            refined.preprocessing_adjustment,
            "soft_background",
        )
        return refined

    return candidate


def save_preprocessed_image(processed: np.ndarray, image_path: Path, file_index: int) -> Path:
    save_path = OUTPUT_DIR / f"{file_index:02d}_{image_path.stem}_preprocessed_28x28.png"
    enlarged = Image.fromarray((processed * 255).astype("uint8")).resize(
        (280, 280), Image.Resampling.NEAREST
    )
    enlarged.save(save_path)
    return save_path


def save_prediction_figure(
    model: keras.Model,
    image_path: Path,
    processed: np.ndarray,
    expected_label: int | None,
    file_index: int,
    rotation_degrees: int = 0,
    threshold: int = 30,
    preprocessing_adjustment: str | None = None,
) -> dict[str, object]:
    probabilities = predict_probabilities(model, processed)
    predicted_label = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_label])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    original = open_exif_corrected_image(image_path, "L")
    if rotation_degrees:
        original = original.rotate(rotation_degrees, expand=True)
    axes[0].imshow(original, cmap="gray")
    original_title = f"Original: {image_path.name}"
    if rotation_degrees:
        original_title += f"\nAuto-rotated: {rotation_degrees} deg"
    if threshold != 30:
        original_title += f"\nAuto-threshold: {threshold}"
    if preprocessing_adjustment:
        original_title += f"\nPreprocess: {preprocessing_adjustment}"
    axes[0].set_title(original_title)
    axes[0].axis("off")

    axes[1].imshow(processed, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Processed 28x28")
    axes[1].axis("off")

    axes[2].bar(np.arange(10), probabilities, color="#4062bb")
    axes[2].set_xticks(np.arange(10))
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel("Digit")
    axes[2].set_ylabel("Probability")
    title = f"Predict: {predicted_label} ({confidence:.2%})"
    if expected_label is not None:
        title += f"\nExpected: {expected_label}"
    axes[2].set_title(title)

    plt.tight_layout()
    figure_path = OUTPUT_DIR / f"{file_index:02d}_{image_path.stem}_prediction.png"
    plt.savefig(figure_path, dpi=160)
    plt.close(fig)

    return {
        "image": image_path.name,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "expected_label": expected_label,
        "auto_rotation_degrees": rotation_degrees,
        "preprocess_threshold": threshold,
        "preprocessing_adjustment": preprocessing_adjustment,
        "probabilities": [float(x) for x in probabilities],
        "figure": figure_path.name,
    }


def save_prediction_reports(
    model_path: Path,
    user_predictions: list[dict[str, object]],
) -> None:
    try:
        display_model_path = model_path.relative_to(PROJECT_DIR)
    except ValueError:
        display_model_path = model_path

    report_path = OUTPUT_DIR / "prediction_summary.txt"
    lines = [
        "Handwritten Digit Recognition - Prediction Summary",
        "=" * 56,
        "",
        f"Loaded model: {display_model_path}",
        "",
        "Local handwritten image predictions:",
    ]

    known_items = [item for item in user_predictions if item["expected_label"] is not None]
    correct_count = sum(
        1
        for item in known_items
        if item["predicted_label"] == item["expected_label"]
    )

    for item in user_predictions:
        expected = item["expected_label"]
        expected_text = "unknown" if expected is None else str(expected)
        rotation = int(item.get("auto_rotation_degrees", 0))
        rotation_text = "none" if rotation == 0 else f"{rotation} degrees"
        threshold = int(item.get("preprocess_threshold", 30))
        adjustment = item.get("preprocessing_adjustment")
        adjustment_text = "" if not adjustment else f", preprocess {adjustment}"
        lines.append(
            "- {image}: predicted {predicted_label}, expected {expected}, "
            "confidence {confidence:.2%}, auto-rotation {rotation}, "
            "threshold {threshold}{adjustment}".format(
                image=item["image"],
                predicted_label=item["predicted_label"],
                expected=expected_text,
                confidence=item["confidence"],
                rotation=rotation_text,
                threshold=threshold,
                adjustment=adjustment_text,
            )
        )

    if known_items:
        lines.extend(
            [
                "",
                f"Known-label local accuracy: {correct_count}/{len(known_items)}",
            ]
        )

    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")

    json_path = OUTPUT_DIR / "prediction_summary.json"
    json_path.write_text(
        json.dumps(
            {
                "loaded_model": str(display_model_path),
                "known_label_accuracy": {
                    "correct": int(correct_count),
                    "total": int(len(known_items)),
                },
                "user_predictions": user_predictions,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load the saved model and predict local handwritten images."
    )
    parser.add_argument(
        "--images",
        nargs="*",
        default=discover_default_images(),
        help="Local handwritten image paths to predict. Default: all t*.jpg/png/bmp in this folder.",
    )
    parser.add_argument(
        "--expected",
        nargs="*",
        default=default_expected_items(),
        help="Optional expected labels, for example: t1.jpg=6 t2.jpg=1",
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Saved model path.",
    )
    args = parser.parse_args()

    ensure_output_dir()
    clear_prediction_outputs()

    model_path = resolve_model_path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved model not found: {model_path}. Run python train.py once first."
        )

    expected_labels = parse_expected_labels(args.expected)
    image_paths = resolve_image_paths(args.images)
    model = keras.models.load_model(model_path)

    user_predictions: list[dict[str, object]] = []
    for image_path in image_paths:
        candidates = preprocess_user_digit_candidates(image_path)
        candidate = choose_candidate_for_model(model, candidates)
        candidate = refine_low_confidence_candidate(model, image_path, candidate)
        processed = candidate.processed
        save_preprocessed_image(processed, image_path, 7)
        prediction = save_prediction_figure(
            model,
            image_path,
            processed,
            expected_labels.get(image_path.name),
            8,
            candidate.rotation_degrees,
            candidate.threshold,
            candidate.preprocessing_adjustment,
        )
        user_predictions.append(prediction)
        print(
            f"{image_path.name}: predicted {prediction['predicted_label']} "
            f"with confidence {prediction['confidence']:.2%}"
        )

    save_prediction_reports(model_path, user_predictions)
    print("")
    print(f"Loaded model: {model_path}")
    print(f"Prediction outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
