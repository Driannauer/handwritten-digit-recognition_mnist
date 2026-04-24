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
    "t1.jpg": 3,
    "t2.jpg": 5,
    "t3.jpg": 4,
    "t4.jpg": 7,
    "t5.jpg": 9,
    "t6.jpg": 2,
    "t7.jpg": 8,
    "t8.jpg": 9,
    "t9.jpg": 3,
    "t10.jpg": 6,
    "t11.jpg": 4,
}


@dataclass
class PreprocessCandidate:
    processed: np.ndarray
    rotation_degrees: int
    orientation_score: float


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
    image = Image.open(image_path)
    return ImageOps.exif_transpose(image).convert(mode)


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
        if aspect_ratio > 8:
            continue
        if (
            ys.start <= 2
            or xs.start <= 2
            or ys.stop >= image_height - 2
            or xs.stop >= image_width - 2
        ) and aspect_ratio > 4:
            continue

        fill_ratio = area / max(1, height * width)
        center_y = (ys.start + ys.stop) / 2
        center_x = (xs.start + xs.stop) / 2
        center_bonus = 1 / (
            1
            + ((center_x - image_width / 2) / (image_width / 2)) ** 2
            + ((center_y - image_height / 2) / (image_height / 2)) ** 2
        )
        shape_bonus = max(0.2, min(height, width) / max(1, max(height, width)))
        upright_bonus = max(0.2, min(1.6, height / max(1, width)))
        ink_strength = float(ink[component_slice][component].sum())
        score = ink_strength * fill_ratio * center_bonus * shape_bonus * upright_bonus

        if best_score is None or score > best_score:
            best_score = score
            best_slice = component_slice

    if best_slice is None:
        return PreprocessCandidate(
            processed=np.zeros((28, 28), dtype=np.float32),
            rotation_degrees=rotation_degrees,
            orientation_score=0.0,
        )

    ys, xs = best_slice
    crop = ink[ys, xs]
    crop_mask = crop > threshold
    y_coords, x_coords = np.nonzero(crop_mask)
    if len(y_coords) == 0 or len(x_coords) == 0:
        return PreprocessCandidate(
            processed=np.zeros((28, 28), dtype=np.float32),
            rotation_degrees=rotation_degrees,
            orientation_score=0.0,
        )

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
    arr = np.clip(arr, 0, 1)

    mass = center_of_mass(arr)
    if not np.isnan(mass[0]) and not np.isnan(mass[1]):
        arr = shift(
            arr,
            shift=(13.5 - mass[0], 13.5 - mass[1]),
            order=1,
            mode="constant",
            cval=0.0,
        )

    return PreprocessCandidate(
        processed=np.clip(arr, 0, 1),
        rotation_degrees=rotation_degrees,
        orientation_score=float(best_score or 0.0),
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


def choose_candidate_for_model(
    model: keras.Model,
    candidates: list[PreprocessCandidate],
) -> PreprocessCandidate:
    max_orientation_score = max((item.orientation_score for item in candidates), default=0.0)
    if max_orientation_score <= 0:
        return candidates[0]

    original_candidate = candidates[0]
    original_probabilities = predict_probabilities(model, original_candidate.processed)
    original_confidence = float(np.max(original_probabilities))
    if (
        original_confidence >= 0.85
        and original_candidate.orientation_score >= 0.7 * max_orientation_score
    ):
        return original_candidate

    best_candidate = candidates[0]
    best_score = -1.0
    for candidate in candidates:
        probabilities = predict_probabilities(model, candidate.processed)
        confidence = float(np.max(probabilities))
        orientation_weight = candidate.orientation_score / max_orientation_score
        combined_score = confidence * (0.55 + 0.45 * orientation_weight)
        if combined_score > best_score:
            best_score = combined_score
            best_candidate = candidate
    return best_candidate


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
        lines.append(
            "- {image}: predicted {predicted_label}, expected {expected}, "
            "confidence {confidence:.2%}, auto-rotation {rotation}".format(
                image=item["image"],
                predicted_label=item["predicted_label"],
                expected=expected_text,
                confidence=item["confidence"],
                rotation=rotation_text,
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
        help="Optional expected labels, for example: t1.jpg=3 t2.jpg=5",
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
        processed = candidate.processed
        save_preprocessed_image(processed, image_path, 7)
        prediction = save_prediction_figure(
            model,
            image_path,
            processed,
            expected_labels.get(image_path.name),
            8,
            candidate.rotation_degrees,
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
