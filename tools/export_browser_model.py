"""Export the saved Keras MNIST model for the static browser frontend.

The Pages app does not run Python, so this script extracts only the inference
weights from models/best_digit_model.keras and writes a small manifest plus a
binary Float32 weight blob under docs/assets/model.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "models" / "best_digit_model.keras"
OUTPUT_DIR = PROJECT_DIR / "docs" / "assets" / "model"
SAMPLE_DIR = PROJECT_DIR / "docs" / "assets" / "samples"
PREDICTION_SUMMARY = PROJECT_DIR / "outputs" / "prediction_summary.json"

WEIGHTS = {
    "conv2d": ("kernel", "bias"),
    "batch_normalization": ("gamma", "beta", "moving_mean", "moving_variance"),
    "conv2d_1": ("kernel", "bias"),
    "batch_normalization_1": ("gamma", "beta", "moving_mean", "moving_variance"),
    "conv2d_2": ("kernel", "bias"),
    "batch_normalization_2": ("gamma", "beta", "moving_mean", "moving_variance"),
    "conv2d_3": ("kernel", "bias"),
    "batch_normalization_3": ("gamma", "beta", "moving_mean", "moving_variance"),
    "conv2d_4": ("kernel", "bias"),
    "batch_normalization_4": ("gamma", "beta", "moving_mean", "moving_variance"),
    "dense": ("kernel", "bias"),
    "batch_normalization_5": ("gamma", "beta", "moving_mean", "moving_variance"),
    "dense_1": ("kernel", "bias"),
}


def read_prediction_summary() -> dict:
    if not PREDICTION_SUMMARY.exists():
        return {}
    return json.loads(PREDICTION_SUMMARY.read_text(encoding="utf-8"))


def save_samples(summary: dict) -> list[dict]:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    samples: list[dict] = []

    for item in summary.get("user_predictions", []):
        image_name = Path(item["image"]).stem
        source = PROJECT_DIR / "outputs" / f"07_{image_name}_preprocessed_28x28.png"
        if not source.exists():
            continue

        target = SAMPLE_DIR / f"{image_name}.png"
        shutil.copyfile(source, target)
        samples.append(
            {
                "name": image_name,
                "label": item.get("expected_label"),
                "predicted": item.get("predicted_label"),
                "confidence": item.get("confidence"),
                "image": f"assets/samples/{target.name}",
            }
        )

    (SAMPLE_DIR / "samples.json").write_text(
        json.dumps({"samples": samples}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return samples


def image_to_28x28(path: Path) -> list[float]:
    image = Image.open(path).convert("L")
    if image.size != (28, 28):
        image = image.resize((28, 28), Image.Resampling.BOX)
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    return pixels.reshape(-1).round(6).tolist()


def save_self_test(samples: list[dict], summary: dict) -> None:
    if not samples:
        return

    by_name = {Path(item["image"]).stem: item for item in summary.get("user_predictions", [])}
    cases = []
    for sample in samples[:3]:
        name = sample["name"]
        prediction = by_name.get(name, {})
        cases.append(
            {
                "name": name,
                "input": image_to_28x28(PROJECT_DIR / "docs" / sample["image"]),
                "expectedTop": prediction.get("predicted_label"),
                "expectedConfidence": prediction.get("confidence"),
            }
        )

    (OUTPUT_DIR / "self-test.json").write_text(
        json.dumps({"cases": cases}, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def export_weights() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "format": "mnist-cnn-float32-v1",
        "source": "models/best_digit_model.keras",
        "inputShape": [28, 28, 1],
        "classes": list(range(10)),
        "epsilon": 0.001,
        "weights": {},
    }

    arrays: list[np.ndarray] = []
    float_offset = 0

    with zipfile.ZipFile(MODEL_PATH) as archive:
        h5_bytes = archive.read("model.weights.h5")

    with h5py.File(BytesIO(h5_bytes), "r") as weights_file:
        weights_manifest = manifest["weights"]
        assert isinstance(weights_manifest, dict)

        for layer_name, variable_names in WEIGHTS.items():
            layer_group = weights_file[f"layers/{layer_name}/vars"]
            for index, variable_name in enumerate(variable_names):
                array = np.asarray(layer_group[str(index)], dtype="<f4")
                key = f"{layer_name}/{variable_name}"
                size = int(array.size)
                weights_manifest[key] = {
                    "shape": list(array.shape),
                    "offset": float_offset,
                    "size": size,
                }
                arrays.append(array.reshape(-1))
                float_offset += size

    weights = np.concatenate(arrays).astype("<f4", copy=False)
    (OUTPUT_DIR / "weights.bin").write_bytes(weights.tobytes())
    (OUTPUT_DIR / "model.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    summary = read_prediction_summary()
    export_weights()
    samples = save_samples(summary)
    save_self_test(samples, summary)
    (PROJECT_DIR / "docs" / ".nojekyll").write_text("", encoding="utf-8")
    print(f"Exported browser model to {OUTPUT_DIR}")
    print(f"Exported {len(samples)} sample digits to {SAMPLE_DIR}")


if __name__ == "__main__":
    main()
