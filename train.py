"""
Train handwritten digit recognition models with TensorFlow.

Workflow:
1. Train an MLP and a CNN on MNIST.
2. Evaluate both models on the MNIST test set.
3. Save training curves, evaluation charts, reports, and the best model.

Use test.py separately when you only want to predict local handwritten images.
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
import tensorflow as tf
from scipy.interpolate import PchipInterpolator


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_DIR = PROJECT_DIR / "models"
RANDOM_STATE = 42
CLASSES = np.arange(10)


@dataclass
class KerasModelResult:
    name: str
    model: keras.Model
    accuracy: float
    predictions: np.ndarray
    history: dict[str, list[float]]


class LearningRateLogger(keras.callbacks.Callback):
    """Record the optimizer learning rate at the end of each epoch."""

    def __init__(self) -> None:
        super().__init__()
        self.learning_rates: list[float] = []

    def on_epoch_end(self, epoch, logs=None):  # noqa: D401
        lr_value = tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        self.learning_rates.append(float(lr_value))


def ensure_dirs(clear_outputs: bool = True) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    if clear_outputs:
        clear_training_outputs()


def clear_training_outputs() -> None:
    generated_prefixes = tuple(f"{index:02d}_" for index in range(1, 7))
    generated_names = {
        "classification_report.txt",
        "results_summary.json",
        "results_summary.txt",
        "training_history.json",
    }
    for path in OUTPUT_DIR.iterdir():
        if not path.is_file():
            continue
        if path.name.startswith(generated_prefixes) or path.name in generated_names:
            path.unlink()


def set_random_seed() -> None:
    np.random.seed(RANDOM_STATE)
    tf.keras.utils.set_random_seed(RANDOM_STATE)


def stratified_train_val_split(
    images: np.ndarray,
    labels: np.ndarray,
    val_fraction: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_STATE)
    train_indices = []
    val_indices = []

    for class_id in CLASSES:
        class_indices = np.where(labels == class_id)[0]
        rng.shuffle(class_indices)
        val_count = max(1, int(round(len(class_indices) * val_fraction)))
        val_indices.append(class_indices[:val_count])
        train_indices.append(class_indices[val_count:])

    train_index_array = np.concatenate(train_indices)
    val_index_array = np.concatenate(val_indices)
    rng.shuffle(train_index_array)
    rng.shuffle(val_index_array)

    return (
        images[train_index_array],
        images[val_index_array],
        labels[train_index_array],
        labels[val_index_array],
    )


def load_mnist_data(train_limit: int = 0):
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train_full = x_train_full.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train, x_val, y_train, y_val = stratified_train_val_split(
        x_train_full,
        y_train_full,
    )

    if train_limit and train_limit < len(x_train):
        x_train = x_train[:train_limit]
        y_train = y_train[:train_limit]

    return x_train_full, y_train_full, x_train, x_val, x_test, y_train, y_val, y_test


def save_digit_examples(images: np.ndarray, labels: np.ndarray) -> None:
    plt.figure(figsize=(10, 4))
    for index in range(12):
        plt.subplot(3, 4, index + 1)
        plt.imshow(images[index], cmap="gray")
        plt.title(f"label: {labels[index]}")
        plt.axis("off")
    plt.suptitle("MNIST Training Data Examples")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_dataset_examples.png", dpi=160)
    plt.close()


def build_mlp() -> keras.Model:
    return keras.Sequential(
        [
            keras.layers.Input(shape=(28, 28)),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )


def build_cnn() -> keras.Model:
    return keras.Sequential(
        [
            keras.layers.Input(shape=(28, 28, 1)),
            keras.layers.RandomRotation(
                0.08,
                fill_mode="constant",
                fill_value=0.0,
                seed=RANDOM_STATE,
            ),
            keras.layers.RandomTranslation(
                0.08,
                0.08,
                fill_mode="constant",
                fill_value=0.0,
                seed=RANDOM_STATE,
            ),
            keras.layers.RandomZoom(
                0.08,
                fill_mode="constant",
                fill_value=0.0,
                seed=RANDOM_STATE,
            ),
            keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(),
            keras.layers.Dropout(0.15),
            keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(),
            keras.layers.Dropout(0.2),
            keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )


def compile_model(model: keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def fit_with_history(
    model: keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    initial_learning_rate: float,
    learning_rate_decay: float,
    add_channel: bool,
) -> dict[str, list[float]]:
    lr_logger = LearningRateLogger()
    scheduler = keras.callbacks.LearningRateScheduler(
        lambda epoch, _: initial_learning_rate * (learning_rate_decay**epoch),
        verbose=0,
    )

    train_input = x_train[..., None] if add_channel else x_train
    val_input = x_val[..., None] if add_channel else x_val

    history = model.fit(
        train_input,
        y_train,
        validation_data=(val_input, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[scheduler, lr_logger],
        verbose=0,
    )
    history_dict = dict(history.history)
    history_dict["learning_rate"] = lr_logger.learning_rates
    history_dict["epoch"] = list(range(1, len(lr_logger.learning_rates) + 1))
    return history_dict


def evaluate_model(
    model: keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    add_channel: bool,
    name: str,
    history: dict[str, list[float]],
) -> KerasModelResult:
    test_input = x_test[..., None] if add_channel else x_test
    probabilities = model.predict(test_input, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    accuracy = np.mean(y_test == predictions)
    print(f"{name} test accuracy: {accuracy:.4f}")
    return KerasModelResult(
        name=name,
        model=model,
        accuracy=float(accuracy),
        predictions=predictions,
        history=history,
    )


def safe_file_stem(name: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", name.lower().replace(" ", "_")).strip("_")


def smooth_metric_curve(
    epochs: list[int],
    values: list[float],
    *,
    clip_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    x_values = np.array(epochs, dtype=np.float32)
    y_values = np.array(values, dtype=np.float32)
    if len(x_values) < 4:
        return x_values, y_values

    smooth_x = np.linspace(float(x_values.min()), float(x_values.max()), 160)
    smooth_y = PchipInterpolator(x_values, y_values)(smooth_x)
    if clip_range is not None:
        smooth_y = np.clip(smooth_y, clip_range[0], clip_range[1])
    return smooth_x, smooth_y


def plot_metric_curve(
    ax,
    epochs: list[int],
    values: list[float],
    label: str,
    *,
    color: str,
    clip_range: tuple[float, float] | None = None,
) -> None:
    smooth_x, smooth_y = smooth_metric_curve(epochs, values, clip_range=clip_range)
    ax.plot(smooth_x, smooth_y, color=color, linewidth=2.2, label=f"{label} smooth")
    ax.plot(
        epochs,
        values,
        color=color,
        marker="o",
        linestyle="",
        markersize=4,
        alpha=0.75,
        label=f"{label} epoch",
    )


def save_training_curves(result: KerasModelResult, file_index: int) -> None:
    history = result.history
    epochs = history["epoch"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    plot_metric_curve(axes[0], epochs, history["loss"], "train_loss", color="#4062bb")
    plot_metric_curve(axes[0], epochs, history["val_loss"], "val_loss", color="#d1495b")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Trend")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    plot_metric_curve(
        axes[1],
        epochs,
        history["accuracy"],
        "train_accuracy",
        color="#4062bb",
        clip_range=(0.0, 1.0),
    )
    plot_metric_curve(
        axes[1],
        epochs,
        history["val_accuracy"],
        "val_accuracy",
        color="#d1495b",
        clip_range=(0.0, 1.0),
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Trend")
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].step(
        epochs,
        history["learning_rate"],
        where="post",
        color="#2a9d8f",
        linewidth=2.2,
        label="learning_rate",
    )
    axes[2].plot(
        epochs,
        history["learning_rate"],
        color="#2a9d8f",
        marker="o",
        linestyle="",
        markersize=4,
    )
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning rate")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].set_yscale("log")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    fig.suptitle(f"{result.name} Training Curves")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"{file_index:02d}_{safe_file_stem(result.name)}_training_curves.png",
        dpi=160,
    )
    plt.close(fig)


def save_training_history_json(results: list[KerasModelResult]) -> None:
    history_path = OUTPUT_DIR / "training_history.json"
    history_path.write_text(
        json.dumps(
            {item.name: item.history for item in results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def save_model_comparison(results: list[KerasModelResult], file_index: int) -> None:
    names = [item.name for item in results]
    scores = [item.accuracy for item in results]

    plt.figure(figsize=(7, 4))
    bars = plt.bar(names, scores, color=["#4062bb", "#59a14f"][: len(names)])
    plt.ylim(max(0.0, min(scores) - 0.05), 1.0)
    plt.ylabel("Test accuracy")
    plt.title("Model Accuracy Comparison")
    for bar, score in zip(bars, scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.003,
            f"{score:.4f}",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{file_index:02d}_model_comparison.png", dpi=160)
    plt.close()


def make_confusion_matrix(y_true: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    for true_label, predicted_label in zip(y_true.astype(int), predictions.astype(int)):
        if 0 <= true_label < len(CLASSES) and 0 <= predicted_label < len(CLASSES):
            matrix[true_label, predicted_label] += 1
    return matrix


def save_confusion_matrix(
    y_test: np.ndarray,
    predictions: np.ndarray,
    model_name: str,
    file_index: int,
) -> None:
    matrix = make_confusion_matrix(y_test, predictions)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(CLASSES)
    ax.set_yticks(CLASSES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"{model_name} Confusion Matrix")

    threshold = matrix.max() / 2 if matrix.max() else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            color = "white" if matrix[row, column] > threshold else "black"
            ax.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{file_index:02d}_confusion_matrix.png", dpi=160)
    plt.close(fig)


def save_wrong_samples(
    images_test: np.ndarray,
    y_test: np.ndarray,
    predictions: np.ndarray,
    model_name: str,
    file_index: int,
    max_count: int = 12,
) -> None:
    wrong_indexes = np.where(y_test != predictions)[0]
    text_path = OUTPUT_DIR / f"{file_index:02d}_wrong_samples.txt"
    image_path = OUTPUT_DIR / f"{file_index:02d}_wrong_samples.png"

    if len(wrong_indexes) == 0:
        text_path.write_text(
            f"{model_name} made no mistakes on this test split.\n",
            encoding="utf-8",
        )
        return

    count = min(max_count, len(wrong_indexes))
    plt.figure(figsize=(10, 6))
    for plot_index, sample_index in enumerate(wrong_indexes[:count]):
        plt.subplot(3, 4, plot_index + 1)
        plt.imshow(images_test[sample_index], cmap="gray")
        plt.title(f"true:{y_test[sample_index]} pred:{predictions[sample_index]}")
        plt.axis("off")
    plt.suptitle(f"{model_name} Wrong Prediction Examples")
    plt.tight_layout()
    plt.savefig(image_path, dpi=160)
    plt.close()


def build_classification_report(y_true: np.ndarray, predictions: np.ndarray) -> str:
    matrix = make_confusion_matrix(y_true, predictions)
    total = int(matrix.sum())
    accuracy = float(np.trace(matrix) / total) if total else 0.0
    rows = [
        "Classification Report",
        "=" * 70,
        f"{'digit':>8} {'precision':>12} {'recall':>12} {'f1-score':>12} {'support':>10}",
    ]

    precisions = []
    recalls = []
    f1_scores = []
    supports = []
    for class_id in CLASSES:
        true_positive = float(matrix[class_id, class_id])
        predicted_total = float(matrix[:, class_id].sum())
        actual_total = float(matrix[class_id, :].sum())
        precision = true_positive / predicted_total if predicted_total else 0.0
        recall = true_positive / actual_total if actual_total else 0.0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        support = int(actual_total)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)
        supports.append(support)
        rows.append(
            f"{class_id:>8} {precision:>12.4f} {recall:>12.4f} "
            f"{f1_score:>12.4f} {support:>10}"
        )

    weighted_precision = float(np.average(precisions, weights=supports)) if total else 0.0
    weighted_recall = float(np.average(recalls, weights=supports)) if total else 0.0
    weighted_f1 = float(np.average(f1_scores, weights=supports)) if total else 0.0

    rows.extend(
        [
            "",
            f"{'accuracy':>8} {'':>12} {'':>12} {accuracy:>12.4f} {total:>10}",
            f"{'macro avg':>8} {np.mean(precisions):>12.4f} {np.mean(recalls):>12.4f} "
            f"{np.mean(f1_scores):>12.4f} {total:>10}",
            f"{'weighted avg':>8} {weighted_precision:>12.4f} {weighted_recall:>12.4f} "
            f"{weighted_f1:>12.4f} {total:>10}",
            "",
        ]
    )
    return "\n".join(rows)


def save_text_reports(
    results: list[KerasModelResult],
    best_result: KerasModelResult,
    y_test: np.ndarray,
) -> None:
    report_path = OUTPUT_DIR / "results_summary.txt"
    lines = [
        "Handwritten Digit Recognition - Training Summary",
        "=" * 52,
        "",
        "Model accuracy on the MNIST test set:",
    ]

    for item in results:
        lines.append(f"- {item.name}: {item.accuracy:.4f}")

    lines.extend(
        [
            "",
            f"Best model: {best_result.name}",
            "",
            "Note:",
            "This file is generated by train.py.",
            "Local handwritten image prediction is handled only by test.py.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")

    classification_path = OUTPUT_DIR / "classification_report.txt"
    classification_path.write_text(
        build_classification_report(y_test, best_result.predictions),
        encoding="utf-8",
    )

    json_path = OUTPUT_DIR / "results_summary.json"
    json_path.write_text(
        json.dumps(
            {
                "model_accuracy": {item.name: item.accuracy for item in results},
                "best_model": best_result.name,
                "test_set_size": int(len(y_test)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def save_model(best_result: KerasModelResult) -> None:
    model_path = MODEL_DIR / "best_digit_model.keras"
    best_result.model.save(model_path)

    metadata_path = MODEL_DIR / "best_digit_model_info.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model_name": best_result.name,
                "input_shape": [28, 28, 1],
                "pixel_range": "0 to 1",
                "trained_on": "MNIST",
                "prediction_script": "test.py",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train MNIST MLP/CNN models and evaluate them on the MNIST test set."
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=0,
        help="How many MNIST training images to use. Default 0 means use all training images.",
    )
    args = parser.parse_args()

    set_random_seed()
    ensure_dirs()

    (
        x_train_full,
        y_train_full,
        x_train,
        x_val,
        x_test,
        y_train,
        y_val,
        y_test,
    ) = load_mnist_data(train_limit=args.train_limit)
    save_digit_examples(x_train_full, y_train_full)

    mlp = build_mlp()
    compile_model(mlp, learning_rate=1e-3)
    mlp_history = fit_with_history(
        mlp,
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=8,
        batch_size=128,
        initial_learning_rate=1e-3,
        learning_rate_decay=0.9,
        add_channel=False,
    )
    mlp_result = evaluate_model(
        mlp,
        x_test,
        y_test,
        add_channel=False,
        name="MLP",
        history=mlp_history,
    )

    cnn = build_cnn()
    compile_model(cnn, learning_rate=8e-4)
    cnn_history = fit_with_history(
        cnn,
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=10,
        batch_size=128,
        initial_learning_rate=8e-4,
        learning_rate_decay=0.9,
        add_channel=True,
    )
    cnn_result = evaluate_model(
        cnn,
        x_test,
        y_test,
        add_channel=True,
        name="CNN",
        history=cnn_history,
    )

    results = [mlp_result, cnn_result]

    for file_index, result in enumerate(results, start=2):
        save_training_curves(result, file_index)

    save_training_history_json(results)
    save_model_comparison(results, 4)

    best_result = max(results, key=lambda item: item.accuracy)
    save_confusion_matrix(y_test, best_result.predictions, best_result.name, 5)
    save_wrong_samples(x_test, y_test, best_result.predictions, best_result.name, 6)
    save_model(best_result)
    save_text_reports(results, best_result, y_test)

    print("")
    print(f"Best model: {best_result.name}")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print(f"Model saved to: {MODEL_DIR / 'best_digit_model.keras'}")


if __name__ == "__main__":
    main()
