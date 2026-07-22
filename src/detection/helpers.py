"""Helpers for the Ultralytics detector-first baseline."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
import yaml


def normalize_split_entry(entry: str) -> str:
    entry = entry.strip().replace("\\", "/")
    entry = entry.replace("./images/", "").replace("images/", "")
    return Path(entry).name


def read_split_file(path: Path) -> list[str]:
    return [normalize_split_entry(line) for line in path.read_text().splitlines() if line.strip()]


def image_path_from_name(image_dir: Path, image_name: str) -> Path:
    return image_dir / image_name


def label_path_from_name(label_dir: Path, image_name: str) -> Path:
    return label_dir / f"{Path(image_name).stem}.txt"


def parse_tile_name(image_name: str) -> dict[str, int | str | None]:
    stem = Path(image_name).stem
    parts = stem.split("_")
    if len(parts) != 3:
        return {"tile_name": stem, "diagram_id": None, "crop_x": None, "crop_y": None}
    diagram_id, crop_x, crop_y = parts
    return {
        "tile_name": stem,
        "diagram_id": diagram_id,
        "crop_x": int(crop_x),
        "crop_y": int(crop_y),
    }


def read_yolo_annotations(label_path: Path) -> pd.DataFrame:
    columns = ["class_id", "x_center", "y_center", "width", "height"]
    if not label_path.exists() or label_path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(label_path, sep=r"\s+", header=None, names=columns)
    df["class_id"] = df["class_id"].astype(int)
    return df


def yolo_to_pixel_boxes(df: pd.DataFrame, image_width: int, image_height: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    pixel_df = df.copy()
    pixel_df["x_center_px"] = pixel_df["x_center"] * image_width
    pixel_df["y_center_px"] = pixel_df["y_center"] * image_height
    pixel_df["width_px"] = pixel_df["width"] * image_width
    pixel_df["height_px"] = pixel_df["height"] * image_height
    pixel_df["x_min"] = pixel_df["x_center_px"] - pixel_df["width_px"] / 2
    pixel_df["y_min"] = pixel_df["y_center_px"] - pixel_df["height_px"] / 2
    pixel_df["x_max"] = pixel_df["x_center_px"] + pixel_df["width_px"] / 2
    pixel_df["y_max"] = pixel_df["y_center_px"] + pixel_df["height_px"] / 2
    return pixel_df


def load_image_rgb(image_path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def draw_boxes(image: np.ndarray, boxes_df: pd.DataFrame, class_name_map: dict[int, str] | None = None) -> np.ndarray:
    canvas = image.copy()
    for _, row in boxes_df.iterrows():
        x_min, y_min, x_max, y_max = map(int, [row.x_min, row.y_min, row.x_max, row.y_max])
        cv2.rectangle(canvas, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
        class_id = int(row.class_id)
        class_label = class_name_map.get(class_id, f"class_{class_id:02d}") if class_name_map else f"class_{class_id:02d}"
        cv2.putText(
            canvas,
            class_label,
            (x_min, max(20, y_min - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def collect_dataset_integrity(image_dir: Path, label_dir: Path, split_names: list[str]) -> dict[str, int]:
    missing_images = 0
    missing_labels = 0
    empty_labels = 0
    invalid_rows = 0

    for image_name in split_names:
        image_path = image_path_from_name(image_dir, image_name)
        label_path = label_path_from_name(label_dir, image_name)

        if not image_path.exists():
            missing_images += 1
        if not label_path.exists():
            missing_labels += 1
            continue

        lines = [line.strip() for line in label_path.read_text().splitlines() if line.strip()]
        if not lines:
            empty_labels += 1
            continue

        for line in lines:
            if len(line.split()) != 5:
                invalid_rows += 1

    return {
        "images_listed": len(split_names),
        "missing_images": missing_images,
        "missing_labels": missing_labels,
        "empty_labels": empty_labels,
        "invalid_rows": invalid_rows,
    }


def scan_class_distribution(label_dir: Path, split_names: list[str]) -> pd.DataFrame:
    counter: Counter[int] = Counter()
    for image_name in split_names:
        ann_df = read_yolo_annotations(label_path_from_name(label_dir, image_name))
        for class_id in ann_df["class_id"].tolist():
            counter[int(class_id)] += 1
    return pd.DataFrame(
        [{"class_id": class_id, "count": count} for class_id, count in sorted(counter.items())]
    )


def build_class_name_map(min_class_id: int = 1, max_class_id: int = 32) -> dict[int, str]:
    return {class_id: f"class_{class_id:02d}" for class_id in range(min_class_id, max_class_id + 1)}


def _remap_annotation_text(raw_text: str) -> str:
    remapped_lines: list[str] = []
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO row: {raw_line}")
        class_id = int(float(parts[0]))
        zero_based = class_id - 1
        remapped_lines.append(" ".join([str(zero_based), *parts[1:]]))
    return "\n".join(remapped_lines) + ("\n" if remapped_lines else "")


def materialize_zero_based_detection_dataset(
    dataset_root: Path,
    prepared_root: Path,
    train_split_path: Path,
    val_split_path: Path,
) -> dict[str, Path]:
    image_dir = dataset_root / "images (3)"
    label_dir = dataset_root / "labels (2)"
    prepared_labels_dir = prepared_root / "labels_zero_based"
    prepared_splits_dir = prepared_root / "splits"
    prepared_labels_dir.mkdir(parents=True, exist_ok=True)
    prepared_splits_dir.mkdir(parents=True, exist_ok=True)

    train_names = read_split_file(train_split_path)
    val_names = read_split_file(val_split_path)
    all_names = sorted(set(train_names + val_names))

    for image_name in all_names:
        source_label = label_path_from_name(label_dir, image_name)
        target_label = prepared_labels_dir / source_label.name
        if source_label.exists():
            target_label.write_text(_remap_annotation_text(source_label.read_text()))
        else:
            target_label.write_text("")

    train_images_txt = prepared_splits_dir / "train_images.txt"
    val_images_txt = prepared_splits_dir / "val_images.txt"
    train_images_txt.write_text("\n".join(str(image_path_from_name(image_dir, name)) for name in train_names) + "\n")
    val_images_txt.write_text("\n".join(str(image_path_from_name(image_dir, name)) for name in val_names) + "\n")

    return {
        "image_dir": image_dir,
        "raw_label_dir": label_dir,
        "prepared_labels_dir": prepared_labels_dir,
        "train_images_txt": train_images_txt,
        "val_images_txt": val_images_txt,
    }


def write_ultralytics_dataset_yaml(
    dataset_yaml_path: Path,
    train_images_txt: Path,
    val_images_txt: Path,
    class_name_map: dict[int, str],
) -> None:
    names = [class_name_map[class_id] for class_id in sorted(class_name_map)]
    payload = {
        "path": str(dataset_yaml_path.parent.resolve()),
        "train": str(train_images_txt.resolve()),
        "val": str(val_images_txt.resolve()),
        "nc": len(names),
        "names": names,
    }
    dataset_yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False))


def normalize_ultralytics_results(results, class_name_map: dict[int, str], run_stage: str) -> list[dict]:
    normalized: list[dict] = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue

        xyxy_values = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
        conf_values = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf
        cls_values = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else boxes.cls

        for bbox_xyxy, conf, cls_idx in zip(xyxy_values, conf_values, cls_values):
            zero_based_id = int(cls_idx)
            raw_class_id = zero_based_id + 1
            normalized.append(
                {
                    "run_stage": run_stage,
                    "class_id": raw_class_id,
                    "class_name": class_name_map.get(raw_class_id, f"class_{raw_class_id:02d}"),
                    "confidence": float(conf),
                    "bbox_xyxy": [float(value) for value in bbox_xyxy.tolist()],
                }
            )
    return normalized


def render_prediction_overlay(image: np.ndarray, predictions: list[dict]) -> np.ndarray:
    canvas = image.copy()
    for pred in predictions:
        x_min, y_min, x_max, y_max = map(int, pred["bbox_xyxy"])
        label = f"{pred['class_name']} {pred['confidence']:.2f}"
        cv2.rectangle(canvas, (x_min, y_min), (x_max, y_max), (255, 99, 71), 2)
        cv2.putText(
            canvas,
            label,
            (x_min, max(20, y_min - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 99, 71),
            1,
            cv2.LINE_AA,
        )
    return canvas


def prediction_summary(
    image_name: str,
    image_path: Path,
    model_name: str,
    run_stage: str,
    predictions: list[dict],
) -> dict:
    return {
        "image_name": image_name,
        "image_path": str(image_path),
        "model_name": model_name,
        "run_stage": run_stage,
        "predictions": predictions,
    }


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))
