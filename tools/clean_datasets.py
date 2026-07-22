"""Audit and build cleaned YOLO datasets from the downloaded Roboflow exports.

The source exports are read-only. Images in the cleaned datasets are hard-linked
when possible, so building the datasets does not duplicate several gigabytes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "test")
TARGET_RATIOS = {"train": 0.80, "valid": 0.10, "test": 0.10}


@dataclass(frozen=True)
class SourceSpec:
    folder: str
    archive: str
    short_name: str
    ball_class: int
    priority: int


BALL_SOURCES = (
    SourceSpec("Pickleball Pen Ball Tracking.v1i.yolov8", "Pickleball Pen Ball Tracking.v1i.yolov8.zip", "pen_v1", 0, 0),
    SourceSpec("Moving Pickleball.v3i.yolov8", "Moving Pickleball.v3i.yolov8.zip", "moving_v3", 0, 1),
    SourceSpec("Ball Detector.v1-roboflow-instant-1--eval-.yolov8", "Ball Detector.v1-roboflow-instant-1--eval-.yolov8.zip", "ball_detector_v1", 0, 2),
    SourceSpec("pickleball-detector-v6-yolov8", "Pickleball Detector.v6i.yolov8.zip", "detector_v6", 0, 3),
    SourceSpec("InAPickle_Core_Tracking.v1-v1-baseline.yolov8", "InAPickle_Core_Tracking.v1-v1-baseline.yolov8.zip", "inapickle_v1", 1, 4),
)

COURT_SOURCE = "Pickleball Court Keypoints.v4i.yolov8"


@dataclass
class Record:
    source: SourceSpec
    original_split: str
    image: Path
    label: Path
    source_key: str
    clip_key: str
    boxes: list[tuple[float, float, float, float]]
    ignored_objects: int
    file_size: int
    archive: Path
    archive_member: str
    image_hash: str = ""
    assigned_split: str = ""

    @property
    def positive(self) -> bool:
        return bool(self.boxes)


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()


def roboflow_source_stem(filename: str) -> str:
    stem = Path(filename).stem
    return re.sub(r"\.rf\.[0-9a-fA-F]{16,}$", "", stem)


def normalized_source_key(spec: SourceSpec, filename: str) -> str:
    stem = roboflow_source_stem(filename).lower()
    # The Pen export embeds the Moving Pickleball dataset with a `kevin_` prefix.
    overlap_stem = re.sub(r"^kevin_", "", stem)
    if re.match(r"pickleball\d+_mp4-", overlap_stem):
        return f"moving_family:{overlap_stem}"
    return f"{spec.short_name}:{stem}"


def clip_from_stem(stem: str) -> str:
    value = stem.lower()
    value = re.sub(r"\.rf\.[0-9a-fA-F]{16,}$", "", value)
    value = re.sub(r"^kevin_", "", value)
    patterns = (
        r"-\d{4}-\d{2}-\d{2}t\d{6}-\d+_(?:jpg|jpeg|png)$",
        r"_\d{2}m\d{2}s_frame_\d+_(?:jpg|jpeg|png)$",
        r"[-_]\d+-?_(?:jpg|jpeg|png)$",
    )
    for pattern in patterns:
        reduced = re.sub(pattern, "", value)
        if reduced != value:
            return reduced
    return value


def normalized_clip_key(spec: SourceSpec, filename: str) -> str:
    clip = clip_from_stem(roboflow_source_stem(filename))
    if re.match(r"pickleball\d+_mp4$", clip):
        return f"moving_family:{clip}"
    return f"{spec.short_name}:{clip}"


def image_files(directory: Path) -> Iterable[Path]:
    if not directory.exists():
        return []
    return (
        Path(entry.path)
        for entry in os.scandir(directory)
        if entry.is_file() and Path(entry.name).suffix.lower() in IMAGE_SUFFIXES
    )


def parse_ball_label_text(
    text: str, target_class: int
) -> tuple[list[tuple[float, float, float, float]], int, int, list[str]]:
    boxes: list[tuple[float, float, float, float]] = []
    ignored = 0
    converted_polygons = 0
    errors: list[str] = []
    seen: set[tuple[float, float, float, float]] = set()
    lines = text.splitlines()

    for line_number, raw in enumerate(lines, 1):
        if not raw.strip():
            continue
        parts = raw.split()
        try:
            class_id = int(float(parts[0]))
            values = tuple(float(value) for value in parts[1:])
        except ValueError:
            errors.append(f"line {line_number}: non-numeric value")
            continue
        if not all(math.isfinite(value) for value in values):
            errors.append(f"line {line_number}: non-finite coordinate")
            continue
        if class_id != target_class:
            ignored += 1
            continue
        if len(parts) == 5:
            x, y, width, height = values
        elif len(parts) >= 7 and len(parts) % 2 == 1:
            xs = values[0::2]
            ys = values[1::2]
            if not all(0.0 <= value <= 1.0 for value in values):
                errors.append(f"line {line_number}: polygon coordinate outside YOLO range")
                continue
            left, right = min(xs), max(xs)
            top, bottom = min(ys), max(ys)
            x, y = (left + right) / 2, (top + bottom) / 2
            width, height = right - left, bottom - top
            converted_polygons += 1
        else:
            errors.append(f"line {line_number}: expected box or polygon values")
            continue
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            errors.append(f"line {line_number}: coordinate outside YOLO range")
            continue
        box = (x, y, width, height)
        if box not in seen:
            seen.add(box)
            boxes.append(box)
    return boxes, ignored, converted_polygons, errors


def scan_ball_sources(dataset_root: Path) -> tuple[list[Record], dict]:
    records: list[Record] = []
    report: dict = {"sources": {}, "errors": []}
    for spec in BALL_SOURCES:
        source_root = dataset_root / spec.folder
        archive_path = dataset_root / spec.archive
        source_stats = Counter()
        if not source_root.exists():
            report["errors"].append(f"Missing source folder: {spec.folder}")
            continue
        if not archive_path.exists():
            report["errors"].append(f"Missing source archive: {spec.archive}")
            continue
        with zipfile.ZipFile(archive_path) as archive:
            entries = {info.filename: info for info in archive.infolist() if not info.is_dir()}
            for member, info in entries.items():
                parts = member.split("/")
                if len(parts) != 3 or parts[0] not in SPLITS or parts[1] != "images":
                    continue
                split = parts[0]
                if Path(parts[2]).suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                image = source_root / split / "images" / parts[2]
                source_stats["images"] += 1
                label = source_root / split / "labels" / f"{image.stem}.txt"
                label_member = f"{split}/labels/{image.stem}.txt"
                if label_member not in entries:
                    boxes, ignored, converted, errors = [], 0, 0, ["missing label"]
                else:
                    try:
                        label_text = archive.read(label_member).decode("utf-8-sig")
                        boxes, ignored, converted, errors = parse_ball_label_text(label_text, spec.ball_class)
                    except (OSError, UnicodeError, zipfile.BadZipFile) as exc:
                        boxes, ignored, converted, errors = [], 0, 0, [f"unreadable label: {exc}"]
                source_stats["ball_boxes"] += len(boxes)
                source_stats["ignored_objects"] += ignored
                source_stats["converted_polygon_boxes"] += converted
                if boxes:
                    source_stats["positive_images"] += 1
                    if len(boxes) > 1:
                        source_stats["multi_ball_images"] += 1
                else:
                    source_stats["negative_images"] += 1
                if errors:
                    source_stats["invalid_images"] += 1
                    if len(report["errors"]) < 100:
                        report["errors"].append(f"{image}: {'; '.join(errors)}")
                    continue
                records.append(
                    Record(
                        source=spec,
                        original_split=split,
                        image=image,
                        label=label,
                        source_key=normalized_source_key(spec, image.name),
                        clip_key=normalized_clip_key(spec, image.name),
                        boxes=boxes,
                        ignored_objects=ignored,
                        file_size=info.file_size,
                        archive=archive_path,
                        archive_member=member,
                        image_hash=f"{info.CRC:08x}:{info.file_size}",
                    )
                )
        report["sources"][spec.short_name] = dict(source_stats)
    report["raw_valid_records"] = len(records)
    return records, report


def deduplicate_source_variants(records: list[Record], report: dict) -> list[Record]:
    groups: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        groups[record.source_key].append(record)

    selected: list[Record] = []
    split_leaks = 0
    label_disagreements = 0
    variants_removed = 0
    for variants in groups.values():
        variants_removed += len(variants) - 1
        if len({record.original_split for record in variants}) > 1:
            split_leaks += 1
        if len({len(record.boxes) for record in variants}) > 1:
            label_disagreements += 1
        # Prefer a positive annotation and a larger JPEG (usually the least blurred variant).
        variants.sort(
            key=lambda record: (
                record.positive,
                len(record.boxes),
                record.file_size,
                stable_hash(str(record.image)),
            ),
            reverse=True,
        )
        selected.append(variants[0])

    report["source_deduplication"] = {
        "unique_source_frames": len(selected),
        "selected_positive_images": sum(record.positive for record in selected),
        "selected_negative_images": sum(not record.positive for record in selected),
        "selected_ball_boxes": sum(len(record.boxes) for record in selected),
        "variants_removed": variants_removed,
        "source_frames_leaking_across_original_splits": split_leaks,
        "variant_groups_with_ball_count_disagreement": label_disagreements,
    }
    return selected


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deduplicate_exact_images(records: list[Record], report: dict) -> list[Record]:
    groups: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        groups[record.image_hash].append(record)

    selected: list[Record] = []
    conflicts = 0
    removed = 0
    for variants in groups.values():
        removed += len(variants) - 1
        if len({tuple(record.boxes) for record in variants}) > 1:
            conflicts += 1
        variants.sort(
            key=lambda record: (
                record.positive,
                len(record.boxes),
                -record.source.priority,
                record.file_size,
            ),
            reverse=True,
        )
        selected.append(variants[0])
    report["exact_deduplication"] = {
        "unique_images": len(selected),
        "exact_duplicates_removed": removed,
        "duplicate_groups_with_label_disagreement": conflicts,
    }
    return selected


def limit_negatives(records: list[Record], report: dict, negative_ratio: float) -> list[Record]:
    positives = [record for record in records if record.positive]
    negatives = [record for record in records if not record.positive]
    maximum = int(len(positives) * negative_ratio)
    negatives.sort(key=lambda record: stable_hash(record.source_key))
    kept_negatives = negatives[:maximum]
    report["negative_sampling"] = {
        "positive_images": len(positives),
        "available_negative_images": len(negatives),
        "kept_negative_images": len(kept_negatives),
        "max_negative_to_positive_ratio": negative_ratio,
    }
    return positives + kept_negatives


def assign_grouped_splits(records: list[Record], report: dict) -> None:
    groups: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        groups[record.clip_key].append(record)

    total = len(records)
    targets = {split: total * ratio for split, ratio in TARGET_RATIOS.items()}
    counts = Counter()
    source_totals = Counter(record.source.short_name for record in records)
    source_targets = {
        source: {split: count * TARGET_RATIOS[split] for split in SPLITS}
        for source, count in source_totals.items()
    }
    source_counts: dict[str, Counter] = defaultdict(Counter)
    group_source_counts = {
        key: Counter(record.source.short_name for record in group)
        for key, group in groups.items()
    }
    source_groups: dict[str, list[str]] = defaultdict(list)
    for key, vector in group_source_counts.items():
        for source in vector:
            source_groups[source].append(key)

    assignments: dict[str, str] = {}

    def place(key: str, split: str) -> None:
        assignments[key] = split
        counts[split] += len(groups[key])
        for source, amount in group_source_counts[key].items():
            source_counts[source][split] += amount

    # Reserve independent evaluation clips for every source that has at least
    # three clips. The closest-sized clips make validation/test meaningful even
    # for small sources such as Moving Pickleball (three videos total).
    for source in sorted(source_groups, key=lambda name: (len(source_groups[name]), name)):
        available = [key for key in source_groups[source] if key not in assignments]
        if len(available) < 3:
            continue
        for split in ("test", "valid"):
            candidates = [key for key in available if key not in assignments]
            if not candidates:
                break
            target_size = source_targets[source][split]
            key = min(
                candidates,
                key=lambda candidate: (
                    abs(group_source_counts[candidate][source] - target_size),
                    stable_hash(f"{source}:{split}:{candidate}"),
                ),
            )
            place(key, split)

    ordered_groups = sorted(
        ((key, group) for key, group in groups.items() if key not in assignments),
        key=lambda item: (-len(item[1]), stable_hash(item[0])),
    )
    for key, group in ordered_groups:
        size = len(group)

        def score(name: str) -> float:
            global_score = sum(
                (
                    (counts[candidate] + (size if candidate == name else 0) - targets[candidate])
                    / max(targets[candidate], 1.0)
                )
                ** 2
                for candidate in SPLITS
            )
            source_score = 0.0
            for source, amount in group_source_counts[key].items():
                source_score += sum(
                    (
                        (
                            source_counts[source][candidate]
                            + (amount if candidate == name else 0)
                            - source_targets[source][candidate]
                        )
                        / max(source_targets[source][candidate], 1.0)
                    )
                    ** 2
                    for candidate in SPLITS
                )
            return global_score + source_score

        split = min(
            SPLITS,
            key=lambda name: (score(name), SPLITS.index(name)),
        )
        place(key, split)

    for key, group in groups.items():
        for record in group:
            record.assigned_split = assignments[key]

    report["cleaned_splits"] = dict(counts)
    report["cleaned_source_splits"] = {
        source: dict(source_counts[source]) for source in sorted(source_counts)
    }
    report["clip_groups"] = len(groups)
    report["clip_group_overlap_after_resplit"] = 0


def hardlink_or_copy(
    source: Path,
    destination: Path,
    archive: zipfile.ZipFile | None = None,
    archive_member: str = "",
) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError as link_error:
        try:
            shutil.copy2(source, destination)
            return "copy"
        except FileNotFoundError:
            if archive is None or not archive_member:
                raise link_error
            with archive.open(archive_member) as input_handle, destination.open("wb") as output_handle:
                shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
            return "zip_extract"


def write_ball_dataset(records: list[Record], output_root: Path, report: dict) -> None:
    if output_root.exists():
        raise FileExistsError(f"Output already exists: {output_root}")
    for split in SPLITS:
        (output_root / split / "images").mkdir(parents=True)
        (output_root / split / "labels").mkdir(parents=True)

    methods = Counter()
    manifest_rows = []
    archives = {path: zipfile.ZipFile(path) for path in {record.archive for record in records}}
    try:
        for record in sorted(records, key=lambda item: (item.assigned_split, item.source_key)):
            short_hash = stable_hash(f"{record.source.short_name}:{record.source_key}:{record.image_hash}")[:16]
            output_stem = f"{record.source.short_name}__{short_hash}"
            output_image = output_root / record.assigned_split / "images" / f"{output_stem}{record.image.suffix.lower()}"
            output_label = output_root / record.assigned_split / "labels" / f"{output_stem}.txt"
            methods[
                hardlink_or_copy(
                    record.image,
                    output_image,
                    archives[record.archive],
                    record.archive_member,
                )
            ] += 1
            lines = [f"0 {x:.8f} {y:.8f} {width:.8f} {height:.8f}" for x, y, width, height in record.boxes]
            output_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="ascii")
            manifest_rows.append(
                {
                    "clean_split": record.assigned_split,
                    "clean_image": output_image.relative_to(output_root).as_posix(),
                    "source_dataset": record.source.short_name,
                    "source_split": record.original_split,
                    "source_image": str(record.image),
                    "source_frame_key": record.source_key,
                    "clip_key": record.clip_key,
                    "ball_count": len(record.boxes),
                    "ignored_non_ball_objects": record.ignored_objects,
                    "content_fingerprint": record.image_hash,
                }
            )
    finally:
        for archive in archives.values():
            archive.close()

    (output_root / "data.yaml").write_text(
        "path: .\ntrain: train/images\nval: valid/images\ntest: test/images\n\nnames:\n  0: pickleball\n",
        encoding="ascii",
    )
    with (output_root / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    report["output_image_methods"] = dict(methods)


def court_clip_key(filename: str) -> str:
    return clip_from_stem(roboflow_source_stem(filename))


def clean_court_keypoints(dataset_root: Path, output_root: Path) -> dict:
    source_root = dataset_root / COURT_SOURCE
    report = Counter()
    if not source_root.exists():
        return {"status": "source folder not found"}
    if output_root.exists():
        raise FileExistsError(f"Output already exists: {output_root}")

    records = []
    errors = []
    for original_split in SPLITS:
        for image in image_files(source_root / original_split / "images"):
            label = source_root / original_split / "labels" / f"{image.stem}.txt"
            if not label.exists():
                errors.append(f"missing label: {image}")
                continue
            lines = label.read_text(encoding="utf-8-sig").splitlines()
            valid = True
            for line_number, line in enumerate(lines, 1):
                parts = line.split()
                if len(parts) != 41:
                    errors.append(f"{label}:{line_number}: expected 41 values, got {len(parts)}")
                    valid = False
                    break
                try:
                    values = [float(value) for value in parts]
                except ValueError:
                    errors.append(f"{label}:{line_number}: non-numeric value")
                    valid = False
                    break
                if int(values[0]) != 0 or not all(math.isfinite(value) for value in values):
                    errors.append(f"{label}:{line_number}: invalid class or coordinate")
                    valid = False
                    break
            if valid:
                records.append((image, label, original_split, court_clip_key(image.name)))

    groups: dict[str, list] = defaultdict(list)
    for record in records:
        groups[record[3]].append(record)
    totals = len(records)
    targets = {split: totals * TARGET_RATIOS[split] for split in SPLITS}
    counts = Counter()
    assignments = {}
    for key, group in sorted(groups.items(), key=lambda item: (-len(item[1]), stable_hash(item[0]))):
        size = len(group)
        split = min(
            SPLITS,
            key=lambda name: sum(
                (counts[candidate] + (size if candidate == name else 0) - targets[candidate]) ** 2
                for candidate in SPLITS
            ),
        )
        assignments[key] = split
        counts[split] += size

    methods = Counter()
    for split in SPLITS:
        (output_root / split / "images").mkdir(parents=True)
        (output_root / split / "labels").mkdir(parents=True)
    for image, label, _, clip_key in records:
        split = assignments[clip_key]
        short_hash = stable_hash(str(image))[:16]
        stem = f"court__{short_hash}"
        methods[hardlink_or_copy(image, output_root / split / "images" / f"{stem}{image.suffix.lower()}")] += 1
        shutil.copy2(label, output_root / split / "labels" / f"{stem}.txt")
    (output_root / "data.yaml").write_text(
        "path: .\ntrain: train/images\nval: valid/images\ntest: test/images\n\n"
        "kpt_shape: [12, 3]\nflip_idx: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]\n\n"
        "names:\n  0: court_points\n",
        encoding="ascii",
    )
    return {
        "status": "built",
        "valid_images": len(records),
        "invalid_images": len(errors),
        "errors": errors[:100],
        "clip_groups": len(groups),
        "splits": dict(counts),
        "output_image_methods": dict(methods),
    }


def write_report(output_root: Path, report: dict) -> None:
    (output_root / "cleaning_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Dataset Cleaning Report",
        "",
        "The original Roboflow exports were not modified. Ball classes were remapped to `pickleball=0`,",
        "source-frame variants and exact duplicates were removed, negatives were capped, and splits",
        "were reassigned by source clip to prevent adjacent-frame leakage.",
        "",
        "## Result",
        "",
        f"- Images: {sum(report.get('cleaned_splits', {}).values())}",
        f"- Splits: {report.get('cleaned_splits', {})}",
        f"- Clip groups: {report.get('clip_groups', 0)}",
        f"- Source variants removed: {report.get('source_deduplication', {}).get('variants_removed', 0)}",
        f"- Exact duplicates removed: {report.get('exact_deduplication', {}).get('exact_duplicates_removed', 0)}",
        f"- Negatives kept: {report.get('negative_sampling', {}).get('kept_negative_images', 0)}",
        "",
        "See `cleaning_report.json` and `manifest.csv` for full provenance and counts.",
    ]
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path(__file__).resolve().parents[1] / "datasets")
    parser.add_argument("--output-name", default="cleaned_ball_detection")
    parser.add_argument("--court-output-name", default="cleaned_court_keypoints")
    parser.add_argument("--negative-ratio", type=float, default=0.25)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--skip-court", action="store_true")
    args = parser.parse_args()

    records, report = scan_ball_sources(args.dataset_root)
    records = deduplicate_source_variants(records, report)
    if args.audit_only:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    records = deduplicate_exact_images(records, report)
    records = limit_negatives(records, report, args.negative_ratio)
    assign_grouped_splits(records, report)
    output_root = args.dataset_root / args.output_name
    write_ball_dataset(records, output_root, report)
    if not args.skip_court:
        report["court_keypoints"] = clean_court_keypoints(
            args.dataset_root, args.dataset_root / args.court_output_name
        )
    write_report(output_root, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
