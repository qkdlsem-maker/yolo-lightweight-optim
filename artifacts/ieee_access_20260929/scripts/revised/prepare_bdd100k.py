"""
BDD100K det_20 JSON 라벨을 YOLO 학습용 txt 포맷으로 변환.

사용:
    python prepare_bdd100k.py \
        --bdd-root ../data/raw/bdd100k \
        --out-root ../data/bdd100k_yolo

출력 구조:
    out-root/
    ├── images/{train,val}/*.jpg   (원본 이미지 심볼릭 링크)
    └── labels/{train,val}/*.txt   (YOLO 포맷: class cx cy w h, 0~1 정규화)
"""
import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm

# Match configs/bdd100k.yaml. The archived experiments used an externally
# converted dataset; this converter is not evidence of its provenance.
CLASS_MAP = {
    "pedestrian": 0,
    "rider": 1,
    "car": 2,
    "truck": 4,
    "bus": 3,
    "train": 9,
    "motorcycle": 6,
    "bicycle": 5,
    "traffic light": 7,
    "traffic sign": 8,
}

IMG_W, IMG_H = 1280, 720  # BDD100K 원본 해상도 고정값


def convert_split(bdd_root: Path, out_root: Path, split: str):
    label_file = bdd_root / "labels" / "det_20" / f"det_{split}.json"
    img_dir = bdd_root / "images" / "100k" / split

    out_img_dir = out_root / "images" / split
    out_lbl_dir = out_root / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    with open(label_file, "r") as f:
        data = json.load(f)

    skipped_no_image = 0
    for item in tqdm(data, desc=f"[{split}] 변환 중"):
        name = item["name"]
        stem = Path(name).stem
        src_img = img_dir / name
        if not src_img.exists():
            skipped_no_image += 1
            continue

        # 이미지 심볼릭 링크 (복사 대신 -> 용량 절약)
        dst_img = out_img_dir / name
        if not dst_img.exists():
            os.symlink(src_img.resolve(), dst_img)

        lines = []
        for label in item.get("labels", []):
            category = label.get("category")
            if category not in CLASS_MAP:
                continue
            box = label.get("box2d")
            if box is None:
                continue
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            cx = (x1 + x2) / 2 / IMG_W
            cy = (y1 + y2) / 2 / IMG_H
            w = (x2 - x1) / IMG_W
            h = (y2 - y1) / IMG_H
            cls_id = CLASS_MAP[category]
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        with open(out_lbl_dir / f"{stem}.txt", "w") as lf:
            lf.write("\n".join(lines))

    print(f"[{split}] 완료. 이미지 누락으로 스킵된 항목: {skipped_no_image}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bdd-root", type=str, required=True)
    parser.add_argument("--out-root", type=str, required=True)
    args = parser.parse_args()

    bdd_root = Path(args.bdd_root)
    out_root = Path(args.out_root)

    for split in ["train", "val"]:
        convert_split(bdd_root, out_root, split)


if __name__ == "__main__":
    main()
