import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm


def convert_bdd_to_coco(bdd_json_path, img_dir, coco_out_path, out_img_dir):
    print(f"\n{'='*60}")
    print(f"Loading BDD JSON from {bdd_json_path}...")
    with open(bdd_json_path, 'r') as f:
        bdd_data = json.load(f)

    class_names = [
        'bike', 'traffic sign', 'train', 'traffic light',
        'motor', 'truck', 'person', 'rider', 'bus', 'car'
    ]

    coco = {
        "info": {"description": "BDD100k converted to COCO for RF-DETR"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": i + 1, "name": name, "supercategory": "none"}
            for i, name in enumerate(class_names)
        ]
    }

    category_to_id = {c['name']: c['id'] for c in coco['categories']}
    annotation_id = 1
    skipped_images = 0
    skipped_labels = 0

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(os.path.dirname(coco_out_path), exist_ok=True)

    print(f"Converting and copying images to {out_img_dir}...")
    for img_id, item in enumerate(tqdm(bdd_data)):
        img_name = item['name']
        src = os.path.join(img_dir, img_name)
        dst = os.path.join(out_img_dir, img_name)

        if not os.path.exists(src):
            skipped_images += 1
            continue

        shutil.copy2(src, dst)

        coco['images'].append({
            "id": img_id,
            "file_name": img_name,
            "width": 1280,
            "height": 720
        })

        for label in item.get('labels', []):
            if 'box2d' not in label:
                continue

            cat_name = label.get('category')
            if cat_name not in category_to_id:
                skipped_labels += 1
                continue

            box2d = label['box2d']
            x1 = max(0, box2d['x1'])
            y1 = max(0, box2d['y1'])
            x2 = min(1280, box2d['x2'])
            y2 = min(720, box2d['y2'])
            w = x2 - x1
            h = y2 - y1

            if w <= 0 or h <= 0:
                skipped_labels += 1
                continue

            coco['annotations'].append({
                "id": annotation_id,
                "image_id": img_id,
                "category_id": category_to_id[cat_name],
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "iscrowd": 0,
                "segmentation": []
            })
            annotation_id += 1

    print(f"Saving COCO JSON to {coco_out_path}...")
    with open(coco_out_path, 'w') as f:
        json.dump(coco, f, indent=2)

    print(f"✓ Images:      {len(coco['images'])}")
    print(f"✓ Annotations: {len(coco['annotations'])}")
    print(f"✗ Skipped images (not found): {skipped_images}")
    print(f"✗ Skipped labels (bad class/bbox): {skipped_labels}")


def create_test_split_symlink(out_dir):
    """RF-DETR wants a test/ folder — symlink valid/ → test/ if no test split exists."""
    test_dir = os.path.join(out_dir, 'test')
    valid_dir = os.path.join(out_dir, 'valid')
    if not os.path.exists(test_dir):
        print(f"\nNo test split found — copying valid/ to test/ for RF-DETR compatibility...")
        shutil.copytree(valid_dir, test_dir)
        print("✓ test/ created from valid/")


if __name__ == '__main__':
    # ── Input paths (auto-detected from script location) ─────────
    base_dir    = str(Path(__file__).resolve().parent)
    labels_dir  = os.path.join(base_dir, 'assignment_data_bdd/bdd100k_labels_release/bdd100k/labels')
    images_dir  = os.path.join(base_dir, 'assignment_data_bdd/bdd100k_images_100k/bdd100k/images/100k')

    # ── Output path (separate folder, RF-DETR ready) ──────────────
    out_dir = os.path.join(base_dir, 'coco_format_full')

    # ── Splits: (bdd_split_name, output_folder_name) ──────────────
    # RF-DETR requires "valid" not "val"
    splits = [
        ('train', 'train'),
        ('val',   'valid'),
    ]

    for bdd_split, out_split in splits:
        print(f"\n>>> Processing split: {bdd_split} → {out_split}/")
        convert_bdd_to_coco(
            bdd_json_path = os.path.join(labels_dir, f'bdd100k_labels_images_{bdd_split}.json'),
            img_dir       = os.path.join(images_dir, bdd_split),
            coco_out_path = os.path.join(out_dir, out_split, '_annotations.coco.json'),
            out_img_dir   = os.path.join(out_dir, out_split),
        )

    # Create test/ split for RF-DETR (copy of valid/)
    create_test_split_symlink(out_dir)

    print(f"\n{'='*60}")
    print(f"✓ Conversion complete!")
    print(f"  Dataset ready at: {out_dir}")
    print(f"  Structure:")
    print(f"    coco_output/")
    print(f"    ├── train/")
    print(f"    │   ├── _annotations.coco.json")
    print(f"    │   └── *.jpg")
    print(f"    ├── valid/")
    print(f"    │   ├── _annotations.coco.json")
    print(f"    │   └── *.jpg")
    print(f"    └── test/  (copy of valid/)")
    print(f"{'='*60}")
    print(f"\nTo train RF-DETR:")
    print(f"  from rfdetr import RFDETRBase")
    print(f"  model = RFDETRBase()")
    print(f"  model.train(dataset_dir='{out_dir}', epochs=50, batch_size=4)")