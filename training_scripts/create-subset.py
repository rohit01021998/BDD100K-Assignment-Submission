import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
import argparse


class BDD100KODSubset:
    def __init__(self, coco_root, output_dir, subset_ratio=0.1, seed=42):
        """
        Initialize OD subset creator.

        Expects a COCO-style root directory with the structure:
            coco_root/
                train/
                    _annotations.coco.json   (or instances_train.json, etc.)
                    *.jpg / *.png ...
                valid/
                    _annotations.coco.json
                    *.jpg / *.png ...
                test/
                    _annotations.coco.json
                    *.jpg / *.png ...

        The subset will mirror this structure under output_dir so it is a
        fully self-contained COCO dataset.

        Args:
            coco_root:    Path to the root COCO directory (contains train/valid/test).
            output_dir:   Output directory for the subset.
            subset_ratio: Fraction of images to keep (0–1).
            seed:         Random seed for reproducibility.
        """
        self.coco_root = Path(coco_root)
        self.output_dir = Path(output_dir)
        self.subset_ratio = subset_ratio

        random.seed(seed)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Supported image extensions
        self.IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

        # Known annotation file name patterns (priority order)
        self.ANNOTATION_CANDIDATES = [
            '_annotations.coco.json',
            'instances_train.json',
            'instances_val.json',
            'instances_test.json',
            'instances_default.json',
            'annotations.json',
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_annotation_file(self, split_dir: Path) -> Path:
        """Return the annotation JSON inside a split directory."""
        # Try known names first
        for name in self.ANNOTATION_CANDIDATES:
            candidate = split_dir / name
            if candidate.exists():
                return candidate
        # Fallback: any .json file
        jsons = list(split_dir.glob('*.json'))
        if jsons:
            return jsons[0]
        raise FileNotFoundError(
            f"No annotation JSON found in {split_dir}. "
            f"Tried: {self.ANNOTATION_CANDIDATES}"
        )

    def _find_splits(self):
        """Return list of split directory names that exist."""
        splits = []
        for name in ['train', 'valid', 'val', 'test']:
            p = self.coco_root / name
            if p.is_dir():
                splits.append(name)
        if not splits:
            raise FileNotFoundError(
                f"No train/valid/val/test directories found under {self.coco_root}"
            )
        return splits

    def _load_split(self, split_dir: Path):
        """Load a COCO JSON from a split directory."""
        ann_file = self._find_annotation_file(split_dir)
        with open(ann_file, 'r') as f:
            data = json.load(f)
        images     = {img['id']: img for img in data.get('images', [])}
        annotations = data.get('annotations', [])
        categories  = {cat['id']: cat for cat in data.get('categories', [])}
        return data, images, annotations, categories, ann_file

    # ------------------------------------------------------------------
    # Stratification
    # ------------------------------------------------------------------

    def stratify_by_category(self, images, annotations, categories):
        """
        Select a stratified subset of image IDs proportional to each category.
        Images that belong to multiple categories may be counted more than once
        during selection but are de-duplicated in the final set.
        """
        category_images = defaultdict(set)
        for ann in annotations:
            category_images[ann['category_id']].add(ann['image_id'])

        print(f"\n  Original distribution:")
        for cat_id in sorted(category_images):
            count = len(category_images[cat_id])
            pct   = 100 * count / max(len(images), 1)
            print(f"    {categories[cat_id]['name']:20s}: {count:5d} images ({pct:.1f}%)")

        selected_ids = set()
        print(f"\n  Selecting subset (ratio={self.subset_ratio}):")
        for cat_id in sorted(category_images):
            pool = list(category_images[cat_id])
            n    = max(1, int(len(pool) * self.subset_ratio))
            chosen = random.sample(pool, n)
            selected_ids.update(chosen)
            print(f"    {categories[cat_id]['name']:20s}: {n}/{len(pool)}")

        return selected_ids

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_subset(self, subset_data, selected_ids, split_name):
        """Validate subset JSON consistency."""
        errors = []

        img_ids = {img['id'] for img in subset_data['images']}
        for ann in subset_data['annotations']:
            if ann['image_id'] not in img_ids:
                errors.append(f"Annotation {ann['id']} references missing image {ann['image_id']}")
            for field in ['id', 'image_id', 'category_id', 'bbox', 'area']:
                if field not in ann:
                    errors.append(f"Annotation {ann.get('id','?')} missing field '{field}'")

        if not subset_data['images']:
            errors.append("No images in subset!")
        if not subset_data['annotations']:
            errors.append("No annotations in subset!")

        if errors:
            print(f"  ✗ [{split_name}] {len(errors)} validation error(s):")
            for e in errors[:5]:
                print(f"      - {e}")
            return False
        print(f"  ✓ [{split_name}] Validation passed.")
        return True

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def compute_statistics(self, original_data, subset_data, split_name):
        orig_imgs  = original_data['images']
        orig_anns  = original_data.get('annotations', [])
        sub_imgs   = subset_data['images']
        sub_anns   = subset_data['annotations']

        print(f"\n  📈 [{split_name}] Statistics:")
        print(f"    Images:      {len(orig_imgs):6d}  →  {len(sub_imgs):6d}  "
              f"({100*len(sub_imgs)/max(len(orig_imgs),1):.1f}%)")
        print(f"    Annotations: {len(orig_anns):6d}  →  {len(sub_anns):6d}  "
              f"({100*len(sub_anns)/max(len(orig_anns),1):.1f}%)")

        orig_cat = defaultdict(int)
        for ann in orig_anns:
            orig_cat[ann['category_id']] += 1
        sub_cat = defaultdict(int)
        for ann in sub_anns:
            sub_cat[ann['category_id']] += 1

        categories = {cat['id']: cat for cat in original_data.get('categories', [])}
        max_diff = 0
        for cat_id in sorted(categories):
            o = 100 * orig_cat[cat_id] / max(len(orig_anns), 1)
            s = 100 * sub_cat[cat_id]  / max(len(sub_anns),  1)
            d = abs(o - s)
            max_diff = max(max_diff, d)
            print(f"    {categories[cat_id]['name']:20s}: {o:5.1f}% → {s:5.1f}%  (Δ {d:.1f}%)")

        if max_diff < 5:
            print(f"    ✓ Excellent distribution match (max Δ {max_diff:.1f}%)")
        elif max_diff < 10:
            print(f"    ✓ Good distribution match (max Δ {max_diff:.1f}%)")
        else:
            print(f"    ⚠ Distribution differs significantly (max Δ {max_diff:.1f}%)")

    # ------------------------------------------------------------------
    # Image copying
    # ------------------------------------------------------------------

    def _copy_images(self, selected_ids, images_map, src_dir: Path, dst_dir: Path, split_name: str):
        """
        Copy selected image files from src_dir to dst_dir.
        Images are looked up by filename stored in the COCO metadata.
        Returns the set of image IDs whose files were actually found & copied.
        """
        dst_dir.mkdir(parents=True, exist_ok=True)
        copied, missing = 0, 0

        for img_id in selected_ids:
            img_meta = images_map.get(img_id)
            if img_meta is None:
                missing += 1
                continue

            # 'file_name' may contain sub-paths; take the basename
            file_name = Path(img_meta['file_name']).name
            src = src_dir / file_name

            # If not found at basename, try the full relative path
            if not src.exists():
                src = src_dir / img_meta['file_name']

            if src.exists():
                shutil.copy2(src, dst_dir / file_name)
                copied += 1
            else:
                missing += 1

        print(f"  📁 [{split_name}] Copied {copied} images "
              f"({missing} not found on disk — JSON-only subset still valid).")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def create_subset(self):
        splits = self._find_splits()
        print(f"\n🔍 Found splits: {splits}")
        print(f"   Source root : {self.coco_root}")
        print(f"   Output root : {self.output_dir}")
        print(f"   Ratio       : {self.subset_ratio}")

        all_valid = True

        for split in splits:
            split_dir = self.coco_root / split
            print(f"\n{'='*60}")
            print(f"  Processing split: [{split}]")
            print(f"{'='*60}")

            out_split_dir = self.output_dir / split

            # ── Non-training splits: copy everything as-is ──────────────
            if split != 'train':
                print(f"  ⏭  Non-training split — copying as-is (no subsetting).")
                if out_split_dir.resolve() != split_dir.resolve():
                    shutil.copytree(split_dir, out_split_dir, dirs_exist_ok=True)
                    print(f"  📁 Copied {split_dir} → {out_split_dir}")
                else:
                    print(f"  ℹ  Source and destination are the same — skipping copy.")
                continue

            # ── Training split: stratify & subset ───────────────────────
            # Load
            original_data, images, annotations, categories, ann_file = \
                self._load_split(split_dir)
            print(f"  ✓ Loaded {len(images)} images, {len(annotations)} annotations, "
                  f"{len(categories)} categories  [{ann_file.name}]")

            # Stratify
            selected_ids = self.stratify_by_category(images, annotations, categories)

            # Build subset JSON
            subset_images = [img for img in original_data['images']
                             if img['id'] in selected_ids]
            subset_anns   = [ann for ann in annotations
                             if ann['image_id'] in selected_ids]

            subset_data = {
                'info':        original_data.get('info', {}),
                'licenses':    original_data.get('licenses', []),
                'categories':  original_data.get('categories', []),
                'images':      subset_images,
                'annotations': subset_anns,
            }

            # Validate
            ok = self.validate_subset(subset_data, selected_ids, split)
            all_valid = all_valid and ok

            # Statistics
            self.compute_statistics(original_data, subset_data, split)

            # Create output split directory
            out_split_dir.mkdir(parents=True, exist_ok=True)

            # Save annotation JSON with the same filename as the source
            out_ann_file = out_split_dir / ann_file.name
            with open(out_ann_file, 'w') as f:
                json.dump(subset_data, f)
            print(f"\n  💾 Annotation saved → {out_ann_file}")

            # Copy only selected image files
            self._copy_images(selected_ids, images, split_dir, out_split_dir, split)

        print(f"\n{'='*60}")
        if all_valid:
            print(f"✅  All splits processed successfully!")
        else:
            print(f"⚠   Some splits had validation issues — check output above.")
        print(f"📂  Self-contained subset at: {self.output_dir}")
        print(f"{'='*60}\n")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            'Create a category-stratified subset of a COCO-format BDD100K dataset.\n\n'
            'The script expects a root directory with train/valid/test sub-folders,\n'
            'each containing images and one COCO annotation JSON file.\n'
            'The output mirrors this structure and is fully self-contained.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'coco_root',
        nargs='?',
        help='Path to root COCO directory (contains train/valid/test sub-folders)',
    )
    parser.add_argument(
        '--coco-root', '--coco_root',
        dest='coco_root_opt',
        help='Path to root COCO directory (flag form)',
    )
    parser.add_argument(
        '--output-dir',
        default='./bdd100k_subset',
        help='Output directory (default: ./bdd100k_subset)',
    )
    parser.add_argument(
        '--ratio',
        type=float,
        default=0.1,
        help='Subset ratio 0–1 (default: 0.1 → 10%%)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)',
    )

    args = parser.parse_args()
    coco_root = args.coco_root_opt or args.coco_root

    if not coco_root:
        parser.error(
            "'coco_root' is required — pass it as a positional argument "
            "or via --coco-root."
        )

    creator = BDD100KODSubset(
        coco_root=coco_root,
        output_dir=args.output_dir,
        subset_ratio=args.ratio,
        seed=args.seed,
    )
    creator.create_subset()


if __name__ == '__main__':
    main()