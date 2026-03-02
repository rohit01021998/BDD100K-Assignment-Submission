import os
import json
import torch
from rfdetr import RFDETRMedium
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from PIL import Image

# ── Config (Match the original script) ────────────────────────────────────────
DATASET_DIR  = '/home/thora/Music/Analysis/bdd100k_subset'
OUTPUT_DIR   = '/home/thora/Music/Analysis/rf_detr_output'
FINAL_WEIGHTS = os.path.join(OUTPUT_DIR, 'rf_detr_medium_thor_3x_ms_final.pth')

NUM_CLASSES  = 10
RESOLUTION   = 640

CONVNEXT_BENCHMARKS = {
    'ConvNeXt-T  + Faster R-CNN (1x)':      33.32,
    'ConvNeXt-T  + Faster R-CNN (3x+MS)':   34.08,
    'ConvNeXt-S  + Faster R-CNN (3x+MS)':   34.79,
    'ConvNeXt-B  + Faster R-CNN (3x+MS)':   33.92,
    'ConvNeXt-T  + Cascade R-CNN (1x)':     35.51,
    'ConvNeXt-T  + Cascade R-CNN (3x+MS)':  35.84,
    'ConvNeXt-S  + Cascade R-CNN (3x+MS)':  36.11,
    'ConvNeXt-B  + Cascade R-CNN (3x+MS)':  35.77,
}

def evaluate_val_coco(model, val_ann_path: str, val_img_dir: str, threshold: float = 0.001):
    print("\n" + "="*65)
    print("Running COCO evaluation on val set...")
    coco_gt = COCO(val_ann_path)
    img_ids = coco_gt.getImgIds()
    results = []

    for img_id in tqdm(img_ids, desc="Evaluating"):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(val_img_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            continue
        img  = Image.open(img_path).convert('RGB')
        dets = model.predict(img, threshold=threshold)
        for i in range(len(dets.xyxy)):
            x1, y1, x2, y2 = dets.xyxy[i]
            results.append({
                'image_id':    img_id,
                'category_id': int(dets.class_id[i]) + 1,
                'bbox':        [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                'score':       float(dets.confidence[i]),
            })

    if not results:
        print("✗ No detections — check threshold or model")
        return None

    results_path = os.path.join(OUTPUT_DIR, 'coco_val_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f)

    coco_dt   = coco_gt.loadRes(results_path)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        'box_ap':       coco_eval.stats[0] * 100,
        'box_ap50':     coco_eval.stats[1] * 100,
        'box_ap75':     coco_eval.stats[2] * 100,
        'box_ap_small': coco_eval.stats[3] * 100,
        'box_ap_med':   coco_eval.stats[4] * 100,
        'box_ap_large': coco_eval.stats[5] * 100,
    }

def print_benchmark_comparison(our_ap: float):
    print("\n" + "="*65)
    print(f"{'Model':<42} {'Box AP':>8}  {'vs Ours':>8}")
    print("-"*65)
    print(f"{'★ RF-DETR Medium (3x+MS) BF16 [OURS]':<42} {our_ap:>8.2f}")
    print("-"*65)
    for name, ap in sorted(CONVNEXT_BENCHMARKS.items(), key=lambda x: x[1]):
        diff = our_ap - ap
        marker = '✓' if our_ap > ap else '✗'
        print(f"{marker} {name:<40} {ap:>8.2f}  {diff:>+7.2f}")
    print("="*65)
    beats = sum(1 for ap in CONVNEXT_BENCHMARKS.values() if our_ap > ap)
    print(f"\nRF-DETR Medium beats {beats}/{len(CONVNEXT_BENCHMARKS)} ConvNeXt benchmarks")

def main():
    print("Loading model and final weights...")
    model = RFDETRMedium(num_classes=NUM_CLASSES, resolution=RESOLUTION)
    
    # We must match the checkpoint (10 classes + 1 background = 11).
    model.model.reinitialize_detection_head(NUM_CLASSES + 1)
    
    model_state = torch.load(FINAL_WEIGHTS, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=False)
    model.model.model.load_state_dict(model_state['model'])
    print(f"✓ Weights loaded from {FINAL_WEIGHTS}")

    metrics = evaluate_val_coco(
        model,
        val_ann_path = os.path.join(DATASET_DIR, 'valid', '_annotations.coco.json'),
        val_img_dir  = os.path.join(DATASET_DIR, 'valid'),
        threshold    = 0.001,
    )

    if metrics:
        print(f"\n{'='*65}")
        print(f"  Box AP  (mAP@50:95) : {metrics['box_ap']:.2f}")
        print(f"  Box AP50 (mAP@50)   : {metrics['box_ap50']:.2f}")
        print(f"  Box AP75 (mAP@75)   : {metrics['box_ap75']:.2f}")
        
        print_benchmark_comparison(metrics['box_ap'])

        out = os.path.join(OUTPUT_DIR, 'val_metrics_restored.json')
        with open(out, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✓ Metrics saved → {out}")

if __name__ == '__main__':
    main()
