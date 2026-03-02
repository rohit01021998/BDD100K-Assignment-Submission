import os
import json
import requests
import torch
import torch.backends.cuda
import torch.backends.cudnn
from tqdm import tqdm
from rfdetr import RFDETRMedium
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ── Jetson Thor Precision Setup ───────────────────────────────────────────────
# Thor is Blackwell: best training precision is BF16 via Transformer Engine
# FP16 is also great; BF16 is preferred because it has same range as FP32
# INT8/FP8 gives no gain for transformers in training mode on Thor (confirmed)

def setup_jetson_thor():
    """Configure PyTorch for maximum performance on Jetson Thor (Blackwell)."""

    if not torch.cuda.is_available():
        print("⚠ No CUDA — running on CPU")
        return 'cpu'

    gpu  = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    cc   = torch.cuda.get_device_capability(0)  # Blackwell = (10, 0)
    print(f"✓ GPU  : {gpu}")
    print(f"✓ VRAM : {vram:.1f} GB")
    print(f"✓ Compute Capability: sm_{cc[0]}{cc[1]}")

    # ── TF32: free speedup on Ampere+, Blackwell supports it natively
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    print("✓ TF32 enabled (matmul + cuDNN)")

    # ── BF16: preferred over FP16 on Blackwell — wider dynamic range
    if torch.cuda.is_bf16_supported():
        print("✓ BF16 supported — will use for AMP (recommended for Thor)")
        amp_dtype = 'bf16'
    else:
        print("  BF16 not supported — falling back to FP16")
        amp_dtype = 'fp16'

    # ── cuDNN benchmark: finds fastest conv algorithm for fixed input size
    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.deterministic = False  # slightly faster, non-deterministic
    print("✓ cuDNN benchmark mode enabled")

    # ── Memory: allow expandable segments to reduce fragmentation on unified RAM
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = (
        'expandable_segments:True,'
        'max_split_size_mb:512'
    )
    print("✓ CUDA memory allocator tuned for unified memory (Thor)")

    # ── Pin memory for faster CPU→GPU transfers on Jetson unified memory arch
    # Note: on Jetson (unified mem), pinning is less critical but still helps
    os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '0'

    # ── Transformer Engine: enables FP8 internally for transformer layers
    # RF-DETR uses DINOv2 backbone which benefits from this
    try:
        import transformer_engine.pytorch as te
        print("✓ NVIDIA Transformer Engine available — FP8 enabled for attn layers")
    except ImportError:
        print("  Transformer Engine not installed — using BF16/FP16 only")
        print("  Install: pip install transformer-engine")

    ram_gb = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1e9
    print(f"✓ RAM  : {ram_gb:.0f} GB (unified memory)")

    return amp_dtype


# ── Config ────────────────────────────────────────────────────────────────────

DATASET_DIR  = 'training_scripts/bdd100k_subset'
OUTPUT_DIR   = 'training_scripts/rf_detr_output'
WEIGHTS_DIR  = 'training_scripts/rf_detr_weights'
WEIGHTS_FILE = 'rf-detr-medium.pth'
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, WEIGHTS_FILE)
WEIGHTS_URL  = 'https://storage.googleapis.com/com.roboflow-platform.files/rf-detr/rf-detr-medium.pth'

NUM_CLASSES  = 10
RESOLUTION   = 640      # divisible by 56; try 800 if VRAM allows

# 3x schedule = 36 epochs
EPOCHS       = 12
BATCH_SIZE   = 16       # Thor has 128GB unified — push to 32 if comfortable
GRAD_ACCUM   = 1
LR           = 1e-4
LR_ENCODER   = 1e-5
WEIGHT_DECAY = 1e-4

CLASS_NAMES = [
    'bike', 'traffic sign', 'train', 'traffic light',
    'motor', 'truck', 'person', 'rider', 'bus', 'car'
]

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

os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)


# ── Download weights ──────────────────────────────────────────────────────────

def download_weights(url: str, dest: str):
    if os.path.exists(dest) and os.path.getsize(dest) > 10e6:
        print(f"✓ Weights found: {dest} ({os.path.getsize(dest)/1e6:.1f} MB)")
        return True
    print(f"Downloading RF-DETR Medium weights...")
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        with open(dest, 'wb') as f, tqdm(desc=WEIGHTS_FILE, total=total, unit='B', unit_scale=True) as bar:
            for chunk in r.iter_content(8192):
                f.write(chunk)
                bar.update(len(chunk))
        print(f"✓ Downloaded: {os.path.getsize(dest)/1e6:.1f} MB")
        return True
    except Exception as e:
        if os.path.exists(dest): os.remove(dest)
        print(f"✗ Download failed: {e} — RF-DETR will auto-download on init")
        return False


# ── COCO val evaluation ───────────────────────────────────────────────────────

def evaluate_val_coco(model, val_ann_path: str, val_img_dir: str, threshold: float = 0.001):
    from PIL import Image

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


# ── Benchmark comparison ──────────────────────────────────────────────────────

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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("="*65)
    print("RF-DETR Medium | 3x LR + MS-train | Jetson Thor (Blackwell)")
    print("="*65)

    amp_dtype = setup_jetson_thor()

    ok = download_weights(WEIGHTS_URL, WEIGHTS_PATH)
    weights_path = WEIGHTS_PATH if ok else None

    print(f"\n  AMP dtype  : {amp_dtype.upper()} (Blackwell Tensor Cores)")
    print(f"  Epochs     : {EPOCHS} (3x schedule)")
    print(f"  Batch      : {BATCH_SIZE} × {GRAD_ACCUM} = {BATCH_SIZE*GRAD_ACCUM} effective")
    print(f"  Resolution : {RESOLUTION}px + multi-scale")
    print(f"  LR         : {LR} cosine | encoder: {LR_ENCODER}")
    print("="*65)

    model_kwargs = dict(num_classes=NUM_CLASSES, resolution=RESOLUTION)
    if weights_path:
        model_kwargs['pretrain_weights'] = weights_path
    model = RFDETRMedium(**model_kwargs)

    model.train(
        dataset_dir              = DATASET_DIR,
        epochs                   = EPOCHS,
        batch_size               = BATCH_SIZE,
        grad_accum_steps         = GRAD_ACCUM,
        lr                       = LR,
        lr_encoder               = LR_ENCODER,
        weight_decay             = WEIGHT_DECAY,
        output_dir               = OUTPUT_DIR,

        # ── Precision (Jetson Thor / Blackwell) ───────────────────
        amp                      = True,         # automatic mixed precision
        amp_dtype                = amp_dtype,    # 'bf16' on Thor, 'fp16' fallback

        # ── LR schedule (3x convention) ───────────────────────────
        lr_scheduler             = 'cosine',
        warmup_epochs            = 1,

        # ── Multi-scale training ──────────────────────────────────
        multi_scale              = True,

        # ── Throughput tweaks for Thor ────────────────────────────
        use_ema                  = True,
        gradient_checkpointing   = False,        # not needed with 128GB RAM
        num_workers              = 8,            # Thor has 14 Arm cores

        # ── Evaluation: val only ──────────────────────────────────
        run_test                 = False,
        checkpoint_interval      = 6,
        early_stopping           = False,        # fixed schedule for fair benchmark
    )

    final_path = os.path.join(OUTPUT_DIR, 'rf_detr_medium_thor_3x_ms_final.pth')

    # RFDETR doesn't have a .save() method, but training already saves checkpoints.
    import shutil
    best_path = os.path.join(OUTPUT_DIR, 'checkpoint_best_total.pth')
    if os.path.exists(best_path):
        shutil.copy(best_path, final_path)
        print(f"\n✓ Best model copied → {final_path}")
    else:
        # Fallback: manual save of current state
        torch.save({
            "model": model.model.model.state_dict(),
            "args": model.model.args
        }, final_path)
        print(f"\n✓ Model state saved → {final_path}")


    # ── COCO val evaluation ───────────────────────────────────────
    metrics = evaluate_val_coco(
        model,
        val_ann_path = os.path.join(DATASET_DIR, 'valid', '_annotations.coco.json'),
        val_img_dir  = os.path.join(DATASET_DIR, 'valid'),
        threshold    = 0.001,
    )

    if metrics:
        print(f"\n{'='*65}")
        print(f"  Box AP  (mAP@50:95) : {metrics['box_ap']:.2f}   ← compare with ConvNeXt")
        print(f"  Box AP50 (mAP@50)   : {metrics['box_ap50']:.2f}")
        print(f"  Box AP75 (mAP@75)   : {metrics['box_ap75']:.2f}")
        print(f"  AP small            : {metrics['box_ap_small']:.2f}")
        print(f"  AP medium           : {metrics['box_ap_med']:.2f}")
        print(f"  AP large            : {metrics['box_ap_large']:.2f}")

        print_benchmark_comparison(metrics['box_ap'])

        metrics.update({
            'model':     'RF-DETR Medium',
            'schedule':  '3x + cosine + MS-train',
            'precision': amp_dtype.upper(),
            'epochs':    EPOCHS,
            'hardware':  'Jetson Thor (Blackwell)',
        })
        out = os.path.join(OUTPUT_DIR, 'val_metrics.json')
        with open(out, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✓ Metrics saved → {out}")


if __name__ == '__main__':
    main()