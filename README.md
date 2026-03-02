# BDD100K Assignment Submission

A comprehensive computer vision pipeline for object detection on the BDD100K dataset, featuring exploratory data analysis (EDA), model training, and evaluation for two state-of-the-art detectors: YOLO11n and RF-DETR.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Task 1: Exploratory Data Analysis (EDA)](#task-1-exploratory-data-analysis-eda)
- [Task 2: Model Training](#task-2-model-training)
- [Task 3: Model Evaluation](#task-3-model-evaluation)
- [Project Structure](#project-structure)

## Prerequisites
- Python 3.8+
- pip or conda package manager

**Note:** Each task has its own setup requirements. See the setup section under each task below.

## Task 1: Exploratory Data Analysis (EDA)

<div style="font-size: 16px; background-color: #FFEB3B; color: #000; padding: 8px; border-radius: 4px; margin-bottom: 15px;"><strong>⚠️ INSTRUCTIONS:</strong> View dashboard.html side by side with Task-1-Report.pdf for a comprehensive analysis overview.</div>

### Setup
```bash
pip install -r requirements.txt
```

### Running EDA

Run the EDA pipeline to analyze the BDD100K dataset:

```bash
python -m eda_pipeline.main
```

This will:
- Analyze dataset statistics and distributions
- Generate visualizations
- Identify data patterns and anomalies
- Create a comprehensive EDA dashboard

**Output:** Analysis reports and visualizations in the `eda_pipeline/` output directory.

<span style="background-color: #FFEB3B; color: #000; padding: 2px 4px; border-radius: 3px;">**Intructions** View `dashboard.html` side by side with `Task-1-Report.pdf` for a comprehensive analysis overview.</span>

---

## Task 2: Model Training

<div style="font-size: 16px; background-color: #FFEB3B; color: #000; padding: 8px; border-radius: 4px; margin-bottom: 15px;"><strong>⚠️ INSTRUCTIONS:</strong> View Task-2-report.pdf, output and description is captured in it</div>

### Setup
```bash
cd training_scripts
pip install -r requirements-training-rfdetr.txt
```

**Note:** 
- YOLO11n was trained in Kaggle's default environment
- RF-DETR was trained on Jetson Thor

### Training Models

Two object detection models were trained on the BDD100K dataset:

#### YOLO11n Training
*Trained in Kaggle's default environment*

```bash
python yolo-11n-train-bdd100k.ipynb
```

#### RF-DETR Training
*Trained on Jetson Thor*

```bash
python rf-detr-finetuning-v2-lr-ms.py
python finish_evaluation.py
```

**Note:** Jetson Thor has its own requirements file as its a ARM based system and training script of RF-DETR may not work in x86 systems. For convinience pretrained weights have been kept here (Both are needed):
- `rf-detr-medium.pth` - Base RF-DETR model
- `rf_detr_medium_thor_3x_ms_final.pth` - Fine-tuned RF-DETR model

Once we have trained the model we will keep the weights in the main folder so that Task-3 scripts can utilise it. For convinience it has been kept before hand.

**Output:** Trained model weights saved in `training_scripts/rf_detr_output/`

---

## Task 3: Model Evaluation

<div style="font-size: 16px; background-color: #FFEB3B; color: #000; padding: 8px; border-radius: 4px; margin-bottom: 15px;"><strong>⚠️ INSTRUCTIONS:</strong> View Task-3-report.pdf for detailed analysis and execution of the task</div>

### Setup
```bash
pip install -r requirements.txt
```

### Evaluating Models

Evaluate both trained models using their respective evaluation pipelines.

### YOLO11n Evaluation

```bash
# Step 1: Run inference (generates predictions JSON)
python -m yolo11s_eval.inference

# Step 2: Generate all plots & metrics
python -m yolo11s_eval.evaluation

# Step 3: Load data into FiftyOne (only needed once)
python -m yolo11s_eval.voxel51_eval
python -m yolo11s_eval.add_scene_metadata

# Step 4: Launch FiftyOne browser app (reusable anytime after step 2)
python -m yolo11s_eval.fo_launch
```

### RF-DETR Evaluation

```bash
# Step 1: Run inference (generates predictions JSON)
python -m rf_detr_eval.inference

# Step 2: Generate all plots & metrics
python -m rf_detr_eval.evaluation

# Step 3: Load data into FiftyOne (only needed once)
python -m rf_detr_eval.voxel51_eval
python -m rf_detr_eval.add_scene_metadata

# Step 4: Launch FiftyOne browser app (reusable anytime after step 2)
python -m rf_detr_eval.fo_launch
```

**Output:**
- Prediction JSON files (`yolo11s_val_predictions.json`, `rf_detr_val_predictions.json`)
- Performance metrics and visualizations
- FiftyOne dataset for interactive exploration

---

## Project Structure

```
.
├── convert2coco.py                 # Dataset format converter
├── requirements.txt                # Base dependencies
├── eda_pipeline/                   # Task 1: EDA
│   ├── main.py
│   ├── analyzer.py
│   ├── data_loader.py
│   ├── outlier_detector.py
│   ├── visualizer.py
│   └── dashboard.py
├── training_scripts/               # Task 2: Training
│   ├── rf-detr-finetuning-v2-lr-ms.py
│   ├── yolo-11n-train-bdd100k.ipynb
│   ├── requirements-training-rfdetr.txt
│   ├── create-subset.py
│   └── rf_detr_output/
├── yolo11s_eval/                   # Task 3: YOLO11n Evaluation
│   ├── inference.py
│   ├── evaluation.py
│   ├── voxel51_eval.py
│   ├── add_scene_metadata.py
│   ├── fo_launch.py
│   └── config.py
├── rf_detr_eval/                   # Task 3: RF-DETR Evaluation
│   ├── inference.py
│   ├── evaluation.py
│   ├── voxel51_eval.py
│   ├── add_scene_metadata.py
│   ├── fo_launch.py
│   └── config.py
└── Model Weights
    ├── rf-detr-medium.pth
    └── rf_detr_medium_thor_3x_ms_final.pth
```

---

## Notes

- Ensure all dependencies are installed before running each task
- Model weights should be placed in the root directory or configured in respective config files
- FiftyOne visualization requires internet access for initial setup
- For dataset conversion to COCO format, use: `python convert2coco.py`
