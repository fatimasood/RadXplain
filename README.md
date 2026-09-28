# RadXplain — Explainable Chest X-Ray Abnormality Detection

RadXplain is an end-to-end computer vision system that detects and localizes
chest X-ray abnormalities, and shows why it flagged each finding using
EigenCAM-based visual explanations. Built as a research-grade, deployable demo
to explore how explainability can be designed into an object detection pipeline
for medical imaging from the start, rather than bolted on afterward.

> **Disclaimer:** This is a research/educational project. It is not validated
> for, and must not be used for, clinical diagnosis.

## Live demo

The model is deployed on Hugging Face Spaces:

https://fatemadevv-explainable-cxr-detector.hf.space

Upload a chest X-ray and you will get back bounding boxes for detected
abnormalities, plus an EigenCAM heatmap for each finding showing which
regions of the image drove the prediction.

## Why this project

Chest X-rays are the most commonly performed diagnostic imaging exam
worldwide, and radiologist time is a real bottleneck. Automated detection
can act as a triage aid or a second opinion, but black-box flagging tools
have a well-known trust problem in clinical settings. RadXplain pairs each
detection with a visual explanation of which pixels drove that prediction,
treating interpretability as a first-class requirement rather than an
afterthought.

## What it does

- Upload a chest X-ray and get bounding boxes for detected abnormalities,
  each labeled with class and confidence.
- Each detection comes with an EigenCAM heatmap overlay showing the regions
  that drove the prediction.
- Available as a web demo (Gradio) on Hugging Face Spaces. A REST API
  endpoint is available at /predict on the same Space for programmatic
  access.

  <img width="981" height="392" alt="image" src="https://github.com/user-attachments/assets/954beb9d-2383-4c12-a043-b23237762215" />

The three panels show, from left to right: the original chest X-ray, the
EigenCAM heatmap highlighting the regions the model attended to, and the
final detections with class labels and confidence scores. In this example
the model detects aortic enlargement (0.68) and pulmonary fibrosis
(0.31-0.55) across both lungs.

## Dataset

VinDr-CXR / VinBigData Chest X-ray Abnormalities Detection:
https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection

18,000 chest radiographs, radiologist-annotated (3 independent readers per
training image, 5-reader consensus on the test set), 14 abnormality classes
plus "No finding." Multiple radiologists' boxes for the same image are fused
with Weighted Boxes Fusion into one consensus set of labels before training.

This project uses the roughly 4,394 abnormal scans (excluding "No finding")
and focuses on the 8 most frequent classes. The remaining classes are
documented as a limitation rather than silently ignored. A free Kaggle
account and accepting the competition rules are required before downloading.
The data is used here for research and portfolio purposes only.

## Method

| Stage          | Approach                                                                 |
|----------------|--------------------------------------------------------------------------|
| Detection      | YOLOv8s (Ultralytics), fine-tuned from COCO-pretrained weights           |
| Input size     | 1024x1024 (higher than the 640 default — small nodules need resolution)  |
| Box fusion     | Weighted Boxes Fusion (ensemble-boxes) across the 3 radiologists         |
| Explainability | EigenCAM (pytorch_grad_cam), adapted for the YOLO detection head         |
| Export         | ONNX + onnxruntime for fast CPU inference                                |
| Serving        | Gradio frontend on Hugging Face Spaces, ONNX Runtime backend             |
| Evaluation     | mAP@0.5 and mAP@0.5:0.95, per-class AP                                   |

## Results

Evaluated on the held-out test set (657 images, 2,908 annotated instances).

| Metric        | Value |
|---------------|-------|
| mAP@0.5       | 0.331 |
| mAP@0.5:0.95  | 0.175 |
| Precision     | 0.633 |
| Recall        | 0.364 |

Per-class AP@0.5:

| Class                | AP    |
|----------------------|-------|
| Aortic enlargement   | 0.878 |
| Cardiomegaly         | 0.864 |
| Pleural effusion     | 0.327 |
| Nodule/Mass          | 0.200 |
| Pulmonary fibrosis   | 0.192 |
| Pleural thickening   | 0.101 |
| Lung Opacity         | 0.060 |
| Other lesion         | 0.026 |

Large anatomical features such as the heart and aorta are detected
robustly. Small, subtle, or ambiguous findings remain challenging even at
1024x1024 resolution. This is consistent with what the broader medical
imaging literature reports, and is treated here as a documented limitation
rather than something the model has solved.

## Reproducing

1. Download VinDr-CXR from Kaggle and accept the competition rules.
2. Run the data pipeline: DICOM to PNG conversion with CLAHE contrast
   enhancement, Weighted Boxes Fusion (IoU 0.4) across radiologist
   annotations, and YOLO format conversion for the 8 selected classes.
3. Train:

       yolo detect train data=dataset.yaml model=yolov8s.pt epochs=100 imgsz=1024 batch=8 optimizer=AdamW cos_lr=True

4. Export to ONNX:

       from ultralytics import YOLO
       model = YOLO("best.pt")
       model.export(format="onnx", imgsz=1024, opset=12)

5. Run the demo locally:

       pip install -r deployment/requirements.txt
       python deployment/app.py

## Limitations

- Research and portfolio project, not a clinical tool. No diagnostic
  claims are made.
- Trained on 8 of the 14 classes. Rare classes (pneumothorax,
  atelectasis, consolidation, calcification, ILD, infiltration) are
  excluded and would require additional work.
- Single-institution dataset (Vietnamese hospitals). Performance on other
  populations, scanners, or acquisition protocols is untested.
- No external validation. Results are on the VinDr-CXR held-out test set
  only.
- Small features such as nodules and lung opacity still underperform.

## Acknowledgements

Dataset: VinDr-CXR (Nguyen et al.), released via the VinBigData Chest
X-ray Abnormalities Detection Kaggle challenge, subject to its own data
use agreement.

## License

Code released under the MIT license. YOLOv8 itself is AGPL-3.0 licensed
by Ultralytics.
