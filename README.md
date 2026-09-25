# RadXplain — Explainable Chest X-Ray Abnormality Detection

RadXplain is an end-to-end computer vision system that detects and localizes 14 types
of thoracic abnormalities on chest X-rays, and shows *why* it flagged each finding
using EigenCAM-based visual explanations. Built as a research-grade, deployable demo
to explore how explainability can be designed into an object detection pipeline for
medical imaging from the start, rather than bolted on afterward.

> ⚠️ **Disclaimer:** This is a research/educational project. It is not validated for,
> and must not be used for, clinical diagnosis.

## Why this project

Chest X-rays are the most commonly performed diagnostic imaging exam worldwide, and
radiologist time is a real bottleneck. Automated detection can act as a triage aid or
a second opinion but black-box flagging tools have a well-known trust problem in
clinical settings. RadXplain pairs each detection with a visual explanation of *which
pixels* drove that prediction, treating interpretability as a first-class requirement
rather than an afterthought.

## What it does

- Upload a chest X-ray → get back bounding boxes for detected abnormalities, each
  labeled with class and confidence.
- Each detection comes with an EigenCAM heatmap overlay showing the regions that
  drove that prediction.
- Available two ways:
  - A web demo (Streamlit)
  - A REST API (`POST /predict`)

## Dataset

[VinDr-CXR / VinBigData Chest X-ray Abnormalities Detection](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection)
— 18,000 chest radiographs, radiologist-annotated (3 independent readers per training
image, 5-reader consensus on the test set), 14 abnormality classes plus "No finding."
Multiple radiologists' boxes for the same image are fused with Weighted Boxes Fusion
into one consensus set of labels before training. You'll need a (free) Kaggle account
and to accept the competition rules before downloading — this project uses the data
for research/portfolio purposes only.

## Method

| Stage          | Approach                                                              |
|----------------|------------------------------------------------------------------------|
| Detection      | YOLOv8 (Ultralytics), fine-tuned from COCO-pretrained weights          |
| Box fusion     | Weighted Boxes Fusion (`ensemble-boxes`) across the 3 radiologists     |
| Explainability | EigenCAM (`pytorch_grad_cam`), adapted for the YOLO detection head     |
| Export         | ONNX + `onnxruntime` for fast CPU inference                            |
| Serving        | FastAPI backend, Streamlit frontend, Docker                            |
| Evaluation     | mAP@0.4 (matching the original challenge's metric), per-class AP       |


## Limitations

- Research/portfolio project, not a clinical tool — no diagnostic claims are made.
- Trained on a single-institution dataset (Vietnamese hospitals); performance on other
  populations, scanners, or acquisition protocols is untested.
- Class imbalance across the 14 findings — rarer classes will underperform without
  further handling (oversampling, focal loss, or narrowing the class list).

## Acknowledgements

Dataset: VinDr-CXR (Nguyen et al.), released via the VinBigData Chest X-ray
Abnormalities Detection Kaggle challenge — subject to its own data use agreement.
Explainability approach builds on EigenCAM (Muhammad & Yeasin, 2020) and its
application to YOLO-style detectors.
