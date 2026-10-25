# DeepLung: Detection and Classification of Lung Cancer using Deep Learning

A project proposal for a developing and evaluating deep learning models for classifying lung CT scans as Normal, Benign or Malignant using the IQ-OTH/NCCD dataset, and to deploy the best-performing model in a web-based application for use by medical professionals.
> **Project status:** Proposal stage. This repository currently contains the proposal, project guidelines, and dataset images. It does not contain training or application code, a trained model, or runnable setup commands.

> **Medical disclaimer:** This is an academic research prototype. It is **not** a certified medical device and must not be used for real clinical diagnosis or treatment decisions.

---

## Table of Contents

- [Background](#background)
- [Aim and Objectives](#aim-and-objectives)
- [Planned Features](#planned-features)
- [Dataset](#dataset)
- [Planned Methodology](#planned-methodology)
- [Proposed System Architecture](#proposed-system-architecture)
- [Proposed Tech Stack](#proposed-tech-stack)
- [Repository Contents](#repository-contents)
- [Running the Project](#running-the-project)
- [Planned Evaluation](#planned-evaluation)
- [Deliverables and Timeline](#deliverables-and-timeline)
- [Results](#results)
- [Scope and Limitations](#scope-and-limitations)
- [Team](#team)
- [Acknowledgements](#acknowledgements)

---

## Background

Lung cancer is one of the leading causes of cancer-related deaths worldwide. CT imaging is the primary tool for spotting pulmonary nodules, but manual review of high-resolution scans is slow and subject to inter-observer variability. In resource-limited referral hospitals in Lesotho, where specialist radiologists and working CT scanners are scarce, long waiting times can lead to late-stage diagnosis and lower survival rates.

DeepLung aims to provide a fast, objective and explainable second opinion for clinicians, in line with Lesotho's National Digital Transformation Strategy and Digital Health Strategy.

## Aim and Objectives

**Aim:** To develop and evaluate a Convolutional Neural Network (CNN) for classifying lung CT scans into Normal, Benign and Malignant categories, with the proposed model integrated into a web application.

1. Source, pre-process and normalise the IQ-OTH/NCCD dataset into a consistent input pipeline.
2. Design, implement and train a CNN for three-class classification.
3. Add Grad-CAM heatmaps to show which lung regions drove each prediction.
4. Evaluate with accuracy, precision, recall, F1-score, ROC-AUC and confusion matrices.
5. Build a web interface to upload scans, view predictions and inspect heatmaps.

## Planned Features

The proposal describes the following features for a future implementation:

- Classify CT slices as Normal, Benign, or Malignant and return a confidence score.
- Generate Grad-CAM overlays and display them beside the original scan and prediction.
- Accept PNG/JPEG images and DICOM uploads, converting DICOM data for inference.
- Remove identifying DICOM metadata before processing or storage.
- Provide authenticated clinician access, role-based permissions, and encrypted data handling.
- Log predictions and model versions, collect clinician feedback, and show model performance to administrators.

## Dataset

This project uses the **IQ-OTH/NCCD lung cancer dataset**, collected in fall 2019 at the Iraq-Oncology Teaching Hospital / National Center for Cancer Diseases. CT scans were acquired on a Siemens SOMATOM scanner (120 kV, 1 mm slice thickness), annotated by oncologists and radiologists, and de-identified before release. It is publicly available on Kaggle.

| Class | Slices (JPEG) |
|-----------|------:|
| Malignant | 561 |
| Normal | 416 |
| Benign | 120 |
| **Total** | **1,097** |

The slices come from 110 patient cases (40 malignant, 15 benign, 55 normal). The slice-level class ratio is roughly 4.7 : 1 (largest to smallest), so imbalance is handled explicitly (see below).

The images are included in this repository under `Dataset/The IQ-OTHNCCD lung cancer dataset/`. The benign folder is spelled `Bengin cases` in the supplied dataset. The dataset description file reports 1,190 images, while the proposal and files currently present indicate 1,097; the table above counts the files in this repository. Verify the source data and labels before training or reporting results.

The proposal cites the [IQ-OTH/NCCD dataset on Mendeley Data](https://doi.org/10.17632/bhmdr45bh2.2). Use and redistribute the images only in accordance with the dataset's own terms; the repository's MIT license does not apply to the dataset.

## Planned Methodology

The project follows an experiment-driven pipeline: data exploration, preprocessing, splitting and augmentation, training, classification, Grad-CAM explanation, evaluation, and web integration.

**Preprocessing**
- Images read with OpenCV; uploaded DICOM files read with Pydicom and converted to PNG
- Resized to 224 × 224, converted to grayscale, then replicated across 3 channels
- ResNet50 `preprocess_input` for the transfer model; scaling to [0, 1] for the baseline CNN
- Optional Gaussian/median denoising (tested, avoiding heavy blur to preserve nodule edges)
- Removal of patient-identifying DICOM tags before storage

**Data handling**
- Stratified 70 / 15 / 15 train / validation / test split
- Patient-level grouping: the dataset has no patient IDs, so adjacent slices are grouped using a correlation check on downsampled pixels, preventing near-duplicate slices from leaking between splits
- Augmentation on the training set only: small rotations (±10°), translations (up to 10%) and zoom (±10%)
- Class-weighted categorical cross-entropy for imbalance
- Test set evaluated once, after model selection is final

**Models**

| Model | Description |
|---|---|
| Baseline CNN | 4 conv blocks (3×3 conv, ReLU, batch norm, 2×2 max-pool; 32/64/128/256 filters), global average pooling, dropout 0.5, 3-way softmax. Trained from scratch with Adam (~1e-3). |
| ResNet50 (transfer learning) | ImageNet weights, new head (GAP, dense + ReLU, dropout 0.5, 3-way softmax). Stage 1: frozen base, head trained at ~1e-3. Stage 2: last conv block fine-tuned at ~1e-5 with batch norm frozen. |

Training uses batch sizes of 16 to 32, 30 to 50 epochs, early stopping on validation loss with best-weight restoration, and learning-rate reduction on plateau. Feature vectors are also projected with PCA and t-SNE for visual analysis of class separability.

**Explainability**

Grad-CAM is computed at the last convolutional layer of ResNet50 (`conv5_block3_out`) using TensorFlow's `GradientTape`, upsampled and overlaid on the scan. Heatmaps are checked for correct and incorrect predictions to confirm the model attends to lung tissue rather than borders, the scanner table or annotations.

## Proposed System Architecture

The system has four layers:

- **Presentation:** clinician web app (login, upload, results view) and an administrator dashboard
- **Application:** preprocessing module, CNN inference engine, Grad-CAM module, logging and metrics service
- **Data:** PostgreSQL database (credentials, de-identified metadata, prediction logs, feedback) and versioned model storage
- **Security:** authentication, role-based access control and encryption, enforcing OSI security services across the other layers

## Proposed Tech Stack

| Area | Technology |
|---|---|
| Language | Python |
| Deep learning | TensorFlow / Keras |
| Image and DICOM handling | OpenCV, Pydicom |
| Backend | FastAPI |
| Database | PostgreSQL |
| Frontend | HTML, CSS, JavaScript |

## Repository Contents

- `Dataset/` - class-organized JPEG images and the supplied dataset description.
- `docs/Detection and classification of lung cancer using deep learning.pdf` - project proposal and planned methodology.
- `docs/Project Guidelines_CS_IS_02112022.pdf` - department project guidelines.
- `requirements.txt` - currently empty; no Python environment is defined yet.
- `LICENSE` - MIT License for repository software materials.

## Running the Project

There are no training scripts or application entry points in the repository yet, so it cannot currently be installed or run. Setup instructions will be added with the implementation.

## Planned Evaluation

The proposal plans to evaluate the selected models on a held-out test set using:

- Accuracy; per-class precision, recall (sensitivity), F1-score and specificity; macro and weighted averages
- One-vs-rest ROC-AUC per class and the confusion matrix
- **Malignant recall is the priority metric**, since a missed cancer is the costliest error
- 95% confidence intervals via bootstrap resampling (1,000 resamples)
- Comparisons: baseline CNN vs ResNet50, with and without denoising and class weighting
- Robustness: patient-grouped k-fold cross-validation
- Error analysis of misclassified slices using Grad-CAM, plus inference time per slice and functional/user acceptance testing

## Deliverables and Timeline

### Deliverables by 15 January 2027

- A documented preprocessing, patient-level splitting, adjacent-slice checking, and augmentation pipeline.
- A trained and evaluated baseline CNN, and a trained and evaluated ResNet50 using a frozen-base stage followed by fine-tuning.
- Grad-CAM heatmaps generated and reviewed for both correct and incorrect predictions.
- One final held-out test-set evaluation, including accuracy, per-class precision, recall, F1-score and specificity, ROC-AUC, a confusion matrix, and bootstrap 95% confidence intervals. Compare the baseline CNN with ResNet50.
- A pipeline report explaining the design choices and their rationale so that each team member can explain the complete pipeline.
- A high-fidelity prototype using the selected model. A user uploads a CT slice and sees the original scan, predicted class with confidence, and Grad-CAM heatmap side by side.

### Work Plan

| Period | Focus | Deliverable |
|---|---|---|
| 14 to 25 Oct | Exploratory Data Analysis (EDA), preprocessing pipeline, patient-level split, augmentation | Documented EDA and data pipeline |
| 26 Oct to 15 Nov | Baseline CNN training, class-weighting and denoising comparisons on the validation set | Baseline selection frozen by 15 Nov |
| 16 Nov to end of exams | Reduced workload (exams and other coursework). Documentation of completed work (data and baseline sections), reading, designing the prototype screens. No new experiments | Data and baseline report sections, prototype screen designs |
| After exams to 10 Dec | ResNet50 stage 1 (frozen base) and stage 2 (fine-tuning) | Best model selected on validation results by 10 Dec |
| 11 Dec to 5 Jan | Grad-CAM, PCA/t-SNE, single final test-set evaluation, bootstrap CIs, error analysis. From about 20 Dec, connect the prototype interface to the selected model | Final evaluation results, Grad-CAM heatmaps, working prototype |
| 6 to 15 Jan | Finish the pipeline report, polish the prototype, review with supervisor, fix gaps | Complete report and prototype. This period is also the buffer if any stage runs late. |

## Results

No model has been trained or evaluated in this repository, so there are no results to report.

## Scope and Limitations

**Proposed scope:** 2D slice-level classification, Grad-CAM overlays, metadata de-identification, a web prototype with upload, prediction and heatmap display, and model evaluation.

**Out of scope:** 3D volumetric CT analysis, and clinical deployment or hospital PACS integration.

**Known limitations:**
- The proposal describes a dataset of 110 cases from one region and scanner type, so generalisation to other populations and equipment is unproven. The supplied dataset description and included image count also differ; see [Dataset](#dataset).
- Benign cases are scarce, which makes the Benign class the hardest to learn reliably.
- Grad-CAM heatmaps are coarse and show where the model looked, not why; they can create false confidence.

## Team

- **Tumelo Leubane**
- **Nkuebe Nkuebe**
- **Supervisor:** Lekuba Ntoane

Department of Mathematics and Computer Science, Faculty of Science and Technology, National University of Lesotho.

## Acknowledgements

- The creators of the IQ-OTH/NCCD lung cancer dataset and the Iraq-Oncology Teaching Hospital / National Center for Cancer Diseases
- Our supervisor for guidance throughout the project
