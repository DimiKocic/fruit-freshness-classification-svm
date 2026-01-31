# Fruit Freshness Classification – Classical ML & Deep Learning Solutions

End-to-end computer vision system for automated fruit freshness classification, providing two complementary approaches:

1. A lightweight classical machine learning pipeline using feature engineering + Support Vector Machines (SVM)  
2. A state-of-the-art deep learning pipeline using ConvNeXt Large with GPU acceleration and production-ready inference

The project demonstrates both traditional and modern approaches to image classification, focusing on performance, interpretability, and deployability.

---

## Overview

Fruit freshness assessment is commonly performed manually, which is subjective and inefficient. Deep learning approaches often require large datasets and specialized hardware.

This repository explores two solutions:

- **Classical ML Solution:** Feature engineering + SVM for efficient, interpretable classification  
- **Advanced Deep Learning Solution:** ConvNeXt Large with mixed precision training and FastAPI deployment  

Both pipelines classify fruit images into six categories:

- freshapples  
- freshbanana  
- freshoranges  
- rottenapples  
- rottenbanana  
- rottenoranges  

---

## Repository Structure

├── firstSolution/ # Classical ML pipeline (SVM + engineered features)
├── cockaSolution/ # prototype
├── advancedSolution/ # Deep Learning pipeline (ConvNeXt + FastAPI)
├── requirements.txt
└── README.md


---

## Solution 1 – Classical Machine Learning (SVM)

Located in `firstSolution/`

A lightweight computer vision pipeline built on engineered features and Support Vector Machines.

### Highlights

- Otsu-based fruit segmentation  
- HSV color histograms for ripeness detection  
- Hu Moments and area for shape analysis  
- Texture statistics for surface irregularities  
- SVM (RBF kernel) with grid-search hyperparameter tuning  
- Modular pipeline: preprocessing → feature extraction → training → inference  

### Performance (current dataset split)

| Metric | Score |
|------|------|
| Accuracy | 98.89% |
| Precision | 98.90% |
| Recall | 98.89% |
| F1-score | 98.89% |

This approach demonstrates how classical ML combined with domain-specific features can rival deep learning while remaining computationally efficient and interpretable.

Detailed documentation is available in `firstSolution/README.md`.

---

## Solution 2 – Advanced Deep Learning (ConvNeXt Large)

Located in `advancedSolution/`

A state-of-the-art CNN-based system designed for scalability and deployment.

### Highlights

- ConvNeXt Large backbone with ImageNet pretrained weights  
- Mixed precision training (float16) for faster GPU execution  
- Two-phase training (warmup + fine-tuning with cosine decay)  
- AdamW optimizer with label smoothing  
- High-throughput tf.data pipeline with cache() and prefetch()  
- Test Time Augmentation (TTA) for robust inference  
- FastAPI service for real-time predictions  

This solution focuses on modern ML engineering practices including transfer learning, efficient data pipelines, and production-style serving.

Detailed documentation is available in `advancedSolution/README.md`.

---

## Dataset Format

Both pipelines expect the dataset organized as:

data/
├── train/
│ ├── freshapples/
│ ├── rottenapples/
│ ├── freshbanana/
│ ├── rottenbanana/
│ ├── freshoranges/
│ └── rottenoranges/
└── test/
├── freshapples/
├── rottenapples/
├── freshbanana/
├── rottenbanana/
├── freshoranges/
└── rottenoranges/


Folder names are automatically mapped to class labels.

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd fruit-freshness-classification


