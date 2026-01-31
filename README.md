# Fruit Freshness Classification (SVM)

End-to-end computer vision pipeline for **fresh vs rotten fruit classification** using an optimized **Support Vector Machine (SVM)** with engineered color, texture, and shape features.

**Performance:** ~**98.89% accuracy** on the project test split.

This system demonstrates how classical machine learning combined with strong feature engineering can achieve high performance without relying on computationally expensive deep learning models.

---

## Problem Statement

Manual quality inspection in agricultural supply chains is slow, subjective, and expensive.  
This project automates fruit freshness classification by analyzing images and predicting whether fruits are fresh or rotten.

The model supports **6 classes**:

- freshapples  
- freshbanana  
- freshoranges  
- rottenapples  
- rottenbanana  
- rottenoranges  

---

## Key Features

### Computer Vision & Feature Engineering

- Adaptive segmentation using **Otsu thresholding**
- HSV color histograms for ripeness and spoilage detection
- Shape descriptors using **Area + Hu Moments (log-transformed)**
- Texture statistics (mean and standard deviation of pixel intensities)

### Machine Learning

- Support Vector Machine (RBF kernel)
- Grid Search hyperparameter tuning (`C`, `gamma`, kernel)
- Standard scaling + label encoding
- Full evaluation with confusion matrix and classification report

### System Capabilities

- End-to-end pipeline (analysis → preprocessing → training → evaluation → inference)
- Inference on unseen and internet images
- Modular Python architecture

---

## Prototype Dataset Creation (cockaSolution)

Before building the final SVM pipeline, I developed an early prototype inside `cockaSolution/`.

In this phase I:

- Implemented web scraping to collect raw fruit images
- Validated dataset structure and labeling strategy
- Tested preprocessing approaches

This prototype was used to bootstrap and validate the dataset before implementing the full feature-engineered SVM pipeline in `firstSolution/`.

> The full dataset is not included in this repository to keep it lightweight and avoid redistribution issues.

---

## Project Structure

Main implementation lives in `firstSolution/`:

| File | Description |
|------|------------|
| `main.py` | Entry point (analysis → training → prediction) |
| `train.py` | Training logic and SVM optimization |
| `process.py` | FeatureExtractor (segmentation + features) |
| `normalization.py` | Image resizing (128×128) |
| `predict.py` | Test-set inference |
| `predict_internet.py` | Inference on unseen images |
| `analyze.py` | Dataset statistics |

Additional folders:

- `cockaSolution/` – early scraping + prototype experiments  
- `advancedSolution/` – extended experiments  
- `results/` – evaluation outputs  
- `assets/` – screenshots and visual results  

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/fruit-freshness-classification-svm.git
cd fruit-freshness-classification-svm
