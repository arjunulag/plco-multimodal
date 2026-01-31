# Lung Cancer AI-Based Diagnosis Through Multi-Modal Integration of Clinical and Imaging Data

[![DOI](https://img.shields.io/badge/DOI-10.59720%2F24--190-blue)](https://doi.org/10.59720/24-190)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-CC%20BY--NC--ND-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

Official implementation of the paper **"Lung cancer AI-based diagnosis through multi-modal integration of clinical and imaging data"**, published in the *Journal of Emerging Investigators* (2025).

> **Authors:** Arjun S. Ulag<sup>1</sup> and Ricardo A. Gonzales<sup>2</sup>  
> <sup>1</sup>Lakeside Upper School  
> <sup>2</sup>Radcliffe Department of Medicine, University of Oxford

---

## Abstract

Lung cancer remains the most lethal form of cancer, primarily due to late-stage diagnoses. Early detection significantly improves survival rates, yet it remains challenging. This study enhances early lung cancer diagnosis by developing and evaluating three models:

- **MLP (Multi-Layer Perceptron)** — for clinical/tabular data
- **CNN (Convolutional Neural Network)** — for chest X-ray imaging data  
- **Hybrid Model** — combining both clinical and imaging modalities

We hypothesized that integrating clinical and imaging data would yield higher diagnostic accuracy than single-modality approaches. Using the NIH's PLCO Cancer Screening Trial dataset, our hybrid model achieved the highest accuracy (**71.58%**), outperforming both the MLP (70.88%) and CNN (58.25%) models.

---

## Results

| Model | Accuracy |
|-------|----------|
| CNN (Images Only) | 58.25% |
| MLP (Clinical Only) | 70.88% |
| **Hybrid (Multimodal)** | **71.58%** |

---

## Dataset

This study uses the **Prostate, Lung, Colorectal, and Ovarian (PLCO) Cancer Screening Trial** dataset from the U.S. National Institutes of Health, which includes:

- **100,000+** chest X-ray images
- Associated clinical records and demographic information

### Clinical Features Used

| Category | Features |
|----------|----------|
| **Demographics** | Age, Sex, Race, Current BMI, BMI at age 20 |
| **Smoking History** | Cigarettes per day, Years smoked, Current smoker status |
| **Family History** | Lung cancer family history |
| **Medications** | Aspirin use, Ibuprofen use |
| **Comorbidities** | Arthritis, Bronchitis, Colon disease, Diabetes, Diverticulitis, Emphysema, Gallbladder disease, Heart attack, Hypertension, Liver disease, Osteoporosis, Polyps, Stroke |

### Data Access

The PLCO dataset is publicly available through the NIH Cancer Data Access System (CDAS):  
🔗 [https://cdas.cancer.gov/plco/](https://cdas.cancer.gov/plco/)

---

## Model Architecture

### MLP (Clinical Data Model)
```
Input (24 features) → Dense(128) → Dropout(0.2) → Dense(64) → Dropout(0.2) → Dense(32) → Dropout(0.2) → Sigmoid
```

### CNN (Imaging Model)
- **Backbone:** ResNet152V2 (pretrained on ImageNet, frozen)
- **Data Augmentation:** Random contrast, rotation, brightness, and zoom
- **Head:** Dense(64) → Dropout(0.2) → Dense(32) → Dropout(0.2) → Sigmoid

### Hybrid Multimodal Model
- **Late Fusion Architecture:** Combines CNN image embeddings with MLP clinical feature embeddings
- **Fusion:** Concatenation of 32-dimensional vectors from each stream
- **Final Layers:** Dense(64) → Dense(32) → Dropout(0.2) → Sigmoid

---

## Repository Structure

```
├── EDA.py              # Exploratory data analysis and preprocessing
├── Train.py            # Model training and evaluation scripts
├── cleaned_data.csv    # Preprocessed dataset (after running EDA.py)
└── README.md
```

---

## Requirements

```
tensorflow>=2.0
pandas
numpy
scikit-learn
matplotlib
```

---

## Usage

### 1. Data Preprocessing

Run the EDA script to clean and prepare the data:

```python
# EDA.py performs:
# - Merging clinical and imaging metadata
# - Feature selection and filtering
# - Missing value handling
# - Class label encoding
# - Saving cleaned dataset
```

### 2. Model Training

The `Train.py` script supports training all three model types:

```python
# Configure model type by setting flags:
include_images = False
include_features = True   # MLP only

include_images = True
include_features = False  # CNN only

include_images = True
include_features = True   # Hybrid multimodal
```

### 3. Training Parameters

| Parameter | Value |
|-----------|-------|
| Batch Size | 16 |
| Epochs | 200 |
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Loss | Binary Cross-Entropy |
| Image Size | 256 × 256 × 3 |
| Train/Val Split | 80/20 |

---

## Key Methods

- **Class Balancing:** Undersampling of majority class (controls) to match positive cases
- **Image Preprocessing:** Aspect-ratio-preserving resize with zero-padding to 256×256
- **Regularization:** L2 regularization + Dropout (0.2) throughout all models
- **Transfer Learning:** Frozen ResNet152V2 backbone pretrained on ImageNet

---

## Citation

If you use this code or find our work helpful, please cite:

```bibtex
@article{ulag2025lungcancer,
  title={Lung cancer AI-based diagnosis through multi-modal integration of clinical and imaging data},
  author={Ulag, Arjun S and Gonzales, Ricardo A},
  journal={Journal of Emerging Investigators},
  year={2025},
  month={June},
  doi={10.59720/24-190},
  url={https://doi.org/10.59720/24-190}
}
```

---

## License

This work is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).

---

## Acknowledgments

- NIH National Cancer Institute for the PLCO dataset
- TensorFlow and Keras teams for deep learning frameworks

---

## Contact

For questions or collaborations, please open an issue or contact the authors through the [published paper](https://doi.org/10.59720/24-190).
