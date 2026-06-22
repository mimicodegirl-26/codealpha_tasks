
# CodeAlpha Machine Learning Internship – Tasks

[![Internship](https://img.shields.io/badge/Internship-CodeAlpha-blue?logo=codeforces)](https://www.codealpha.tech)
[![Batch](https://img.shields.io/badge/Batch-June%202026-brightgreen)](https://www.codealpha.tech)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> This repository contains the projects completed during the **CodeAlpha Machine Learning Internship (June 2026 Batch)**.  
> Two end‑to‑end machine learning solutions showcasing **classification on tabular data** and **deep learning on audio data**, both deployed as interactive web apps.

---

## 📋 Completed Tasks

| # | Task | Domain | Key Techniques | Accuracy | Demo |
|---|------|--------|----------------|----------|------|
| 1 | **Credit Scoring Model** | Fintech / Tabular ML | Logistic Regression, Random Forest, XGBoost, SMOTE, SHAP | ROC‑AUC ~0.80 | [▶️ Live Demo](https://github.com/mimicodegirl-26/codealpha_tasks/tree/main/Task1_CreditScoring#live-demo) |
| 2 | **Emotion Recognition from Speech** | Speech / Deep Learning | CNN‑Bidirectional LSTM, MFCC, Data Augmentation | 70.1% (8 classes) | [▶️ Live Demo](https://github.com/mimicodegirl-26/codealpha_tasks/tree/main/Task2_EmotionRecognition#live-demo) |

---

## 🗂️ Repository Structure

```
codealpha_tasks/
├── Task1_CreditScoring/
│   ├── app.py                   # Streamlit web app
│   ├── credit_scoring.py        # Model training & evaluation
│   ├── models/                  # Saved model & preprocessor
│   ├── data/                    # German Credit dataset
│   ├── README.md                # Detailed task documentation
│   └── ...
├── Task2_EmotionRecognition/
│   ├── app.py                   # Streamlit web app
│   ├── train_emotion.py         # Model training & evaluation
│   ├── emotion_model.h5         # Trained Keras model
│   ├── README.md                # Detailed task documentation
│   └── ...
└── README.md                    # This file
```

---

## 🚀 Quick Start

Each task folder contains a complete `README.md` with installation and usage instructions.  
To run any project locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/mimicodegirl-26/codealpha_tasks.git
   cd codealpha_tasks
   ```

2. Navigate to the task folder (e.g., `Task1_CreditScoring`) and follow its **README**.

---

## 📌 Submission Details

- **Internship Provider:** [CodeAlpha](https://www.codealpha.tech)  
- **Batch:** June 2026  
- **Tasks Completed:** 2 (Credit Scoring, Emotion Recognition)  
- **GitHub Repository:** [codealpha_tasks](https://github.com/mimicodegirl-26/codealpha_tasks)  
- **Status:** ✅ Submitted  

---

## 🙏 Acknowledgements

- **CodeAlpha** for the mentorship and opportunity.
- Open‑source libraries: Scikit‑learn, TensorFlow, Keras, Streamlit, Librosa, and many more.

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/mimicodegirl-26">@mimicodegirl-26</a>
</p>
