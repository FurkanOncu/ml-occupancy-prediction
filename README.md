# ML Occupancy Prediction

> Tilburg University — Machine Learning Challenge (Resit) | Grade: **8.47 / 10**

## Project Overview

This project predicts the **occupancy** of rental properties using structured tabular data. It was developed as part of the Machine Learning Challenge at Tilburg University and submitted to the [Codabench leaderboard](https://www.codabench.org/competitions/12244/).

The goal was to outperform both the baseline and reference solutions on the leaderboard, evaluated using **Mean Absolute Error (MAE)** — lower is better.

---

## Task

| Property | Detail |
|---|---|
| **Type** | Supervised regression |
| **Target variable** | `occupancy` (continuous, non-negative) |
| **Evaluation metric** | Mean Absolute Error (MAE) |
| **Data format** | JSON (train.json / test.json) |
| **Submission format** | ZIP containing predicted.json |

---

## Approach

### 1. Data Loading
Training and test data were loaded from `train.json` and `test.json` using pandas. The dataset contains a mix of numeric and categorical features.

### 2. Preprocessing
A `ColumnTransformer` pipeline was used to handle both feature types automatically:

- **Numeric features**: Median imputation + StandardScaler
- **Categorical features**: Most-frequent imputation + OneHotEncoder (unknown categories ignored)

### 3. Model
The core model is a **HistGradientBoostingRegressor** from scikit-learn, chosen for its speed, robustness to missing values, and strong performance on tabular data.

Key hyperparameters:
```python
HistGradientBoostingRegressor(
    loss='absolute_error',   # Directly optimises MAE
    max_iter=500,
    learning_rate=0.07,
    max_depth=10,
    l2_regularization=1.0,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42
)
```

### 4. Evaluation
Model performance was estimated locally using an 80/20 train-validation split before submission to the leaderboard.

---

## Results

- **Leaderboard**: Outperformed both the baseline (`ML_Challenge_Baseline`) and reference (`ML_Challenge_Reference`) solutions
- **Assignment grade**: 8.47 / 10

---

## Tech Stack

| Tool | Version |
|---|---|
| Python | 3.10+ |
| pandas | — |
| NumPy | — |
| scikit-learn | — |

---

## Files

```
├── baseline_1.py       # Full pipeline: preprocessing, model training, prediction, submission
├── README.md
```

> Note: `train.json` and `test.json` are not included in this repository as they are course materials from Tilburg University.

---

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/FurkanOncu/ml-occupancy-prediction.git
cd ml-occupancy-prediction
```

2. Install dependencies:
```bash
pip install pandas numpy scikit-learn
```

3. Place `train.json` and `test.json` in the project directory.

4. Run the script:
```bash
python baseline_1.py
```

This will output a `submission.zip` file containing `predicted.json` ready for leaderboard submission.

---

## Author

**Furkan Öncü**  
MSc Data Science and Society — Tilburg University  
[GitHub](https://github.com/FurkanOncu)
