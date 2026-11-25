
# TrustLens
### Explainable, Robust and Fair Machine Learning Evaluation Framework

TrustLens is a modular machine learning evaluation framework designed to examine model performance, explainability, robustness, fairness and deep learning experiments through an interactive Streamlit dashboard.

The project combines classical machine learning, neural networks, explainable AI, robustness testing, fairness evaluation and transfer learning in one evaluation environment.

## Project Objectives

- Evaluate predictive performance using multiple classification metrics.
- Interpret model behaviour using SHAP.
- Examine model sensitivity to Gaussian noise and adversarial perturbations.
- Analyse group-wise performance and fairness differences.
- Compare custom CNN experiments and ResNet18 transfer learning.
- Present experimental results through an interactive dashboard.

## Features

### 1. Model Performance
- Logistic Regression
- Random Forest
- PyTorch Neural Network
- Accuracy, precision, recall and F1
- Confusion matrix analysis

Dataset: Breast Cancer Wisconsin.

### 2. Explainability
- SHAP-based global feature importance
- Random Forest explanations
- Logistic Regression explanations
- Top influential feature visualizations

Dataset: Breast Cancer Wisconsin.

### 3. Robustness Evaluation
- Gaussian noise perturbation
- FGSM adversarial evaluation
- Accuracy comparison across perturbation levels
- Adversarial loss analysis

Dataset: Breast Cancer Wisconsin.

### 4. Fairness Evaluation
- Demographic Parity Difference
- Equal Opportunity Difference
- Group-wise accuracy, precision, recall and F1
- Selection rates
- True-positive rates

Dataset: Adult Census Income.

Sensitive attributes are retained for auditing and excluded from the model's input features.

### 5. Deep Learning
- Custom CNN experiments
- Data augmentation
- Regularization
- ResNet18 feature extraction
- ResNet18 fine-tuning
- Class-wise CIFAR-10 evaluation

Dataset: CIFAR-10.

### 6. Interactive Dashboard

The Streamlit dashboard provides six sections:

1. Overview
2. Model Performance
3. Explainability
4. Robustness
5. Fairness
6. Deep Learning

## Experimental Results

### Classical Machine Learning

Dataset: Breast Cancer Wisconsin.

| Model | Test Accuracy |
|---|---:|
| Logistic Regression | 98.25% |
| Random Forest | 94.74% |
| PyTorch Neural Network | 96.49% |

These results use the project's fixed 80/20 split with random state 42.

### Gaussian Noise Robustness

At noise level 0.50:

| Model | Accuracy |
|---|---:|
| Logistic Regression | 93.86% |
| Random Forest | 93.86% |
| PyTorch Neural Network | 92.98% |

### FGSM Robustness

For the PyTorch Neural Network:

| Epsilon | Accuracy |
|---|---:|
| 0.00 | 96.49% |
| 0.01 | 95.61% |
| 0.03 | 94.74% |
| 0.05 | 93.86% |
| 0.10 | 85.09% |
| 0.20 | 70.18% |
| 0.30 | 59.65% |

Epsilon is measured in standardized feature space.

### Fairness Evaluation

| Attribute | Demographic Parity Difference | Equal Opportunity Difference |
|---|---:|---:|
| Sex | 0.166916 | 0.073824 |
| Race | 0.141747 | 0.130667 |

These values represent measured group disparities, not proof of discrimination or a complete assessment of fairness.

### Custom CNN Experiments

Dataset: CIFAR-10.

| Experiment | Test Accuracy | Macro F1 |
|---|---:|---:|
| Baseline CNN | 71.52% | 0.7153 |
| CNN + Augmentation | 73.12% | 0.7305 |
| CNN + Augmentation + Regularization | 73.19% | 0.7298 |

### ResNet18 Experiments

| Experiment | Accuracy | Macro F1 |
|---|---:|---:|
| Feature Extraction | 86.72% | 0.8669 |
| Fine-Tuning | 91.02% | 0.9102 |

ResNet18 uses ImageNet pretraining, 224×224 resized CIFAR-10 images, and fine-tuning of the final ResNet block and classification head.

CNN and ResNet18 results represent separate experimental configurations and are not a controlled architecture-only comparison.

## Project Structure

```text
TrustLens/
│
├── dashboard/
│   ├── app.py
│   └── pages/
│       ├── performance.py
│       ├── explainability.py
│       ├── robustness.py
│       ├── fairness.py
│       └── deep_learning.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── reports/
│   ├── model_comparison.csv
│   ├── neural_network_results.csv
│   ├── random_forest_shap_importance.csv
│   ├── logistic_regression_shap_importance.csv
│   ├── gaussian_noise_robustness.csv
│   ├── fgsm_robustness.csv
│   ├── fairness_summary.csv
│   ├── fairness_by_sex.csv
│   ├── fairness_by_race.csv
│   ├── cnn_experiment_comparison.csv
│   └── resnet18_experiment_comparison.csv
│
├── src/
│   ├── data/
│   ├── models/
│   ├── evaluation/
│   ├── explainability/
│   ├── robustness/
│   ├── fairness/
│   └── visualization/
│
├── tests/
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/rithiksivakumar07/TrustLens.git

cd TrustLens
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it.

Windows:

```bash
venv\Scripts\activate
```

Linux / macOS:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Dashboard

From the project root directory:

```bash
streamlit run dashboard/app.py
```

The dashboard will be available at:

```text
http://localhost:8501
```

The dashboard uses saved CSV experiment reports from the `reports/` directory. Running the dashboard does not automatically reproduce the training experiments.

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- PyTorch
- SHAP
- Fairlearn
- Streamlit
- Plotly
- Matplotlib
- Seaborn

## Limitations

- The Breast Cancer experiments use a fixed train/test split; results should not be interpreted as definitive generalization estimates.
- The test data was inspected during iterative experimentation, so the reported results are exploratory rather than untouched final benchmark estimates.
- SHAP values explain model behaviour and do not establish causal relationships.
- Gaussian noise and FGSM represent specific perturbation experiments, not every possible real-world threat.
- Fairness metrics capture selected definitions of group disparities and do not establish fairness under every criterion.
- Some demographic groups have relatively small sample sizes, making their estimates less stable.
- CIFAR-10 CNN and ResNet18 experiments use different configurations.
- Further repeated experiments, statistical analysis and independent evaluation are needed.

## Future Improvements

- Repeated experiments across multiple random seeds.
- More comprehensive hyperparameter optimization.
- Additional robustness attacks and corruption benchmarks.
- Expanded fairness metrics and subgroup analysis.
- Improved automated testing.
- Reproducible experiment configuration and model versioning.
- More detailed individual prediction explanations.

## Author

**S Rithik**

GitHub: [rithiksivakumar07](https://github.com/rithiksivakumar07)

Project: [TrustLens](https://github.com/rithiksivakumar07/TrustLens)

---

TrustLens is an experimental machine learning evaluation project developed for educational, engineering and research exploration.
