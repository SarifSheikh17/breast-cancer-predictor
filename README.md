# Breast Cancer Predictor (ML + Streamlit)

This project uses machine learning models to predict whether a breast tumor is **benign** or **malignant** based on diagnostic features from the Breast Cancer Wisconsin dataset.

It includes:
- Logistic Regression
- MLP Classifier (Multi-layer Perceptron / Neural Network)
- Interactive Streamlit Web Application

---

## Features Used

The following nine features were selected based on their correlation with the target variable:

- worst concave points  
- mean concave points  
- mean perimeter  
- mean radius  
- mean concavity  
- worst perimeter  
- worst radius  
- worst area  
- mean area  

---

## Model Performance

| Model               | Accuracy | Precision | Recall | F1 Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression| ~0.95+   | ~0.95+    | ~0.95+ | ~0.95+   |
| MLP Classifier     | ~0.97+   | ~0.97+    | ~0.97+ | ~0.97+   |

Visualizations include confusion matrices using Seaborn and Matplotlib.

---

## Streamlit Application

### App Features
- Sliders for entering medical diagnostic values
- Predicts using both Logistic Regression and MLP
- Displays prediction probability confidence
- Clean, responsive layout

---

## Getting Started

### Step 1: Clone the repository

```bash
git clone https://github.com/SarifSheikh17/breast-cancer-predictor.git
cd breast-cancer-predictor
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Train the models

```bash
python main.py
```

This will save the following files inside the `models/` directory:

- `logistic_regression_model.pkl`  
- `mlp_classifier_model.pkl`  
- `scaler.pkl`  
- `selected_features.pkl`

### Step 4: Run the Streamlit application

```bash
streamlit run streamlit_app.py
```

---

## Project Structure

```
breast-cancer-predictor/
├── main.py
├── streamlit_app.py
├── models/
│   ├── logistic_regression_model.pkl
│   ├── mlp_classifier_model.pkl
│   ├── scaler.pkl
│   └── selected_features.pkl
├── requirements.txt
├── README.md
└── demo.png (optional)
```

---

## Acknowledgements

- Dataset: `load_breast_cancer` from `sklearn.datasets`
- Libraries: Scikit-learn, Matplotlib, Seaborn, Streamlit
- Developed through hands-on practice with guidance from Online Resources

---

## Author

**Sarif Sheikh**  