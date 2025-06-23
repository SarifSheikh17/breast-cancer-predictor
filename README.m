# 🔬 Breast Cancer Predictor (ML + Streamlit)

This project uses machine learning models to predict whether a breast tumor is **benign** or **malignant** based on key diagnostic features from the [Breast Cancer Wisconsin dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html).

It includes:
- Logistic Regression
- MLP Classifier (Multi-layer Perceptron / Neural Network)
- Interactive Streamlit web app

---

## 📌 Features Used

From the dataset, the **9 most correlated features** were selected based on their correlation with the target label (`benign`/`malignant`):

- `worst concave points`
- `mean concave points`
- `mean perimeter`
- `mean radius`
- `mean concavity`
- `worst perimeter`
- `worst radius`
- `worst area`
- `mean area`

---

## 📈 Model Performance

Both models were trained and evaluated using accuracy, precision, recall, F1 score, and confusion matrix.

| Model               | Accuracy | Precision | Recall | F1 Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression| ~0.95+   | ~0.95+    | ~0.95+ | ~0.95+   |
| MLP Classifier     | ~0.97+   | ~0.97+    | ~0.97+ | ~0.97+   |

> 📊 Charts generated using **Seaborn** and **Matplotlib** to compare confusion matrices.

---

## 💻 Streamlit App

### 🎮 Features:
- Responsive sliders for medical inputs
- Real-time predictions from both models
- Probability confidence levels shown
- Clean two-column layout with sidebar info

### 🖼️ Demo Screenshot:
*(Include a screenshot here like `demo.png`)*

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/SarifSheikh17/breast-cancer-predictor.git
cd breast-cancer-predictor
2. Install requirements
bash
Copy
Edit
pip install -r requirements.txt
3. Train the model
bash
Copy
Edit
python main.py
4. Run Streamlit app
bash
Copy
Edit
streamlit run streamlit_app.py
🧠 Files & Structure
File/Folder	Purpose
main.py	Training & evaluation of both models
streamlit_app.py	Frontend app using Streamlit
models/	Saved .pkl models & scaler
requirements.txt	Project dependencies
README.md	Documentation
demo.png (optional)	Screenshot of the app

📚 Acknowledgements
Dataset: load_breast_cancer from sklearn.datasets

Visualization: seaborn, matplotlib

UI: Inspired by Streamlit documentation

🙋‍♂️ Author
Sarif Sheikh
ML Intern @ IIITDM Kancheepuram
Built with guidance from reading, experimentin.