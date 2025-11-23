# 🌸 Iris Flower Classification — Machine Learning Project

## 🚀 Live Streamlit App  
This project is live and running online!  
Click the link below to open it instantly:

🔗 **https://iris-flower-classification-ikma9qifmbvvvg9requmjm.streamlit.app/**

### 📘 Overview  
This project is a complete **end-to-end Machine Learning pipeline** built using **Python, Scikit-learn, and Streamlit**.  
It predicts the **species of an Iris flower** based on four key features — *Sepal Length, Sepal Width, Petal Length,* and *Petal Width.*

> 🚀 *An elegant ML + Streamlit project designed for learning, demonstration, and portfolio presentation.*

---

## 🧠 Objective
The goal of this project is to classify Iris flowers into one of the following species:
- **Setosa**
- **Versicolor**
- **Virginica**

This dataset is widely used for introducing classification concepts in Machine Learning due to its simplicity and interpretability.

---

## ⚙️ Tech Stack

| Category | Tools/Frameworks |
|-----------|------------------|
| Language | Python |
| ML Library | Scikit-learn |
| UI Framework | Streamlit |
| Data Handling | Pandas, NumPy |
| Model Persistence | Joblib |

---

## 🧩 Project Structure


iris-flower-classification/
│
├── iris_pipeline.py # Model training and evaluation
├── iris_app.py # Streamlit app for interactive prediction
├── iris_model.joblib # Saved trained model
├── iris_scaler.joblib # StandardScaler saved for normalization
├── requirements.txt # All dependencies
└── README.md # Project documentation (this file)


---

## 🧪 How It Works

1️⃣ **Data Loading** — Iris dataset imported from `sklearn.datasets`  
2️⃣ **Data Preprocessing** — Standard scaling applied using `StandardScaler`  
3️⃣ **Model Training** — SVM (Support Vector Machine) trained for classification  
4️⃣ **Evaluation** — Achieved ~96 % accuracy  
5️⃣ **Model Persistence** — Model and scaler saved via `joblib`  
6️⃣ **App Interface** — Streamlit UI built for real-time predictions  

---

## 💻 Run Locally

### 🔹 Step 1 — Clone the Repository
```bash
git clone https://github.com/<your-username>/iris-flower-classification.git
cd iris-flower-classification

🔹 Step 2 — Install Dependencies
pip install -r requirements.txt

🔹 Step 3 — Train the Model
python iris_pipeline.py

🔹 Step 4 — Launch the Streamlit App
streamlit run iris_app.py


🌐 Visit → http://localhost:8501

🌺 Example Output

When you move the sliders and click Predict,
you’ll see the predicted flower species with a friendly success message 🌸

📊 Model Performance
Metric	Score
Accuracy	96 %
Precision	95 %
Recall	96 %
F1-Score	95.5 %


🪴 Author 

              👩‍💻 Nandini
