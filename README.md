# 🧠 Breast Cancer Detection using Machine Learning

This project predicts whether a breast tumor is **malignant** or **benign** using various supervised machine learning models.  
It uses the **Breast Cancer Wisconsin dataset** from `scikit-learn` and compares different algorithms to find the most accurate model.

---

## 📊 Dataset
- **Source:** `sklearn.datasets.load_breast_cancer`
- **Samples:** 569
- **Features:** 30 numeric features (mean radius, mean texture, mean area, etc.)
- **Target:**  
  - `0` → Malignant  
  - `1` → Benign

---

## ⚙️ Workflow

1. **Import libraries**
   - Numpy, Pandas, Matplotlib, Seaborn, Scikit-learn, XGBoost

2. **Load dataset**
   - Used `load_breast_cancer()` from sklearn

3. **Data analysis & visualization**
   - Pairplots and heatmaps to explore relationships
   - Checked for missing values and distribution of target classes

4. **Data preprocessing**
   - Split into train (80%) and test (20%)
   - Standardized features using `StandardScaler`

5. **Model training**
   - Trained and evaluated multiple classifiers:
     - Support Vector Classifier (SVC)
     - Logistic Regression
     - K-Nearest Neighbors (KNN)
     - Random Forest
     - XGBoost Classifier

6. **Evaluation**
   - Used metrics:
     - Accuracy Score  
     - Confusion Matrix  
     - Classification Report  
     - Cross Validation (cv = 10)

7. **Model Saving**
   - Final trained model: **XGBoost Classifier**
   - Saved as `breast_cancer_detector.pickle` using `pickle`

---

## 🧾 Results

| Model | Accuracy (Scaled Data) |
|--------|-------------------------|
| Logistic Regression | ~98% |
| SVM | ~98% |
| KNN | ~97% |
| Random Forest | ~97% |
| **XGBoost** | **~99%** ✅ |

The XGBoost model performed best with around **99% accuracy** on test data.

---

## Tools & Libraries Used
- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Pickle

---

##  How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/breast-cancer-detection.git
   cd breast-cancer-detection
Install dependencies:

pip install -r requirements.txt


Run the Jupyter Notebook or Python script:

jupyter notebook breast_cancer.ipynb


The trained model (breast_cancer_detector.pickle) can be loaded and used to make new predictions.

📈 Visualization Samples

Pairplot of main features

Heatmap of correlation matrix

Confusion matrix of final model

💾 Model Deployment

The trained model is saved as a .pickle file and can be integrated into a web app using:

Streamlit

Flask

FastAPI

Example:

import pickle
model = pickle.load(open('breast_cancer_detector.pickle', 'rb'))
prediction = model.predict(new_data_scaled)

🧑‍💻 Author

Mudasir Iqbal
Machine Learning & Data Science Enthusiast
📧 Contact:ranamudasir687@gmail.com
🔗 GitHub: MudasirIqbal-Rana
