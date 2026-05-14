# 🏦 Future of Loan Approvals with Explainable AI

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.7+-blue?style=flat-square&logo=python">
  <img alt="License" src="https://img.shields.io/badge/License-Educational-green?style=flat-square">
  <img alt="Status" src="https://img.shields.io/badge/Status-Active-success?style=flat-square">
</p>

> A powerful machine learning application for predicting loan approval status and rejection reasons using Random Forest classifiers with explainable AI (SHAP) visualization.

This desktop application provides an **intuitive GUI** for data exploration, model training, and **interpretable predictions** with complete explainability.

---

## 🤝 Contributors

<table>
<tr>
<td align="center"><b>T. Varshitha</b></td>
<td align="center"><b>S. Shiva Sai</b></td>
<td align="center"><b>P. Harikrishna</b></td>
<td align="center"><b>Y. Prasuna</b></td>
<td align="center"><b>T. Koushik</b></td>
</tr>
</table>

---

## 📋 Table of Contents

- [✨ Features](#-features)
- [🚀 Quick Start](#-quick-start)
- [📦 Requirements](#-requirements)
- [💻 Installation](#-installation)
- [📖 Usage Guide](#-usage-guide)
- [📊 Dataset Format](#-dataset-format)
- [🏗️ Project Structure](#️-project-structure)
- [🛠️ Technologies](#️-technologies)
- [🔍 Key Features Explanation](#-key-features-explanation)
- [📈 Performance Metrics](#-performance-metrics)
- [❓ Troubleshooting](#-troubleshooting)

---

## ✨ Features

### 📊 Data Management
- ✅ **Dataset Upload**: Load loan application datasets in CSV format
- ✅ **Data Preprocessing**: Automatic handling of missing values and categorical encoding
- ✅ **Train-Test Split**: Configurable dataset splitting (80/20) for model validation

### 🤖 Machine Learning Models
- ✅ **Approval Prediction**: Random Forest classifier for loan approval status
- ✅ **Rejection Analysis**: Separate model for predicting rejection reasons
- ✅ **Comprehensive Metrics**: Accuracy, Precision, Recall, and F1-Score calculations

### 📈 Visualization & Analysis
- ✅ **Interactive Dashboards**: Real-time plotting of loan status and rejection distributions
- ✅ **Feature Distribution**: Histogram and KDE plots for individual features
- ✅ **Correlation Analysis**: Heatmap visualization of feature correlations
- ✅ **3D Visualizations**: Scatter and surface plots for multi-dimensional data exploration
- ✅ **Confusion Matrices**: Visual representation of model performance
- ✅ **SHAP Explainability**: Feature importance and decision explanation plots

### 🎯 Prediction & Results
- ✅ **Batch Predictions**: Process test datasets and generate predictions
- ✅ **Interactive Results Table**: View approval status and rejection reasons with sortable columns
- ✅ **Export Results**: Save selected predictions to CSV format
- ✅ **Logging**: Comprehensive application logs for tracking operations

### 🎨 Modern UI
- ✅ **Dark Hacker Theme**: Professional dark interface with neon accents
- ✅ **Tabbed Navigation**: Organized workflow across Dashboard, Models, and Predict/Logs tabs
- ✅ **Responsive Layouts**: Adaptive interface that scales with window resizing
- ✅ **Gradient Buttons**: Modern interactive button styling with hover effects

---

## 🚀 Quick Start

Get up and running in 3 steps:

```bash
# 1. Clone/navigate to project
cd d:\Projects\Loan

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# 3. Install and run
pip install -r requirements.txt
python LoanStatus.py
```

> **💡 Tip**: Use `run.bat` for one-click startup on Windows!

---

## 📦 Requirements

| Component | Minimum Version |
|-----------|-----------------|
| Python | 3.7+ |
| NumPy | >= 1.19.2 |
| Pandas | >= 1.0.0 |
| Scikit-learn | >= 0.24.0 |
| Matplotlib | >= 3.2.0 |
| Seaborn | >= 0.11.0 |
| SHAP | >= 0.40.0 |
| Jupyter | >= 1.0.0 |

---

## 💻 Installation

### Step 1️⃣ Clone or Navigate to Repository
```bash
cd d:\Projects\Loan
```

### Step 2️⃣ Create Virtual Environment (Recommended)
```bash
python -m venv .venv
.venv\Scripts\activate
```

> **ℹ️ Why virtual environments?** They isolate your project dependencies and prevent conflicts with other projects.

### Step 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

> **✨ Pro Tip**: All dependencies are listed in `requirements.txt` for easy installation.

---

## 📖 Usage Guide

### ▶️ Running the Application

You have two options:

**Option 1: Quick Start with Batch File** (Windows)
```bash
run.bat
```

**Option 2: Direct Python Execution**
```bash
python LoanStatus.py
```

### 📋 Workflow Guide


Follow these steps to get predictions:

#### **📥 Step 1: Load Dataset**
1. Navigate to the **Dashboard** tab
2. Click **Upload Dataset** button
3. Select a loan application CSV file from the Dataset folder
4. View the initial data distribution in the summary bar plots

#### **⚙️ Step 2: Preprocess Data**
1. Click **Preprocess Dataset**
2. The application will:
   - Handle missing values
   - Encode categorical variables
   - Normalize numerical features
3. View available features in the Feature Selector

#### **🔍 Step 3: Explore Data**
1. Select features from the Feature Selector listbox
2. Use visualization buttons:
   - **Plot Distribution**: Histogram with KDE for selected feature
   - **Plot Correlation**: Correlation heatmap of all features
   - **3D Scatter**: Interactive 3D scatter plot
   - **3D Surface**: 3D surface plot for trend visualization

#### **✂️ Step 4: Prepare Data for Training**
1. Click **Split Train/Test**
2. Dataset is split into 80% training and 20% testing sets

#### **🏋️ Step 5: Train Models**
1. Navigate to the **Models** tab
2. Click **Train Approval Model** to train the loan status classifier
3. Click **Train Reject Model** to train the rejection reason classifier
4. View real-time metrics in the Model Metrics table

#### **🔬 Step 6: Interpret Model Decisions**
1. Click **Explainable AI (SHAP)** to generate feature importance plots
2. SHAP summary plots show which features most influence predictions

#### **🎯 Step 7: Make Predictions**
1. Navigate to the **Predict / Logs** tab
2. Click **Predict (Select Test CSV)** and choose a test dataset
3. Results are displayed in the interactive Results table
4. Double-click any result row to view detailed information
5. Click **Save Selected** to export predictions as CSV

#### **📝 Step 8: View Logs**
- Application logs and operation summaries appear in the Results table
- Click **Save Log** to export the complete operation log

---

---

## 📊 Dataset Format

### Expected CSV Columns
```
NAME_CONTRACT_STATUS  → Loan approval status (target variable)
CODE_REJECT_REASON    → Reason for rejection (target variable)
Additional features   → Numerical and categorical attributes for prediction
```

### 📂 Sample Files
Example dataset files are provided in the `Dataset/` folder:
- 📄 `loan_application.csv` - Training dataset
- 📄 `testData.csv` - Test dataset for predictions

---

## 🏗️ Project Structure

```
Loan/
├── 📄 LoanStatus.py              # Main application entry point
├── 📄 requirements.txt           # Python dependencies
├── 📄 run.bat                    # Windows batch runner (quick start)
├── 📄 DatasetLink.txt            # Dataset reference information
├── 📄 README.md                  # This file
└── 📁 Dataset/
    ├── loan_application.csv
    └── testData.csv
```

---

## 🛠️ Technologies

| Category | Technology |
|----------|-----------|
| **GUI Framework** | Tkinter (Python standard library) |
| **Machine Learning** | Scikit-learn (Random Forest) |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Explainability** | SHAP (SHapley Additive exPlanations) |
| **Numerical Computing** | SciPy |

---

## 🔍 Key Features Explanation

### 🌲 Random Forest Classifier

Why Random Forest?

- ✅ Handles mixed data types effectively
- ✅ Provides robust feature importance rankings
- ✅ Resistant to overfitting with proper parameters
- ✅ Fast training and prediction on large datasets

### 📊 SHAP Explainability

SHAP values provide:
- **🎯 Feature Importance**: Which factors most influence decisions
- **🔎 Instance-level Explanations**: Why a specific prediction was made
- **📈 Global Insights**: Overall model behavior patterns
- **✨ Trust & Transparency**: Understand model decisions for regulatory compliance

---

## 📈 Performance Metrics

The application calculates:
- **Accuracy**: Overall correctness of predictions
- **Precision**: Accuracy of positive predictions
- **Recall**: Coverage of actual positive cases
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of prediction types

---

## ❓ Troubleshooting

### 🔴 Missing Dependencies
```bash
pip install -r requirements.txt --upgrade
```

**Solution**: Reinstall all dependencies to ensure compatibility.

### 🔴 Virtual Environment Issues
```bash
python -m venv .venv --clear
.venv\Scripts\activate
pip install -r requirements.txt
```

**Solution**: Clear and recreate your virtual environment.

### 🔴 Dataset Not Found
Ensure the following:
- ✅ CSV files are in the `Dataset/` folder
- ✅ File names and format match expected structure
- ✅ Required columns exist (`NAME_CONTRACT_STATUS`, `CODE_REJECT_REASON`)

### 🔴 SHAP Visualization Issues
- ✅ Ensure matplotlib backend is properly configured
- ✅ Try retraining the model before running SHAP analysis
- ✅ Check that you have at least 200 test samples

---

## 📝 Notes

> **⚙️ Threading**: The application runs with threading to prevent UI freezing during heavy computations

> **💾 Model Storage**: Processed models are stored in memory during the session

> **⏱️ Processing Time**: Large datasets (>100K rows) may require longer processing times

> **📦 Optional Imports**: mplcursors is optionally imported for enhanced interactive plotting

---

## 🚀 Future Enhancements

- [ ] Support for additional machine learning algorithms
- [ ] Model persistence (save/load trained models)
- [ ] Cross-validation support
- [ ] Hyperparameter tuning interface
- [ ] Real-time batch prediction streaming
- [ ] Database integration for data storage

---

## 📜 License

This project is provided for **educational and research purposes**.

---

## 💬 Contact & Support

For questions or issues, please contact the project contributors listed above.

---

<p align="center">
  <strong>Last Updated: May 2026</strong> | 
  <a href="#-future-of-loan-approvals-with-explainable-ai">Back to Top</a>
</p>
