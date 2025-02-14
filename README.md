# 🚀 Global Health Statistics - Machine Learning Analysis

## 📌 Project Overview
This project aims to analyze global health statistics using various machine learning techniques. The goal is to predict health-related outcomes using classification models such as Random Forest, XGBoost, K-Nearest Neighbors (KNN), and Support Vector Machines (SVM).

## 📊 Dataset Information
We use the **Global Health Statistics** dataset, which includes various health indicators. 

### 📂 Datasets Used
1. **Branding Dataset** (`unicef_image_analysis_with_color_and_text.csv`): Contains information on branding elements in social media posts, such as slogans, color schemes, and logo presence.
2. **Engagement Dataset** (`social_media_postings.csv`): Includes engagement metrics such as impressions, likes, comments, and shares.

## 🔧 Installation
To set up the environment, install the required R packages:
```r
install.packages(c("dplyr", "fastDummies", "ggplot2", "caret", "caTools", "randomForest", "smotefamily", "factoextra", "reshape2", "xgboost", "e1071", "class"))
```

## 📌 Workflow
### Step 1️⃣: Load the Datasets
We begin by reading both datasets into Pandas dataframes and inspecting them for consistency.

### Step 2️⃣: Data Preprocessing
- **Handling Missing Values**: Numeric variables are imputed with the median, and categorical variables are imputed with the mode.
- **Encoding Categorical Variables**: Converted to factors and then dummy encoded.
- **Scaling and Normalization**: Standardization of numeric variables.
- **Outlier Handling**: Capped using the interquartile range (IQR) method.
- **Feature Selection**: Constant and duplicate columns are removed.

### Step 3️⃣: Exploratory Data Analysis
- Summary statistics and dataset structure analysis.
- Missing value inspection.
- **Correlation Heatmap** to visualize relationships between features.

### Step 4️⃣: Dimensionality Reduction
- **Principal Component Analysis (PCA)**: Used to reduce dimensionality while preserving at least 70% variance.
- **Scree Plot Visualization** to determine the number of components to retain.

### Step 5️⃣: Machine Learning Models
We implement various classification models:

#### 🌲 Random Forest
- Initial model training.
- Feature importance analysis.
- Model tuning with cross-validation.

#### ⚡ XGBoost
- Data transformation into a matrix format.
- Model training and hyperparameter tuning.
- Performance evaluation using a confusion matrix.

#### 🔍 K-Nearest Neighbors (KNN)
- Data scaling before training.
- Testing multiple `k` values to find the optimal one.
- Cross-validation for accuracy comparison.

#### 🎯 Support Vector Machines (SVM)
- Training models with different kernels (`linear`, `polynomial`, `radial`, `sigmoid`).
- Hyperparameter tuning using grid search.
- Final model evaluation.

### Step 6️⃣: Model Evaluation
- **Metrics Used**: Accuracy, Sensitivity, Specificity, Balanced Accuracy.
- **Feature Importance Ranking** for interpretability.

## 📌 Results & Insights
- A comparison of different classification models.
- Identification of key health indicators affecting mortality rates.
- Feature importance rankings to understand critical factors in global health trends.

## 🛠️ Usage
Clone the repository and run the R script:
```sh
git clone https://github.com/yourusername/health-statistics-ml.git
cd health-statistics-ml
Rscript analysis.R
```

## 📜 License
This project is licensed under the MIT License. See the LICENSE file for more details.

## 👤 Author
**Your Name**  
🔗 [GitHub Profile](https://github.com/yourusername)

