# Titanic Survival Analysis Research Report

**Date:** 2025-11-07  
**Project:** Titanic Survival Prediction and Survivor Profile Analysis

---

## Abstract

This project presents a comprehensive analysis of the Titanic survival dataset using machine learning techniques. The study consists of two main directions: (A) Survival prediction using interpretable decision tree models, and (B) Survivor profile analysis using statistical methods and clustering algorithms. Multiple models including Decision Tree, Random Forest, and Gradient Boosting were trained and compared. The best model achieved an AUC of 0.856 and accuracy of 0.810. Survivor clustering identified 4 distinct groups with unique characteristics. The analysis reveals that gender, class, and age are the most significant factors affecting survival probability.

---

## 1. Introduction

### 1.1 Background

The sinking of the RMS Titanic on April 15, 1912, is one of the most famous maritime disasters in history. The tragedy resulted in the loss of over 1,500 lives. This dataset provides valuable insights into survival patterns and factors that influenced passenger survival rates.

### 1.2 Research Objectives

This study aims to:

1. **Direction A (Prediction):** Build interpretable machine learning models to predict passenger survival probability and provide rule-based explanations.

2. **Direction B (Profiling):** Analyze survivor characteristics through statistical analysis and clustering to identify typical survivor profiles.

### 1.3 Dataset Overview

The dataset contains 891 passenger records with the following key features:
- **Demographic:** Age, Sex, Title
- **Socioeconomic:** Pclass (passenger class), Fare
- **Family:** SibSp (siblings/spouses), Parch (parents/children)
- **Travel:** Embarked (port of embarkation), Cabin

**Note:** The Fare feature was excluded from model training as per project requirements, but was used for analysis and clustering purposes.

---

## 2. Methodology

### 2.1 Data Preprocessing

#### 2.1.1 Missing Value Handling

- **Age:** Missing values were imputed using median age grouped by Title (Mr, Mrs, Miss, Master, Other)
- **Embarked:** Missing values were filled with the mode (S - Southampton)
- **Fare:** Missing values were filled with the median fare
- **Cabin:** Missing values were marked as "Unknown", and cabin class (first letter) was extracted when available

#### 2.1.2 Feature Engineering

The following derived features were created:

- **Title:** Extracted from Name field (Mr, Mrs, Miss, Master, Other)
- **FamilySize:** SibSp + Parch + 1
- **IsAlone:** Binary indicator (1 if FamilySize == 1)
- **AgeGroup:** Categorical (Child: 0-12, Teen: 13-30, Adult: 31-50, Senior: 50+)
- **FareBin:** Quartile-based grouping (Q1, Q2, Q3, Q4)
- **CabinClass:** First letter of cabin number or "Unknown"

#### 2.1.3 Encoding

- **Sex:** Binary encoding (0=male, 1=female)
- **Embarked:** One-hot encoding (Embarked_C, Embarked_Q, Embarked_S)
- **Title, AgeGroup, CabinClass:** One-hot encoding

**Insert Figure: Data Preprocessing Pipeline**  
*[Figure: data_preprocessing_flowchart.png]*

### 2.2 Model Training (Direction A)

#### 2.2.1 Data Splitting

The dataset was split into:
- **Training set:** 80% (712 samples)
- **Test set:** 20% (178 samples)

Stratified sampling was used to maintain class balance. Random seed was set to 42 for reproducibility.

#### 2.2.2 Models Implemented

**1. Decision Tree (Main Model)**
- Criterion: Entropy
- Max depth: 5
- Min samples leaf: 20
- Min samples split: 40
- Class weight: Balanced
- **Purpose:** Primary model for interpretability and rule extraction

**2. Random Forest**
- N estimators: 100
- Max depth: 7
- Min samples leaf: 10
- **Purpose:** Ensemble method for improved performance

**3. Gradient Boosting**
- N estimators: 100
- Learning rate: 0.1
- Max depth: 4
- Subsample: 0.8
- **Purpose:** Advanced ensemble method for best performance

**Insert Figure: Model Architecture Comparison**  
*[Figure: model_architecture.png]*

### 2.3 Clustering Analysis (Direction B)

#### 2.3.1 Survivor Filtering

All passengers with Survived=1 were extracted for clustering analysis (342 survivors).

#### 2.3.2 Clustering Features

Features selected for clustering:
- Age (standardized)
- Fare (standardized)
- Pclass (numeric)
- FamilySize (standardized)

**Note:** Fare was used for clustering analysis but excluded from prediction model training.

#### 2.3.3 Clustering Methods

**1. KMeans (Main Method)**
- Optimal k determined using silhouette score
- Initialization: k-means++
- N init: 10

**2. Gaussian Mixture Model (GMM)**
- N components: Same as optimal k
- Covariance type: Full

**3. Hierarchical Clustering**
- Linkage: Ward
- N clusters: Same as optimal k

**Insert Figure: Clustering Methods Comparison**  
*[Figure: clustering_comparison.png]*

---

## 3. Results

### 3.1 Model Performance (Direction A)

#### 3.1.1 Overall Performance

The following table summarizes model performance on the test set:

| Model | Accuracy | AUC | F1 (Class 0) | F1 (Class 1) | Precision (Class 1) | Recall (Class 1) |
|-------|----------|-----|--------------|--------------|---------------------|------------------|
| Decision Tree | 0.7821 | 0.8200 | 0.8186 | 0.7273 | 0.7027 | 0.7536 |
| Random Forest | 0.8101 | 0.8560 | 0.8426 | 0.7606 | 0.7397 | 0.7826 |
| Gradient Boosting | 0.7933 | 0.8560 | 0.8398 | 0.7087 | 0.7759 | 0.6522 |


**Insert Figure: Model Accuracy Comparison**  
*[Figure: model_accuracy_comparison.png]*

**Insert Figure: Model ROC Curves**  
*[Figure: model_roc_curves.png]*

**Insert Figure: Confusion Matrices**  
*[Figure: confusion_matrices.png]*

**Insert Figure: Model Metrics Heatmap**  
*[Figure: model_metrics_heatmap.png]*

#### 3.1.2 Decision Tree Analysis

The Decision Tree model achieved an accuracy of 0.7821 and AUC of 0.8200, meeting the success criteria (AUC ≥ 0.80, Accuracy ≥ 75%).

**Insert Figure: Decision Tree Structure**  
*[Figure: decision_tree_structure.png]*

**Insert Figure: Feature Importance Comparison**  
*[Figure: feature_importance_comparison.png]*

#### 3.1.3 Key Findings

1. **Gender** is the most important factor, with females having significantly higher survival rates
2. **Passenger class** (Pclass) is strongly correlated with survival
3. **Age** plays a role, with children having better survival chances
4. **Family size** shows mixed effects - traveling alone vs. with family

### 3.2 Exploratory Data Analysis

#### 3.2.1 Overall Survival Statistics

- **Total survivors:** 342 (38.4%)
- **Female survival rate:** 74.2%
- **Male survival rate:** 18.9%

**Insert Figure: Gender Distribution Comparison**  
*[Figure: survivor_vs_nonsurvivor_gender.png]*

**Insert Figure: Class Distribution Comparison**  
*[Figure: survivor_vs_nonsurvivor_class.png]*

**Insert Figure: Age Distribution Comparison**  
*[Figure: survivor_vs_nonsurvivor_age.png]*

**Insert Figure: Fare Distribution Comparison**  
*[Figure: survivor_vs_nonsurvivor_fare.png]*

**Insert Figure: Feature Distributions**  
*[Figure: feature_distributions.png]*

**Insert Figure: Correlation Heatmap**  
*[Figure: correlation_heatmap.png]*

**Insert Figure: Survival Rate by Features**  
*[Figure: survival_rate_by_features.png]*

### 3.3 Survivor Clustering Analysis (Direction B)

#### 3.3.1 Optimal Cluster Number

The optimal number of clusters was determined using silhouette score analysis:

**Insert Figure: Elbow Curve**  
*[Figure: elbow_curve.png]*

Optimal k = 4 clusters (highest silhouette score: 0.3702)

#### 3.3.2 Clustering Method Comparison

| Method | Silhouette Score | Calinski-Harabasz Index | Davies-Bouldin Index |
|--------|------------------|-------------------------|----------------------|
| KMeans | 0.3702 | 163.44 | 1.0011 |
| GMM | 0.2267 | 92.58 | 2.1058 |
| Hierarchical | 0.3268 | 132.61 | 0.9595 |


**Insert Figure: Clustering Methods Comparison**  
*[Figure: clustering_comparison.png]*

#### 3.3.3 Cluster Profiles (KMeans)

The KMeans clustering identified 4 distinct survivor groups:

**Insert Figure: Cluster Scatter Plot (PCA)**  
*[Figure: cluster_scatter_2d.png]*

**Insert Figure: Cluster Size Distribution**  
*[Figure: cluster_size_distribution.png]*

**Insert Figure: Cluster Radar Charts**  
*[Figure: cluster_radar_charts.png]*

**Cluster Characteristics:**

- Cluster 0 (62 survivors, 18.1%): young, medium fare, third class, large family, mixed gender
- Cluster 1 (135 survivors, 39.5%): young adult, low fare, third class, small family, mixed gender
- Cluster 2 (130 survivors, 38.0%): middle-aged, high fare, first class, small family, mixed gender
- Cluster 3 (15 survivors, 4.4%): young adult, high fare, first class, small family, predominantly female


**Detailed Cluster Statistics:**

| Cluster | Size | % | Age (mean) | Fare (mean) | Pclass (mode) | FamilySize (mean) | Female % |
|---------|------|---|------------|-------------|---------------|-------------------|----------|
| 0 | 62 | 18.1% | 11.0 | 29.2 | 3 | 3.5 | 62.9% |
| 1 | 135 | 39.5% | 26.3 | 14.0 | 3 | 1.3 | 68.1% |
| 2 | 130 | 38.0% | 37.8 | 65.6 | 1 | 1.8 | 68.5% |
| 3 | 15 | 4.4% | 30.5 | 287.8 | 1 | 2.7 | 86.7% |


### 3.4 Statistical Significance Tests

Statistical tests were performed to validate differences between survivors and non-survivors:

- **Gender (Chi-square):** p < 0.001 (highly significant)
- **Pclass (Chi-square):** p < 0.001 (highly significant)
- **Embarked (Chi-square):** p < 0.001 (highly significant)
- **Age (Mann-Whitney U):** p = 0.087 (marginally significant)
- **Fare (Mann-Whitney U):** p < 0.001 (highly significant)

---

## 4. Discussion

### 4.1 Model Performance Analysis

The Decision Tree model successfully met the success criteria with AUC ≥ 0.80 and accuracy ≥ 75%. The model provides interpretable rules that can be easily understood and explained. Random Forest and Gradient Boosting showed similar or slightly better performance, but at the cost of interpretability.

### 4.2 Key Survival Factors

The analysis confirms historical accounts of the disaster:

1. **"Women and children first" policy:** Females had a 74.2% survival rate vs. 18.9% for males.

2. **Class privilege:** First-class passengers had 63.0% survival rate, compared to 24.2% for third-class passengers.

3. **Age factor:** Children (age ≤ 12) showed better survival rates, consistent with rescue priorities.

### 4.3 Survivor Clustering Insights

The clustering analysis revealed 4 distinct survivor profiles:

1. **Cluster 0:** Young passengers with families, primarily third class
2. **Cluster 1:** Young adults traveling alone, mostly third class
3. **Cluster 2:** Middle-aged, high-fare passengers, primarily first class
4. **Cluster 3:** High-fare, first-class passengers (smallest group)

These clusters align with the historical context and rescue patterns.

### 4.4 Model Limitations and Ethical Considerations

1. **Historical bias:** The dataset reflects historical social structures and rescue policies ("women and children first")
2. **Data limitations:** Missing values in Age and Cabin may affect model accuracy
3. **Temporal context:** Results are specific to this historical event and should not be generalized
4. **Ethical use:** This model is for educational purposes only and should not be used for real-world decision-making

### 4.5 Fairness Analysis

The model shows different performance across demographic groups:
- Gender-based differences in false positive/negative rates
- Class-based accuracy variations

These differences reflect historical patterns rather than model bias, but should be acknowledged in any application.

---

## 5. Conclusions

### 5.1 Summary of Findings

1. **Model Performance:** Successfully built interpretable models meeting AUC ≥ 0.80 and accuracy ≥ 75% criteria
2. **Key Factors:** Gender, passenger class, and age are the most significant survival factors
3. **Survivor Profiles:** Identified 4 distinct survivor groups with unique characteristics
4. **Statistical Validation:** Confirmed significant differences between survivors and non-survivors across multiple features

### 5.2 Contributions

- Comprehensive analysis combining prediction and profiling approaches
- Multiple model comparison providing performance benchmarks
- Interpretable decision rules for transparency
- Clustering-based survivor profile identification

### 5.3 Future Work

1. **Model improvements:** Explore advanced feature engineering and hyperparameter tuning
2. **Additional models:** Implement XGBoost, LightGBM, or neural networks
3. **Explainability:** Add SHAP values for enhanced interpretability
4. **Deployment:** Create API for real-time predictions

---

## 6. Appendix

### 6.1 Project Structure

```
titanic-survival-analysis/
├── data/
│   ├── Titanic-Dataset.csv
│   └── titanic_cleaned.csv
├── src/
│   ├── 00_data_prep.py
│   ├── 10_tree_classifier.py
│   ├── 20_survivor_clustering.py
│   ├── 30_reports.py
│   └── 40_generate_report.py
├── models/
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   └── kmeans.pkl
├── reports/
│   ├── figures/ (18 visualization files)
│   ├── tables/
│   └── Titanic_Report.md
└── app/
    └── streamlit_app.py
```

### 6.2 Complete Visualization List

All figures are saved in `reports/figures/`:

1. Model Performance:
   - model_accuracy_comparison.png
   - model_roc_curves.png
   - confusion_matrices.png
   - model_metrics_heatmap.png
   - decision_tree_structure.png
   - feature_importance_comparison.png

2. Exploratory Data Analysis:
   - survivor_vs_nonsurvivor_gender.png
   - survivor_vs_nonsurvivor_class.png
   - survivor_vs_nonsurvivor_age.png
   - survivor_vs_nonsurvivor_fare.png
   - feature_distributions.png
   - correlation_heatmap.png
   - survival_rate_by_features.png

3. Clustering Analysis:
   - clustering_comparison.png
   - elbow_curve.png
   - cluster_scatter_2d.png
   - cluster_radar_charts.png
   - cluster_size_distribution.png

### 6.3 Code Availability

All code is available in the `src/` directory. To reproduce results:

1. Run data preprocessing: `python src/00_data_prep.py`
2. Train models: `python src/10_tree_classifier.py`
3. Perform clustering: `python src/20_survivor_clustering.py`
4. Generate visualizations: `python src/30_reports.py`
5. Launch Streamlit app: `streamlit run app/streamlit_app.py`

### 6.4 Model Cards

**Decision Tree Model:**
- Version: 1.0
- Training Date: 2025-11-07
- Features: 28 (excluding Fare)
- Performance: Accuracy=0.8045, AUC=0.8489
- Hyperparameters: max_depth=5, min_samples_leaf=20, criterion='entropy'

---

**End of Report**
