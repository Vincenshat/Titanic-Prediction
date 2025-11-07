# Titanic Survival Analysis Project

A comprehensive machine learning project analyzing the Titanic survival dataset with two main directions:
- **Direction A:** Survival prediction using interpretable decision tree models
- **Direction B:** Survivor profile analysis using statistical methods and clustering

## Project Overview

This project implements a complete data science pipeline including:
- Data preprocessing and feature engineering
- Multiple classification models (Decision Tree, Random Forest, Gradient Boosting)
- Survivor clustering analysis (KMeans, GMM, Hierarchical)
- Comprehensive visualizations (18+ figures)
- Interactive Streamlit web application
- Research report with figure placeholders

## Project Structure

```
Myproject/
├── data/
│   ├── Titanic-Dataset.csv          # Original dataset
│   ├── titanic_cleaned.csv          # Processed data
│   └── feature_metadata.json        # Feature engineering metadata
├── src/
│   ├── 00_data_prep.py             # Data preprocessing
│   ├── 10_tree_classifier.py       # Model training (Direction A)
│   ├── 20_survivor_clustering.py   # Clustering analysis (Direction B)
│   ├── 30_reports.py               # Visualization generation
│   └── 40_generate_report.py       # Report generation
├── models/
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   ├── kmeans.pkl
│   └── scaler.pkl
├── reports/
│   ├── figures/                    # All visualization files (18 figures)
│   ├── tables/                      # Statistical tables
│   └── Titanic_Report.md           # Research report
├── app/
│   └── streamlit_app.py            # Interactive web application
└── requirements.txt
```

## Installation

1. Clone or download this repository
2. Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Data Preprocessing

```bash
python src/00_data_prep.py
```

This will:
- Load and clean the raw data
- Handle missing values
- Perform feature engineering
- Save cleaned data to `data/titanic_cleaned.csv`

### Step 2: Train Prediction Models

```bash
python src/10_tree_classifier.py
```

This will:
- Train Decision Tree, Random Forest, and Gradient Boosting models
- Evaluate model performance
- Extract decision rules
- Save models to `models/` directory
- Generate model comparison table

**Note:** Fare feature is excluded from model training as per requirements.

### Step 3: Perform Clustering Analysis

```bash
python src/20_survivor_clustering.py
```

This will:
- Filter survivors (Survived=1)
- Perform statistical analysis
- Run clustering algorithms (KMeans, GMM, Hierarchical)
- Generate cluster profiles
- Save clustering models and results

### Step 4: Generate Visualizations

```bash
python src/30_reports.py
```

This will generate all 18 visualization figures in `reports/figures/`:
- Model performance comparisons
- Decision tree visualizations
- EDA plots
- Clustering visualizations

### Step 5: Generate Report

```bash
python src/40_generate_report.py
```

This will generate a comprehensive Markdown report in `reports/Titanic_Report.md` with figure placeholders.

### Step 6: Launch Interactive Application

```bash
streamlit run app/streamlit_app.py
```

This will launch a web interface with three tabs:
- **Tab 1:** Survival Prediction - Enter passenger info to get survival probability
- **Tab 2:** Survivor Profiles - View statistical analysis and clustering results
- **Tab 3:** Model Comparison - Compare model performance metrics

## Model Performance

### Best Model Results

| Model | Accuracy | AUC | F1 (Class 1) |
|-------|----------|-----|--------------|
| Decision Tree | 0.8045 | 0.8489 | 0.7651 |
| Random Forest | 0.8212 | 0.8431 | 0.7808 |
| Gradient Boosting | 0.8045 | 0.8602 | 0.7200 |

All models meet the success criteria: **AUC ≥ 0.80, Accuracy ≥ 75%**

## Key Findings

1. **Gender** is the most important survival factor (74.2% female survival vs. 18.9% male)
2. **Passenger class** strongly correlates with survival (63% first class vs. 24% third class)
3. **Age** plays a role, with children having better survival chances
4. **Clustering** identified 4 distinct survivor groups with unique characteristics

## Features

- **Interpretable Models:** Decision tree provides clear IF-THEN rules
- **Multiple Model Comparison:** Decision Tree, Random Forest, Gradient Boosting
- **Comprehensive Visualizations:** 18+ figures covering all aspects
- **Statistical Analysis:** Chi-square and Mann-Whitney U tests
- **Clustering Analysis:** KMeans, GMM, and Hierarchical methods
- **Interactive Demo:** Streamlit web application
- **Complete Report:** Research report with figure placeholders

## Important Notes

1. **Fare Feature:** Excluded from model training but used for analysis and clustering
2. **Reproducibility:** Random seed set to 42 throughout
3. **Ethical Considerations:** Model reflects historical patterns, not recommended for real-world use
4. **All code and comments are in English** as per requirements

## Dependencies

- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- streamlit >= 1.10.0
- scipy >= 1.7.0

## License

This project is for educational purposes only.

## Contact

For questions or issues, please refer to the project documentation in `reports/Titanic_Report.md`.

