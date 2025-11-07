# Design Document

## Overview

本设计文档描述 Titanic 生存分析系统的技术架构和实现方案。系统采用轻量级、可解释的机器学习方法，包含两个核心研究方向：

- **方向A（预测）**：使用多个模型变体预测乘客生存概率，进行性能对比，并提供基于规则的解释
- **方向B（画像）**：使用统计分析和多种聚类方法识别幸存者群体特征

系统设计遵循以下原则：
1. **可解释性优先**：主模型使用决策树，确保每个预测都有清晰的规则路径
2. **模型对比**：实现多个模型变体（决策树、随机森林、逻辑回归、集成方法），展示不同方法的优劣
3. **模块化设计**：数据处理、建模、可视化、界面各自独立，便于维护和测试
4. **可复现性**：固定随机种子，保存所有中间结果和模型文件
5. **轻量级实现**：避免过度工程化，使用成熟的 scikit-learn 和 pandas 工具链

## Architecture

### System Architecture

系统采用经典的数据科学项目架构，分为五个主要层次：

```
┌─────────────────────────────────────────────────────────┐
│                   Presentation Layer                     │
│              (Streamlit Web Interface)                   │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                   Application Layer                      │
│         (Prediction Service, Analysis Service)           │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                     Model Layer                          │
│        (Decision Tree, KMeans, Scaler)                   │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                  Data Processing Layer                   │
│    (Cleaning, Feature Engineering, Transformation)       │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                      Data Layer                          │
│           (CSV Files, Pickle Models)                     │
└─────────────────────────────────────────────────────────┘
```

### Data Flow


#### Training Pipeline

```
Titanic-Dataset.csv
    │
    ├─> 00_data_prep.py
    │   ├─> Clean missing values
    │   ├─> Encode categorical features
    │   ├─> Engineer derived features
    │   └─> Save: titanic_cleaned.csv
    │
    ├─> 10_tree_classifier.py (Direction A)
    │   ├─> Train/test split (80/20)
    │   ├─> Train DecisionTreeClassifier
    │   ├─> Evaluate metrics (Accuracy, AUC, F1)
    │   ├─> Extract decision rules
    │   └─> Save: tree_model.pkl
    │
    ├─> 20_survivor_clustering.py (Direction B)
    │   ├─> Filter survivors (Survived=1)
    │   ├─> Statistical analysis
    │   ├─> Standardize features
    │   ├─> KMeans clustering (k=3)
    │   ├─> Generate cluster profiles
    │   └─> Save: kmeans.pkl, scaler.pkl
    │
    └─> 30_reports.py
        ├─> Generate all visualizations
        ├─> Create comparison tables
        ├─> Export figures and tables
        └─> Save: reports/figures/, reports/tables/
```

#### Inference Pipeline

```
User Input (Streamlit)
    │
    ├─> Load Models
    │   ├─> tree_model.pkl
    │   ├─> kmeans.pkl
    │   └─> scaler.pkl
    │
    ├─> Prediction Service
    │   ├─> Preprocess input
    │   ├─> Apply feature engineering
    │   ├─> Predict with decision tree
    │   ├─> Extract decision path
    │   └─> Generate rule-based explanation
    │
    └─> Display Results
        ├─> Survival probability
        ├─> Decision rules
        └─> Counterfactual suggestions
```



## Components and Interfaces

### 1. Data Processing Module (00_data_prep.py)

**职责**：数据清洗、特征工程、数据转换

**输入**：
- `data/Titanic-Dataset.csv` - 原始数据集

**输出**：
- `data/titanic_cleaned.csv` - 清洗后的完整数据集
- `data/feature_metadata.json` - 特征工程的元数据（分位数阈值、编码映射等）

**核心函数**：

```python
def load_raw_data(filepath: str) -> pd.DataFrame
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame
def extract_title(name: str) -> str
def engineer_features(df: pd.DataFrame) -> pd.DataFrame
def encode_categorical(df: pd.DataFrame) -> pd.DataFrame
def save_cleaned_data(df: pd.DataFrame, filepath: str) -> None
```

**特征工程细节**：

| 原始特征 | 处理方法 | 派生特征 |
|---------|---------|---------|
| Age | 按 Title 中位数填补 | AgeGroup (0-12, 13-30, 31-50, 50+) |
| Name | 正则提取称谓 | Title (Mr, Mrs, Miss, Master, Other) |
| SibSp, Parch | 求和 + 1 | FamilySize, IsAlone |
| Fare | 四分位数分组 | FareBin (Q1, Q2, Q3, Q4) |
| Sex | 二值编码 | Sex_encoded (0=male, 1=female) |
| Embarked | One-hot 编码 | Embarked_C, Embarked_Q, Embarked_S |
| Cabin | 提取首字母 | CabinClass (A-G, Unknown) |

### 2. Model Training Module (10_tree_classifier.py)

**职责**：训练多个分类模型变体，评估性能，选择最佳模型

**输入**：
- `data/titanic_cleaned.csv` - 清洗后的数据

**输出**：
- `models/decision_tree.pkl` - 决策树模型（主模型，用于解释）
- `models/random_forest.pkl` - 随机森林模型
- `models/gradient_boosting.pkl` - 梯度提升模型
- `models/train_test_indices.pkl` - 训练/测试集索引
- `reports/model_comparison.csv` - 模型性能对比表

**模型变体设计**：

#### Model 1: Decision Tree (主模型 - 可解释性)
```python
DecisionTreeClassifier(
    criterion='entropy',
    max_depth=5,
    min_samples_leaf=20,
    min_samples_split=40,
    class_weight='balanced',
    random_state=42
)
```
- **优势**：完全可解释，可提取规则
- **用途**：主预测模型，规则解释，决策路径可视化
- **预期性能**：AUC ~0.82, Accuracy ~0.80

#### Model 2: Random Forest (集成方法 - 性能提升)
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=7,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
```
- **优势**：更高准确率，特征重要性稳定，减少过拟合
- **用途**：性能对比，特征重要性分析
- **预期性能**：AUC ~0.86, Accuracy ~0.84

#### Model 3: Gradient Boosting (高级方法 - 最佳性能)
```python
GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    min_samples_leaf=15,
    subsample=0.8,
    random_state=42
)
```
- **优势**：通常最高准确率，逐步优化错误
- **用途**：性能上限参考，展示高级集成方法
- **预期性能**：AUC ~0.88, Accuracy ~0.86

**核心函数**：

```python
def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]
def split_data(X: pd.DataFrame, y: pd.Series) -> Tuple
def train_decision_tree(X_train, y_train) -> DecisionTreeClassifier
def train_random_forest(X_train, y_train) -> RandomForestClassifier
def train_gradient_boosting(X_train, y_train) -> GradientBoostingClassifier
def evaluate_model(model, X_test, y_test) -> dict
def extract_decision_rules(tree_model, feature_names: list) -> list
def get_decision_path(tree_model, sample: np.array) -> str
def compare_models(models: dict, X_test, y_test) -> pd.DataFrame
def save_models(models: dict, output_dir: str) -> None
```

**评估指标**：

所有模型将在以下指标上进行对比：
- Accuracy
- Precision (class 0 & 1)
- Recall (class 0 & 1)
- F1-score (class 0 & 1)
- ROC-AUC
- Training time
- Prediction time



### 3. Clustering Analysis Module (20_survivor_clustering.py)

**职责**：对幸存者进行聚类分析，使用多种聚类方法进行对比

**输入**：
- `data/titanic_cleaned.csv` - 清洗后的数据

**输出**：
- `models/kmeans.pkl` - KMeans 聚类模型（主模型）
- `models/gmm.pkl` - 高斯混合模型
- `models/hierarchical_labels.pkl` - 层次聚类结果
- `models/scaler.pkl` - 特征标准化器
- `reports/cluster_comparison.csv` - 聚类方法对比表
- `reports/cluster_profiles.csv` - 各簇特征统计

**聚类方法设计**：

#### Method 1: KMeans (主方法 - 简单高效)
```python
KMeans(
    n_clusters=3,
    init='k-means++',
    n_init=10,
    random_state=42
)
```
- **优势**：快速、结果稳定、易于解释
- **用途**：主聚类方法，生成幸存者画像

#### Method 2: Gaussian Mixture Model (概率模型)
```python
GaussianMixture(
    n_components=3,
    covariance_type='full',
    random_state=42
)
```
- **优势**：软聚类、考虑特征分布、提供概率
- **用途**：对比 KMeans，展示概率聚类

#### Method 3: Hierarchical Clustering (层次结构)
```python
AgglomerativeClustering(
    n_clusters=3,
    linkage='ward'
)
```
- **优势**：可视化树状图、不需预设簇数
- **用途**：探索最优簇数，生成树状图

**聚类特征选择**：
- Age (标准化)
- Fare (标准化)
- Pclass (已是数值)
- FamilySize (标准化)

**聚类评估指标**：
- Silhouette Score (轮廓系数)
- Calinski-Harabasz Index (CH 指数)
- Davies-Bouldin Index (DB 指数)
- Inertia (仅 KMeans)

**核心函数**：

```python
def filter_survivors(df: pd.DataFrame) -> pd.DataFrame
def select_clustering_features(df: pd.DataFrame) -> pd.DataFrame
def standardize_features(X: pd.DataFrame) -> Tuple[np.array, StandardScaler]
def determine_optimal_k(X: np.array, k_range: range) -> dict
def train_kmeans(X: np.array, n_clusters: int) -> KMeans
def train_gmm(X: np.array, n_components: int) -> GaussianMixture
def train_hierarchical(X: np.array, n_clusters: int) -> np.array
def evaluate_clustering(X: np.array, labels: np.array) -> dict
def generate_cluster_profiles(df: pd.DataFrame, labels: np.array) -> pd.DataFrame
def create_cluster_descriptions(profiles: pd.DataFrame) -> dict
def compare_clustering_methods(X: np.array, methods: dict) -> pd.DataFrame
def save_clustering_models(models: dict, output_dir: str) -> None
```

### 4. Reporting and Visualization Module (30_reports.py)

**职责**：生成所有统计图表、对比表格、可视化结果

**输入**：
- `data/titanic_cleaned.csv` - 清洗后的数据
- `models/*.pkl` - 所有训练好的模型
- `reports/model_comparison.csv` - 模型对比数据
- `reports/cluster_comparison.csv` - 聚类对比数据

**输出**：
- `reports/figures/` - 所有可视化图表
- `reports/tables/` - 所有统计表格

**可视化清单**：

#### 方向A - 预测模型可视化

1. **模型性能对比**
   - `model_accuracy_comparison.png` - 各模型准确率柱状图
   - `model_roc_curves.png` - 所有模型的 ROC 曲线对比
   - `model_metrics_heatmap.png` - 多指标热力图
   - `confusion_matrices.png` - 所有模型的混淆矩阵（子图）

2. **决策树可视化**
   - `decision_tree_structure.png` - 决策树结构图
   - `feature_importance_comparison.png` - 不同模型的特征重要性对比
   - `decision_rules_text.txt` - 提取的决策规则文本

3. **特征分析**
   - `feature_distributions.png` - 各特征分布（按生存情况分组）
   - `correlation_heatmap.png` - 特征相关性热力图
   - `survival_rate_by_features.png` - 各特征对生存率的影响

#### 方向B - 幸存者画像可视化

4. **统计对比**
   - `survivor_vs_nonsurvivor_gender.png` - 性别分布对比
   - `survivor_vs_nonsurvivor_class.png` - 舱位分布对比
   - `survivor_vs_nonsurvivor_age.png` - 年龄分布对比（直方图+箱型图）
   - `survivor_vs_nonsurvivor_fare.png` - 票价分布对比

5. **聚类可视化**
   - `clustering_methods_comparison.png` - 不同聚类方法的轮廓系数对比
   - `elbow_curve.png` - KMeans 肘部曲线（确定最优 k）
   - `cluster_scatter_2d.png` - 聚类散点图（Age-Fare，PCA 降维）
   - `cluster_radar_charts.png` - 各簇特征雷达图
   - `hierarchical_dendrogram.png` - 层次聚类树状图

6. **群体画像**
   - `cluster_profiles_table.png` - 各簇统计特征表格可视化
   - `cluster_size_distribution.png` - 各簇人数分布饼图

**核心函数**：

```python
def plot_model_comparison(comparison_df: pd.DataFrame) -> None
def plot_roc_curves(models: dict, X_test, y_test) -> None
def plot_confusion_matrices(models: dict, X_test, y_test) -> None
def plot_decision_tree(tree_model, feature_names: list) -> None
def plot_feature_importance_comparison(models: dict, feature_names: list) -> None
def plot_survival_analysis(df: pd.DataFrame) -> None
def plot_clustering_comparison(comparison_df: pd.DataFrame) -> None
def plot_cluster_scatter(X: np.array, labels: np.array, feature_names: list) -> None
def plot_cluster_radar_charts(profiles: pd.DataFrame) -> None
def plot_dendrogram(X: np.array) -> None
def generate_all_visualizations() -> None
```



### 5. Streamlit Application (app/streamlit_app.py)

**职责**：提供 Web 交互界面，展示预测和分析结果

**界面设计**：

#### Tab 1: 生存预测

**布局**：
```
┌─────────────────────────────────────────────────────────┐
│  🚢 Titanic 生存预测系统                                  │
├─────────────────────────────────────────────────────────┤
│  [Tab: 生存预测] [Tab: 幸存者画像] [Tab: 模型对比]        │
├─────────────────────────────────────────────────────────┤
│  输入乘客信息：                                           │
│  ┌─────────────┬─────────────┬─────────────┐            │
│  │ 性别: [▼]   │ 年龄: [__]  │ 舱位: [▼]   │            │
│  │ 票价: [__]  │ 兄弟姐妹: [▼]│ 父母子女: [▼]│            │
│  │ 登船港口: [▼]│             │             │            │
│  └─────────────┴─────────────┴─────────────┘            │
│                                                          │
│  [预测] 按钮                                              │
│                                                          │
│  预测结果：                                               │
│  ┌─────────────────────────────────────────────────┐    │
│  │ 生存概率: ████████░░ 82%                         │    │
│  │ 预测结果: ✅ 幸存                                 │    │
│  └─────────────────────────────────────────────────┘    │
│                                                          │
│  决策路径：                                               │
│  ┌─────────────────────────────────────────────────┐    │
│  │ 1. Sex = female ✓                                │    │
│  │ 2. Pclass <= 2 ✓                                 │    │
│  │ 3. Age <= 35 ✓                                   │    │
│  │ → 预测: 幸存 (置信度: 0.92)                       │    │
│  └─────────────────────────────────────────────────┘    │
│                                                          │
│  改善建议：                                               │
│  ┌─────────────────────────────────────────────────┐    │
│  │ • 当前预测为幸存，主要因素：                       │    │
│  │   - 性别为女性 (影响最大)                         │    │
│  │   - 舱位等级较高 (1或2等)                         │    │
│  │ • 如果改变条件：                                   │    │
│  │   - 若舱位降至3等 → 生存概率降至 65%              │    │
│  │   - 若年龄增至50+ → 生存概率降至 75%              │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

#### Tab 2: 幸存者画像

**布局**：
```
┌─────────────────────────────────────────────────────────┐
│  📊 幸存者特征分析                                        │
├─────────────────────────────────────────────────────────┤
│  基本统计：                                               │
│  ┌─────────────────────────────────────────────────┐    │
│  │ 总幸存人数: 342 (38.4%)                          │    │
│  │ 女性幸存率: 74.2%  |  男性幸存率: 18.9%          │    │
│  │ 1等舱: 63%  |  2等舱: 47%  |  3等舱: 24%         │    │
│  └─────────────────────────────────────────────────┘    │
│                                                          │
│  统计对比图表：                                           │
│  ┌──────────────┬──────────────┐                        │
│  │ 性别分布对比  │ 舱位分布对比  │                        │
│  │ [柱状图]     │ [柱状图]     │                        │
│  └──────────────┴──────────────┘                        │
│  ┌──────────────┬──────────────┐                        │
│  │ 年龄分布对比  │ 票价分布对比  │                        │
│  │ [箱型图]     │ [箱型图]     │                        │
│  └──────────────┴──────────────┘                        │
│                                                          │
│  聚类分析结果：                                           │
│  ┌─────────────────────────────────────────────────┐    │
│  │ 使用 KMeans 识别出 3 个典型幸存者群体：           │    │
│  │                                                  │    │
│  │ 群体 1 (35%): 富裕家庭与青年                     │    │
│  │   • 平均年龄: 28岁                               │    │
│  │   • 平均票价: £65                                │    │
│  │   • 主要舱位: 1等舱 (78%)                        │    │
│  │   • 家庭规模: 2.8人                              │    │
│  │                                                  │    │
│  │ 群体 2 (42%): 中产阶级女性                       │    │
│  │   • 平均年龄: 35岁                               │    │
│  │   • 平均票价: £28                                │    │
│  │   • 主要舱位: 2等舱 (65%)                        │    │
│  │   • 家庭规模: 1.5人                              │    │
│  │                                                  │    │
│  │ 群体 3 (23%): 儿童与家庭                         │    │
│  │   • 平均年龄: 12岁                               │    │
│  │   • 平均票价: £35                                │    │
│  │   • 主要舱位: 混合                               │    │
│  │   • 家庭规模: 4.2人                              │    │
│  └─────────────────────────────────────────────────┘    │
│                                                          │
│  聚类可视化：                                             │
│  ┌──────────────┬──────────────┐                        │
│  │ 散点图       │ 雷达图       │                        │
│  │ [Age-Fare]  │ [多维特征]   │                        │
│  └──────────────┴──────────────┘                        │
└─────────────────────────────────────────────────────────┘
```

#### Tab 3: 模型对比

**布局**：
```
┌─────────────────────────────────────────────────────────┐
│  🔬 模型性能对比                                          │
├─────────────────────────────────────────────────────────┤
│  分类模型对比：                                           │
│  ┌─────────────────────────────────────────────────┐    │
│  │ 模型          准确率  AUC   F1    训练时间       │    │
│  │ ─────────────────────────────────────────────   │    │
│  │ Decision Tree  0.80  0.82  0.78   0.05s        │    │
│  │ Random Forest  0.84  0.86  0.82   0.32s        │    │
│  │ Gradient Boost 0.86  0.88  0.84   0.45s        │    │
│  └─────────────────────────────────────────────────┘    │
│                                                          │
│  ROC 曲线对比：                                           │
│  ┌─────────────────────────────────────────────────┐    │
│  │ [所有模型的 ROC 曲线叠加图]                       │    │
│  └─────────────────────────────────────────────────┘    │
│                                                          │
│  特征重要性对比：                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │ [不同模型的特征重要性柱状图对比]                   │    │
│  └─────────────────────────────────────────────────┘    │
│                                                          │
│  聚类方法对比：                                           │
│  ┌─────────────────────────────────────────────────┐    │
│  │ 方法          Silhouette  CH Index  DB Index    │    │
│  │ ─────────────────────────────────────────────   │    │
│  │ KMeans         0.45       285.3     0.82        │    │
│  │ GMM            0.43       272.1     0.85        │    │
│  │ Hierarchical   0.44       278.5     0.83        │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

**核心函数**：

```python
def load_models() -> dict
def load_data() -> pd.DataFrame
def preprocess_input(user_input: dict) -> pd.DataFrame
def predict_with_explanation(model, sample: pd.DataFrame) -> dict
def generate_counterfactuals(tree_model, sample: pd.DataFrame) -> list
def display_prediction_tab() -> None
def display_survivor_profile_tab() -> None
def display_model_comparison_tab() -> None
def main() -> None
```



## Data Models

### Input Data Schema

**原始数据 (Titanic-Dataset.csv)**

| 字段 | 类型 | 说明 | 缺失值处理 |
|-----|------|------|-----------|
| PassengerId | int | 乘客ID | 无缺失 |
| Survived | int | 是否幸存 (0/1) | 无缺失（目标变量）|
| Pclass | int | 舱位等级 (1/2/3) | 无缺失 |
| Name | str | 姓名 | 无缺失 |
| Sex | str | 性别 (male/female) | 无缺失 |
| Age | float | 年龄 | 按 Title 中位数填补 |
| SibSp | int | 同行兄弟姐妹/配偶数 | 无缺失 |
| Parch | int | 同行父母/子女数 | 无缺失 |
| Ticket | str | 船票编号 | 无缺失（不使用）|
| Fare | float | 票价 | 中位数填补 |
| Cabin | str | 舱房号 | 提取首字母或标记 Unknown |
| Embarked | str | 登船港口 (C/Q/S) | 众数填补 |

### Processed Data Schema

**清洗后数据 (titanic_cleaned.csv)**

| 字段 | 类型 | 说明 | 取值范围 |
|-----|------|------|---------|
| Survived | int | 目标变量 | 0, 1 |
| Pclass | int | 舱位等级 | 1, 2, 3 |
| Sex_encoded | int | 性别编码 | 0 (male), 1 (female) |
| Age | float | 年龄（填补后）| 0.42 - 80 |
| SibSp | int | 兄弟姐妹/配偶数 | 0 - 8 |
| Parch | int | 父母/子女数 | 0 - 6 |
| Fare | float | 票价（填补后）| 0 - 512.33 |
| Embarked_C | int | 登船港口 C | 0, 1 |
| Embarked_Q | int | 登船港口 Q | 0, 1 |
| Embarked_S | int | 登船港口 S | 0, 1 |
| Title | str | 称谓 | Mr, Mrs, Miss, Master, Other |
| FamilySize | int | 家庭规模 | 1 - 11 |
| IsAlone | int | 是否独行 | 0, 1 |
| AgeGroup | str | 年龄段 | Child, Teen, Adult, Senior |
| FareBin | str | 票价分组 | Q1, Q2, Q3, Q4 |
| CabinClass | str | 舱位区域 | A-G, Unknown |

### Model Output Schema

**预测结果**

```python
{
    "survival_probability": float,  # 0.0 - 1.0
    "prediction": int,              # 0 or 1
    "decision_path": list[str],     # 决策规则列表
    "confidence": float,            # 0.0 - 1.0
    "counterfactuals": list[dict]   # 反事实建议
}
```

**聚类结果**

```python
{
    "cluster_id": int,              # 0, 1, 2
    "cluster_name": str,            # 群体名称
    "cluster_size": int,            # 该簇人数
    "cluster_percentage": float,    # 占比
    "profile": {
        "avg_age": float,
        "avg_fare": float,
        "pclass_distribution": dict,
        "gender_distribution": dict,
        "avg_family_size": float
    }
}
```

## Error Handling

### 数据处理错误

1. **缺失值过多**
   - 检测：如果某列缺失率 > 50%，发出警告
   - 处理：记录日志，使用默认填补策略

2. **异常值检测**
   - Age < 0 或 Age > 100 → 标记为异常，使用中位数替换
   - Fare < 0 → 标记为异常，使用中位数替换
   - 记录所有异常值到日志文件

3. **编码错误**
   - 遇到未知类别 → 映射到 "Other" 类别
   - 记录未知类别到日志

### 模型训练错误

1. **训练失败**
   - 捕获异常，记录错误信息
   - 跳过该模型，继续训练其他模型
   - 在报告中标记失败的模型

2. **性能过低**
   - 如果 AUC < 0.6，发出警告
   - 建议检查数据质量和特征工程

3. **过拟合检测**
   - 如果训练集准确率 - 测试集准确率 > 0.15，发出警告
   - 建议调整模型复杂度

### 界面错误

1. **模型加载失败**
   - 显示友好错误信息
   - 提示用户先运行训练脚本

2. **输入验证**
   - Age: 0-100
   - Fare: >= 0
   - 其他字段：下拉选择，避免输入错误

3. **预测失败**
   - 捕获异常，显示错误信息
   - 提供默认预测结果

## Testing Strategy

### 单元测试

**数据处理模块**
- 测试缺失值填补逻辑
- 测试特征工程函数
- 测试编码转换

**模型训练模块**
- 测试每个模型的训练和预测
- 测试决策规则提取
- 测试模型保存和加载

**聚类模块**
- 测试聚类算法
- 测试评估指标计算
- 测试画像生成

### 集成测试

- 端到端数据流测试
- 模型训练到预测的完整流程
- Streamlit 界面交互测试

### 性能测试

- 训练时间测试（应 < 5分钟）
- 预测延迟测试（应 < 100ms）
- 内存使用测试（应 < 2GB）

### 验证测试

- 交叉验证（5-fold）
- 不同随机种子的稳定性测试
- 模型性能基线验证（AUC >= 0.80）

## Deployment Considerations

### 环境要求

```
Python >= 3.8
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
streamlit >= 1.10.0
```

### 运行流程

1. **数据准备**
   ```bash
   python src/00_data_prep.py
   ```

2. **模型训练**
   ```bash
   python src/10_tree_classifier.py
   python src/20_survivor_clustering.py
   ```

3. **生成报告**
   ```bash
   python src/30_reports.py
   ```

4. **启动界面**
   ```bash
   streamlit run app/streamlit_app.py
   ```

### 文件依赖关系

```
00_data_prep.py
    ↓ (生成 titanic_cleaned.csv)
    ├─> 10_tree_classifier.py
    │       ↓ (生成 models/*.pkl)
    │       └─> streamlit_app.py
    │
    ├─> 20_survivor_clustering.py
    │       ↓ (生成 models/*.pkl)
    │       └─> streamlit_app.py
    │
    └─> 30_reports.py
            ↓ (生成 reports/*)
            └─> streamlit_app.py
```

## Performance Optimization

### 数据处理优化

- 使用 pandas 向量化操作，避免循环
- 缓存中间结果，避免重复计算
- 使用 pickle 保存处理后的数据

### 模型训练优化

- 并行训练多个模型（使用 joblib）
- 使用 n_jobs=-1 利用多核 CPU
- 缓存训练好的模型

### 界面优化

- 使用 @st.cache_data 缓存数据加载
- 使用 @st.cache_resource 缓存模型加载
- 延迟加载大型可视化

## Future Enhancements

1. **模型改进**
   - 添加 XGBoost、LightGBM 等高级模型
   - 实现自动超参数调优（Optuna）
   - 添加 SHAP 值解释

2. **特征工程**
   - 从 Ticket 提取更多信息
   - 添加特征交互项
   - 自动特征选择

3. **界面增强**
   - 添加批量预测功能
   - 导出预测结果为 CSV
   - 添加模型性能监控面板

4. **部署**
   - Docker 容器化
   - 云端部署（Streamlit Cloud）
   - API 接口（FastAPI）

