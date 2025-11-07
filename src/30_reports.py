"""
Visualization and Reporting Module

This module generates all visualizations and reports for the Titanic survival analysis.
All code and comments are in English.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Set random seed
np.random.seed(42)

def load_data_and_models():
    """Load all necessary data and models."""
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    models_dir = base_dir / 'models'
    reports_dir = base_dir / 'reports'
    
    # Load data
    df = pd.read_csv(data_dir / 'titanic_cleaned.csv')
    
    # Load models
    with open(models_dir / 'decision_tree.pkl', 'rb') as f:
        dt_model = pickle.load(f)
    with open(models_dir / 'random_forest.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    with open(models_dir / 'gradient_boosting.pkl', 'rb') as f:
        gb_model = pickle.load(f)
    
    # Load feature names
    with open(models_dir / 'feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    # Load train/test indices
    with open(models_dir / 'train_test_indices.pkl', 'rb') as f:
        indices = pickle.load(f)
    
    # Load clustering models
    with open(models_dir / 'kmeans.pkl', 'rb') as f:
        kmeans_model = pickle.load(f)
    with open(models_dir / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(models_dir / 'clustering_labels.pkl', 'rb') as f:
        clustering_labels = pickle.load(f)
    with open(models_dir / 'clustering_feature_names.pkl', 'rb') as f:
        clustering_feature_names = pickle.load(f)
    
    # Load comparison data
    model_comparison = pd.read_csv(reports_dir / 'model_comparison.csv')
    cluster_comparison = pd.read_csv(reports_dir / 'cluster_comparison.csv')
    cluster_profiles = pd.read_csv(reports_dir / 'cluster_profiles.csv')
    
    # Prepare test data - need to recreate features as models expect
    # Recreate one-hot encoded features
    df_processed = df.copy()
    
    # One-hot encode Title
    if 'Title' in df_processed.columns:
        title_dummies = pd.get_dummies(df_processed['Title'], prefix='Title')
        df_processed = pd.concat([df_processed, title_dummies], axis=1)
    
    # One-hot encode AgeGroup
    if 'AgeGroup' in df_processed.columns:
        agegroup_dummies = pd.get_dummies(df_processed['AgeGroup'], prefix='AgeGroup')
        df_processed = pd.concat([df_processed, agegroup_dummies], axis=1)
    
    # One-hot encode CabinClass
    if 'CabinClass' in df_processed.columns:
        cabin_dummies = pd.get_dummies(df_processed['CabinClass'], prefix='Cabin')
        df_processed = pd.concat([df_processed, cabin_dummies], axis=1)
    
    # Select only available features
    available_features = [f for f in feature_names if f in df_processed.columns]
    X_test = df_processed.loc[indices['test_indices']][available_features]
    y_test = df.loc[indices['test_indices']]['Survived']
    
    # Prepare survivor data for clustering
    survivors = df[df['Survived'] == 1]
    X_cluster = survivors[clustering_feature_names].fillna(survivors[clustering_feature_names].median())
    X_cluster_scaled = scaler.transform(X_cluster)
    
    return {
        'df': df,
        'survivors': survivors,
        'models': {
            'Decision Tree': dt_model,
            'Random Forest': rf_model,
            'Gradient Boosting': gb_model
        },
        'feature_names': feature_names,
        'X_test': X_test,
        'y_test': y_test,
        'kmeans_model': kmeans_model,
        'X_cluster_scaled': X_cluster_scaled,
        'clustering_labels': clustering_labels,
        'clustering_feature_names': clustering_feature_names,
        'model_comparison': model_comparison,
        'cluster_comparison': cluster_comparison,
        'cluster_profiles': cluster_profiles
    }

def plot_model_accuracy_comparison(comparison_df: pd.DataFrame, output_path: str):
    """Plot model accuracy comparison bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    comparison_df.plot(x='Model', y='Accuracy', kind='bar', ax=ax, color='steelblue')
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.legend().remove()
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_model_roc_curves(models: dict, X_test: pd.DataFrame, y_test: pd.Series, 
                          output_path: str):
    """Plot ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = np.trapz(tpr, fpr)
        ax.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_confusion_matrices(models: dict, X_test: pd.DataFrame, y_test: pd.Series,
                           output_path: str):
    """Plot confusion matrices for all models."""
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   cbar_kws={'label': 'Count'})
        axes[idx].set_xlabel('Predicted', fontsize=11)
        axes[idx].set_ylabel('Actual', fontsize=11)
        axes[idx].set_title(f'{name}', fontsize=12, fontweight='bold')
        axes[idx].set_xticklabels(['Not Survived', 'Survived'])
        axes[idx].set_yticklabels(['Not Survived', 'Survived'])
    
    plt.suptitle('Confusion Matrices Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_decision_tree_structure(tree_model, feature_names: list, output_path: str):
    """Plot decision tree structure."""
    from sklearn.tree import plot_tree
    
    fig, ax = plt.subplots(figsize=(20, 12))
    plot_tree(tree_model, feature_names=feature_names, filled=True, 
             class_names=['Not Survived', 'Survived'], ax=ax, fontsize=8)
    ax.set_title('Decision Tree Structure', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_feature_importance_comparison(models: dict, feature_names: list, 
                                      output_path: str):
    """Plot feature importance comparison across models."""
    fig, axes = plt.subplots(1, len(models), figsize=(6*len(models), 8))
    
    if len(models) == 1:
        axes = [axes]
    
    for idx, (name, model) in enumerate(models.items()):
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'tree_'):
            importances = model.tree_.compute_feature_importances(normalize=True)
        else:
            continue
        
        # Sort by importance
        indices = np.argsort(importances)[::-1]
        top_n = min(15, len(importances))
        
        axes[idx].barh(range(top_n), importances[indices][:top_n], color='steelblue')
        axes[idx].set_yticks(range(top_n))
        axes[idx].set_yticklabels([feature_names[i] for i in indices[:top_n]], fontsize=9)
        axes[idx].set_xlabel('Importance', fontsize=11)
        axes[idx].set_title(f'{name}\nFeature Importance', fontsize=12, fontweight='bold')
        axes[idx].invert_yaxis()
        axes[idx].grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Feature Importance Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_survivor_vs_nonsurvivor_gender(df: pd.DataFrame, output_path: str):
    """Plot gender distribution comparison."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    comparison = pd.crosstab(df['Survived'], df['Sex'])
    comparison.plot(kind='bar', ax=ax, color=['coral', 'steelblue'])
    ax.set_xlabel('Survival Status', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Gender Distribution: Survivors vs Non-Survivors', fontsize=14, fontweight='bold')
    ax.set_xticklabels(['Not Survived', 'Survived'], rotation=0)
    ax.legend(title='Gender', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_survivor_vs_nonsurvivor_class(df: pd.DataFrame, output_path: str):
    """Plot class distribution comparison."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    comparison = pd.crosstab(df['Survived'], df['Pclass'])
    comparison.plot(kind='bar', ax=ax, color=['coral', 'steelblue', 'lightgreen'])
    ax.set_xlabel('Survival Status', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Class Distribution: Survivors vs Non-Survivors', fontsize=14, fontweight='bold')
    ax.set_xticklabels(['Not Survived', 'Survived'], rotation=0)
    ax.legend(title='Class', labels=['1st', '2nd', '3rd'], fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_survivor_vs_nonsurvivor_age(df: pd.DataFrame, output_path: str):
    """Plot age distribution comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    survivors_age = df[df['Survived'] == 1]['Age']
    nonsurvivors_age = df[df['Survived'] == 0]['Age']
    
    # Histogram
    axes[0].hist([nonsurvivors_age, survivors_age], bins=20, 
                label=['Not Survived', 'Survived'], alpha=0.7, color=['coral', 'steelblue'])
    axes[0].set_xlabel('Age', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Age Distribution', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    data_to_plot = [nonsurvivors_age.dropna(), survivors_age.dropna()]
    bp = axes[1].boxplot(data_to_plot, labels=['Not Survived', 'Survived'], 
                         patch_artist=True)
    bp['boxes'][0].set_facecolor('coral')
    bp['boxes'][1].set_facecolor('steelblue')
    axes[1].set_ylabel('Age', fontsize=12)
    axes[1].set_title('Age Distribution (Box Plot)', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Age Distribution: Survivors vs Non-Survivors', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_survivor_vs_nonsurvivor_fare(df: pd.DataFrame, output_path: str):
    """Plot fare distribution comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    survivors_fare = df[df['Survived'] == 1]['Fare']
    nonsurvivors_fare = df[df['Survived'] == 0]['Fare']
    
    data_to_plot = [nonsurvivors_fare.dropna(), survivors_fare.dropna()]
    bp = ax.boxplot(data_to_plot, labels=['Not Survived', 'Survived'], patch_artist=True)
    bp['boxes'][0].set_facecolor('coral')
    bp['boxes'][1].set_facecolor('steelblue')
    ax.set_ylabel('Fare', fontsize=12)
    ax.set_xlabel('Survival Status', fontsize=12)
    ax.set_title('Fare Distribution: Survivors vs Non-Survivors', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_feature_distributions(df: pd.DataFrame, output_path: str):
    """Plot feature distributions grouped by survival status."""
    features = ['Age', 'Fare', 'FamilySize', 'SibSp', 'Parch']
    available_features = [f for f in features if f in df.columns]
    
    n_features = len(available_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, feature in enumerate(available_features):
        survivors_data = df[df['Survived'] == 1][feature].dropna()
        nonsurvivors_data = df[df['Survived'] == 0][feature].dropna()
        
        axes[idx].hist([nonsurvivors_data, survivors_data], bins=20,
                      label=['Not Survived', 'Survived'], alpha=0.7, 
                      color=['coral', 'steelblue'])
        axes[idx].set_xlabel(feature, fontsize=11)
        axes[idx].set_ylabel('Frequency', fontsize=11)
        axes[idx].set_title(f'{feature} Distribution', fontsize=12, fontweight='bold')
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Feature Distributions by Survival Status', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_correlation_heatmap(df: pd.DataFrame, output_path: str):
    """Plot correlation heatmap."""
    # Select numerical features
    numerical_features = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 
                        'FamilySize', 'Sex_encoded']
    if 'Fare' in df.columns:
        numerical_features.append('Fare')
    
    available_features = [f for f in numerical_features if f in df.columns]
    corr_matrix = df[available_features].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
               center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_survival_rate_by_features(df: pd.DataFrame, output_path: str):
    """Plot survival rate by different features."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # By Sex
    if 'Sex' in df.columns:
        survival_by_sex = df.groupby('Sex')['Survived'].mean()
        axes[0].bar(survival_by_sex.index, survival_by_sex.values, color=['steelblue', 'coral'])
        axes[0].set_ylabel('Survival Rate', fontsize=11)
        axes[0].set_title('Survival Rate by Gender', fontsize=12, fontweight='bold')
        axes[0].set_ylim([0, 1])
        axes[0].grid(True, alpha=0.3, axis='y')
    
    # By Pclass
    if 'Pclass' in df.columns:
        survival_by_class = df.groupby('Pclass')['Survived'].mean()
        axes[1].bar(survival_by_class.index, survival_by_class.values, color='steelblue')
        axes[1].set_xlabel('Class', fontsize=11)
        axes[1].set_ylabel('Survival Rate', fontsize=11)
        axes[1].set_title('Survival Rate by Class', fontsize=12, fontweight='bold')
        axes[1].set_ylim([0, 1])
        axes[1].grid(True, alpha=0.3, axis='y')
    
    # By Embarked
    if 'Embarked' in df.columns:
        survival_by_embarked = df.groupby('Embarked')['Survived'].mean()
        axes[2].bar(survival_by_embarked.index, survival_by_embarked.values, 
                   color=['steelblue', 'coral', 'lightgreen'])
        axes[2].set_xlabel('Embarked', fontsize=11)
        axes[2].set_ylabel('Survival Rate', fontsize=11)
        axes[2].set_title('Survival Rate by Embarked', fontsize=12, fontweight='bold')
        axes[2].set_ylim([0, 1])
        axes[2].grid(True, alpha=0.3, axis='y')
    
    # By IsAlone
    if 'IsAlone' in df.columns:
        survival_by_alone = df.groupby('IsAlone')['Survived'].mean()
        axes[3].bar(['With Family', 'Alone'], survival_by_alone.values, 
                   color=['steelblue', 'coral'])
        axes[3].set_ylabel('Survival Rate', fontsize=11)
        axes[3].set_title('Survival Rate by Traveling Alone', fontsize=12, fontweight='bold')
        axes[3].set_ylim([0, 1])
        axes[3].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Survival Rate by Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_clustering_comparison(comparison_df: pd.DataFrame, output_path: str):
    """Plot clustering methods comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['Silhouette_Score', 'Calinski_Harabasz', 'Davies_Bouldin']
    metric_labels = ['Silhouette Score', 'Calinski-Harabasz Index', 'Davies-Bouldin Index']
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        comparison_df.plot(x='Method', y=metric, kind='bar', ax=axes[idx], 
                          color='steelblue')
        axes[idx].set_ylabel(label, fontsize=11)
        axes[idx].set_xlabel('Method', fontsize=11)
        axes[idx].set_title(label, fontsize=12, fontweight='bold')
        axes[idx].legend().remove()
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Clustering Methods Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_elbow_curve(kmeans_model, X_scaled: np.array, output_path: str):
    """Plot elbow curve for KMeans."""
    inertias = []
    k_range = range(2, 8)
    
    for k in k_range:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Inertia', fontsize=12)
    ax.set_title('Elbow Curve for KMeans Clustering', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_cluster_scatter(X_scaled: np.array, labels: np.array, 
                        feature_names: list, output_path: str):
    """Plot cluster scatter plot using PCA."""
    # Reduce to 2D using PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                  c=[color], label=f'Cluster {label}', alpha=0.6, s=50)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
    ax.set_title('Cluster Visualization (PCA)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_cluster_radar_charts(profiles_df: pd.DataFrame, feature_names: list,
                             output_path: str):
    """Plot radar charts for cluster profiles."""
    from math import pi
    
    # Select features for radar chart
    radar_features = ['Age_mean', 'Fare_mean', 'Pclass_mode', 'FamilySize_mean']
    available_features = [f for f in radar_features if f in profiles_df.columns]
    
    n_clusters = len(profiles_df)
    n_cols = min(3, n_clusters)
    n_rows = (n_clusters + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows),
                            subplot_kw=dict(projection='polar'))
    
    if n_clusters == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Normalize features for radar chart
    for idx, row in profiles_df.iterrows():
        cluster_id = int(row['Cluster'])
        values = []
        labels = []
        
        for feature in available_features:
            if pd.notna(row[feature]):
                values.append(row[feature])
                labels.append(feature.replace('_mean', '').replace('_mode', ''))
        
        if len(values) == 0:
            continue
        
        # Normalize to 0-1 scale
        values = np.array(values)
        if values.max() > values.min():
            values = (values - values.min()) / (values.max() - values.min())
        
        # Complete the circle
        angles = [n / len(values) * 2 * pi for n in range(len(values))]
        angles += angles[:1]
        values = list(values) + [values[0]]
        
        axes[idx].plot(angles, values, 'o-', linewidth=2, label=f'Cluster {cluster_id}')
        axes[idx].fill(angles, values, alpha=0.25)
        axes[idx].set_xticks(angles[:-1])
        axes[idx].set_xticklabels(labels, fontsize=9)
        axes[idx].set_ylim([0, 1])
        axes[idx].set_title(f'Cluster {cluster_id} Profile', fontsize=12, fontweight='bold', pad=20)
        axes[idx].grid(True)
    
    # Hide unused subplots
    for idx in range(n_clusters, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Cluster Profiles (Radar Charts)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_dendrogram(linkage_matrix: np.array, output_path: str):
    """Plot hierarchical clustering dendrogram."""
    fig, ax = plt.subplots(figsize=(12, 8))
    dendrogram(linkage_matrix, ax=ax, leaf_rotation=90, leaf_font_size=8)
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Distance', fontsize=12)
    ax.set_title('Hierarchical Clustering Dendrogram', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_cluster_size_distribution(profiles_df: pd.DataFrame, output_path: str):
    """Plot cluster size distribution pie chart."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.pie(profiles_df['Size'], labels=[f'Cluster {int(c)}' for c in profiles_df['Cluster']],
          autopct='%1.1f%%', startangle=90, colors=plt.cm.tab10.colors[:len(profiles_df)])
    ax.set_title('Cluster Size Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_model_metrics_heatmap(comparison_df: pd.DataFrame, output_path: str):
    """Plot model metrics heatmap."""
    metrics = ['Accuracy', 'AUC', 'F1_Class0', 'F1_Class1', 
              'Precision_Class1', 'Recall_Class1']
    available_metrics = [m for m in metrics if m in comparison_df.columns]
    
    heatmap_data = comparison_df.set_index('Model')[available_metrics]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', 
               ax=ax, cbar_kws={'label': 'Score'})
    ax.set_title('Model Performance Metrics Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def generate_all_visualizations():
    """Generate all visualizations."""
    print("="*50)
    print("GENERATING ALL VISUALIZATIONS")
    print("="*50)
    
    # Load data
    data = load_data_and_models()
    
    # Create output directory
    figures_dir = Path(__file__).parent.parent / 'reports' / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    # Model performance visualizations
    print("\n--- Model Performance Visualizations ---")
    plot_model_accuracy_comparison(data['model_comparison'], 
                                  str(figures_dir / 'model_accuracy_comparison.png'))
    plot_model_roc_curves(data['models'], data['X_test'], data['y_test'],
                          str(figures_dir / 'model_roc_curves.png'))
    plot_confusion_matrices(data['models'], data['X_test'], data['y_test'],
                           str(figures_dir / 'confusion_matrices.png'))
    plot_model_metrics_heatmap(data['model_comparison'],
                              str(figures_dir / 'model_metrics_heatmap.png'))
    
    # Decision tree visualization
    print("\n--- Decision Tree Visualizations ---")
    plot_decision_tree_structure(data['models']['Decision Tree'], 
                                 data['feature_names'],
                                 str(figures_dir / 'decision_tree_structure.png'))
    plot_feature_importance_comparison(data['models'], data['feature_names'],
                                       str(figures_dir / 'feature_importance_comparison.png'))
    
    # EDA visualizations
    print("\n--- Exploratory Data Analysis Visualizations ---")
    plot_survivor_vs_nonsurvivor_gender(data['df'],
                                       str(figures_dir / 'survivor_vs_nonsurvivor_gender.png'))
    plot_survivor_vs_nonsurvivor_class(data['df'],
                                      str(figures_dir / 'survivor_vs_nonsurvivor_class.png'))
    plot_survivor_vs_nonsurvivor_age(data['df'],
                                    str(figures_dir / 'survivor_vs_nonsurvivor_age.png'))
    plot_survivor_vs_nonsurvivor_fare(data['df'],
                                      str(figures_dir / 'survivor_vs_nonsurvivor_fare.png'))
    plot_feature_distributions(data['df'],
                              str(figures_dir / 'feature_distributions.png'))
    plot_correlation_heatmap(data['df'],
                            str(figures_dir / 'correlation_heatmap.png'))
    plot_survival_rate_by_features(data['df'],
                                  str(figures_dir / 'survival_rate_by_features.png'))
    
    # Clustering visualizations
    print("\n--- Clustering Visualizations ---")
    plot_clustering_comparison(data['cluster_comparison'],
                              str(figures_dir / 'clustering_comparison.png'))
    plot_elbow_curve(data['kmeans_model'], data['X_cluster_scaled'],
                    str(figures_dir / 'elbow_curve.png'))
    plot_cluster_scatter(data['X_cluster_scaled'], 
                        data['clustering_labels']['kmeans'],
                        data['clustering_feature_names'],
                        str(figures_dir / 'cluster_scatter_2d.png'))
    plot_cluster_radar_charts(data['cluster_profiles'],
                             data['clustering_feature_names'],
                             str(figures_dir / 'cluster_radar_charts.png'))
    plot_cluster_size_distribution(data['cluster_profiles'],
                                  str(figures_dir / 'cluster_size_distribution.png'))
    
    # Load and plot dendrogram if available
    try:
        models_dir = Path(__file__).parent.parent / 'models'
        with open(models_dir / 'linkage_matrix.pkl', 'rb') as f:
            linkage_matrix = pickle.load(f)
        plot_dendrogram(linkage_matrix,
                       str(figures_dir / 'hierarchical_dendrogram.png'))
    except:
        print("Warning: Could not load linkage matrix for dendrogram")
    
    print("\n" + "="*50)
    print("ALL VISUALIZATIONS GENERATED")
    print("="*50)
    print(f"Total figures saved to: {figures_dir}")

if __name__ == '__main__':
    generate_all_visualizations()

