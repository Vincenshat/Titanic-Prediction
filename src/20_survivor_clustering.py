"""
Survivor Clustering Analysis Module

This module performs statistical analysis and clustering on survivors (Survived=1).
It implements multiple clustering methods:
- KMeans (main method)
- Gaussian Mixture Model (GMM)
- Hierarchical Clustering

All code and comments are in English.
"""

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)
from scipy.stats import chi2_contingency, mannwhitneyu
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def load_cleaned_data(filepath: str) -> pd.DataFrame:
    """
    Load cleaned dataset.
    
    Args:
        filepath: Path to cleaned CSV file
        
    Returns:
        DataFrame with cleaned data
    """
    print(f"Loading cleaned data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def filter_survivors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter survivors (Survived=1).
    
    Args:
        df: DataFrame with all data
        
    Returns:
        DataFrame with only survivors
    """
    survivors = df[df['Survived'] == 1].copy()
    print(f"\nFiltered survivors: {len(survivors)} out of {len(df)} ({len(survivors)/len(df)*100:.1f}%)")
    return survivors

def survivor_statistical_analysis(df: pd.DataFrame, survivors: pd.DataFrame) -> dict:
    """
    Perform statistical analysis comparing survivors and non-survivors.
    
    Args:
        df: Full dataset
        survivors: Survivor subset
        
    Returns:
        Dictionary with statistical results
    """
    non_survivors = df[df['Survived'] == 0]
    
    stats = {}
    
    # Basic statistics
    stats['total_survivors'] = len(survivors)
    stats['survival_rate'] = len(survivors) / len(df)
    
    # Gender distribution
    if 'Sex' in survivors.columns:
        stats['gender_dist'] = {
            'survivors': survivors['Sex'].value_counts().to_dict(),
            'non_survivors': non_survivors['Sex'].value_counts().to_dict()
        }
        stats['gender_survival_rate'] = {
            'female': len(survivors[survivors['Sex'] == 'female']) / len(df[df['Sex'] == 'female']),
            'male': len(survivors[survivors['Sex'] == 'male']) / len(df[df['Sex'] == 'male'])
        }
    
    # Age statistics
    if 'Age' in survivors.columns:
        stats['age'] = {
            'survivors_mean': survivors['Age'].mean(),
            'survivors_median': survivors['Age'].median(),
            'non_survivors_mean': non_survivors['Age'].mean(),
            'non_survivors_median': non_survivors['Age'].median()
        }
    
    # Class distribution
    if 'Pclass' in survivors.columns:
        stats['class_dist'] = {
            'survivors': survivors['Pclass'].value_counts().to_dict(),
            'non_survivors': non_survivors['Pclass'].value_counts().to_dict()
        }
        stats['class_survival_rate'] = {}
        for pclass in [1, 2, 3]:
            class_total = len(df[df['Pclass'] == pclass])
            class_survived = len(survivors[survivors['Pclass'] == pclass])
            if class_total > 0:
                stats['class_survival_rate'][pclass] = class_survived / class_total
    
    # Fare statistics (for analysis, not training)
    if 'Fare' in survivors.columns:
        stats['fare'] = {
            'survivors_mean': survivors['Fare'].mean(),
            'survivors_median': survivors['Fare'].median(),
            'non_survivors_mean': non_survivors['Fare'].mean(),
            'non_survivors_median': non_survivors['Fare'].median()
        }
    
    # Family size
    if 'FamilySize' in survivors.columns:
        stats['family_size'] = {
            'survivors_mean': survivors['FamilySize'].mean(),
            'survivors_median': survivors['FamilySize'].median(),
            'non_survivors_mean': non_survivors['FamilySize'].mean(),
            'non_survivors_median': non_survivors['FamilySize'].median()
        }
        stats['alone_rate'] = {
            'survivors': (survivors['IsAlone'] == 1).sum() / len(survivors),
            'non_survivors': (non_survivors['IsAlone'] == 1).sum() / len(non_survivors)
        }
    
    # Embarked distribution
    if 'Embarked' in survivors.columns:
        stats['embarked_dist'] = {
            'survivors': survivors['Embarked'].value_counts().to_dict(),
            'non_survivors': non_survivors['Embarked'].value_counts().to_dict()
        }
    
    return stats

def statistical_significance_tests(df: pd.DataFrame, survivors: pd.DataFrame) -> dict:
    """
    Perform statistical significance tests.
    
    Args:
        df: Full dataset
        survivors: Survivor subset
        
    Returns:
        Dictionary with test results
    """
    non_survivors = df[df['Survived'] == 0]
    test_results = {}
    
    # Chi-square test for categorical variables
    if 'Sex' in df.columns:
        contingency = pd.crosstab(df['Survived'], df['Sex'])
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        test_results['sex_chi2'] = {
            'chi2': chi2,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    if 'Pclass' in df.columns:
        contingency = pd.crosstab(df['Survived'], df['Pclass'])
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        test_results['pclass_chi2'] = {
            'chi2': chi2,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    if 'Embarked' in df.columns:
        contingency = pd.crosstab(df['Survived'], df['Embarked'])
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        test_results['embarked_chi2'] = {
            'chi2': chi2,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    # Mann-Whitney U test for continuous variables
    if 'Age' in df.columns:
        u_stat, p_value = mannwhitneyu(
            survivors['Age'].dropna(),
            non_survivors['Age'].dropna(),
            alternative='two-sided'
        )
        test_results['age_mannwhitney'] = {
            'u_stat': u_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    if 'Fare' in df.columns:
        u_stat, p_value = mannwhitneyu(
            survivors['Fare'].dropna(),
            non_survivors['Fare'].dropna(),
            alternative='two-sided'
        )
        test_results['fare_mannwhitney'] = {
            'u_stat': u_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    return test_results

def select_clustering_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select features for clustering.
    
    Args:
        df: DataFrame with survivor data
        
    Returns:
        DataFrame with selected features
    """
    # Select features: Age, Fare, Pclass, FamilySize
    # Note: Fare is used for clustering analysis, not training
    feature_cols = ['Age', 'Fare', 'Pclass', 'FamilySize']
    
    # Check which features are available
    available_features = [col for col in feature_cols if col in df.columns]
    
    X = df[available_features].copy()
    
    # Handle any remaining missing values
    X = X.fillna(X.median())
    
    print(f"\nSelected {len(available_features)} features for clustering: {available_features}")
    print(f"Feature statistics:")
    print(X.describe())
    
    return X

def standardize_features(X: pd.DataFrame) -> tuple:
    """
    Standardize features for clustering.
    
    Args:
        X: Features DataFrame
        
    Returns:
        Tuple of (standardized array, scaler)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"\nFeatures standardized")
    print(f"Mean: {X_scaled.mean(axis=0).round(4)}")
    print(f"Std: {X_scaled.std(axis=0).round(4)}")
    
    return X_scaled, scaler

def determine_optimal_k(X: np.array, k_range: range = range(2, 7)) -> dict:
    """
    Determine optimal number of clusters using silhouette score.
    
    Args:
        X: Standardized feature array
        k_range: Range of k values to test
        
    Returns:
        Dictionary with k values and scores
    """
    results = {}
    
    print("\nDetermining optimal number of clusters...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        sil_score = silhouette_score(X, labels)
        ch_score = calinski_harabasz_score(X, labels)
        db_score = davies_bouldin_score(X, labels)
        inertia = kmeans.inertia_
        
        results[k] = {
            'silhouette': sil_score,
            'calinski_harabasz': ch_score,
            'davies_bouldin': db_score,
            'inertia': inertia
        }
        
        print(f"  k={k}: Silhouette={sil_score:.4f}, CH={ch_score:.2f}, DB={db_score:.4f}, Inertia={inertia:.2f}")
    
    # Find optimal k (highest silhouette score)
    optimal_k = max(results.keys(), key=lambda k: results[k]['silhouette'])
    print(f"\nOptimal k: {optimal_k} (highest silhouette score)")
    
    return results, optimal_k

def train_kmeans(X: np.array, n_clusters: int = 3) -> tuple:
    """
    Train KMeans clustering model.
    
    Args:
        X: Standardized feature array
        n_clusters: Number of clusters
        
    Returns:
        Tuple of (model, labels)
    """
    print(f"\nTraining KMeans with {n_clusters} clusters...")
    
    model = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=10,
        random_state=42
    )
    
    labels = model.fit_predict(X)
    
    # Evaluate
    sil_score = silhouette_score(X, labels)
    ch_score = calinski_harabasz_score(X, labels)
    db_score = davies_bouldin_score(X, labels)
    
    print(f"KMeans Performance:")
    print(f"  Silhouette Score: {sil_score:.4f}")
    print(f"  Calinski-Harabasz Index: {ch_score:.2f}")
    print(f"  Davies-Bouldin Index: {db_score:.4f}")
    print(f"  Inertia: {model.inertia_:.2f}")
    
    return model, labels

def train_gmm(X: np.array, n_components: int = 3) -> tuple:
    """
    Train Gaussian Mixture Model.
    
    Args:
        X: Standardized feature array
        n_components: Number of components
        
    Returns:
        Tuple of (model, labels)
    """
    print(f"\nTraining GMM with {n_components} components...")
    
    model = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        random_state=42
    )
    
    labels = model.fit_predict(X)
    
    # Evaluate
    sil_score = silhouette_score(X, labels)
    ch_score = calinski_harabasz_score(X, labels)
    db_score = davies_bouldin_score(X, labels)
    
    print(f"GMM Performance:")
    print(f"  Silhouette Score: {sil_score:.4f}")
    print(f"  Calinski-Harabasz Index: {ch_score:.2f}")
    print(f"  Davies-Bouldin Index: {db_score:.4f}")
    print(f"  Log Likelihood: {model.score(X):.2f}")
    
    return model, labels

def train_hierarchical(X: np.array, n_clusters: int = 3) -> tuple:
    """
    Train Hierarchical Clustering model.
    
    Args:
        X: Standardized feature array
        n_clusters: Number of clusters
        
    Returns:
        Tuple of (model, labels, linkage_matrix)
    """
    print(f"\nTraining Hierarchical Clustering with {n_clusters} clusters...")
    
    # Compute linkage matrix
    linkage_matrix = linkage(X, method='ward')
    
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='ward'
    )
    
    labels = model.fit_predict(X)
    
    # Evaluate
    sil_score = silhouette_score(X, labels)
    ch_score = calinski_harabasz_score(X, labels)
    db_score = davies_bouldin_score(X, labels)
    
    print(f"Hierarchical Clustering Performance:")
    print(f"  Silhouette Score: {sil_score:.4f}")
    print(f"  Calinski-Harabasz Index: {ch_score:.2f}")
    print(f"  Davies-Bouldin Index: {db_score:.4f}")
    
    return model, labels, linkage_matrix

def generate_cluster_profiles(df: pd.DataFrame, labels: np.array, 
                              feature_names: list) -> pd.DataFrame:
    """
    Generate statistical profiles for each cluster.
    
    Args:
        df: Original survivor DataFrame
        labels: Cluster labels
        feature_names: Names of clustering features
        
    Returns:
        DataFrame with cluster profiles
    """
    df_clustered = df.copy()
    df_clustered['Cluster'] = labels
    
    profiles = []
    
    for cluster_id in np.unique(labels):
        cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
        
        profile = {
            'Cluster': cluster_id,
            'Size': len(cluster_data),
            'Percentage': len(cluster_data) / len(df_clustered) * 100
        }
        
        # Statistics for clustering features
        for feature in feature_names:
            if feature in cluster_data.columns:
                profile[f'{feature}_mean'] = cluster_data[feature].mean()
                profile[f'{feature}_median'] = cluster_data[feature].median()
                profile[f'{feature}_std'] = cluster_data[feature].std()
        
        # Additional statistics
        if 'Age' in cluster_data.columns:
            profile['Age_mean'] = cluster_data['Age'].mean()
            profile['Age_median'] = cluster_data['Age'].median()
        
        if 'Fare' in cluster_data.columns:
            profile['Fare_mean'] = cluster_data['Fare'].mean()
            profile['Fare_median'] = cluster_data['Fare'].median()
        
        if 'Pclass' in cluster_data.columns:
            profile['Pclass_mode'] = cluster_data['Pclass'].mode()[0] if len(cluster_data['Pclass'].mode()) > 0 else np.nan
            profile['Pclass_1_rate'] = (cluster_data['Pclass'] == 1).sum() / len(cluster_data)
            profile['Pclass_2_rate'] = (cluster_data['Pclass'] == 2).sum() / len(cluster_data)
            profile['Pclass_3_rate'] = (cluster_data['Pclass'] == 3).sum() / len(cluster_data)
        
        if 'FamilySize' in cluster_data.columns:
            profile['FamilySize_mean'] = cluster_data['FamilySize'].mean()
            profile['FamilySize_median'] = cluster_data['FamilySize'].median()
        
        if 'Sex' in cluster_data.columns:
            profile['Female_rate'] = (cluster_data['Sex'] == 'female').sum() / len(cluster_data)
            profile['Male_rate'] = (cluster_data['Sex'] == 'male').sum() / len(cluster_data)
        
        if 'IsAlone' in cluster_data.columns:
            profile['Alone_rate'] = (cluster_data['IsAlone'] == 1).sum() / len(cluster_data)
        
        profiles.append(profile)
    
    profiles_df = pd.DataFrame(profiles)
    return profiles_df

def create_cluster_descriptions(profiles_df: pd.DataFrame) -> dict:
    """
    Create human-readable descriptions for each cluster.
    
    Args:
        profiles_df: DataFrame with cluster profiles
        
    Returns:
        Dictionary mapping cluster_id to description
    """
    descriptions = {}
    
    for _, row in profiles_df.iterrows():
        cluster_id = int(row['Cluster'])
        size = int(row['Size'])
        pct = row['Percentage']
        
        # Build description based on characteristics
        desc_parts = []
        
        # Age category
        if 'Age_mean' in row:
            age = row['Age_mean']
            if age < 18:
                age_cat = "young"
            elif age < 35:
                age_cat = "young adult"
            elif age < 50:
                age_cat = "middle-aged"
            else:
                age_cat = "elderly"
            desc_parts.append(age_cat)
        
        # Fare category
        if 'Fare_mean' in row:
            fare = row['Fare_mean']
            if fare > 50:
                fare_cat = "high fare"
            elif fare > 20:
                fare_cat = "medium fare"
            else:
                fare_cat = "low fare"
            desc_parts.append(fare_cat)
        
        # Class
        if 'Pclass_mode' in row and pd.notna(row['Pclass_mode']):
            pclass = int(row['Pclass_mode'])
            if pclass == 1:
                class_cat = "first class"
            elif pclass == 2:
                class_cat = "second class"
            else:
                class_cat = "third class"
            desc_parts.append(class_cat)
        
        # Family
        if 'FamilySize_mean' in row:
            fam_size = row['FamilySize_mean']
            if fam_size > 3:
                fam_cat = "large family"
            elif fam_size > 1:
                fam_cat = "small family"
            else:
                fam_cat = "traveling alone"
            desc_parts.append(fam_cat)
        
        # Gender
        if 'Female_rate' in row:
            if row['Female_rate'] > 0.7:
                gender_cat = "predominantly female"
            elif row['Female_rate'] < 0.3:
                gender_cat = "predominantly male"
            else:
                gender_cat = "mixed gender"
            desc_parts.append(gender_cat)
        
        description = f"Cluster {cluster_id} ({size} survivors, {pct:.1f}%): " + ", ".join(desc_parts)
        descriptions[cluster_id] = description
    
    return descriptions

def compare_clustering_methods(X: np.array, methods: dict) -> pd.DataFrame:
    """
    Compare different clustering methods.
    
    Args:
        X: Standardized feature array
        methods: Dictionary of method_name: (model, labels) pairs
        
    Returns:
        DataFrame with comparison metrics
    """
    comparison = []
    
    for name, (model, labels) in methods.items():
        sil_score = silhouette_score(X, labels)
        ch_score = calinski_harabasz_score(X, labels)
        db_score = davies_bouldin_score(X, labels)
        
        comparison.append({
            'Method': name,
            'Silhouette_Score': sil_score,
            'Calinski_Harabasz': ch_score,
            'Davies_Bouldin': db_score
        })
    
    comparison_df = pd.DataFrame(comparison)
    return comparison_df

def save_clustering_models(models: dict, scaler: StandardScaler, 
                          output_dir: str) -> None:
    """
    Save clustering models and scaler.
    
    Args:
        models: Dictionary of model_name: model pairs
        scaler: StandardScaler used for preprocessing
        output_dir: Directory to save models
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for name, model in models.items():
        filename = f"{name.lower().replace(' ', '_')}.pkl"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved {name} to {filepath}")
    
    # Save scaler
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to {scaler_path}")

def main():
    """Main function to run clustering analysis pipeline."""
    # Set up paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    models_dir = base_dir / 'models'
    reports_dir = base_dir / 'reports'
    
    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)
    
    # Load cleaned data
    data_file = data_dir / 'titanic_cleaned.csv'
    df = load_cleaned_data(str(data_file))
    
    # Filter survivors
    survivors = filter_survivors(df)
    
    # Statistical analysis
    print("\n" + "="*50)
    print("STATISTICAL ANALYSIS")
    print("="*50)
    stats = survivor_statistical_analysis(df, survivors)
    print("\nSurvivor Statistics:")
    print(f"  Total survivors: {stats['total_survivors']}")
    print(f"  Survival rate: {stats['survival_rate']*100:.1f}%")
    if 'gender_survival_rate' in stats:
        print(f"  Female survival rate: {stats['gender_survival_rate']['female']*100:.1f}%")
        print(f"  Male survival rate: {stats['gender_survival_rate']['male']*100:.1f}%")
    if 'class_survival_rate' in stats:
        for pclass, rate in stats['class_survival_rate'].items():
            print(f"  Class {pclass} survival rate: {rate*100:.1f}%")
    
    # Statistical significance tests
    print("\n" + "="*50)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("="*50)
    test_results = statistical_significance_tests(df, survivors)
    print("\nTest Results:")
    for test_name, result in test_results.items():
        if isinstance(result, dict):
            print(f"\n{test_name}:")
            for key, value in result.items():
                if isinstance(value, bool):
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {value:.4f}")
    
    # Save statistical results
    stats_file = reports_dir / 'survivor_statistics.csv'
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(stats_file, index=False)
    
    test_results_file = reports_dir / 'statistical_tests.csv'
    test_results_df = pd.DataFrame([test_results])
    test_results_df.to_csv(test_results_file, index=False)
    
    # Prepare clustering data
    print("\n" + "="*50)
    print("CLUSTERING ANALYSIS")
    print("="*50)
    X = select_clustering_features(survivors)
    X_scaled, scaler = standardize_features(X)
    
    # Determine optimal k
    k_results, optimal_k = determine_optimal_k(X_scaled, k_range=range(2, 7))
    
    # Train clustering models
    clustering_models = {}
    clustering_labels = {}
    
    # KMeans (main method)
    kmeans_model, kmeans_labels = train_kmeans(X_scaled, n_clusters=optimal_k)
    clustering_models['kmeans'] = kmeans_model
    clustering_labels['kmeans'] = kmeans_labels
    
    # GMM
    gmm_model, gmm_labels = train_gmm(X_scaled, n_components=optimal_k)
    clustering_models['gmm'] = gmm_model
    clustering_labels['gmm'] = gmm_labels
    
    # Hierarchical
    hier_model, hier_labels, linkage_matrix = train_hierarchical(X_scaled, n_clusters=optimal_k)
    clustering_models['hierarchical'] = hier_model
    clustering_labels['hierarchical'] = hier_labels
    
    # Save linkage matrix for dendrogram
    linkage_file = models_dir / 'linkage_matrix.pkl'
    with open(linkage_file, 'wb') as f:
        pickle.dump(linkage_matrix, f)
    
    # Compare clustering methods
    methods_dict = {
        'KMeans': (kmeans_model, kmeans_labels),
        'GMM': (gmm_model, gmm_labels),
        'Hierarchical': (hier_model, hier_labels)
    }
    clustering_comparison = compare_clustering_methods(X_scaled, methods_dict)
    comparison_file = reports_dir / 'cluster_comparison.csv'
    clustering_comparison.to_csv(comparison_file, index=False)
    print(f"\nClustering comparison saved to {comparison_file}")
    print("\nClustering Comparison:")
    print(clustering_comparison.to_string(index=False))
    
    # Generate cluster profiles (using KMeans as main method)
    print("\n" + "="*50)
    print("CLUSTER PROFILES (KMeans)")
    print("="*50)
    profiles = generate_cluster_profiles(survivors, kmeans_labels, list(X.columns))
    profiles_file = reports_dir / 'cluster_profiles.csv'
    profiles.to_csv(profiles_file, index=False)
    print(f"\nCluster profiles saved to {profiles_file}")
    print("\nCluster Profiles:")
    print(profiles.to_string(index=False))
    
    # Create cluster descriptions
    descriptions = create_cluster_descriptions(profiles)
    print("\nCluster Descriptions:")
    for cluster_id, desc in descriptions.items():
        print(f"  {desc}")
    
    # Save descriptions
    descriptions_file = reports_dir / 'cluster_descriptions.txt'
    with open(descriptions_file, 'w') as f:
        for cluster_id, desc in descriptions.items():
            f.write(f"{desc}\n")
    
    # Save models
    save_clustering_models(clustering_models, scaler, str(models_dir))
    
    # Save labels and feature names for visualization
    labels_file = models_dir / 'clustering_labels.pkl'
    with open(labels_file, 'wb') as f:
        pickle.dump({
            'kmeans': kmeans_labels,
            'gmm': gmm_labels,
            'hierarchical': hier_labels
        }, f)
    
    feature_names_file = models_dir / 'clustering_feature_names.pkl'
    with open(feature_names_file, 'wb') as f:
        pickle.dump(list(X.columns), f)
    
    print("\n" + "="*50)
    print("CLUSTERING ANALYSIS COMPLETED")
    print("="*50)

if __name__ == '__main__':
    main()

