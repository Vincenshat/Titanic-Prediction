"""
Classification Models for Titanic Survival Prediction

This module implements multiple classification models:
- Decision Tree (main model for interpretability)
- Random Forest (ensemble method)
- Gradient Boosting (advanced ensemble)

Note: Fare feature is excluded from training as per requirements.
All code and comments are in English.
"""

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import time
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

def prepare_features(df: pd.DataFrame, exclude_fare: bool = True) -> tuple:
    """
    Prepare features and target for modeling.
    
    Args:
        df: DataFrame with cleaned data
        exclude_fare: Whether to exclude Fare and FareBin features
        
    Returns:
        Tuple of (X, y) where X is features and y is target
    """
    # Target variable
    y = df['Survived']
    
    # Select features - exclude Fare and FareBin as per requirements
    # Use Sex_encoded and Embarked one-hot encoding
    # Exclude Cabin features (missing means "not recorded", not "no cabin")
    feature_cols = [
        'Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch',
        'Embarked_C', 'Embarked_Q', 'Embarked_S',  # One-hot encoded
        'FamilySize', 'IsAlone'
    ]
    
    X = df[feature_cols].copy()
    
    # Add AgeGroup if available (one-hot encoded for now, but could be label encoded)
    if 'AgeGroup' in df.columns:
        agegroup_dummies = pd.get_dummies(df['AgeGroup'], prefix='AgeGroup')
        X = pd.concat([X, agegroup_dummies], axis=1)
    
    # Explicitly exclude Fare and FareBin
    if exclude_fare:
        X = X[[col for col in X.columns if 'Fare' not in col]]
    
    print(f"\nSelected {X.shape[1]} features for modeling")
    print(f"Features: {list(X.columns)}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, 
               random_state: int = 42) -> tuple:
    """
    Split data into training and testing sets.
    
    Args:
        X: Features
        y: Target
        test_size: Proportion of test set
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, 
        stratify=y
    )
    
    print(f"\nData split:")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    
    # Save train/test indices
    train_indices = X_train.index.tolist()
    test_indices = X_test.index.tolist()
    
    return X_train, X_test, y_train, y_test, train_indices, test_indices

def train_decision_tree(X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
    """
    Train Decision Tree classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        Trained DecisionTreeClassifier
    """
    print("\n" + "="*50)
    print("Training Decision Tree...")
    print("="*50)
    
    model = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=5,
        min_samples_leaf=20,
        min_samples_split=40,
        class_weight='balanced',
        random_state=42
    )
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.3f} seconds")
    print(f"Tree depth: {model.tree_.max_depth}")
    print(f"Number of leaves: {model.tree_.n_leaves}")
    
    return model

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Train Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        Trained RandomForestClassifier
    """
    print("\n" + "="*50)
    print("Training Random Forest...")
    print("="*50)
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=7,
        min_samples_leaf=10,
        min_samples_split=20,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.3f} seconds")
    print(f"Number of trees: {len(model.estimators_)}")
    
    return model

def train_gradient_boosting(X_train: pd.DataFrame, y_train: pd.Series) -> GradientBoostingClassifier:
    """
    Train Gradient Boosting classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        Trained GradientBoostingClassifier
    """
    print("\n" + "="*50)
    print("Training Gradient Boosting...")
    print("="*50)
    
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        min_samples_leaf=15,
        min_samples_split=30,
        subsample=0.8,
        random_state=42
    )
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.3f} seconds")
    print(f"Number of estimators: {len(model.estimators_)}")
    
    return model

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, 
                   model_name: str = "Model") -> dict:
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        model_name: Name of the model
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    f1 = f1_score(y_test, y_pred, average=None)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision_0': precision[0],
        'precision_1': precision[1],
        'recall_0': recall[0],
        'recall_1': recall[1],
        'f1_0': f1[0],
        'f1_1': f1[1],
        'auc': auc,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    print(f"\n{model_name} Performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision (Class 0): {precision[0]:.4f}")
    print(f"  Precision (Class 1): {precision[1]:.4f}")
    print(f"  Recall (Class 0): {recall[0]:.4f}")
    print(f"  Recall (Class 1): {recall[1]:.4f}")
    print(f"  F1-score (Class 0): {f1[0]:.4f}")
    print(f"  F1-score (Class 1): {f1[1]:.4f}")
    print(f"  ROC-AUC: {auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)
    
    return metrics

def extract_decision_rules(tree_model: DecisionTreeClassifier, 
                           feature_names: list) -> list:
    """
    Extract decision rules from decision tree.
    
    Args:
        tree_model: Trained DecisionTreeClassifier
        feature_names: List of feature names
        
    Returns:
        List of decision rules as strings
    """
    from sklearn.tree import _tree
    
    tree = tree_model.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree.feature
    ]
    
    rules = []
    
    def recurse(node, depth, parent_rule=""):
        indent = "  " * depth
        if tree.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree.threshold[node]
            rule = f"{name} <= {threshold:.2f}"
            
            if parent_rule:
                rule = f"{parent_rule} AND {rule}"
            
            recurse(tree.children_left[node], depth + 1, rule)
            recurse(tree.children_right[node], depth + 1, 
                   parent_rule.replace(f" <= {threshold:.2f}", f" > {threshold:.2f}") 
                   if parent_rule else f"{name} > {threshold:.2f}")
        else:
            # Leaf node
            samples = tree.n_node_samples[node]
            value = tree.value[node][0]
            class_pred = np.argmax(value)
            prob = value[class_pred] / samples
            
            rule_str = f"{parent_rule} → Class {class_pred} (prob={prob:.3f}, samples={samples})"
            rules.append(rule_str)
    
    recurse(0, 0)
    return rules

def get_decision_path(tree_model: DecisionTreeClassifier, 
                      sample: np.array, feature_names: list) -> str:
    """
    Get decision path for a single sample.
    
    Args:
        tree_model: Trained DecisionTreeClassifier
        sample: Single sample as array
        feature_names: List of feature names
        
    Returns:
        Decision path as string
    """
    path = tree_model.decision_path(sample.reshape(1, -1))
    node_indicator = path.toarray()[0]
    
    leaf_id = tree_model.apply(sample.reshape(1, -1))[0]
    
    path_str = "Decision Path:\n"
    for node_id in np.where(node_indicator == 1)[0]:
        if leaf_id == node_id:
            continue
        if tree_model.tree_.children_left[node_id] != tree_model.tree_.children_right[node_id]:
            feature_idx = tree_model.tree_.feature[node_id]
            threshold = tree_model.tree_.threshold[node_id]
            feature_name = feature_names[feature_idx]
            value = sample[feature_idx]
            
            if value <= threshold:
                path_str += f"  {feature_name} = {value:.2f} <= {threshold:.2f} ✓\n"
            else:
                path_str += f"  {feature_name} = {value:.2f} > {threshold:.2f} ✓\n"
    
    return path_str

def calculate_fairness_metrics(model, X_test: pd.DataFrame, y_test: pd.Series,
                              df_test: pd.DataFrame) -> dict:
    """
    Calculate fairness metrics for different groups.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        df_test: Original test dataframe with demographic info
        
    Returns:
        Dictionary with fairness metrics
    """
    y_pred = model.predict(X_test)
    
    fairness = {}
    
    # By gender
    if 'Sex' in df_test.columns:
        male_mask = df_test['Sex'] == 'male'
        female_mask = df_test['Sex'] == 'female'
        
        male_fpr = np.sum((y_pred[male_mask] == 1) & (y_test[male_mask] == 0)) / np.sum(y_test[male_mask] == 0)
        male_fnr = np.sum((y_pred[male_mask] == 0) & (y_test[male_mask] == 1)) / np.sum(y_test[male_mask] == 1)
        female_fpr = np.sum((y_pred[female_mask] == 1) & (y_test[female_mask] == 0)) / np.sum(y_test[female_mask] == 0)
        female_fnr = np.sum((y_pred[female_mask] == 0) & (y_test[female_mask] == 1)) / np.sum(y_test[female_mask] == 1)
        
        fairness['gender'] = {
            'male_fpr': male_fpr,
            'male_fnr': male_fnr,
            'female_fpr': female_fpr,
            'female_fnr': female_fnr
        }
    
    # By class
    if 'Pclass' in df_test.columns:
        for pclass in [1, 2, 3]:
            class_mask = df_test['Pclass'] == pclass
            if class_mask.sum() > 0:
                accuracy = accuracy_score(y_test[class_mask], y_pred[class_mask])
                fairness[f'class_{pclass}_accuracy'] = accuracy
    
    return fairness

def compare_models(models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Compare performance of multiple models.
    
    Args:
        models: Dictionary of model_name: model pairs
        X_test: Test features
        y_test: Test target
        
    Returns:
        DataFrame with comparison metrics
    """
    comparison = []
    
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, name)
        comparison.append({
            'Model': name,
            'Accuracy': metrics['accuracy'],
            'AUC': metrics['auc'],
            'F1_Class0': metrics['f1_0'],
            'F1_Class1': metrics['f1_1'],
            'Precision_Class1': metrics['precision_1'],
            'Recall_Class1': metrics['recall_1']
        })
    
    comparison_df = pd.DataFrame(comparison)
    return comparison_df

def save_models(models: dict, output_dir: str, train_indices: list, 
                test_indices: list) -> None:
    """
    Save trained models and indices.
    
    Args:
        models: Dictionary of model_name: model pairs
        output_dir: Directory to save models
        train_indices: Training set indices
        test_indices: Test set indices
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for name, model in models.items():
        filename = f"{name.lower().replace(' ', '_')}.pkl"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved {name} to {filepath}")
    
    # Save indices
    indices = {
        'train_indices': train_indices,
        'test_indices': test_indices
    }
    indices_path = os.path.join(output_dir, 'train_test_indices.pkl')
    with open(indices_path, 'wb') as f:
        pickle.dump(indices, f)
    print(f"Saved train/test indices to {indices_path}")

def main():
    """Main function to run model training pipeline."""
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
    
    # Prepare features (excluding Fare)
    X, y = prepare_features(df, exclude_fare=True)
    
    # Split data
    X_train, X_test, y_train, y_test, train_indices, test_indices = split_data(
        X, y, test_size=0.2, random_state=42
    )
    
    # Get original test dataframe for fairness analysis
    df_test = df.loc[test_indices].copy()
    
    # Train models
    models = {}
    model_metrics = {}
    
    # Decision Tree
    dt_model = train_decision_tree(X_train, y_train)
    models['Decision Tree'] = dt_model
    model_metrics['Decision Tree'] = evaluate_model(dt_model, X_test, y_test, "Decision Tree")
    
    # Random Forest
    rf_model = train_random_forest(X_train, y_train)
    models['Random Forest'] = rf_model
    model_metrics['Random Forest'] = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    
    # Gradient Boosting
    gb_model = train_gradient_boosting(X_train, y_train)
    models['Gradient Boosting'] = gb_model
    model_metrics['Gradient Boosting'] = evaluate_model(gb_model, X_test, y_test, "Gradient Boosting")
    
    # Compare models
    comparison_df = compare_models(models, X_test, y_test)
    comparison_file = reports_dir / 'model_comparison.csv'
    comparison_df.to_csv(comparison_file, index=False)
    print(f"\nModel comparison saved to {comparison_file}")
    print("\nModel Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Extract decision rules from Decision Tree
    print("\n" + "="*50)
    print("DECISION TREE RULES")
    print("="*50)
    rules = extract_decision_rules(dt_model, list(X.columns))
    print(f"\nExtracted {len(rules)} decision rules")
    print("\nSample rules (first 10):")
    for i, rule in enumerate(rules[:10]):
        print(f"{i+1}. {rule}")
    
    # Calculate fairness metrics for Decision Tree
    print("\n" + "="*50)
    print("FAIRNESS ANALYSIS (Decision Tree)")
    print("="*50)
    fairness = calculate_fairness_metrics(dt_model, X_test, y_test, df_test)
    print("\nFairness Metrics:")
    for key, value in fairness.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v:.4f}")
        else:
            print(f"{key}: {value:.4f}")
    
    # Save models
    save_models(models, str(models_dir), train_indices, test_indices)
    
    # Save feature names for later use
    feature_names_file = models_dir / 'feature_names.pkl'
    with open(feature_names_file, 'wb') as f:
        pickle.dump(list(X.columns), f)
    print(f"\nSaved feature names to {feature_names_file}")
    
    print("\n" + "="*50)
    print("MODEL TRAINING COMPLETED")
    print("="*50)

if __name__ == '__main__':
    main()

