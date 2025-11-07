"""
Check Decision Tree training and feature selection logic.
Verify that the tree uses information gain (entropy) to select features.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.tree import _tree
import matplotlib.pyplot as plt

print("="*70)
print("CHECKING DECISION TREE TRAINING AND FEATURE SELECTION")
print("="*70)

# Load data
base_dir = Path(__file__).parent
data_dir = base_dir / 'data'
models_dir = base_dir / 'models'

df = pd.read_csv(data_dir / 'titanic_cleaned.csv')

# Load feature names
with open(models_dir / 'feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Recreate features exactly as in training
def prepare_features(df):
    feature_cols = [
        'Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch',
        'Embarked_C', 'Embarked_Q', 'Embarked_S',
        'FamilySize', 'IsAlone'
    ]
    
    if 'Title' in df.columns:
        title_dummies = pd.get_dummies(df['Title'], prefix='Title')
        feature_cols.extend(title_dummies.columns.tolist())
        X = df[feature_cols[:-len(title_dummies.columns)]].copy()
        X = pd.concat([X, title_dummies], axis=1)
    else:
        X = df[feature_cols].copy()
    
    if 'AgeGroup' in df.columns:
        agegroup_dummies = pd.get_dummies(df['AgeGroup'], prefix='AgeGroup')
        X = pd.concat([X, agegroup_dummies], axis=1)
    
    if 'CabinClass' in df.columns:
        cabin_dummies = pd.get_dummies(df['CabinClass'], prefix='Cabin')
        X = pd.concat([X, cabin_dummies], axis=1)
    
    X = X[[col for col in X.columns if 'Fare' not in col]]
    return X

X = prepare_features(df)
y = df['Survived']

print("\n1. CHECKING TRAINING PARAMETERS")
print("-" * 70)

# Load existing model
with open(models_dir / 'decision_tree.pkl', 'rb') as f:
    dt_model = pickle.load(f)

print(f"Model criterion: {dt_model.criterion}")
print(f"Max depth: {dt_model.max_depth}")
print(f"Min samples leaf: {dt_model.min_samples_leaf}")
print(f"Min samples split: {dt_model.min_samples_split}")
print(f"Class weight: {dt_model.class_weight}")

if dt_model.criterion != 'entropy':
    print("⚠️  WARNING: Criterion should be 'entropy' for information gain!")

print("\n2. CHECKING ROOT NODE FEATURE SELECTION")
print("-" * 70)

tree = dt_model.tree_
root_feature_idx = tree.feature[0]
root_feature_name = feature_names[root_feature_idx]
root_threshold = tree.threshold[0]
root_impurity = tree.impurity[0]
root_samples = tree.n_node_samples[0]

print(f"Root node:")
print(f"  Feature: {root_feature_name} (index {root_feature_idx})")
print(f"  Threshold: {root_threshold:.4f}")
print(f"  Impurity (entropy): {root_impurity:.4f}")
print(f"  Samples: {root_samples}")

print("\n3. CALCULATING FEATURE IMPORTANCE (INFORMATION GAIN)")
print("-" * 70)

# Get feature importances
importances = dt_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("Top 10 most important features (by information gain):")
print(feature_importance_df.head(10).to_string(index=False))

print("\n4. VERIFYING DECISION PATH LOGIC")
print("-" * 70)

# Test sample
sample_data = {
    'Pclass': 1,
    'Sex_encoded': 1,
    'Age': 25,
    'SibSp': 0,
    'Parch': 0,
    'Embarked_C': 1,
    'Embarked_Q': 0,
    'Embarked_S': 0,
    'FamilySize': 1,
    'IsAlone': 1,
    'Title_Master': 0,
    'Title_Miss': 1,
    'Title_Mr': 0,
    'Title_Mrs': 0,
    'Title_Other': 0,
    'AgeGroup_Adult': 0,
    'AgeGroup_Child': 0,
    'AgeGroup_Senior': 0,
    'AgeGroup_Teen': 1,
    'Cabin_A': 0,
    'Cabin_B': 0,
    'Cabin_C': 0,
    'Cabin_D': 0,
    'Cabin_E': 0,
    'Cabin_F': 0,
    'Cabin_G': 0,
    'Cabin_T': 0,
    'Cabin_Unknown': 1
}

sample_df = pd.DataFrame([sample_data])[feature_names]
sample_array = sample_df.values[0]

# Get decision path
path = dt_model.decision_path(sample_df)
node_indicator = path.toarray()[0]
leaf_id = dt_model.apply(sample_df)[0]

print(f"Sample: Female, Age 25, 1st Class")
print(f"Leaf node: {leaf_id}")
print("\nDecision path (in order of tree traversal):")

# Get all nodes in path (excluding leaf)
path_nodes = []
for node_id in np.where(node_indicator == 1)[0]:
    if node_id == leaf_id:
        continue
    if tree.children_left[node_id] != tree.children_right[node_id]:  # Not a leaf
        feature_idx = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        impurity = tree.impurity[node_id]
        samples = tree.n_node_samples[node_id]
        
        path_nodes.append({
            'node_id': node_id,
            'feature': feature_names[feature_idx],
            'threshold': threshold,
            'impurity': impurity,
            'samples': samples,
            'value': sample_array[feature_idx]
        })

# Sort by node_id to show traversal order
path_nodes.sort(key=lambda x: x['node_id'])

for i, node in enumerate(path_nodes):
    direction = "<=" if node['value'] <= node['threshold'] else ">"
    print(f"  Step {i+1}: Node {node['node_id']}")
    print(f"    Feature: {node['feature']} (importance: {importances[feature_names.index(node['feature'])]:.4f})")
    print(f"    Condition: {node['value']:.2f} {direction} {node['threshold']:.2f}")
    print(f"    Impurity: {node['impurity']:.4f}, Samples: {node['samples']}")

print("\n5. VERIFYING TREE STRUCTURE")
print("-" * 70)

print(f"Total nodes: {tree.node_count}")
print(f"Total leaves: {tree.n_leaves}")
print(f"Max depth: {tree.max_depth}")

# Check if root node uses the most important feature
root_importance = importances[root_feature_idx]
print(f"\nRoot feature importance: {root_importance:.4f}")
print(f"Is root feature the most important? {root_importance == importances.max()}")

if root_importance != importances.max():
    max_importance_feature = feature_names[np.argmax(importances)]
    print(f"⚠️  Most important feature is: {max_importance_feature} (importance: {importances.max():.4f})")
    print("   Note: Root node may not use the most important feature due to tree structure")

print("\n6. MANUAL INFORMATION GAIN CALCULATION (for root node)")
print("-" * 70)

# Calculate information gain for root split manually
def entropy(y):
    """Calculate entropy."""
    if len(y) == 0:
        return 0
    p = np.bincount(y) / len(y)
    p = p[p > 0]  # Remove zeros
    return -np.sum(p * np.log2(p))

# Root feature and threshold
root_feature_values = X.iloc[:, root_feature_idx].values
root_feature_name = feature_names[root_feature_idx]

# Split at root threshold
left_mask = root_feature_values <= root_threshold
right_mask = root_feature_values > root_threshold

left_y = y[left_mask].values
right_y = y[right_mask].values

# Calculate entropies
parent_entropy = entropy(y.values)
left_entropy = entropy(left_y)
right_entropy = entropy(right_y)

# Calculate information gain
left_weight = len(left_y) / len(y)
right_weight = len(right_y) / len(y)
information_gain = parent_entropy - (left_weight * left_entropy + right_weight * right_entropy)

print(f"Root feature: {root_feature_name}")
print(f"Root threshold: {root_threshold:.4f}")
print(f"Parent entropy: {parent_entropy:.4f}")
print(f"Left child entropy: {left_entropy:.4f} (samples: {len(left_y)})")
print(f"Right child entropy: {right_entropy:.4f} (samples: {len(right_y)})")
print(f"Information gain: {information_gain:.4f}")

print("\n" + "="*70)
print("VERIFICATION COMPLETE")
print("="*70)

