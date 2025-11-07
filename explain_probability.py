"""
Detailed explanation of how probability is calculated in Decision Tree models.

This script demonstrates step-by-step how predict_proba() works.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier

# Load model
base_dir = Path(__file__).parent
models_dir = base_dir / 'models'

print("="*70)
print("HOW DECISION TREE CALCULATES PROBABILITY")
print("="*70)

with open(models_dir / 'decision_tree.pkl', 'rb') as f:
    dt_model = pickle.load(f)
with open(models_dir / 'feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Create a test sample: Female, Age 25, 1st Class
print("\n" + "="*70)
print("EXAMPLE: Female, Age 25, 1st Class Passenger")
print("="*70)

# Create sample manually
sample_data = {
    'Pclass': 1,
    'Sex_encoded': 1,  # female
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

print("\n1. SAMPLE FEATURES:")
print("-" * 70)
for i, (name, value) in enumerate(zip(feature_names, sample_array)):
    if value != 0 or i < 10:  # Show non-zero and first 10 features
        print(f"  {name:25s} = {value}")

# Get prediction
prediction = dt_model.predict(sample_df)[0]
probability = dt_model.predict_proba(sample_df)[0]

print(f"\n2. MODEL PREDICTION:")
print("-" * 70)
print(f"  Prediction: {prediction} ({'Survived' if prediction == 1 else 'Not Survived'})")
print(f"  Probability: [Not Survived: {probability[0]:.4f}, Survived: {probability[1]:.4f}]")

# Now explain HOW it calculates this
print("\n" + "="*70)
print("HOW THE PROBABILITY IS CALCULATED:")
print("="*70)

print("\nStep 1: Traverse the Decision Tree")
print("-" * 70)
print("The model follows the decision path through the tree based on feature values:")

# Get decision path
path = dt_model.decision_path(sample_df)
node_indicator = path.toarray()[0]
leaf_id = dt_model.apply(sample_df)[0]

print(f"\n  Leaf node reached: {leaf_id}")

# Show the path
print("\n  Decision path:")
for node_id in np.where(node_indicator == 1)[0]:
    if leaf_id == node_id:
        continue
    if dt_model.tree_.children_left[node_id] != dt_model.tree_.children_right[node_id]:
        feature_idx = dt_model.tree_.feature[node_id]
        threshold = dt_model.tree_.threshold[node_id]
        feature_name = feature_names[feature_idx]
        value = sample_array[feature_idx]
        
        direction = "<=" if value <= threshold else ">"
        print(f"    Node {node_id}: {feature_name} = {value:.2f} {direction} {threshold:.2f}")

print(f"\n  → Reached leaf node {leaf_id}")

print("\nStep 2: Get Leaf Node Statistics")
print("-" * 70)

# Get leaf node value
leaf_value = dt_model.tree_.value[leaf_id][0]
leaf_samples = dt_model.tree_.n_node_samples[leaf_id]

print(f"  Leaf node {leaf_id} contains:")
print(f"    - Total training samples that reached this leaf: {leaf_samples}")
print(f"    - Class distribution (weighted): {leaf_value}")
print(f"      * Class 0 (Not Survived): {leaf_value[0]:.4f} (weighted)")
print(f"      * Class 1 (Survived): {leaf_value[1]:.4f} (weighted)")

print("\nStep 3: Calculate Probability")
print("-" * 70)
print("Note: Because class_weight='balanced' was used, the values are weighted.")
print("Probability = Normalized class distribution in the leaf node")
print("(sklearn normalizes the leaf values to sum to 1)")

# The leaf_value is already normalized to probabilities
# sklearn's predict_proba returns normalized leaf values
prob_not_survived = leaf_value[0] / (leaf_value[0] + leaf_value[1])
prob_survived = leaf_value[1] / (leaf_value[0] + leaf_value[1])

print(f"\n  P(Not Survived) = {leaf_value[0]:.4f} / ({leaf_value[0]:.4f} + {leaf_value[1]:.4f}) = {prob_not_survived:.4f}")
print(f"  P(Survived) = {leaf_value[1]:.4f} / ({leaf_value[0]:.4f} + {leaf_value[1]:.4f}) = {prob_survived:.4f}")

print(f"\n  Verification:")
print(f"    Model output: [{probability[0]:.4f}, {probability[1]:.4f}]")
print(f"    Calculated:   [{prob_not_survived:.4f}, {prob_survived:.4f}]")
print(f"    Match: {np.allclose(probability, [prob_not_survived, prob_survived], atol=1e-6)}")
print(f"\n  Note: The leaf_value array stores normalized probabilities directly.")
print(f"       sklearn's predict_proba() returns these normalized values.")

print("\n" + "="*70)
print("KEY INSIGHTS:")
print("="*70)
print("""
1. Probability is NOT a learned parameter - it's calculated from training data statistics
2. Each leaf node stores the class distribution of training samples that reached it
3. The probability = proportion of each class in that leaf node
4. This is why decision trees are interpretable - you can see exactly why a prediction was made
5. The more samples in a leaf, the more reliable the probability estimate

Example interpretation:
- If a leaf has 100 samples: 80 survived, 20 didn't
- Then P(Survived) = 80/100 = 0.80 (80%)
- This means 80% of similar passengers in training data survived
""")

print("\n" + "="*70)
print("COMPARISON WITH OTHER MODELS:")
print("="*70)

# Load other models
with open(models_dir / 'random_forest.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open(models_dir / 'gradient_boosting.pkl', 'rb') as f:
    gb_model = pickle.load(f)

print("\nSame sample, different models:")
print("-" * 70)

dt_proba = dt_model.predict_proba(sample_df)[0][1]
rf_proba = rf_model.predict_proba(sample_df)[0][1]
gb_proba = gb_model.predict_proba(sample_df)[0][1]

print(f"  Decision Tree:      {dt_proba:.4f} ({dt_proba*100:.1f}%)")
print(f"  Random Forest:      {rf_proba:.4f} ({rf_proba*100:.1f}%)")
print(f"  Gradient Boosting:  {gb_proba:.4f} ({gb_proba*100:.1f}%)")

print("\nWhy they differ:")
print("""
- Decision Tree: Single tree, probability from one leaf node
- Random Forest: Average of 100 trees, each tree votes with its leaf probability
- Gradient Boosting: Weighted combination of 100 trees, probabilities are calibrated
""")

print("\n" + "="*70)
print("END OF EXPLANATION")
print("="*70)

