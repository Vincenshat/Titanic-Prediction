"""
Test script to verify model predictions are working correctly.
"""

import pandas as pd
import pickle
from pathlib import Path
import sys
sys.path.append('app')
from streamlit_app import preprocess_input

# Load models
base_dir = Path(__file__).parent
models_dir = base_dir / 'models'

print("Loading models...")
with open(models_dir / 'decision_tree.pkl', 'rb') as f:
    dt_model = pickle.load(f)
with open(models_dir / 'feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Load test data
data_dir = base_dir / 'data'
df = pd.read_csv(data_dir / 'titanic_cleaned.csv')

# Get a few test samples
print("\n" + "="*50)
print("TESTING PREDICTIONS")
print("="*50)

# Test case 1: Female, 1st class, young
print("\nTest Case 1: Female, Age 25, 1st Class")
user_input1 = {
    'Sex': 'female',
    'Age': 25,
    'Pclass': 1,
    'SibSp': 0,
    'Parch': 0,
    'Embarked': 'C'
}
sample1 = preprocess_input(user_input1, feature_names)
pred1 = dt_model.predict(sample1)[0]
proba1 = dt_model.predict_proba(sample1)[0]
print(f"  Prediction: {pred1} (Survived={pred1==1})")
print(f"  Probability: Survived={proba1[1]:.3f}, Not Survived={proba1[0]:.3f}")
print(f"  Expected: Should be high survival probability (female, 1st class)")

# Test case 2: Male, 3rd class, adult
print("\nTest Case 2: Male, Age 35, 3rd Class")
user_input2 = {
    'Sex': 'male',
    'Age': 35,
    'Pclass': 3,
    'SibSp': 0,
    'Parch': 0,
    'Embarked': 'S'
}
sample2 = preprocess_input(user_input2, feature_names)
pred2 = dt_model.predict(sample2)[0]
proba2 = dt_model.predict_proba(sample2)[0]
print(f"  Prediction: {pred2} (Survived={pred2==1})")
print(f"  Probability: Survived={proba2[1]:.3f}, Not Survived={proba2[0]:.3f}")
print(f"  Expected: Should be low survival probability (male, 3rd class)")

# Test case 3: Child, 2nd class
print("\nTest Case 3: Child (Age 8), 2nd Class")
user_input3 = {
    'Sex': 'male',
    'Age': 8,
    'Pclass': 2,
    'SibSp': 1,
    'Parch': 1,
    'Embarked': 'S'
}
sample3 = preprocess_input(user_input3, feature_names)
pred3 = dt_model.predict(sample3)[0]
proba3 = dt_model.predict_proba(sample3)[0]
print(f"  Prediction: {pred3} (Survived={pred3==1})")
print(f"  Probability: Survived={proba3[1]:.3f}, Not Survived={proba3[0]:.3f}")
print(f"  Expected: Should have decent survival probability (child)")

# Compare with actual data
print("\n" + "="*50)
print("COMPARING WITH ACTUAL DATA")
print("="*50)

# Find similar cases in actual data
print("\nSimilar cases in actual data:")
print("\nFemale, 1st class examples:")
female_1st = df[(df['Sex']=='female') & (df['Pclass']==1) & (df['Age']>=20) & (df['Age']<=30)]
if len(female_1st) > 0:
    print(f"  Found {len(female_1st)} examples")
    print(f"  Survival rate: {female_1st['Survived'].mean():.1%}")
    print(f"  Sample: Age={female_1st.iloc[0]['Age']:.0f}, Survived={female_1st.iloc[0]['Survived']}")

print("\nMale, 3rd class examples:")
male_3rd = df[(df['Sex']=='male') & (df['Pclass']==3) & (df['Age']>=30) & (df['Age']<=40)]
if len(male_3rd) > 0:
    print(f"  Found {len(male_3rd)} examples")
    print(f"  Survival rate: {male_3rd['Survived'].mean():.1%}")
    print(f"  Sample: Age={male_3rd.iloc[0]['Age']:.0f}, Survived={male_3rd.iloc[0]['Survived']}")

print("\n" + "="*50)
print("TEST COMPLETED")
print("="*50)
print("\nIf predictions match expected patterns, the model is working correctly!")
print("If not, check feature preprocessing logic.")

