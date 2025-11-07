"""
Streamlit Web Application for Titanic Survival Analysis

This application provides interactive interfaces for:
1. Survival Prediction (Tab 1)
2. Survivor Profile Analysis (Tab 2)
3. Model Comparison (Tab 3)

All code and comments are in English.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Analysis",
    page_icon="🚢",
    layout="wide"
)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

@st.cache_resource
def load_models():
    """Load all trained models with individual error handling."""
    base_dir = Path(__file__).parent.parent
    models_dir = base_dir / 'models'
    
    models = {}
    errors = []
    
    # Define model files to load
    model_files = {
        'Decision Tree': 'decision_tree.pkl',
        'Random Forest': 'random_forest.pkl',
        'Gradient Boosting': 'gradient_boosting.pkl',
        'KMeans': 'kmeans.pkl',
        'Scaler': 'scaler.pkl',
        'feature_names': 'feature_names.pkl',
        'clustering_feature_names': 'clustering_feature_names.pkl'
    }
    
    # Load each model individually
    for model_name, filename in model_files.items():
        try:
            filepath = models_dir / filename
            if not filepath.exists():
                errors.append(f"{model_name}: File not found ({filename})")
                continue
                
            with open(filepath, 'rb') as f:
                models[model_name] = pickle.load(f)
        except Exception as e:
            error_msg = f"{model_name}: {str(e)}"
            errors.append(error_msg)
            # Continue loading other models even if one fails
    
    # Check if essential models are loaded
    if 'Decision Tree' not in models:
        st.error("❌ Critical Error: Decision Tree model could not be loaded!")
        st.info("Please run the training scripts first: python src/10_tree_classifier.py")
        if errors:
            with st.expander("Error Details", expanded=True):
                for error in errors:
                    st.text(error)
        return None
    
    if 'feature_names' not in models:
        st.error("❌ Critical Error: Feature names could not be loaded!")
        st.info("Please run the training scripts first: python src/10_tree_classifier.py")
        if errors:
            with st.expander("Error Details", expanded=True):
                for error in errors:
                    st.text(error)
        return None
    
    # Show warnings for non-critical model failures
    if errors:
        import sys
        import sklearn
        
        # Display environment info
        with st.expander("🔍 Environment Diagnostics", expanded=False):
            st.code(f"""
Python Version: {sys.version.split()[0]}
scikit-learn Version: {sklearn.__version__}
            """)
        
        missing_models = []
        version_errors = []
        other_errors = []
        
        for error in errors:
            model_name = error.split(':')[0]
            error_msg = error.split(':', 1)[1].strip() if ':' in error else error
            
            if 'not found' in error_msg.lower():
                missing_models.append((model_name, error_msg))
            elif 'CyHalfBinomialLoss' in error_msg or '__pyx_unpickle' in error_msg or '_loss' in error_msg or 'No module named' in error_msg:
                version_errors.append((model_name, error_msg))
            else:
                other_errors.append((model_name, error_msg))
        
        if version_errors:
            st.error("⚠️ **Version Compatibility Error**")
            for model_name, error_msg in version_errors:
                st.error(f"**{model_name}**: {error_msg}")
            st.info("""
**Solution**: This application uses Python 3.13, which requires scikit-learn>=1.6.0. 
Your models were trained with an older version and need to be retrained:

```bash
# Install compatible versions
pip install -r requirements.txt

# Retrain all models
python src/10_tree_classifier.py
python src/20_survivor_clustering.py

# Commit and push the new model files
git add models/
git commit -m "Retrain models for Python 3.13 compatibility"
git push
```

**Note**: After retraining, commit the new model files to your repository.
            """)
        
        if missing_models:
            st.warning("⚠️ **Model Files Not Found**")
            for model_name, error_msg in missing_models:
                st.warning(f"**{model_name}**: {error_msg}")
            st.info("Please run the training script first: `python src/10_tree_classifier.py`")
        
        if other_errors:
            st.warning("⚠️ **Other Errors**")
            for model_name, error_msg in other_errors:
                st.warning(f"**{model_name}**: {error_msg}")
        
        st.success("✅ **Application Status**: The app will continue with available models. Core functionality (Decision Tree prediction) is still available.")
    
    return models

@st.cache_data
def load_data():
    """Load cleaned dataset."""
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    
    try:
        df = pd.read_csv(data_dir / 'titanic_cleaned.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def preprocess_input(user_input: dict, feature_names: list) -> pd.DataFrame:
    """
    Preprocess user input to match model features exactly as in training.
    
    Args:
        user_input: Dictionary with user inputs
        feature_names: List of feature names expected by model (in exact order)
        
    Returns:
        DataFrame with preprocessed features in the exact order expected by model
    """
    import pandas as pd
    
    # Calculate derived features first
    family_size = user_input['SibSp'] + user_input['Parch'] + 1
    is_alone = 1 if family_size == 1 else 0
    sex_encoded = 1 if user_input['Sex'] == 'female' else 0
    
    # Determine AgeGroup (matching training logic: 0-12, 13-30, 31-50, 50+)
    if user_input['Age'] <= 12:
        agegroup = 'Child'
    elif user_input['Age'] <= 30:
        agegroup = 'Teen'
    elif user_input['Age'] <= 50:
        agegroup = 'Adult'
    else:
        agegroup = 'Senior'
    
    # Create a temporary dataframe (matching training)
    temp_df = pd.DataFrame([{
        'Pclass': user_input['Pclass'],
        'Sex_encoded': sex_encoded,
        'Age': user_input['Age'],
        'SibSp': user_input['SibSp'],
        'Parch': user_input['Parch'],
        'Embarked': user_input['Embarked'],
        'FamilySize': family_size,
        'IsAlone': is_alone,
        'AgeGroup': agegroup
    }])
    
    # One-hot encode Embarked (matching training)
    embarked_dummies = pd.get_dummies(temp_df['Embarked'], prefix='Embarked')
    temp_df = pd.concat([temp_df, embarked_dummies], axis=1)
    
    # One-hot encode AgeGroup (matching training)
    agegroup_dummies = pd.get_dummies(temp_df['AgeGroup'], prefix='AgeGroup')
    temp_df = pd.concat([temp_df, agegroup_dummies], axis=1)
    
    # Build feature vector in exact order expected by model
    feature_values = []
    for feat_name in feature_names:
        if feat_name in temp_df.columns:
            feature_values.append(temp_df[feat_name].values[0])
        else:
            # If feature doesn't exist (e.g., missing one-hot category), set to 0
            feature_values.append(0)
    
    # Create DataFrame with features in exact order
    feature_df = pd.DataFrame([feature_values], columns=feature_names)
    
    # Ensure all values are numeric
    for col in feature_df.columns:
        feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce').fillna(0)
    
    return feature_df

def get_decision_path(tree_model, sample: pd.DataFrame, feature_names: list) -> list:
    """
    Get decision path for a sample, showing features in order of tree traversal.
    Each feature was selected based on information gain (entropy criterion).
    """
    sample_array = sample.values[0]
    path = tree_model.decision_path(sample_array.reshape(1, -1))
    node_indicator = path.toarray()[0]
    
    leaf_id = tree_model.apply(sample_array.reshape(1, -1))[0]
    tree = tree_model.tree_
    
    # Get feature importances for display
    importances = tree_model.feature_importances_
    
    path_rules = []
    path_nodes = []
    
    # Collect all nodes in the path (excluding leaf)
    for node_id in np.where(node_indicator == 1)[0]:
        if node_id == leaf_id:
            continue
        if tree.children_left[node_id] != tree.children_right[node_id]:  # Not a leaf
            feature_idx = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            feature_name = feature_names[feature_idx]
            value = sample_array[feature_idx]
            importance = importances[feature_idx]
            impurity = tree.impurity[node_id]
            samples = tree.n_node_samples[node_id]
            
            path_nodes.append({
                'node_id': node_id,
                'feature': feature_name,
                'importance': importance,
                'threshold': threshold,
                'value': value,
                'impurity': impurity,
                'samples': samples
            })
    
    # Sort by node_id to show traversal order (root to leaf)
    path_nodes.sort(key=lambda x: x['node_id'])
    
    # Format decision path with feature importance information
    for i, node in enumerate(path_nodes):
        direction = "<=" if node['value'] <= node['threshold'] else ">"
        rule = f"Step {i+1}: {node['feature']} = {node['value']:.2f} {direction} {node['threshold']:.2f}"
        rule += f" (IG: {node['importance']:.3f}, Entropy: {node['impurity']:.3f})"
        path_rules.append(rule)
    
    return path_rules

def predict_with_explanation(model, sample: pd.DataFrame, feature_names: list) -> dict:
    """
    Predict survival probability and provide explanation.
    
    Args:
        model: Trained model
        sample: Preprocessed sample DataFrame
        feature_names: List of feature names
        
    Returns:
        Dictionary with prediction results
    """
    proba = model.predict_proba(sample)[0]
    prediction = model.predict(sample)[0]
    
    result = {
        'survival_probability': proba[1],
        'prediction': int(prediction),
        'confidence': max(proba)
    }
    
    # Get decision path if Decision Tree
    if hasattr(model, 'tree_'):
        result['decision_path'] = get_decision_path(model, sample, feature_names)
    
    return result

def generate_counterfactuals(tree_model, sample: pd.DataFrame, feature_names: list) -> list:
    """Generate counterfactual suggestions."""
    suggestions = []
    
    # Get current prediction
    current_pred = tree_model.predict(sample)[0]
    current_proba = tree_model.predict_proba(sample)[0][1]
    
    # Try changing key features
    sample_copy = sample.copy()
    
    # Change Sex
    if 'Sex_encoded' in sample.columns:
        if sample['Sex_encoded'].values[0] == 0:  # male
            sample_copy = sample.copy()
            sample_copy['Sex_encoded'] = 1  # female
            new_proba = tree_model.predict_proba(sample_copy)[0][1]
            if new_proba > current_proba:
                suggestions.append(f"If gender was female: survival probability would be {new_proba:.1%} (current: {current_proba:.1%})")
        else:  # female
            sample_copy = sample.copy()
            sample_copy['Sex_encoded'] = 0  # male
            new_proba = tree_model.predict_proba(sample_copy)[0][1]
            if new_proba < current_proba:
                suggestions.append(f"If gender was male: survival probability would be {new_proba:.1%} (current: {current_proba:.1%})")
    
    # Change Pclass
    if sample['Pclass'].values[0] > 1:
        sample_copy = sample.copy()
        sample_copy['Pclass'] = 1
        new_proba = tree_model.predict_proba(sample_copy)[0][1]
        if new_proba > current_proba:
            suggestions.append(f"If class was 1st: survival probability would be {new_proba:.1%} (current: {current_proba:.1%})")
    
    return suggestions

def main():
    """Main application function."""
    st.title("🚢 Titanic Survival Analysis System")
    st.markdown("---")
    
    # Load models and data
    models = load_models()
    if models is None:
        return
    
    df = load_data()
    if df is None:
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Survival Prediction", "Survivor Profiles", "Model Comparison"])
    
    # Tab 1: Survival Prediction
    with tab1:
        st.header("Survival Prediction")
        st.markdown("Enter passenger information to predict survival probability.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sex = st.selectbox("Gender", ["male", "female"])
            age = st.slider("Age", 0, 100, 30)
            pclass = st.selectbox("Class", [1, 2, 3])
        
        with col2:
            sibsp = st.slider("Siblings/Spouses", 0, 8, 0)
            parch = st.slider("Parents/Children", 0, 6, 0)
            embarked = st.selectbox("Embarked", ["C", "Q", "S"])
        
        if st.button("Predict Survival", type="primary"):
            # Prepare input
            user_input = {
                'Sex': sex,
                'Age': age,
                'Pclass': pclass,
                'SibSp': sibsp,
                'Parch': parch,
                'Embarked': embarked
            }
            
            # Preprocess
            sample = preprocess_input(user_input, models['feature_names'])
            
            # Debug: Show feature values (expandable)
            with st.expander("🔍 Debug: Feature Values", expanded=False):
                st.write("Input features:")
                st.json(user_input)
                st.write("Processed feature vector:")
                st.dataframe(sample.T.rename(columns={0: 'Value'}))
                st.write(f"Feature shape: {sample.shape}")
                st.write(f"Expected features: {len(models['feature_names'])}")
                st.write(f"Actual features: {len(sample.columns)}")
            
            # Predict with Decision Tree
            try:
                result = predict_with_explanation(
                    models['Decision Tree'], 
                    sample, 
                    models['feature_names']
                )
                
                # Display results
                st.markdown("### Prediction Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Survival Probability", f"{result['survival_probability']:.1%}")
                    st.progress(result['survival_probability'])
                    
                    prediction_text = "✅ Survived" if result['prediction'] == 1 else "❌ Not Survived"
                    st.markdown(f"### {prediction_text}")
                
                with col2:
                    st.metric("Confidence", f"{result['confidence']:.1%}")
                    st.metric("Prediction", result['prediction'])
                
                # Decision path
                if 'decision_path' in result and result['decision_path']:
                    st.markdown("### Decision Path")
                    st.caption("Features are selected based on information gain (entropy). "
                             "The tree chooses the feature that maximizes information gain at each split.")
                    
                    # Show feature importance summary
                    importances = models['Decision Tree'].feature_importances_
                    top_features = pd.DataFrame({
                        'Feature': models['feature_names'],
                        'Importance': importances
                    }).sort_values('Importance', ascending=False).head(5)
                    
                    with st.expander("📊 Top 5 Most Important Features (by Information Gain)", expanded=False):
                        st.dataframe(top_features, use_container_width=True)
                    
                    st.markdown("**Decision Path (from root to leaf):**")
                    for rule in result['decision_path']:
                        st.text(f"  {rule}")
                    
                    # Explain the path
                    st.info("""
                    **How it works:**
                    - Each step selects the feature with highest information gain
                    - Information gain measures how well a feature separates classes
                    - Higher information gain = better feature for classification
                    - The tree automatically finds the optimal feature order
                    """)
                else:
                    st.info("Decision path not available for this sample.")
                
                # Counterfactuals
                suggestions = generate_counterfactuals(
                    models['Decision Tree'],
                    sample,
                    models['feature_names']
                )
                
                if suggestions:
                    st.markdown("### Improvement Suggestions")
                    for suggestion in suggestions:
                        st.info(suggestion)
                
                # Also show predictions from other models for comparison
                with st.expander("📊 Compare All Models", expanded=False):
                    # Build list of available models
                    available_models = []
                    if 'Decision Tree' in models:
                        available_models.append(('Decision Tree', models['Decision Tree']))
                    if 'Random Forest' in models:
                        available_models.append(('Random Forest', models['Random Forest']))
                    if 'Gradient Boosting' in models:
                        available_models.append(('Gradient Boosting', models['Gradient Boosting']))
                    
                    if available_models:
                        cols = st.columns(len(available_models))
                        for idx, (model_name, model) in enumerate(available_models):
                            with cols[idx]:
                                try:
                                    proba = model.predict_proba(sample)[0]
                                    pred = model.predict(sample)[0]
                                    st.metric(model_name, f"{proba[1]:.1%}")
                                    st.caption(f"Prediction: {'Survived' if pred == 1 else 'Not Survived'}")
                                except Exception as e:
                                    st.error(f"Error: {e}")
                    else:
                        st.warning("No additional models available for comparison.")
            
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.exception(e)
                st.info("Please check the debug information above to see feature values.")
    
    # Tab 2: Survivor Profiles
    with tab2:
        st.header("Survivor Profile Analysis")
        
        survivors = df[df['Survived'] == 1]
        
        # Statistics
        st.markdown("### Basic Statistics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Survivors", len(survivors))
        col2.metric("Survival Rate", f"{len(survivors)/len(df)*100:.1f}%")
        col3.metric("Female Survival Rate", 
                   f"{(survivors['Sex']=='female').sum()/(df['Sex']=='female').sum()*100:.1f}%")
        col4.metric("Male Survival Rate",
                   f"{(survivors['Sex']=='male').sum()/(df['Sex']=='male').sum()*100:.1f}%")
        
        # Visualizations
        st.markdown("### Visualizations")
        
        # Gender distribution
        fig, ax = plt.subplots(figsize=(8, 5))
        comparison = pd.crosstab(df['Survived'], df['Sex'])
        comparison.plot(kind='bar', ax=ax, color=['coral', 'steelblue'])
        ax.set_xlabel('Survival Status')
        ax.set_ylabel('Count')
        ax.set_title('Gender Distribution: Survivors vs Non-Survivors')
        ax.set_xticklabels(['Not Survived', 'Survived'], rotation=0)
        ax.legend(title='Gender')
        st.pyplot(fig)
        
        # Class distribution
        fig, ax = plt.subplots(figsize=(8, 5))
        comparison = pd.crosstab(df['Survived'], df['Pclass'])
        comparison.plot(kind='bar', ax=ax, color=['coral', 'steelblue', 'lightgreen'])
        ax.set_xlabel('Survival Status')
        ax.set_ylabel('Count')
        ax.set_title('Class Distribution: Survivors vs Non-Survivors')
        ax.set_xticklabels(['Not Survived', 'Survived'], rotation=0)
        ax.legend(title='Class', labels=['1st', '2nd', '3rd'])
        st.pyplot(fig)
        
        # Clustering results
        st.markdown("### Clustering Analysis")
        
        try:
            # Load cluster profiles
            base_dir = Path(__file__).parent.parent
            cluster_profiles = pd.read_csv(base_dir / 'reports' / 'cluster_profiles.csv')
            
            st.dataframe(cluster_profiles[['Cluster', 'Size', 'Percentage', 
                                          'Age_mean', 'Fare_mean', 'Pclass_mode', 
                                          'FamilySize_mean']].round(2))
            
            # Load cluster descriptions
            with open(base_dir / 'reports' / 'cluster_descriptions.txt', 'r') as f:
                descriptions = f.read()
            
            st.markdown("#### Cluster Descriptions")
            for line in descriptions.strip().split('\n'):
                st.text(line)
        
        except Exception as e:
            st.warning(f"Could not load clustering results: {e}")
    
    # Tab 3: Model Comparison
    with tab3:
        st.header("Model Comparison")
        
        try:
            # Load comparison data
            base_dir = Path(__file__).parent.parent
            model_comparison = pd.read_csv(base_dir / 'reports' / 'model_comparison.csv')
            
            st.markdown("### Performance Metrics")
            st.dataframe(model_comparison.round(4))
            
            # Visualizations
            st.markdown("### Visualizations")
            
            # Accuracy comparison
            fig, ax = plt.subplots(figsize=(8, 5))
            model_comparison.plot(x='Model', y='Accuracy', kind='bar', ax=ax, color='steelblue')
            ax.set_ylabel('Accuracy')
            ax.set_title('Model Accuracy Comparison')
            ax.set_ylim([0, 1])
            ax.legend().remove()
            plt.xticks(rotation=0)
            st.pyplot(fig)
            
            # ROC curves
            st.markdown("#### ROC Curves")
            st.image(str(base_dir / 'reports' / 'figures' / 'model_roc_curves.png'))
            
            # Feature importance
            st.markdown("#### Feature Importance")
            st.image(str(base_dir / 'reports' / 'figures' / 'feature_importance_comparison.png'))
        
        except Exception as e:
            st.error(f"Error loading comparison data: {e}")

if __name__ == '__main__':
    main()

