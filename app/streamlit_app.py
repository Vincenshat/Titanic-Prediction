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
    """Load all trained models."""
    base_dir = Path(__file__).parent.parent
    models_dir = base_dir / 'models'
    
    models = {}
    try:
        with open(models_dir / 'decision_tree.pkl', 'rb') as f:
            models['Decision Tree'] = pickle.load(f)
        with open(models_dir / 'random_forest.pkl', 'rb') as f:
            models['Random Forest'] = pickle.load(f)
        with open(models_dir / 'gradient_boosting.pkl', 'rb') as f:
            models['Gradient Boosting'] = pickle.load(f)
        with open(models_dir / 'kmeans.pkl', 'rb') as f:
            models['KMeans'] = pickle.load(f)
        with open(models_dir / 'scaler.pkl', 'rb') as f:
            models['Scaler'] = pickle.load(f)
        with open(models_dir / 'feature_names.pkl', 'rb') as f:
            models['feature_names'] = pickle.load(f)
        with open(models_dir / 'clustering_feature_names.pkl', 'rb') as f:
            models['clustering_feature_names'] = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please run the training scripts first: python src/10_tree_classifier.py")
        return None
    
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
                    col1, col2, col3 = st.columns(3)
                    for idx, (model_name, model) in enumerate([
                        ('Decision Tree', models['Decision Tree']),
                        ('Random Forest', models['Random Forest']),
                        ('Gradient Boosting', models['Gradient Boosting'])
                    ]):
                        with [col1, col2, col3][idx]:
                            try:
                                proba = model.predict_proba(sample)[0]
                                pred = model.predict(sample)[0]
                                st.metric(model_name, f"{proba[1]:.1%}")
                                st.caption(f"Prediction: {'Survived' if pred == 1 else 'Not Survived'}")
                            except Exception as e:
                                st.error(f"Error: {e}")
            
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

