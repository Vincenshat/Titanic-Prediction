"""
Data Preprocessing Module for Titanic Survival Analysis

This module handles data loading, cleaning, feature engineering, and encoding.
All code and comments are in English.
"""

import pandas as pd
import numpy as np
import json
import os
import re
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load raw Titanic dataset from CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame containing raw data
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def display_data_info(df: pd.DataFrame) -> None:
    """
    Display basic information about the dataset.
    
    Args:
        df: DataFrame to analyze
    """
    print("\n" + "="*50)
    print("DATASET INFORMATION")
    print("="*50)
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumn names: {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing Count'] > 0])
    print(f"\nNumerical summary:")
    print(df.describe())

def extract_title(name: str) -> str:
    """
    Extract title from passenger name.
    
    Args:
        name: Full name string
        
    Returns:
        Extracted title (Mr, Mrs, Miss, Master, Other)
    """
    if pd.isna(name):
        return 'Other'
    
    # Extract title using regex
    title_match = re.search(r',\s*([^\.]+)\.', name)
    if title_match:
        title = title_match.group(1).strip()
    else:
        return 'Other'
    
    # Map rare titles to common categories
    title_mapping = {
        'Mr': 'Mr',
        'Mrs': 'Mrs',
        'Miss': 'Miss',
        'Master': 'Master',
        'Mlle': 'Miss',  # Mademoiselle -> Miss
        'Ms': 'Miss',
        'Mme': 'Mrs',    # Madame -> Mrs
        'Don': 'Mr',
        'Dona': 'Mrs',
        'Rev': 'Other',
        'Dr': 'Other',
        'Col': 'Other',
        'Major': 'Other',
        'Capt': 'Other',
        'Sir': 'Other',
        'Jonkheer': 'Other',
        'Lady': 'Mrs',
        'Countess': 'Mrs'
    }
    
    return title_mapping.get(title, 'Other')

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: DataFrame with missing values
        
    Returns:
        DataFrame with missing values filled
    """
    df = df.copy()
    
    # Extract title first for Age imputation
    df['Title'] = df['Name'].apply(extract_title)
    
    # Fill Age based on Title median
    print("\nFilling Age missing values based on Title median...")
    for title in df['Title'].unique():
        title_median = df[df['Title'] == title]['Age'].median()
        if pd.notna(title_median):
            df.loc[(df['Age'].isna()) & (df['Title'] == title), 'Age'] = title_median
    
    # Fill remaining Age with overall median
    if df['Age'].isna().any():
        overall_median = df['Age'].median()
        df['Age'].fillna(overall_median, inplace=True)
        print(f"Filled remaining Age missing values with overall median: {overall_median}")
    
    # Fill Embarked with mode
    print("\nFilling Embarked missing values with mode...")
    embarked_mode = df['Embarked'].mode()[0]
    df['Embarked'].fillna(embarked_mode, inplace=True)
    print(f"Filled Embarked with mode: {embarked_mode}")
    
    # Fill Fare with median
    print("\nFilling Fare missing values with median...")
    fare_median = df['Fare'].median()
    df['Fare'].fillna(fare_median, inplace=True)
    print(f"Filled Fare with median: {fare_median}")
    
    # Handle Cabin - missing values mean "not recorded", not "no cabin"
    # Since 77% of data is missing, we'll exclude Cabin feature entirely
    print("\nProcessing Cabin...")
    print(f"  Cabin missing: {df['Cabin'].isna().sum()} passengers ({df['Cabin'].isna().mean()*100:.1f}%)")
    print("  Note: Missing Cabin means 'not recorded', not 'no cabin'. Excluding Cabin feature.")
    
    # Detect and handle outliers
    print("\nDetecting outliers...")
    # Age outliers
    age_outliers = (df['Age'] < 0) | (df['Age'] > 100)
    if age_outliers.any():
        print(f"Found {age_outliers.sum()} Age outliers, replacing with median")
        df.loc[age_outliers, 'Age'] = df['Age'].median()
    
    # Fare outliers
    fare_outliers = df['Fare'] < 0
    if fare_outliers.any():
        print(f"Found {fare_outliers.sum()} Fare outliers, replacing with median")
        df.loc[fare_outliers, 'Fare'] = df['Fare'].median()
    
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features from existing columns.
    
    Args:
        df: DataFrame with cleaned data
        
    Returns:
        DataFrame with new features added
    """
    df = df.copy()
    
    # Family size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Is alone
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Age groups
    df['AgeGroup'] = pd.cut(
        df['Age'],
        bins=[0, 12, 30, 50, 100],
        labels=['Child', 'Teen', 'Adult', 'Senior']
    )
    df['AgeGroup'] = df['AgeGroup'].astype(str)
    
    # Fare bins (quartiles)
    fare_quartiles = df['Fare'].quantile([0.25, 0.5, 0.75])
    df['FareBin'] = pd.cut(
        df['Fare'],
        bins=[-np.inf, fare_quartiles[0.25], fare_quartiles[0.5], 
              fare_quartiles[0.75], np.inf],
        labels=['Q1', 'Q2', 'Q3', 'Q4']
    )
    df['FareBin'] = df['FareBin'].astype(str)
    
    print("\nFeature engineering completed:")
    print(f"  - FamilySize: {df['FamilySize'].min()} to {df['FamilySize'].max()}")
    print(f"  - IsAlone: {df['IsAlone'].sum()} passengers ({df['IsAlone'].mean()*100:.1f}%)")
    print(f"  - AgeGroup distribution:\n{df['AgeGroup'].value_counts()}")
    print(f"  - FareBin distribution:\n{df['FareBin'].value_counts()}")
    
    return df

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical variables.
    
    Args:
        df: DataFrame with categorical features
        
    Returns:
        DataFrame with encoded features
    """
    df = df.copy()
    
    # Encode Sex (binary: 0=male, 1=female)
    df['Sex_encoded'] = (df['Sex'] == 'female').astype(int)
    print("\nEncoded Sex: male=0, female=1")
    
    # One-hot encode Embarked
    embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
    df = pd.concat([df, embarked_dummies], axis=1)
    print(f"One-hot encoded Embarked: {list(embarked_dummies.columns)}")
    
    return df

def save_cleaned_data(df: pd.DataFrame, filepath: str, metadata_path: str = None) -> None:
    """
    Save cleaned dataset and metadata.
    
    Args:
        df: Cleaned DataFrame
        filepath: Path to save cleaned CSV
        metadata_path: Path to save feature engineering metadata
    """
    # Drop columns that won't be used for modeling
    columns_to_drop = ['PassengerId', 'Ticket', 'Name', 'Cabin']
    df_cleaned = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    # Save cleaned data
    df_cleaned.to_csv(filepath, index=False)
    print(f"\nCleaned data saved to {filepath}")
    print(f"Final shape: {df_cleaned.shape[0]} rows × {df_cleaned.shape[1]} columns")
    
    # Save metadata
    if metadata_path:
        metadata = {
            'fare_quartiles': {
                'Q1': float(df['Fare'].quantile(0.25)),
                'Q2': float(df['Fare'].quantile(0.5)),
                'Q3': float(df['Fare'].quantile(0.75))
            },
            'age_bins': [0, 12, 30, 50, 100],
            'features': list(df_cleaned.columns),
            'target': 'Survived'
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {metadata_path}")

def main():
    """Main function to run data preprocessing pipeline."""
    # Set up paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    data_dir.mkdir(exist_ok=True)
    
    input_file = data_dir / 'Titanic-Dataset.csv'
    output_file = data_dir / 'titanic_cleaned.csv'
    metadata_file = data_dir / 'feature_metadata.json'
    
    # Load raw data
    df = load_raw_data(str(input_file))
    
    # Display data information
    display_data_info(df)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Engineer features
    df = engineer_features(df)
    
    # Encode categorical variables
    df = encode_categorical(df)
    
    # Save cleaned data
    save_cleaned_data(df, str(output_file), str(metadata_file))
    
    print("\n" + "="*50)
    print("DATA PREPROCESSING COMPLETED")
    print("="*50)

if __name__ == '__main__':
    main()

