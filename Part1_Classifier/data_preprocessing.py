##############################################################################################################
# Script: data_preprocessing.py
# This script performs the data cleaning, feature engineering, and stratified splitting of the census dataset.
##############################################################################################################

import pandas as pd
from sklearn.model_selection import train_test_split
import os


def clean_data():
    # Cleaning and initial preprocessing steps
    with open('../census-bureau.columns', 'r') as f:
        column_names = [line.strip() for line in f]

    df = pd.read_csv('../census-bureau.data', header=None, names=column_names, skipinitialspace=True)
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    df['hispanic origin'] = df['hispanic origin'].fillna('Unknown')
    df.replace('?', 'Unknown', inplace=True)

    df['label'] = df['label'].map({'- 50000.': 0, '50000+.': 1})

    return df


def feature_engineering(df):
    #  Performing feature engineering as outlined in the report

    # Combining Gains + Dividends - Losses into one net value
    df['total_investment'] = df['capital gains'] + df['dividends from stocks'] - df['capital losses']

    # Ordinal Education Mapping
    # Mapping categorical strings to a numeric hierarchy 0-16
    edu_map = {
        'Children': 0, 'Less than 1st grade': 1, '1st 2nd 3rd or 4th grade': 2,
        '5th or 6th grade': 3, '7th and 8th grade': 4, '9th grade': 5,
        '10th grade': 6, '11th grade': 7, '12th grade no diploma': 8,
        'High school graduate': 9, 'Some college but no degree': 10,
        'Associates degree-occup/vocational': 11, 'Associates degree-academic program': 12,
        'Bachelors degree(BA AB BS)': 13, 'Masters degree(MA MS MEng MBA MSW LLS)': 14,
        'Prof school degree (MD DDS DVM LLB JD)': 15, 'Doctorate degree(PhD EdD)': 16
    }
    df['education_num'] = df['education'].map(edu_map).fillna(9)

    # Lineage Tracking
    # Flag for U.S.-born citizens with at least one foreign-born parent
    us_code = 'United-States'
    df['is_second_generation'] = (
        (df['country of birth self'] == us_code) & 
        ((df['country of birth father'] != us_code) | (df['country of birth mother'] != us_code))
    ).astype(int)

    # Removing redundant, low-signal, and administrative columns
    cols_to_drop = [
        # Redundancies
        'capital gains', 'capital losses', 'dividends from stocks', # Replaced by total_investment
        'education', # Replaced by education_num
        'detailed industry recode', 'detailed occupation recode', # Use Major codes instead
        'own business or self employed', # Lower NMI than 'class of worker'
        
        # High Missingness / Noise 
        # Dropped 5 out of 6 migration columns, keeping only 'state of previous residence'
        'migration code-change in msa', 'migration code-change in reg', 
        'migration code-move within reg', 'migration prev res in sunbelt',
        
        'region of previous residence',
        
        # Lineage Consolidation through is_second_generation
        'country of birth father', 'country of birth mother', 'country of birth self',
        
        # Complexity Reduction
        'detailed household and family stat', # Kept 'detailed household summary'
        
        # Admin Artifacts
        'year', 'weight', 
        "fill inc questionnaire for veteran's admin"
    ]

    # Execute drop safely
    df_final = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    return df_final


def main():
    base_df = clean_data()
    final_df = feature_engineering(base_df)

    # Performing the Stratified Split (80/20)
    # We split the dataframe itself so X and y stay together
    train_df, test_df = train_test_split(
        final_df, 
        test_size=0.15, 
        random_state=42, 
        stratify=final_df['label']
    )

    os.makedirs('../processed_datas', exist_ok=True)

    # Saving the consolidated files
    train_df.to_csv('../processed_datas/census_train.csv', index=False)
    test_df.to_csv('../processed_datas/census_test.csv', index=False)
    print("Preprocessed CSV files saved successfully!")

    print(f"Training set: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
    print(f"Testing set: {test_df.shape[0]} rows, {test_df.shape[1]} columns")


if __name__ == "__main__":
    main()