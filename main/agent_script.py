from pyexpat import features
import pandas as pd
import json
import requests
import kagglehub
from kagglehub import KaggleDatasetAdapter


# ==================
# 1. LOAD DATASET
# ==================

def load_credit_card_dataset():
    """
    Load the UCI Credit Card dataset from Kaggle.
    
    Returns:
        pd.DataFrame: The loaded dataset with generic column names
    """
    file_path = "UCI_Credit_Card.csv" 
    
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "uciml/default-of-credit-card-clients-dataset",
        file_path,
    )
    
    # Rename columns to generic names
    df = rename_columns_to_generic(df)
    
    return df


def rename_columns_to_generic(df):
    """
    Rename all columns to generic names (column_1, column_2, column_3, ...).
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        pd.DataFrame: Dataframe with renamed columns
    """
    # Create generic column names
    new_column_names = [f"column_{i+1}" for i in range(len(df.columns))]
    
    # Rename columns
    df.columns = new_column_names

    print("-----------Renamed Column Names---------")
    print(df.columns.tolist())
    
    return df



# ================================
# 2. EXTRACT DATASET INFORMATION
# ================================

def get_dataset_info(df):
    """
    Extract comprehensive information about the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        dict: Dictionary containing dataset metadata and column information
    """
    info = {
        "number_of_rows": len(df),
        "columns": []
    }
    
    for column in df.columns:
        column_info = {
            "name": column,
            "dtype": str(df[column].dtype),
            "missing_values": int(df[column].isna().sum()),
            "unique_values": int(df[column].nunique()),
            "sample_values": (
                df[column]
                .dropna()
                .head(5)
                .tolist()
            )
        }
        
        info["columns"].append(column_info)
    
    return info



# =======================================
# 3. GENERATE FEATURE ENGINEERING PROMPT
# =======================================

def generate_feature_prompt(dataset_info):
    """
    Generate a prompt for the LLM to analyze the dataset and
    select the 5 most relevant input columns for feature engineering.

    Args:
        dataset_info (dict): Dataset information dictionary

    Returns:
        str: Formatted prompt string
    """

    prompt = f"""
You are an expert machine learning feature engineering agent.

Your task is to analyze the dataset information provided below and
identify the 5 most relevant existing input columns that could be
used for feature engineering to improve prediction of the target
variable.

DATASET INFORMATION:
{json.dumps(dataset_info, indent=2, default=str)}

TARGET VARIABLE:
"column_25"

IMPORTANT RULES:

1. Carefully analyze and understand the dataset before selecting columns.

2. Identify the meaning, role, and data type of each column.

3. Determine which existing input columns are most relevant for
   predicting the target variable.

4. Select EXACTLY 5 existing input columns.

5. The target variable "column_25" MUST NOT be selected.

6. Do NOT use "column_25" directly or indirectly.

7. Only select columns that already exist in the dataset.

8. Do NOT create new features.

9. Do NOT perform mathematical operations or transformations.

10. The purpose of this task is ONLY to identify the most relevant
    existing columns that could later be used to construct new features.

11. Prefer columns that have a meaningful relationship with the target
    and provide useful information for feature construction.

12. Avoid selecting redundant columns when possible.

13. Do not select columns simply because they are numerical.
    Consider their meaning and potential predictive usefulness.

14. The selected columns should be suitable candidates for creating
    derived features such as ratios, differences, interactions,
    aggregations, or other transformations in a later step.

Return EXACTLY 5 columns.

Return ONLY valid JSON.

Do NOT include:
- Explanations
- Markdown
- ```json code fences
- Introductory text
- Concluding text
- New feature names
- Mathematical operations
- The target variable "column_25"

Use exactly this format:

[
    {{
        "column": "column_1"
    }},
    {{
        "column": "column_2"
    }},
    {{
        "column": "column_3"
    }},
    {{
        "column": "column_4"
    }},
    {{
        "column": "column_5"
    }}
]
"""

    return prompt




# ==========================================
# 4. SEND DATASET INFORMATION TO LOCAL LLM
# ==========================================

def get_llm_feature_suggestions(prompt, model="llama3.1", host="http://localhost:11434"):
    """
    Send prompt to local LLM and get feature suggestions.
    
    Args:
        prompt (str): The prompt to send to LLM
        model (str): Name of the model to use
        host (str): Ollama server address
    
    Returns:
        list: List of suggested features as dictionaries
    
    Raises:
        requests.exceptions.RequestException: If API call fails
        json.JSONDecodeError: If response is not valid JSON
    """
    response = requests.post(
        f"{host}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    
    response.raise_for_status()
    
    result = response.json()
    llm_response = result["response"]
    
    try:
        features = json.loads(llm_response)
        return features
    except json.JSONDecodeError as e:
        print("LLM returned invalid JSON.")
        print("\nRaw LLM response:")
        print(llm_response)
        raise e


# ======================================
# 5. DISPLAY COLUMNS IDENTIFIED BY LLM
# ======================================

def display_features(features):
    """
    Display the columns identified by the LLM
    for feature engineering.

    Args:
        features (list): List of dictionaries containing column names.
    """

    print("\n========================================")
    print("COLUMNS IDENTIFIED BY LLM")
    print("========================================")

    if isinstance(features, list):
        for i, feature in enumerate(features, start=1):
            print(f"{i}. {feature.get('column', 'N/A')}")
    else:
        print(features)



# ===================
# 6. MAIN FUNCTION
# ===================

def main():
    """
    Main function to orchestrate the entire workflow.
    Each function is called on a separate line for clarity.
    """
    # Load dataset
    df = load_credit_card_dataset()
    
    # Print basic dataset info
    print("Dataset loaded successfully!")
    print(f"\nDataset shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nColumns:")
    print(df.columns.tolist())
    
    # Extract dataset information
    dataset_info = get_dataset_info(df)
    
    # Generate prompt for LLM
    prompt = generate_feature_prompt(dataset_info)
    
    # Get feature suggestions from LLM
    features = get_llm_feature_suggestions(prompt)
    
    # Display the suggested features
    display_features(features)
    
    # Return results
    return df, dataset_info, features
   
# ========================
# 7. SCRIPT EXECUTION
# ========================

if __name__ == "__main__":
    # Call main function
    df, dataset_info, suggested_features = main()
