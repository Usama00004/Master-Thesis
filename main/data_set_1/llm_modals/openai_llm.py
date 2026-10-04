import pandas as pd
import json
import kagglehub
from kagglehub import KaggleDatasetAdapter
from openai import OpenAI


# ============================================================
# 1. LOAD DATASET
# ============================================================

def load_credit_card_dataset():

    file_path = "UCI_Credit_Card.csv"

    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "uciml/default-of-credit-card-clients-dataset",
        file_path,
    )

    # Rename columns to German names
    df = rename_columns_to_german(df)

    return df


# ============================================================
# 2. RENAME COLUMNS
# ============================================================

def rename_columns_to_german(df):

    german_names = [
        "ID",
        "Kreditlimit",
        "Geschlecht",
        "Bildungsniveau",
        "Familienstand",
        "Alter",

        "Zahlungsstatus_Monat_0",
        "Zahlungsstatus_Monat_2",
        "Zahlungsstatus_Monat_3",
        "Zahlungsstatus_Monat_4",
        "Zahlungsstatus_Monat_5",
        "Zahlungsstatus_Monat_6",

        "Rechnungsbetrag_Monat_1",
        "Rechnungsbetrag_Monat_2",
        "Rechnungsbetrag_Monat_3",
        "Rechnungsbetrag_Monat_4",
        "Rechnungsbetrag_Monat_5",
        "Rechnungsbetrag_Monat_6",

        "Zahlungsbetrag_Monat_1",
        "Zahlungsbetrag_Monat_2",
        "Zahlungsbetrag_Monat_3",
        "Zahlungsbetrag_Monat_4",
        "Zahlungsbetrag_Monat_5",
        "Zahlungsbetrag_Monat_6",

        "Kreditausfall"
    ]

    if len(df.columns) != len(german_names):
        raise ValueError(
            f"Expected {len(german_names)} columns, "
            f"but dataset contains {len(df.columns)} columns."
        )

    df.columns = german_names

    return df


# ============================================================
# 3. EXTRACT DATASET INFORMATION
# ============================================================

def get_dataset_info(df):

    info = {
        "number_of_rows": len(df),
        "number_of_columns": len(df.columns),
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


# ============================================================
# 4. GENERATE FEATURE SELECTION PROMPT
# ============================================================

def generate_feature_prompt(dataset_info):

    long_prompt = f"""
    You are an expert machine learning feature engineering agent.

    Your task is to analyze the dataset information provided below and
    identify the 5 most relevant existing input columns that could be
    used for feature engineering to improve prediction of the target
    variable.

    DATASET INFORMATION:
    {json.dumps(dataset_info, indent=2, default=str)}

    TARGET VARIABLE:
    "Kreditausfall"

    IMPORTANT RULES:
    1. Carefully analyze and understand the dataset before selecting columns.
    2. Identify the meaning, role, and data type of each column.
    3. Determine which existing input columns are most relevant for predicting the target variable.
    4. Select EXACTLY 5 existing input columns.
    5. The target variable "Kreditausfall" MUST NOT be selected.
    6. Do NOT use "Kreditausfall" directly or indirectly.
    7. Only select columns that already exist in the dataset.
    8. Do NOT create new features.
    9. Do NOT perform mathematical operations or transformations.
    10. The purpose of this task is ONLY to identify the most relevant existing columns that could later be used to construct new features.
    11. Prefer columns that have a meaningful relationship with the target and provide useful information for feature construction.
    12. Avoid selecting redundant columns when possible.
    13. Do not select columns simply because they are numerical.Consider their meaning and potential predictive usefulness.
    14. The selected columns should be suitable candidates for creating derived features such as ratios, differences, interactions, aggregations, or other transformations in a later step.

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
    - The target variable "Kreditausfall"

    Use exactly this format:
    [
        {{
            "column": "column name"
        }},
        {{
            "column": "column name"
        }},
        {{
            "column": "column name"
        }},
        {{
            "column": "column name"
        }},
        {{
            "column": "column name"
        }}
    ]
    """

    short_prompt = f"""
        Select exactly 5 existing input columns most relevant for predicting "Kreditausfall".

        Dataset:
        {json.dumps(dataset_info, indent=2, default=str)}

        Rules:
        - Exclude "Kreditausfall".
        - Use only existing columns.
        - No new features or transformations.
        - Prefer meaningful, predictive, non-redundant columns useful for future feature engineering.
        - Return ONLY valid JSON. and do not include explanations, markdown, or code fences.

        [
            {{"column": "column name"}},
            {{"column": "column name"}},
            {{"column": "column name"}},
            {{"column": "column name"}},
            {{"column": "column name"}}
        ]
        """

    return short_prompt


# ============================================================
# 5. SEND PROMPT TO OPENAI
# ============================================================

def get_llm_feature_suggestions(
    prompt,
    model="gpt-5-mini"
):

    client = OpenAI()

    response = client.responses.create(
        model=model,
        input=prompt,

        text={
            "format": {
                "type": "json_schema",
                "name": "feature_selection",
                "strict": True,

                "schema": {
                    "type": "object",

                    "properties": {
                        "features": {
                            "type": "array",

                            "items": {
                                "type": "object",

                                "properties": {
                                    "column": {
                                        "type": "string"
                                    }
                                },

                                "required": [
                                    "column"
                                ],

                                "additionalProperties": False
                            },

                            "minItems": 5,
                            "maxItems": 5
                        }
                    },

                    "required": [
                        "features"
                    ],

                    "additionalProperties": False
                }
            }
        },

        store=False
    )

    llm_response = response.output_text.strip()

    print("\nRaw OpenAI response:")
    print(llm_response)

    try:

        result = json.loads(llm_response)

        features = result["features"]

        return features

    except (json.JSONDecodeError, KeyError) as e:

        print("\nOpenAI returned an unexpected response:")
        print(llm_response)

        raise e


# ============================================================
# 6. VALIDATE LLM FEATURE SELECTION
# ============================================================

def validate_features(features, df):

    # Check that output is a list
    if not isinstance(features, list):
        raise ValueError(
            "LLM response is not a list."
        )

    # Check exactly 5 features
    if len(features) != 5:
        raise ValueError(
            f"Expected exactly 5 features, "
            f"but received {len(features)}."
        )

    selected_columns = []

    for feature in features:

        if "column" not in feature:
            raise ValueError(
                "LLM response contains an object without 'column'."
            )

        column_name = feature["column"]

        # Check that column exists
        if column_name not in df.columns:
            raise ValueError(
                f"LLM selected a column that does not exist: "
                f"{column_name}"
            )

        # Check target is not selected
        if column_name == "Kreditausfall":
            raise ValueError(
                "LLM selected the target variable 'Kreditausfall'."
            )

        selected_columns.append(column_name)

    # Check duplicates
    if len(selected_columns) != len(set(selected_columns)):
        raise ValueError(
            "LLM selected duplicate columns."
        )

    return True


# ============================================================
# 7. DISPLAY SELECTED FEATURES
# ============================================================

def display_features(features):

    print("\n")
    print("=" * 50)
    print("COLUMNS IDENTIFIED BY OPENAI")
    print("=" * 50)

    for i, feature in enumerate(features, start=1):

        print(
            f"{i}. {feature['column']}"
        )

    print("=" * 50)


# ============================================================
# 8. DISPLAY SELECTED COLUMN INFORMATION
# ============================================================

def display_selected_column_information(
    features,
    df
):

    print("\n")
    print("=" * 70)
    print("SELECTED COLUMN INFORMATION")
    print("=" * 70)

    for feature in features:

        column = feature["column"]

        print(f"\nColumn: {column}")
        print(f"Data type: {df[column].dtype}")
        print(f"Missing values: {df[column].isna().sum()}")
        print(f"Unique values: {df[column].nunique()}")

        print(
            "Sample values:",
            df[column]
            .dropna()
            .head(5)
            .tolist()
        )

    print("=" * 70)


# ============================================================
# 9. MAIN FUNCTION
# ============================================================

def main():

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = load_credit_card_dataset()

    print("\nDataset loaded successfully!")

    print(
        f"\nDataset shape: {df.shape}"
    )

    print("\nFirst 5 rows:")

    print(
        df.head()
    )

    print("\nColumns:")

    print(
        df.columns.tolist()
    )


    # --------------------------------------------------------
    # Extract dataset information
    # --------------------------------------------------------

    dataset_info = get_dataset_info(df)


    # --------------------------------------------------------
    # Generate prompt
    # --------------------------------------------------------

    prompt = generate_feature_prompt(
        dataset_info
    )


    # --------------------------------------------------------
    # Send dataset information to OpenAI
    # --------------------------------------------------------

    print("\n")
    print("=" * 50)
    print("SENDING DATASET INFORMATION TO OPENAI...")
    print("=" * 50)

    features = get_llm_feature_suggestions(
        prompt,
        model="gpt-5-mini"
    )


    # --------------------------------------------------------
    # Validate response
    # --------------------------------------------------------

    validate_features(
        features,
        df
    )


    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    display_features(
        features
    )

    display_selected_column_information(
        features,
        df
    )


    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return (
        df,
        dataset_info,
        features
    )


# ============================================================
# 10. SCRIPT EXECUTION
# ============================================================

if __name__ == "__main__":

    df, dataset_info, suggested_features = main()
