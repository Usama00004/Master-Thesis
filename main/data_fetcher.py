import kagglehub
from kagglehub import KaggleDatasetAdapter

def load_credit_card_dataset():
    file_path = "UCI_Credit_Card.csv"
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "uciml/default-of-credit-card-clients-dataset",
        file_path,
    )
    return df

def display_dataset_info(df):
    print("First 5 records:")
    print(df.head())

def main():
    df = load_credit_card_dataset()
    display_dataset_info(df)

if __name__ == "__main__":
    main()



















# import kagglehub
# from kagglehub import KaggleDatasetAdapter

# file_path = "UCI_Credit_Card.csv"

# df = kagglehub.load_dataset(
#     KaggleDatasetAdapter.PANDAS,
#     "uciml/default-of-credit-card-clients-dataset",
#     file_path,
# )

# print("First 5 records:")
# print(df.head())