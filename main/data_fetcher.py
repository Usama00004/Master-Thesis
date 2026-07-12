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
    print("-----------Orignal Column Names---------")
    print(df.columns.tolist())
    return df


def data_columns_changer(df):
    df.columns = [f"col{i}" for i in range(1, len(df.columns) + 1)]
    print("-----------Dummy Column Names---------")
    print(df.columns.tolist())
    
def main():
    df = load_credit_card_dataset()
    display_dataset_info(df)
    data_columns_changer(df)

if __name__ == "__main__":
    main()

