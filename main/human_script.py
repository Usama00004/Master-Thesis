import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)


def load_credit_card_dataset():
    file_path = "UCI_Credit_Card.csv"

    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "uciml/default-of-credit-card-clients-dataset",
        file_path,
    )
    return df


def display_dataset_info(df):
    print("----------- Original Column Names -----------")
    print(df.columns.tolist())

    print("\n----------- Dataset Shape -----------")
    print(df.shape)

    print("\n----------- Missing Values -----------")
    print(df.isnull().sum())


def prepare_data(df):

    target_column = "default.payment.next.month"

    selected_features = [
        "LIMIT_BAL",
        "AGE",
        "PAY_AMT6",
        "BILL_AMT6",
        "PAY_6"
    ]

    X = df[selected_features]
    y = df[target_column]

    return X, y


def train_model(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probability)

    print("\n----------- Model Evaluation ----------")

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    print("\n----------- Classification Report -----------")
    print(classification_report(y_test, y_pred))


def main():

    # Load dataset
    df = load_credit_card_dataset()

    # Dataset information
    display_dataset_info(df)

    # Prepare features and target
    X, y = prepare_data(df)

    print("\n----------- Feature Shape -----------")
    print(X.shape)

    print("\n----------- Target Distribution -----------")
    print(y.value_counts())

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples :", len(X_test))

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()