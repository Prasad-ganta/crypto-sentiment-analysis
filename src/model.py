import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score
)
from xgboost import XGBClassifier

def train_model(df):

    data = df.copy()

    data['target'] = (
        data['Closed PnL'] > 0
    ).astype(int)

    data = pd.get_dummies(
        data,
        columns=[
            'Side',
            'classification'
        ]
    )

    drop_cols = [
        'target',
        'Closed PnL',
        'Account',
        'Coin',
        'Timestamp',
        'Timestamp IST',
        'Date',
        'date',
        'timestamp',
        'Direction',
        'Transaction Hash'
    ]

    features = [
        col for col in data.columns
        if col not in drop_cols
    ]

    X = data[features]

    X = X.select_dtypes(
        include=[
            'int64',
            'float64',
            'bool'
        ]
    )

    y = data['target']

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )
    )

    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    report = classification_report(
        y_test,
        predictions
    )

    prediction_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': predictions
    })

    prediction_df.to_csv(
        'outputs/predictions.csv',
        index=False
    )

    return accuracy, report