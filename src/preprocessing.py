import pandas as pd

def preprocess_data(trades, sentiment):

    trades.columns = trades.columns.str.strip()
    sentiment.columns = sentiment.columns.str.strip()

    trades['Timestamp'] = pd.to_datetime(
        trades['Timestamp'],
        unit='ms',
        errors='coerce'
    )

    sentiment['date'] = pd.to_datetime(
        sentiment['date'],
        errors='coerce'
    )

    trades['Date'] = trades['Timestamp'].dt.date

    sentiment['date'] = sentiment['date'].dt.date

    trades['Closed PnL'] = pd.to_numeric(
        trades['Closed PnL'],
        errors='coerce'
    )

    trades['Size USD'] = pd.to_numeric(
        trades['Size USD'],
        errors='coerce'
    )

    trades.dropna(inplace=True)

    merged = trades.merge(
        sentiment,
        left_on='Date',
        right_on='date',
        how='left'
    )

    return merged