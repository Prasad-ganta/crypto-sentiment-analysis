import pandas as pd

def load_data():
    trades = pd.read_csv("data/historical_data.csv")
    sentiment = pd.read_csv("data/fear_greed_index.csv")
    return trades, sentiment