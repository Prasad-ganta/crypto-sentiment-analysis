import pandas as pd

def save_merged_data(df):
    df.to_csv(
        'outputs/merged_data.csv',
        index=False
    )