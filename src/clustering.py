import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns


def cluster_traders(df):

    trader_stats = (
        df.groupby('Account')
        .agg({
            'Closed PnL': 'mean',
            'Size USD': 'mean'
        })
    )

    trader_stats.columns = [
        'avgPnL',
        'avgSizeUSD'
    ]

    trader_stats.dropna(inplace=True)

    scaler = StandardScaler()

    scaled = scaler.fit_transform(
        trader_stats
    )

    kmeans = KMeans(
        n_clusters=3,
        random_state=42
    )

    trader_stats['Cluster'] = (
        kmeans.fit_predict(scaled)
    )

    os.makedirs(
        'visuals',
        exist_ok=True
    )

    os.makedirs(
        'outputs',
        exist_ok=True
    )

    plt.figure(figsize=(10,6))

    sns.scatterplot(
        x='avgSizeUSD',
        y='avgPnL',
        hue='Cluster',
        data=trader_stats,
        palette='Set1'
    )

    plt.savefig(
        'visuals/trader_clusters.png'
    )

    trader_stats.to_csv(
        'outputs/clustered_traders.csv',
        index=True
    )

    return trader_stats