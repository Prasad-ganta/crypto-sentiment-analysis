import matplotlib.pyplot as plt
import seaborn as sns

def plot_pnl(df):

    plt.figure(figsize=(10,6))

    sns.boxplot(
        x='classification',
        y='Closed PnL',
        data=df
    )

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        'visuals/pnl_vs_sentiment.png'
    )

def plot_sentiment_distribution(df):

    plt.figure(figsize=(8,8))

    df['classification'].value_counts().plot(
        kind='pie',
        autopct='%1.1f%%'
    )

    plt.ylabel('')

    plt.savefig(
        'visuals/sentiment_distribution.png'
    )

def plot_correlation(df):

    plt.figure(figsize=(8,6))

    corr = df[
        ['Closed PnL', 'Size USD']
    ].corr()

    sns.heatmap(
        corr,
        annot=True,
        cmap='coolwarm'
    )

    plt.tight_layout()

    plt.savefig(
        'visuals/correlation_matrix.png'
    )

def plot_daily_pnl(df):

    daily = df.groupby(
        'Date'
    )['Closed PnL'].mean()

    plt.figure(figsize=(12,6))

    daily.plot()

    plt.tight_layout()

    plt.savefig(
        'visuals/daily_pnl_trend.png'
    )