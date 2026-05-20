import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.analysis import (
    sentiment_analysis,
    anova_test
)
from src.clustering import cluster_traders
from src.model import train_model

st.set_page_config(
    page_title="Crypto Sentiment Analysis",
    layout="wide"
)

st.title("Crypto Market Sentiment Analysis Dashboard")

trades, sentiment = load_data()

merged = preprocess_data(
    trades,
    sentiment
)

st.subheader("Merged Dataset")

st.dataframe(
    merged.head()
)

pnl, win_rate, volume = (
    sentiment_analysis(merged)
)

anova = anova_test(merged)

accuracy, report = train_model(merged)

st.subheader("Average PnL by Sentiment")

st.write(pnl)

fig1, ax1 = plt.subplots(
    figsize=(10,6)
)

sns.boxplot(
    x='classification',
    y='Closed PnL',
    data=merged,
    ax=ax1
)

plt.xticks(rotation=45)

st.pyplot(fig1)

st.subheader("Win Rate")

st.write(win_rate)

st.subheader("Average Trading Volume")

st.write(volume)

st.subheader("Correlation Heatmap")

numeric_df = merged.select_dtypes(
    include=['float64', 'int64']
)

fig2, ax2 = plt.subplots(
    figsize=(8,6)
)

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap='coolwarm',
    ax=ax2
)

st.pyplot(fig2)

st.subheader("Daily PnL Trend")

daily = merged.groupby(
    'Date'
)['Closed PnL'].mean()

fig3, ax3 = plt.subplots(
    figsize=(12,6)
)

daily.plot(ax=ax3)

st.pyplot(fig3)

st.subheader("Trader Clustering")

clusters = cluster_traders(merged)

fig4, ax4 = plt.subplots(
    figsize=(10,6)
)

sns.scatterplot(
    x='avgSizeUSD',
    y='avgPnL',
    hue='Cluster',
    data=clusters,
    palette='Set1',
    ax=ax4
)

st.pyplot(fig4)

st.subheader("ANOVA Test Result")

st.write(anova)

st.subheader("Model Accuracy")

st.write(accuracy)

st.subheader("Classification Report")

st.text(report)

st.success(
    "Dashboard Running Successfully"
)