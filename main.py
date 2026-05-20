from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.analysis import (
    sentiment_analysis,
    anova_test
)
from src.visualization import (
    plot_pnl,
    plot_sentiment_distribution,
    plot_correlation,
    plot_daily_pnl
)
from src.clustering import cluster_traders
from src.model import train_model
from src.utils import save_merged_data
from src.report_generator import generate_report

trades, sentiment = load_data()

merged = preprocess_data(
    trades,
    sentiment
)

save_merged_data(merged)

pnl, win_rate, volume = (
    sentiment_analysis(merged)
)

anova = anova_test(merged)

plot_pnl(merged)

plot_sentiment_distribution(merged)

plot_correlation(merged)

plot_daily_pnl(merged)

clusters = cluster_traders(merged)

accuracy, report = train_model(merged)

generate_report(
    accuracy,
    anova,
    pnl,
    win_rate,
    volume
)

print("\nAverage PnL")
print(pnl)

print("\nWin Rate")
print(win_rate)

print("\nAverage Volume")
print(volume)

print("\nANOVA Test")
print(anova)

print("\nModel Accuracy")
print(accuracy)

print("\nClassification Report")
print(report)

print("\nProject Completed")