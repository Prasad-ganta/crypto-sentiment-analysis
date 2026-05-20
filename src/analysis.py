from scipy.stats import f_oneway

def sentiment_analysis(df):

    pnl = df.groupby(
        'classification'
    )['Closed PnL'].mean()

    win_rate = (
        df.groupby('classification')
        ['Closed PnL']
        .apply(lambda x: (x > 0).mean() * 100)
    )

    volume = df.groupby(
        'classification'
    )['Size USD'].mean()

    return pnl, win_rate, volume

def anova_test(df):

    groups = []

    for sentiment in df[
        'classification'
    ].dropna().unique():

        groups.append(
            df[
                df['classification'] == sentiment
            ]['Closed PnL']
        )

    return f_oneway(*groups)