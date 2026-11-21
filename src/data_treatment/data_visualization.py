import seaborn as sns
import matplotlib.pyplot as plt

def Corr_Heatmap(df):
    plt.figure(figsize=(12,12))
    sns.heatmap(
        df.corr(),
        annot=True,
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        linewidths=0.5
    )
    plt.title('Heatmap')
    plt.show()

def Pairplot(df, columns=None, title="Pairwise Feature Relationships"):
    """
    Plots a pairplot grid to observe joint relationships and distributions.
    """
    numeric_df = df.select_dtypes(include="number")

    if columns:
        numeric_df = numeric_df[columns]

    # Drop NAs to avoid plotting errors
    clean_data = numeric_df.dropna()

    # Create pairplot using seaborn
    g = sns.pairplot(clean_data, diag_kind="kde", corner=True, plot_kws={'alpha': 0.5})
    g.fig.suptitle(title, y=1.02, fontsize=14)
    plt.show()

def Rolling_correlation(df, col1, col2, window=30):
    """
    Plots the rolling correlation between two columns over a specified window (e.g., 30 periods).
    """
    # Calculate rolling correlation
    rolling_corr = df[col1].rolling(window=window).corr(df[col2])

    # Plotting
    plt.figure(figsize=(10, 4))
    plt.plot(rolling_corr.index, rolling_corr, color='purple', linewidth=1.5)
    plt.axhline(0, color='gray', linestyle='--', alpha=0.7)
    plt.title(f"{window}-Period Rolling Correlation between {col1} and {col2}", fontsize=12)
    plt.xlabel("Timestamp", fontsize=10)
    plt.ylabel("Correlation Coefficient", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()