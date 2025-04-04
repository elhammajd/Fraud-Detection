import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import wilcoxon

#Plot a boxplot
def plot_boxplot(data, title="Boxplot", xlabel="Features", ylabel="Values"):
    if isinstance(data, pd.DataFrame):
        data.plot(kind='box', figsize=(10, 6))
    else:
        plt.boxplot(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# Wilcoxon signed-rank test on two paired samples
def perform_wilcoxon_test(data1, data2):
    stat, p_value = wilcoxon(data1, data2)
    return stat, p_value
