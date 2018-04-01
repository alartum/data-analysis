import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

def stacked_heatmap(nrows, data, xlabel, ylabel, title, xticks, yticks, **kwargs):
    """
    Splits values into vertically stacked heatmaps and then plots them.
    """
    fig, axes = plt.subplots(ncols=1, nrows=nrows, figsize=(10, 8))
    data_local = data
    if nrows == 1:
        axes = [axes]
    if type(data) == pd.core.frame.DataFrame:
        data_local = data.values
    cols = data_local.shape[1]
    data_max = data_local.max().max()
    data_min = data_local.min().min()
    
    idx = [cols//nrows*k for k in range(nrows)]
    idx += [cols]
    for i in range(cols%nrows): idx[i+1] += i+1
    
    for i in range(nrows):
        sns.heatmap(data=data_local[:, idx[i]:idx[i+1]], ax=axes[i], 
                    xticklabels=xticks[idx[i]:idx[i+1]], 
                    yticklabels=yticks,
                    linewidths=0.025, linecolor='black',
                    vmin=data_min, vmax=data_max, **kwargs)
        axes[i].set_ylabel(ylabel)
    
    axes[0].set_title(title, fontsize=14)
    axes[-1].set_xlabel(xlabel)
    plt.tight_layout()
    plt.show()
    
def cluster_quality(transformed):
    """
    Given transformed vectors, returns minimum of (distance to nearest cluster)/(distance to other cluster) over all the vectors.
    """
    check_clustered = transformed[np.all(transformed > 0, axis=1), :]
    check_clustered = check_clustered/(check_clustered.min(axis=1))[:, None]
    
    return np.sort(check_clustered.ravel())[check_clustered.shape[0]]

def cross_val_F1(estimator, X, y):
    F1 = cross_val_score(estimator, X, y, scoring='f1_macro')
    print('F1 = %.2f \u00b1 %.2f' % (F1.mean(), F1.std()))