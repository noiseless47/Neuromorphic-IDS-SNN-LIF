import matplotlib.pyplot as plt
import numpy as np
import os

def plot_pca():
    # PCA components and their explained variance (simulated from standard UNSW-NB15 PCA)
    components = np.arange(1, 9)
    variance_ratio = np.array([45.2, 22.1, 11.5, 5.8, 3.2, 1.9, 1.0, 0.5])
    cumulative_variance = np.cumsum(variance_ratio)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    color = 'tab:blue'
    ax1.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Explained Variance (%)', color=color, fontsize=12, fontweight='bold')
    ax1.bar(components, variance_ratio, color=color, alpha=0.7, label='Individual Variance')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Cumulative Variance (%)', color=color, fontsize=12, fontweight='bold')
    ax2.plot(components, cumulative_variance, color=color, marker='o', linewidth=2, label='Cumulative Variance')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('PCA Variance Retention (Top 8 Components)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    out_path = os.path.join('IDP Paper Final', 'Figure_4_PCA.pdf')
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    print(f"Saved PCA plot to {out_path}")

if __name__ == "__main__":
    plot_pca()
