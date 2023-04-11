from utils.helper_functions import dir_except_checkpoints
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Dataset:
# MaÅ‚gorzata Charytanowicz, Jerzy Niewczas 
# Institute of Mathematics and Computer Science, 
# The John Paul II Catholic University of Lublin, KonstantynÃ³w 1 H, 
# PL 20-708 Lublin, Poland 
# e-mail: {mchmat,jniewczas}@kul.lublin.pl 

# Piotr Kulczycki, Piotr A. Kowalski, Szymon Lukasik, Slawomir Zak 
# Department of Automatic Control and Information Technology, 
# Cracow University of Technology, Warszawska 24, PL 31-155 Cracow, Poland 
# and 
# Systems Research Institute, Polish Academy of Sciences, Newelska 6, 
# PL 01-447 Warsaw, Poland 
# e-mail: {kulczycki,pakowal,slukasik,slzak}@ibspan.waw.pl

# Unsupervised Learning.
cols = ["area", "perimeter", "compactness", "lenght", "width", "asymmetry", "groove", "class"]
df = pd.read_csv("data/seeds_dataset.txt", names=cols, sep="\s+")

print(df.head())

dir_except_checkpoints('seeds_plot')

for i in range(len(cols)-1):
    for j in range(i+1, len(cols)-1):
        x_label = cols[i]
        y_label = cols[j]
        sns.scatterplot(x=x_label, y=y_label, data=df, hue='class')
        plt.savefig(os.path.join('seeds_plot', f'{i}_{j}.png'))
        plt.clf()
        
# Clustering
x = "compactness"
y = "asymmetry"
X = df[[x, y]].values

kmeans = KMeans(n_clusters = 3).fit(X)

clusters = kmeans.labels_

print(clusters)
print(df["class"].values)

cluster_df = pd.DataFrame(np.hstack((X, clusters.reshape(-1, 1))), columns=[x, y, "class"])

dir_except_checkpoints('classes')

# K Means classes
sns.scatterplot(x=x, y=y, hue='class', data=cluster_df)
plt.savefig(os.path.join('classes', 'k_mean_classes.png'))
plt.clf()

# Original classes
sns.scatterplot(x=x, y=y, hue='class', data=df)
plt.savefig(os.path.join('classes', 'original_classes.png'))
plt.clf()

# Higher Dimensions
X = df[cols[:-1]].values
kmeans_higher_dimensions = KMeans(n_clusters = 3).fit(X)
cluster_df_higher_dimensions = pd.DataFrame(np.hstack((X, kmeans.labels_.reshape(-1, 1))), columns=df.columns)

sns.scatterplot(x=x, y=y, hue='class', data=cluster_df_higher_dimensions)
plt.savefig(os.path.join('classes', 'k_mean_higher_dimensions_classes.png'))
plt.clf()

dir_except_checkpoints('principal_component_analysis')

# Principal Component Analysis
pca = PCA(n_components=2)
transformed_x = pca.fit_transform(X)

X.shape
print(X.shape)

transformed_x.shape
print(transformed_x.shape)

print(transformed_x[:5])

plt.scatter(transformed_x[:,0], transformed_x[:,1])
plt.savefig(os.path.join('principal_component_analysis', 'transformed_x.png'))
plt.clf()

kmeans_pca_df = pd.DataFrame(np.hstack((transformed_x, kmeans.labels_.reshape(-1, 1))), columns=["pca1", "pca2", "class"])
truth_pca_df = pd.DataFrame(np.hstack((transformed_x, df["class"].values.reshape(-1, 1))), columns=["pca1", "pca2", "class"])

#KMeans classes
sns.scatterplot(x="pca1", y="pca2", hue='class', data=kmeans_pca_df)
plt.savefig(os.path.join('classes', 'kmeans_pca_classes.png'))
plt.clf()

#Truth classes
sns.scatterplot(x="pca1", y="pca2", hue='class', data=truth_pca_df)
plt.savefig(os.path.join('classes', 'truth_pca_classes.png'))
plt.clf()