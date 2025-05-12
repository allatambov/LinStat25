import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pizza = pd.read_csv("https://raw.githubusercontent.com/allatambov/LinStat25/refs/heads/main/pizza.csv")
pizza.head()

X = pizza.select_dtypes(float)
X.head()

X.corr()
pd.plotting.scatter_matrix(X, figsize = (14, 14));

X_scaled = StandardScaler().fit_transform(X)
X_scaled = pd.DataFrame(X_scaled)
X_scaled.columns = X.columns
X_scaled.head()

X_scaled.describe().round(2)
X_scaled.corr()
pd.plotting.scatter_matrix(X_scaled, figsize = (14, 14));

print(X_scaled.shape)
p = X_scaled.shape[1]

pca_names = ["PC" + str(i) for i in range(1, p + 1)]
print(pca_names)

pca = PCA(n_components = p)
print(pca)

pca_res = pd.DataFrame(pca.fit_transform(X_scaled))
print(pca_res)

pca_res.columns = pca_names
print(pca_res)

pca_var = pca.explained_variance_
print(pca_var.round(2))

plt.plot(range(1, p + 1), pca_var, "o-");
plt.title("Scree plot");
plt.xlabel("Number of components");
plt.ylabel("Eigenvalues (variances)");

pca_var_ratio = pca.explained_variance_ratio_
print((pca_var_ratio * 100).round(2))

print(np.cumsum((pca_var_ratio * 100).round(2)))

print(pca.components_)
pd.DataFrame(pca.components_)

rotation_matrix = pd.DataFrame(pca.components_).T
rotation_matrix.columns = pca_names
rotation_matrix.index = X.columns
rotation_matrix

print((rotation_matrix["PC1"] ** 2).sum()) 
print((rotation_matrix["PC2"] ** 2).sum())

pd.DataFrame(np.dot(X_scaled, rotation_matrix))

print(X.corr().round(3))
print(pca_res.corr().round(3))

rotation_matrix

pizza["Index1"] = pca_res["PC1"]
pizza["Index2"] = pca_res["PC2"]

pizza[["Index1", "Index2"]].describe().round(2)
pizza.groupby("brand")["Index1"].mean().sort_values(ascending = False)
pizza.groupby("brand")["Index2"].mean().sort_values(ascending = False)
pizza.plot.scatter("Index1", "Index2");
pizza["brand_int"] = pizza["brand"].map({"A" : 0, "B" : 1, "C" : 2,
                                         "D" : 3, "E" : 4, "F" : 5, 
                                         "G" : 6, "H" : 7, "I" : 8, 
                                         "J" : 9})
pizza.plot.scatter("Index1", "Index2", 
                   c = "brand_int", 
                   cmap = "tab10");
