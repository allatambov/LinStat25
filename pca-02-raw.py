# !pip install pca

import pandas as pd
from pca import pca

df = pd.read_csv("https://raw.githubusercontent.com/allatambov/LinStat25/refs/heads/main/city24.csv")
df = df.dropna()
df.head()

data = df.select_dtypes([float, int])
data = data.drop(columns = ["Unnamed: 0", "Happiness_Score"])

pc = pca(normalize = True)
res = pc.fit_transform(data, col_labels = data.columns)
res

print(res["variance_ratio"])
print(res["explained_var"])

pc.plot();

res["loadings"].T

pc.biplot();
pc.biplot(cmap = None);
pc.biplot(density = True);
# https://socr.umich.edu/HTML5/BivariateNormal/

df["Index"] = -res["PC"]["PC1"]
df.sort_values("Index", ascending = False).head(5)
df.sort_values("Index", ascending = False).tail(5)
df[["Index", "Happiness_Score"]].corr()


