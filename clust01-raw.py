import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree

### LOAD DATA ###

df = pd.read_csv("https://raw.githubusercontent.com/allatambov/LinStat25/refs/heads/main/flats_cian_upd.csv")
print(df.shape)
df.head()
df.describe()

# choose some rows - to save time
tab = df["station"].value_counts()
stations = tab[tab > 30].index

chosen = df[df["station"].isin(stations)]
flats = chosen.groupby("station").sample(30, random_state = 1234)
print(flats.shape)
flats.head()

### PREPARE DATA ###

small = flats[["lprice", "square", "rooms", "floor", "dmetro"]]
small.head()

scaler = StandardScaler()
X = scaler.fit_transform(small)

### CLUSTERING ###

# try different methods
hc = linkage(X)

# dendrogram
plt.figure(figsize = (16, 9))
dendrogram(hc);

# cut dendrogram and get groups
clusters_ = cut_tree(hc, n_clusters = 3).reshape(-1, )
flats["cluster_c"] = clusters_.astype(str)

### EXPLORE RESULTS ###

print(flats.groupby("cluster_c")["price"].agg(["count", "mean", "min", "max"]))
print(flats.groupby("cluster_c")["square"].agg(["count", "mean", "min", "max"]))
print(flats.groupby("cluster_c")["floor"].agg(["count", "mean", "min", "max"]))

# graphs & tests

### GEO DATA ###

!pip install geopandas
import geopandas as gpd

df_geo = gpd.read_file("Москва_Moscow.geojson")
df_geo.boundary.plot();

out = ["Троицкий административный округ", "район Кунцево", "район Силино", 
       "район Старое Крюково", "район Крюково",
       "район Матушкино", "район Савелки"]

df_geo = df_geo[~df_geo["district"].isin(out)]

fig, axes = plt.subplots(figsize = (50, 50))

df_geo.boundary.plot(ax = axes);
sns.scatterplot(data = flats, 
                x = "lon", y = "lat", 
                hue = "cluster_c", 
                s = 120, ax = axes);
