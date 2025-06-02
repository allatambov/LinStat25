import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score

df = pd.read_csv("https://raw.githubusercontent.com/allatambov/LinStat25/refs/heads/main/flats_cian_upd.csv")
df.head()

# aggregate
df_agg = df.groupby("station")[["price", "square", "rooms", 
                                "floor", "dmetro"]].mean()
df_agg.head()

# scale
scaler = StandardScaler()
X = scaler.fit_transform(small)

# cluster with Ward
hc = linkage(X, method = "ward")
dendrogram(hc);
clusters_ward = cut_tree(hc, n_clusters = 3).reshape(-1, )
print(clusters_ward)

# cluster with Kmeans
kmeans = KMeans(n_clusters=3, random_state=1234).fit(X)
print(kmeans.cluster_centers_)

clusters_kmeans = kmeans.labels_
print(clusters_kmeans)

# add all labels and compare
small["ward"] =  clusters_ward
small["kmeans"] = clusters_kmeans
small.head()

cluster0_ward = small[small["ward"] == 0]
cluster1_ward = small[small["ward"] == 1]
cluster2_ward = small[small["ward"] == 2]

print("0", sorted(cluster0_ward.index))
print("1", sorted(cluster1_ward.index))
print("2", sorted(cluster2_ward.index))

cluster0_kmeans = small[small["kmeans"] == 0]
cluster1_kmeans = small[small["kmeans"] == 1]
cluster2_kmeans = small[small["kmeans"] == 2]

print("0", sorted(cluster0_kmeans.index))
print("1", sorted(cluster1_kmeans.index))
print("2", sorted(cluster2_kmeans.index))

# compare sets
set0_ward = set(cluster0_ward.index)
set1_ward = set(cluster1_ward.index)
set2_ward = set(cluster2_ward.index)

set0_kmeans = set(cluster0_kmeans.index)
set1_kmeans = set(cluster1_kmeans.index)
set2_kmeans = set(cluster2_kmeans.index)

print(len(set0_ward.intersection(set0_kmeans)))
print(len(set0_ward.intersection(set1_kmeans)))
print(len(set0_ward.intersection(set2_kmeans)))

print(set0_ward.difference(set0_kmeans))
print(set0_kmeans.difference(set0_ward))

adjusted_rand_score(clusters_kmeans, clusters_ward)
