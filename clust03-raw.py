import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree
from sklearn.preprocessing import MinMaxScaler

coffee = pd.read_csv("coffee_and_code.csv")
coffee.head()

# Solution 01: recoding, minmax scaling and cityblock distance

coffee["Male"] = (coffee["Gender"] == "Male").astype(int)
coffee["WithoutCoffee"] = coffee["CodingWithoutCoffee"].map({"Yes" : 2,  "Sometimes" : 1, "No" : 0})
add = pd.get_dummies(coffee["CoffeeType"])

chosen = coffee[["CodingHours", "CoffeeCupsPerDay", "Male", "WithoutCoffee"]]
to_clust = pd.concat([chosen, add], axis = 1)

scaler = MinMaxScaler()
X = scaler.fit_transform(to_clust)

hc0 = linkage(X, method = "complete", metric = "cityblock")
dendrogram(hc0);

labels0 = cut_tree(hc0, n_clusters = 5).reshape(-1, )
to_clust["group_cb"] = labels0

to_clust.groupby("group_cb")["CodingHours"].describe()
to_clust.groupby("group_cb")["CoffeeCupsPerDay"].describe()
to_clust.groupby("group_cb")["Male"].describe()
to_clust.groupby("group_cb")["WithoutCoffee"].describe()
to_clust.groupby("group_cb")[list(add.columns)].mean()

# Solution 02: Gower distance for mixed-type data

!pip install gower
import gower

data = coffee[["CodingHours", "CoffeeCupsPerDay", "Gender", 
              "CodingWithoutCoffee", "CoffeeType"]]
D = gower.gower_matrix(data)

hc1 = linkage(D, method = 'complete')
dendrogram(hc1);
labels1 = cut_tree(hc1, n_clusters = 5).reshape(-1, )
to_clust["group_gr"] = labels1

print("Coding Hours\n", to_clust.groupby("group_gr")["CodingHours"].describe())
print("Coffee Cups\n", to_clust.groupby("group_gr")["CoffeeCupsPerDay"].describe())
print("Male\n", to_clust.groupby("group_gr")["Male"].describe())
print("Without Coffee\n", to_clust.groupby("group_gr")["WithoutCoffee"].describe())

to_clust.groupby("group_gr")[list(add.columns)].mean()

