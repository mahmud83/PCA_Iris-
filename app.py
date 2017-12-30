import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# load dataset into Pandas DataFrame
df = pd.read_csv(url, names=['sepal length', 'sepal width',
                             'petal length', 'petal width', 'target'])


features = ['sepal length', 'sepal width', 'petal length', 'petal width']

# Seperate out features
x = df.loc[:, features].values

#seperate out the target
y = df.loc[:, ["target"]].values


# standardize the features
x = StandardScaler().fit_transform(x)

# PCA projection
pca = PCA(n_components=2)
principal_components = pca.fit_transform(x)
principal_df = pd.DataFrame(data=principal_components, columns=["PC 1", "PC 2"])
final_df = pd.concat([principal_df, df[['target']]], axis=1)


# Plot 
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_title('2 component PCA', fontsize=20)

targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']

for target, color in zip(targets, colors):
    indices_to_keep = final_df["target"] == target
    ax.scatter(final_df.loc[indices_to_keep, 'PC 1'],
               final_df.loc[indices_to_keep, 'PC 2'], c=color, s=50)

ax.legend(targets)
ax.grid()
plt.show()
