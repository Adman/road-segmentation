import umap
import matplotlib.pyplot as plt

from data_generator import load_data_memory


X, _ = load_data_memory(['data/datasets/deggendorf'], 'image', 'masks', aug=False)
# reshape to vectors
X = X.reshape(344, 921600)

embedding = umap.UMAP(metric='euclidean').fit_transform(X)

plt.scatter(embedding[:, 0], embedding[:, 1])
plt.gca().set_aspect('equal', 'datalim')
plt.savefig('plots_other/dataset_embedding.png')
