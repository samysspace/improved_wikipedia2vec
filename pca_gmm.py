import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.manifold import TSNE

train_cat_graph = np.loadtxt(f'data_train.csv', delimiter=',')
test_cat_graph = np.loadtxt(f'data_test.csv', delimiter=',')

def pca(cat_graph, k=500):
	pca = PCA(n_components=k)
	pca.fit(cat_graph)
	print(f"Explained Variance Ratio: {sum(pca.explained_variance_ratio)}")
	embeddings = pca.transform(cat_graph)
	np.savetxt('PCA_MODEL_FILE', embeddings, delimiter=',')
	return embeddings

def gmm_log_likelihood_score(estimator, X):
    return estimator.score(X)

def gmm_fit(train_cat_graph, test_cat_graph):
	# This function takes an eternity!
	param_grid = {
    	"n_components": range(1, 25001),
	}
	grid_search = GridSearchCV(
    	GaussianMixture(), param_grid=param_grid, scoring=gmm_log_likelihood_score
	)
	grid_search.fit(train_cat_graph)
	test_log = grid_search.score(test_cat_graph)

	maxdex = test_log.index(max(test_log))
	df = pd.DataFrame(grid_search.cv_results_)[["param_n_components", "mean_test_score"]]
	plt.plot(df["param_n_components"], df["mean_test_score"], label='Training')
	plt.plot(df["param_n_components"], test_log, label='Test')
	plt.xlabel("Number of clusters")
	plt.ylabel("Per-sample Log-Likelihood")
	plt.title("Clusters vs Log-Likelihood of Data")
	plt.axvline(x=maxdex, color='r', linestyle='--')
	plt.text(14800, -300, f'x={maxdex}')
	plt.text(14800, -350, 'minimizes')
	plt.text(14800, -400, 'test error')
	plt.legend()
	plt.tight_layout()
	plt.savefig("cluster-mize.png")

	return maxdex

def gmm_embeddings(number_clusters, cat_graph):
	model = GaussianMixture(n_components=number_clusters)
	model.fit(cat_graph)
	predictions = model.predict(cat_graph)
	embeddings_dict = {}
	for idx, pred in enumerate(predictions):
		embeddings_dict[pred] = embeddings_dict.get(pred, []) + [idx]
	cluster_embeddings = []
	for i in range(number_clusters):
		centroid_so_far = np.array([0 for _ in range(len(cat_graph[0]))])
		for row in embeddings_dict[i]:
			centroid_so_far = centroid_so_far + cat_graph[row]
		centroid_so_far = centroid_so_far/len(embeddings_dict[i])
		cluster_embeddings.append(centroid_so_far)
	embeddings = []
	for pred in enumerate(predictions):
		embeddings.append(cluster_embeddings[pred])

	embeddings = np.array(embeddings)
	np.savetxt('GMM_MODEL_FILE', embeddings, delimiter=',')
	return embeddings

def nearest_neighbors(embeddings_file, version, entity='United_States', up_to=5)
	embeddings = np.load_txt(embeddings_file, delimiter=',')
	embeddings = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(embeddings)
    article_list = open("final_article_list.txt", "r")
    article_list = article_list.readlines()
    idx = article_list.index(entity)
    target_embed = embeddings[idx]
    cos_sim = [np.dot(target_embed, embedding)/(np.linalg.norm(target_embed) * np.linalg.norm(embedding)) for embedding in embeddings]
    indices = sorted(range(len(cos_sim)), key=lambda x: cos_sim[x])[-1*(1+up_to):-1]
    articles = [article_list[index] for index in indices]

    fig, ax = plt.subplots(figsize=(4,4))
    top_embeddings = [embeddings[index] for index in indices]
	x = [x[0] for x in top_embeddings]
	y = [x[1] for x in top_embeddings]
	plt.scatter(x, y)

	for i, txt in enumerate(articles):
		ax.annotate(txt, (x[i]-0.25, y[i]-0.05))

	ax = plt.gca()

	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	plt.title(f"USA Nearest Neighbors {version}")
	plt.savefig(f"{version}.png")



