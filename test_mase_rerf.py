from SupervisedLearning import MASEPipeline
from sklearn.decomposition import PCA
from rerf.rerfClassifier import rerfClassifier
#Testing
if __name__ == '__main__':
	import numpy as np
	from graspy.simulations import sbm
	from graspy.plot import heatmap, pairplot
	n_verts = 100
	nums = 32
	p1 = 0.8
	p2 = 0.81
	labels_sbm = nums * [0] + nums * [1]
	P1 = np.array([[p1, 1.0-p1], [1.0-p1, p1]])
	P2 = np.array([[p2, 1.0-p2], [1.0-p2, p2]])
	undirected_sbms = []
	for i in range(nums):
	    undirected_sbms.append(sbm(2 * [n_verts], P1))
	for i in range(nums):
		undirected_sbms.append(sbm(2 * [n_verts], P2))
	G = np.array(undirected_sbms)
	print(G.shape)
	def plotRerF(clf, X, y):
		import matplotlib
		import matplotlib.pyplot as plt
		#from mpl_toolkits.mplot3d import Axes3D
		#from sklearn.model_selection import GridSearchCV
		x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
		y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

		matplotlib.rc('figure', figsize=[12, 8], dpi=300)
		plt.figure(figsize=(9, 4))
		plt.clf()

		# Plot the training points
		plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')
		plt.xlabel('Dimension 1')
		plt.ylabel('Dimension 2')

		plt.xlim(x_min, x_max)
		plt.ylim(y_min, y_max)

		plt.show()
	MASEP = MASEPipeline([('pca', PCA(n_components=4)), ('rerf', rerfClassifier(n_estimators=10, max_depth=2))], plot_method=plotREF)
	MASEP.set_params(MASE__n_components=6, MASE__algorithm='full')
	MASEP.fit(undirected_sbms, labels_sbm)
	print(MASEP.predict(undirected_sbms))
	cvs, _ = MASEP.cross_val_score(undirected_sbms, labels_sbm)
	print(cvs)
	
