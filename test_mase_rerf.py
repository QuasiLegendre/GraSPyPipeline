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
		h = 0.001
		x_min, x_max = X[:, 0].min() - 0.01, X[:, 0].max() + 0.01
		y_min, y_max = X[:, 1].min() - 0.01, X[:, 1].max() + 0.01
		xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
							 np.arange(y_min, y_max, h))
		import matplotlib
		matplotlib.use('QT5Agg')
		import matplotlib.pyplot as plt
		# Plot the decision boundary. For that, we will assign a color to each
		# point in the mesh [x_min, x_max]x[y_min, y_max].
		plt.subplots(figsize=(10, 10))
		Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
		# Put the result into a color plot
		Z = np.array(Z)
		Z = Z.reshape(xx.shape)
		plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
		# Plot also the training points
		plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolor='k')
		plt.xlabel('Dimension 1')
		plt.ylabel('Dimension 2')
		plt.xlim(xx.min(), xx.max())
		plt.ylim(yy.min(), yy.max())
		plt.xticks(())
		plt.yticks(())
		plt.show()
	MASEP = MASEPipeline([('pca', PCA(n_components=4)), ('rerf', rerfClassifier(n_estimators=10, max_depth=2))], plot_method=plotRerF)
	MASEP.set_params(MASE__n_components=6, MASE__algorithm='full')
	MASEP.fit(undirected_sbms, labels_sbm)
	print(type(MASEP.predict(undirected_sbms)))
	cvs, _ = MASEP.cross_val_score(undirected_sbms, labels_sbm)
	print(cvs)
	MASEP.plot(undirected_sbms, labels_sbm)
	
