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
	MASEP = MASEPipeline([('pca', PCA(n_components=4)), ('rerf', rerfClassifier(n_estimators=10, max_depth=2))])
	MASEP.set_params(MASE__n_components=6, MASE__algorithm='full')
	MASEP.fit(undirected_sbms, labels_sbm)
	cvs, _ = MASEP.cross_val_score(undirected_sbms, labels_sbm)
	print(cvs)
	
