from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from abc import ABCMeta, abstractmethod
class SupervisedLearning(Pipeline, metaclass=ABCMeta):
	def __init__(self, 
				steps, 
				memory=None, 
				verbose=False, 
				plot_method=None, 
				kfold = KFold(n_splits=5)
				):
		super(SupervisedLearning, self).__init__(steps=steps, memory=memory, verbose=verbose)
		if plot_method is not None:
			self.plot = plot_method
		if kfold is not None:
			self.kfold = kfold
	@abstractmethod
	def cross_val_score(self, dataset, labels):
		pass
			
class MASELearning(SupervisedLearning):
	def __init__(self, 
				steps, 
				memory=None, 
				verbose=False, 
				plot_method=None, 
				kfold = KFold(n_splits=5)
				):
		super(MASELearning, self).__init__(steps=steps, memory=memory, verbose=verbose, plot_method=plot_method, kfold=kfold)
		if plot_method is not None:
			self.plot = plot_method
		if kfold is not None:
			self.kfold = kfold
	def cross_val_score(self, dataset, labels):
		test_results = []
		for train_index, test_index in self.kfold.split(dataset):
			dataset= np.array(dataset)
			dataset_train, dataset_test = dataset[train_index], dataset[test_index]
			self.fit(dataset_train, labels)
			test_results.append(self.score(dataset_test, labels))
		avg_score = sum(test_results)/len(test_results)
		return avg_score, test_results




#Testing	
if __name__ == '__main__':
	import numpy as np
	from graspy.embed import MultipleASE
	from graspy.simulations import sbm
	from graspy.plot import heatmap, pairplot
	n_verts = 100
	p = 0.8
	labels_sbm = n_verts * [0] + n_verts * [1]
	P = np.array([[p, 1.0-p], [1.0-p, p]])
	undirected_sbms = []
	for i in range(32):
	    undirected_sbms.append(sbm(2 * [n_verts], P))
	def plotSVC(Xhat, clf):	
		h = 0.0002
		x_min, x_max = Xhat[:, 0].min() - 0.01, Xhat[:, 0].max() + 0.01
		y_min, y_max = Xhat[:, 1].min() - 0.01, Xhat[:, 1].max() + 0.01
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
		Z = Z.reshape(xx.shape)
		plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
		# Plot also the training points
		plt.scatter(Xhat[:, 0], Xhat[:, 1], c=labels_sbm, cmap=plt.cm.coolwarm)
		plt.xlabel('Sepal length')
		plt.ylabel('Sepal width')
		plt.xlim(xx.min(), xx.max())
		plt.ylim(yy.min(), yy.max())
		plt.xticks(())
		plt.yticks(())
		plt.show()
	from sklearn.svm import SVC
	SVL = MASELearning([('mase', MultipleASE(n_components=2, algorithm='full')), ('svc', SVC(gamma='scale', kernel='linear'))], plot_method=plotSVC).fit(undirected_sbms, labels_sbm)
	SVL.plot(SVL['mase'].transform(undirected_sbms), SVL['svc'])
	cvs, cvl = SVL.cross_val_score(undirected_sbms, labels_sbm)
