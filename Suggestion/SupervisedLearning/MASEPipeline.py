import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import PCA
from graspy.embed import MASEClassifier
from .base import SupervisedLearningPipeline

def PCASeq(l):
	ll = []
	pca = PCA(n_components=4)
	for i in range(l.shape[0]):
		ll.append(pca.fit(l[i]).singular_values_)
	return np.array(ll)
def Flat(l):
	return l.reshape(l.shape[0], -1)
class MASEPipeline(SupervisedLearningPipeline):
	def __init__(self, 
				learning_method, 
				memory=None, 
				verbose=False, 
				plot_method=None, 
				kfold = KFold(n_splits=32)
				):
		super(MASEPipeline, self).__init__(steps=learning_method, memory=memory, verbose=verbose, plot_method=plot_method, kfold=kfold)
		self.LM = learning_method[0][1]
		if not isinstance(self.steps[0][1], MASEClassifier):
			self.steps = [('MASE', MASEClassifier()), ('Flat', FunctionTransformer(PCASeq, validate=False))] + self.steps
		if plot_method is not None:
			self.plot = plot_method
		if kfold is not None:
			self.kfold = kfold
	def get_scores(self):
		return self['MASE'].get_scores()
	def cross_val_score(self, dataset, labels):
		test_results = []
		dataset, labels = np.array(dataset), np.array(labels)
		for train_index, test_index in self.kfold.split(dataset):
			dataset_train, dataset_test = dataset[train_index], dataset[test_index]
			labels_train, labels_test = labels[train_index], labels[test_index]
			#self.MF = self[:-1]
			#self.LM.fit(self.MF.fit_transform(dataset_train), labels_train)
			#test_results.append(self.LM.score(self.MF.fit_transform(dataset_test), labels_test))
			self.fit(dataset_train, labels_train)
			test_results.append(self.score(dataset_test, labels_test))
		avg_score = sum(test_results)/len(test_results)
		return avg_score, test_results
