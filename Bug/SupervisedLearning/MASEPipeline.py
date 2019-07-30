import numpy as np
from sklearn.model_selection import KFold
from graspy.embed import MultipleASE
from .base import SupervisedLearningPipeline

class MASEPipeline(SupervisedLearningPipeline):
	def __init__(self, 
				learning_method, 
				memory=None, 
				verbose=False, 
				plot_method=None, 
				kfold = KFold(n_splits=5)
				):
		super(MASEPipeline, self).__init__(steps=[('MASE', MultipleASE())]+learning_method, memory=memory, verbose=verbose, plot_method=plot_method, kfold=kfold)
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
