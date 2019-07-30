import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import FunctionTransformer
from graspy.embed import OmnibusEmbed
from .base import SupervisedLearningPipeline

class OmnibusPipeline(SupervisedLearningPipeline):
	def __init__(self, 
				learning_method, 
				memory=None, 
				verbose=False, 
				plot_method=None, 
				kfold = KFold(n_splits=4, shuffle=True)
				):
		super(OmnibusPipeline, self).__init__(steps=[('Omni', OmnibusEmbed()), ('Flat', FunctionTransformer(lambda x: x.reshape((x.shape[0], -1)), validate=False))], memory=memory, verbose=verbose, plot_method=plot_method, kfold=kfold)
		if plot_method is not None:
			self.plot = plot_method
		if kfold is not None:
			self.kfold = kfold
		self.LM = learning_method[0][1]
	def cross_val_score(self, dataset, labels):
		test_results = []
		dataset, labels = np.array(dataset), np.array(labels)
		for train_index, test_index in self.kfold.split(dataset):
			dataset_train, dataset_test = dataset[train_index], dataset[test_index]
			labels_train, labels_test = labels[train_index], labels[test_index]
			self.LM.fit(self.fit_transform(dataset_train), labels_train)
			test_results.append(self.LM.score(self.fit_transform(dataset_test), labels_test))
		avg_score = sum(test_results)/len(test_results)
		return avg_score, test_results
