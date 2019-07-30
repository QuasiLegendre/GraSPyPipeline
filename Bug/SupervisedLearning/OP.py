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
		super(OmnibusPipeline, self).__init__(steps=learning_method, memory=None, verbose=verbose, plot_method=plot_method, kfold=kfold)
		if self.steps[0][0] != 'Omni':
			self.steps =[('Omni', OmnibusEmbed()), ('Flat', FunctionTransformer(lambda x: x.reshape((x.shape[0], -1)), validate=False))] + self.steps
		if plot_method is not None:
			self.plot = plot_method
		if kfold is not None:
			self.kfold = kfold
	def cross_val_score(self, dataset, labels):
		test_results = []
		dataset, labels = np.array(dataset), np.array(labels)
		for train_index, test_index in self.kfold.split(dataset):
			dataset_train, dataset_test = dataset[train_index], dataset[test_index]
			labels_train, labels_test = labels[train_index], labels[test_index]
			self.fit(dataset_train, labels_train)
			#print(self.predict(dataset_test[0]))
			#self.set_params(Flat__func=lambda x: x.reshape(x.shape[0], -1), validate=False)
			print(dataset_test.shape)
			print(labels_test.shape)
			test_results.append(self.score(dataset_test, labels_test))
		avg_score = sum(test_results)/len(test_results)
		return avg_score, test_results
