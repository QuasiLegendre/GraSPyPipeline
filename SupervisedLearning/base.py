from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from abc import ABCMeta, abstractmethod

class SupervisedLearningPipeline(Pipeline, metaclass=ABCMeta):
	def __init__(self, 
				steps, 
				memory=None, 
				verbose=False, 
				plot_method=None, 
				kfold = KFold(n_splits=5)
				):
		super(SupervisedLearningPipeline, self).__init__(steps, memory, verbose=verbose)
		if plot_method is not None:
			self.plot = plot_method
		if kfold is not None:
			self.kfold = kfold
	@abstractmethod
	def cross_val_score(self, dataset, labels):
		pass
