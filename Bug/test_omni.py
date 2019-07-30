from SupervisedLearning import OmnibusPipeline

#Testing
if __name__ == '__main__':
	import numpy as np
	import graspy
	import matplotlib.pyplot as plt
	import numpy as np
	from sklearn.svm import SVC
	from graspy.simulations import sbm
	nums = 32
	n = [25, 25]
	P1 = [[.3, .1],
		  [.1, .7]]
	P2 = [[.3, .1],
		  [.1, .3]]
	labels = [1]*nums + [2]*nums
	#labels = np.matrix([labels])
	#labels = labels.transpose(1, 0)
	np.random.seed(8)
	Gs = []
	for i in range(nums):
		Gs.append(sbm(n, P1))
	for i in range(nums):
		Gs.append(sbm(n, P2))
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
	OP = OmnibusPipeline(learning_method=[('svc', SVC(gamma='scale', kernel='linear'))])#.fit(Gs, labels)
	from graspy.embed import OmnibusEmbed
	from sklearn.preprocessing import FunctionTransformer
	OPT = OmnibusPipeline([('Omni', OmnibusEmbed()), ('Flat', FunctionTransformer(lambda x: x.reshape(x.shape[0], -1), validate=False))])
	L = OPT.fit_transform(Gs)
	sc = SVC(gamma='scale', kernel='linear')
	sc.fit(L, labels)
	print(sc.predict(L[30:34]))
	print('\n')
	print(L.shape)
	#OP.fit(Gs, labels)
	#print(OP.predict([1]))
	print(OP.cross_val_score(Gs, labels))
