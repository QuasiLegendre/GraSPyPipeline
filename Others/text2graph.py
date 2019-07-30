def Text2List(file_name):
	text_file = open(file_name, 'r')
	graph_list = list(map(lambda x: tuple(map(lambda x: int(x),x.split('\n')[0].split(' '))), text_file.readlines()))
	text_file.close()
	return graph_list
def List2Graph(graph_list):
	import networkx as nx
	graph = nx.Graph()
	graph.add_weighted_edges_from(graph_list)
	return graph
def Graph2Matrix(graph):
	import networkx as nx
	import numpy as np
	mat = nx.convert_matrix.to_numpy_array(graph)
	return mat
	
	
if __name__ == '__main__':
	name = 'Graph.ssv'
	M = Graph2Matrix(List2Graph(Text2List(name)))
	print(M[21][32])
	GM = open('GraphMatrix.txt', 'w')
	GM.write(str(M))
	GM.close()
	
