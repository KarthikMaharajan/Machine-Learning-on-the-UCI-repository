import graphlab as gl
import numpy as np

def getdistance(src, edge, dst):
	edge['distance'] = np.linalg.norm(np.subtract(src['features'],dst['features']))
	if(src['calculatedLabel'] == -1):
		if(src['initialDistance'] > edge['distance']):
			src['label'] = dst['label']
			src['initialDistance'] = edge['distance']
	else:
		if(dst['initialDistance'] > edge['distance']):
			dst['label'] = src['label']
			dst['initialDistance'] = edge['distance']
	return(src,edge,dst)

def populate(featurefile, testfile):
	global currentDistance
	global currentLabel
	ffile = open(featurefile, 'r')
	tfile = open(testfile, 'r')

	Graph = gl.SGraph()

	vertices = []
	i = 0
	d = {}

	vertices.append(gl.Vertex(i, attr=d))

	for line in ffile:
		d = {}

		featureSet = line.split(',')
		label = featureSet[-1]
		features = []

		for item in featureSet[:-1]:
			features.append(float(item))

		features = np.array(features)

		d['label'] = int(label)
		d['features'] = features
		print features,d['features'],d['label']
		d['calculatedLabel'] = int(label)

		vertices.append(gl.Vertex(i, attr=d))
		i  = i + 1

	Graph = Graph.add_vertices(vertices)

	edges = []
	for j in range(1,i):
		edges.append(gl.Edge(j, 0, attr = {'distance':0}))

	Graph = Graph.add_edges(edges)

	print(Graph)

	error = 0
	for line in tfile:
		currentLabel = -1;
		currentDistance = 100000000;

		featureSet = line.split(',')
		label = int(featureSet[-1])
		features = []

		for item in featureSet[:-1]:
			features.append(float(item))
		
		vertex = []
		d = {}

		d['label'] = -1
		d['features'] = np.array(features)
		print features,d['features'],label,d['label']
		d['calculatedLabel'] = -1
		d['initialDistance'] = 10000000

		vertex.append(gl.Vertex(0, attr=d))
		Graph = Graph.add_vertices(vertex)

		Graph = Graph.triple_apply(getdistance, mutated_fields=['distance','label','calculatedLabel','initialDistance'],input_fields=None)

		if label != Graph.get_vertices(0)['label'][0]:
			print label,Graph.get_vertices(0)['label'][0]
			error = error + 1
		else:
			print "same!"

	return error

error = populate('/Users/annu/Desktop/mlProject/Breast-Cancer-train.csv', '/Users/annu/Desktop/mlProject/Breast-Cancer-test.csv')

print error