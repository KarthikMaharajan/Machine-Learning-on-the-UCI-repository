import graphlab as gl
import numpy as np

def getdistance(src, edge, dst):
	edge['distance'] = np.linalg.norm(np.subtract(src['features'],dst['features']))
	if(src['calculatedLabel'] == -1):
		if(max(src['initialDistance']) > edge['distance']):
			src['label'][src['initialDistance'].index(max(src['initialDistance']))] = dst['label'][0]
			src['initialDistance'][src['initialDistance'].index(max(src['initialDistance']))] = edge['distance']
	else:
		if(max(dst['initialDistance']) > edge['distance']):
			dst['label'][dst['initialDistance'].index(max(dst['initialDistance']))] = src['label'][0]
			dst['initialDistance'][dst['initialDistance'].index(max(dst['initialDistance']))] = edge['distance']
	return(src,edge,dst)

def populate(featurefile, testfile, K):
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

		d['label'] = [int(label)]
		d['features'] = features
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

		d['label'] = [-1] * K
		d['features'] = np.array(features)
		d['calculatedLabel'] = -1
		d['initialDistance'] = [10000000] * K

		vertex.append(gl.Vertex(0, attr=d))
		Graph = Graph.add_vertices(vertex)

		Graph = Graph.triple_apply(getdistance, mutated_fields=['distance','label','calculatedLabel','initialDistance'],input_fields=None)

		c = [0] * 10
		for i in range(0, K):
			c[int(Graph.get_vertices(0)['label'][0][i])] = c[int(Graph.get_vertices(0)['label'][0][i])] + 1;

		newLabel = c.index(max(c));

		if label != newLabel:
			print label,newLabel
			error = error + 1
		else:
			print "same!"

	return error

error = populate('optdigits.tra', 'optdigits.tes', 1)

print error