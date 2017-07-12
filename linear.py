import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from scipy.spatial import distance

#algo1 : random sampling from a datamatrix -> graph -> tree, no prior
#no prior corresponds to uniform sampling which is proven to work ( see : Ghoshdastidar and dukkipatti )
#A reasonable prior obtained by previous experience ( such as in CV ) or unsupervised k-means can be used for better perf

def helper(arr1,arr2):
	mat = np.corrcoef(arr1,arr2)
	return mat[0][1]

def randomgraph(datamat, nsam, nvar):
	order = np.random.permutation(nvar)
	edgnum = int(np.rint(np.log(nvar)*np.log(nvar)+1))
	G = nx.Graph()
	for i in range(0,nvar):
		G.add_node(i)
	for idx in range(0,nvar):
		curr = order[idx]
		for i in range(0,edgnum):
			rand = np.random.randint(0,nvar)
			while((rand==curr)or(G.has_edge(rand,curr))or(G.has_edge(curr,rand))):
				rand = np.random.randint(0,nvar)
			G.add_edge(curr,rand,weight=-abs(helper(datamat[:,curr],datamat[:,rand])))
	G = G.to_undirected()
	return (nx.minimum_spanning_tree(G).to_undirected())

#test

ab = np.genfromtxt('../SD.txt',delimiter=" ")
ab = ab[:,:48]

X=randomgraph(ab,len(ab),48)
labels = (X.edges(data='weight'))
print((labels))

#algo2 : Prior being that "nearby" stuff should be prioritized, along with a check across symmetry axes
#alpha = number of proxchecks, 1-alpha = symchecks
#for images, pass imagewidth and imageheight as params
#assume that the arg datamat is a matrix of axes nsam, ht*wd and indexing is row-prioritized

def randnear(idx, ht, wd):
	vdown = idx/wd
	hright = idx%wd
	idx1 = int(np.rint(np.random.normal))
	idx2 = int(np.rint(np.random.normal))	
	while(((idx1+vdown)>=ht)or(idx1<vdown):
		idx1 = int(np.rint(np.random.normal))
	while(((idx2+hright)>=wd)or(idx2<hright):
		idx2 = int(np.rint(np.random.normal))
	return(wd*(vdown+idx1)+(idx2+hright))

def symnear(idx,ht,wd):
	vdown = idx/wd
	hright = idx%wd
	refvdown = ht - 1 - vdown
	refhright = wd - 1 - hright
	return (randnear(wd*(refvdown)+refhright,ht,wd))


def proximitygraph(datamat, nsam, ht, wd, alpha = 0.8):
	order = np.random.permutation(nvar)
	nvar = ht*wd
	edgnum1 = int(alpha*np.rint(np.log(nvar)*np.log(nvar)+1))
	edgnum2 = int((1-alpha)*np.rint(np.log(nvar)*np.log(nvar)+1))
	order = np.random.permutation(nvar)
	G = nx.graph()
	for i in range(0,nvar):
		G.add_node(i)
	for idx in range(0,nvar):
		curr = order[idx]
		for i in range(0,edgnum1):
			rand = randnear(curr,ht,wd)	
			while((rand==curr)or(G.has_edge(rand,curr))or(G.has_edge(curr,rand))):
				rand = randnear(curr,ht,wd)
			G.add_edge(curr,rand,weight=-abs(helper(datamat[:,curr],datamat[:,rand])))
		for i in range(0,edgnum2):
			rand = symnear(curr,ht,wd)	
			while((rand==curr)or(G.has_edge(rand,curr))or(G.has_edge(curr,rand))):
				rand = symnear(curr,ht,wd)
			G.add_edge(curr,rand,weight=-abs(helper(datamat[:,curr],datamat[:,rand])))

	G = G.to_undirected()
	return (nx.minimum_spanning_tree(G).to_undirected())

#algo3 : use arbit prior. The prior is compressed so that it's unspecified and does not incur quadratic cost.
#To use this algo, specify a prior in the form of func(chosenidx,nsam) that outputs another idx in the form P(newidx|chosenidx)

def genericgraph(datamat, nsam, nvar, prior):
	order = np.random.permutation(nvar)
	edgnum = int(np.rint(np.log(nvar)*np.log(nvar)+1))
	G = nx.Graph()
	for i in range(0,nvar):
		G.add_node(i)
	for idx in range(0,nvar):
		curr = order[idx]
		for i in range(0,edgnum):
			rand = prior(idx, nsam)
			while((rand==curr)or(G.has_edge(rand,curr))or(G.has_edge(curr,rand))):
				rand = prior(idx, nsam)
			G.add_edge(curr,rand,weight=-abs(helper(datamat[:,curr],datamat[:,rand])))
	G = G.to_undirected()
	return (nx.minimum_spanning_tree(G).to_undirected())

#algo4 : Use unsupervised k-means to pre-cluster and obtain a prior on the variables to sample edges

def kfoldaffinity(datamat,k,nsam,nvar):
	kmeans = KMeans(n_clusters=k, random_state=0).fit(np.transpose(datamat))
	arr = kmeans.cluster_centers_
	labels = kmeans.labels_
	mat = np.zeros(k*k)
	mat = np.reshape(mat,(k,k))
	sums = np.zeros(k)
	for i in range(0,k):
		for j in range(0,k):	
			mat[i][j] = np.exp(-distance.euclidean(arr[i],arr[j]))
			sums[i] = sums[i] + mat[i][j]
	edgnum = int(np.rint(np.log(nvar)*np.log(nvar)+1))
	G = nx.Graph()
	for i in range(0,nvar):
		G.add_node(i)
	for idx in range(0,nvar):
		label = int(labels[idx])
		for i in range(0,edgnum):
			draw = np.random.uniform(high=sums[label])
			for j in range(0,k):
				draw = draw - mat[label][j]
				if(draw<=0):
					pick = j
					break
			rand = np.random.randint(0,nvar)
			while((rand==idx)or(G.has_edge(rand,idx))or(G.has_edge(idx,rand))or(labels[rand]!=pick)):
				rand = np.random.randint(0,nvar)
			G.add_edge(idx,rand,weight=-abs(helper(datamat[:,idx],datamat[:,rand])))
	G = G.to_undirected()
	return (nx.minimum_spanning_tree(G).to_undirected())
				
				



			


