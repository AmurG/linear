import numpy as np
import networkx as nx

#algo1 : random sampling from a datamatrix -> graph -> tree, no prior
#no prior corresponds to uniform sampling which is proven to work ( see : Ghoshdastidar and dukkipatti )
#A reasonable prior obtained by previous experience ( such as in CV ) or unsupervised k-means can be used for better perf

def randomgraph(datamat, nsam, nvar):
	order = np.random.permutation(nvar)
	edgnum = int(np.rint((np.log(nvar))*(np.log(nvar))+1))
	G = nx.Graph()
	for i in range(0,nvar):
		G.add_node(i)
	for idx in range(0,nvar):
		curr = order[idx]
		for i in range(0,edgnum):
			rand = np.random.randint(0,nvar)
			count = 0
			while((rand==curr)or(G.has_edge(rand,curr))or(G.has_edge(curr,rand))):
				rand = np.random.randint(0,nvar)
				count = count+1
				if(count>nvar/2):
					break
			G.add_edge(curr,rand,weight=abs(np.corrcoef(datamat[:,curr],datamat[:,rand])))
	G = G.to_undirected()
	return G

#test

ab = np.genfromtxt('../AB.dat',delimiter=",")
ab = ab[:,1:]

X=randomgraph(ab,len(ab),8)
print(X.nodes())



