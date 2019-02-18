import networkx as nx
import numpy as np
import itertools
import sys
import os
import random
import igraph as ig
from networkx.algorithms import community
import networkx.algorithms.isolate
import community as cm
import math
import random
import argparse


def check_consensus_graph(G, n_p, delta):
	'''
	This function checks if the networkx graph has converged. 
	Input:
	G: networkx graph
	n_p: number of partitions while creating G
	delta: if more than delta fraction of the edges have weight != n_p then returns False, else True
	'''



	count = 0
	
	for wt in nx.get_edge_attributes(G, 'weight').values():
		if wt != 0 and wt != n_p:
			count += 1

	if count > delta*G.number_of_edges():
		return False

	return True



def nx_to_igraph(Gnx):
	'''
	Function takes in a network Graph, Gnx and returns the equivalent
	igraph graph g
	'''
	g = ig.Graph()
	g.add_vertices(sorted(Gnx.nodes()))
	g.add_edges(sorted(Gnx.edges()))
	g.es["weight"] = 1.0
	for edge in Gnx.edges():
		g[edge[0], edge[1]] = Gnx[edge[0]][edge[1]]['weight']
	return g


def group_to_partition(partition):
	'''
	Takes in a partition, dictionary in the format {node: community_membership}
	Returns a nested list of communities [[comm1], [comm2], ...... [comm_n]]
	'''

	part_dict = {}

	for index, value in partition.items():

		if value in part_dict:
			part_dict[value].append(index)
		else:
			part_dict[value] = [index]


	return part_dict.values()

def check_arguments(args):

	if(args.d > 0.2):
		print('delta is too high. Allowed values are between 0.02 and 0.2')
		return False
	if(args.d < 0.02):
		print('delta is too low. Allowed values are between 0.02 and 0.2')
		return False
	if(args.alg not in ('louvain', 'lpm', 'cnm', 'infomap')):
		print('Incorrect algorithm entered. run with -h for help')
		return False
	if (args.t > 1 or args.t < 0):
		print('Incorrect tau. run with -h for help')
		return False

	return True


def fast_consensus(G,  algorithm = 'louvain', n_p = 20, thresh = 0.2, delta = 0.02):

	graph = G.copy()
	L = G.number_of_edges()
	N = G.number_of_nodes()

	for u,v in graph.edges():
		graph[u][v]['weight'] = 1.0

	while(True):


		if (algorithm == 'louvain'):

			nextgraph = graph.copy()
			L = G.number_of_edges()
			for u,v in nextgraph.edges():
				nextgraph[u][v]['weight'] = 0.0

			communities_all = [cm.partition_at_level(cm.generate_dendrogram(graph, randomize = True, weight = 'weight'), 0) for i in range(n_p)]

			for node,nbr in graph.edges():
						
				if (node,nbr) in graph.edges() or (nbr, node) in graph.edges():
					if graph[node][nbr]['weight'] not in (0,n_p):
						for i in range(n_p):
							communities = communities_all[i]
							if communities[node] == communities[nbr]:
								nextgraph[node][nbr]['weight'] += 1
							else:
								nextgraph[node][nbr]['weight'] = graph[node][nbr]['weight']



			remove_edges = []
			for u,v in nextgraph.edges():
				if nextgraph[u][v]['weight'] < thresh*n_p:
					remove_edges.append((u, v))

			nextgraph.remove_edges_from(remove_edges)



			if check_consensus_graph(nextgraph, n_p = n_p, delta = delta):
				break




			for _ in range(L):

				node = np.random.choice(nextgraph.nodes())
				neighbors = [a[1] for a in nextgraph.edges(node)]

				if (len(neighbors) >= 2):
					a, b = random.sample(set(neighbors), 2)

					if not nextgraph.has_edge(a, b):
						nextgraph.add_edge(a, b, weight = 0)

						for i in range(n_p):
							communities = communities_all[i]
						
							if communities[a] == communities[b]:
								nextgraph[a][b]['weight'] += 1


			for node in nx.isolates(nextgraph):
					nbr, weight = sorted(graph[node].items(), key=lambda edge: edge[1]['weight'])[0]
					nextgraph.add_edge(node, nbr, weight = weight['weight'])
				
				
			graph = nextgraph.copy()


			if check_consensus_graph(nextgraph, n_p = n_p, delta = delta):
				break

		elif (algorithm in ('infomap', 'lpm')):

			nextgraph = graph.copy()
			
			for u,v in nextgraph.edges():
				nextgraph[u][v]['weight'] = 0.0

			if algorithm == 'infomap':
				communities = [{frozenset(c) for c in nx_to_igraph(graph).community_infomap().as_cover()} for _ in range(n_p)]
			if algorithm == 'lpm':
				communities = [{frozenset(c) for c in nx_to_igraph(graph).community_label_propagation().as_cover()} for _ in range(n_p)]


			for node, nbr in graph.edges():

				for i in range(n_p):
					for c in communities[i]:
						if node in c and nbr in c:
							if not nextgraph.has_edge(node,nbr):
								nextgraph.add_edge(node, nbr, weight = 0)
							nextgraph[node][nbr]['weight'] += 1



			remove_edges = []
			for u,v in nextgraph.edges():
				if nextgraph[u][v]['weight'] < thresh*n_p:
					remove_edges.append((u, v))
			nextgraph.remove_edges_from(remove_edges)



			for _ in range(L):
				node = np.random.choice(nextgraph.nodes())
				neighbors = [a[1] for a in nextgraph.edges(node)]

				if (len(neighbors) >= 2):
					a, b = random.sample(set(neighbors), 2)

					if not nextgraph.has_edge(a, b):
						nextgraph.add_edge(a, b, weight = 0)

						for i in range(n_p):
							if a in communities[i] and b in communities[i]:
								nextgraph[a][b]['weight'] += 1


			graph = nextgraph.copy()

			if check_consensus_graph(nextgraph, n_p = n_p, delta = delta):
				break

		elif (algorithm == 'cnm'):

			nextgraph = graph.copy()
			
			for u,v in nextgraph.edges():
				nextgraph[u][v]['weight'] = 0.0

			communities = []
			mapping = []
			inv_map = []


			for _ in range(n_p):

				order = list(range(N))
				random.shuffle(order)
				maps = dict(zip(range(N), order))
				
				mapping.append(maps)
				inv_map.append({v: k for k, v in maps.items()})
				G_c = nx.relabel_nodes(graph, mapping = maps, copy = True)
				G_igraph = nx_to_igraph(G_c)

				communities.append(G_igraph.community_fastgreedy(weights = 'weight').as_clustering())


			for i in range(n_p):
				
				edge_list = [(mapping[i][j], mapping[i][k]) for j,k in graph.edges()]
				
				for node,nbr in edge_list:
					a, b = inv_map[i][node], inv_map[i][nbr]

					if graph[a][b] not in (0, n_p):
						for c in communities[i]:
							if node in c and nbr in c:
								nextgraph[a][b]['weight'] += 1

					else:
						nextgraph[a][b]['weight'] = graph[a][b]['weight']


			remove_edges = []
			for u,v in nextgraph.edges():
				if nextgraph[u][v]['weight'] < thresh*n_p:
					remove_edges.append((u, v))
			
			nextgraph.remove_edges_from(remove_edges)


			for _ in range(L):
				node = np.random.choice(nextgraph.nodes())
				neighbors = [a[1] for a in nextgraph.edges(node)]

				if (len(neighbors) >= 2):
					a, b = random.sample(set(neighbors), 2)
					if not nextgraph.has_edge(a, b):
						nextgraph.add_edge(a, b, weight = 0)

						for i in range(n_p):
							for c in communities[i]:
								if mapping[i][a] in c and mapping[i][b] in c:
								
									nextgraph[a][b]['weight'] += 1
			
			if check_consensus_graph(nextgraph, n_p, delta):
				break

		else:
			break

	if (algorithm == 'louvain'):
		return [cm.partition_at_level(cm.generate_dendrogram(graph, randomize = True, weight = 'weight'), 0) for _ in range(n_p)]
	if algorithm == 'infomap':
		return [{frozenset(c) for c in nx_to_igraph(graph).community_infomap().as_cover()} for _ in range(n_p)]
	if algorithm == 'lpm':
		return [{frozenset(c) for c in nx_to_igraph(graph).community_label_propagation().as_cover()} for _ in range(n_p)]
	if algorithm == 'cnm':

		communities = []
		mapping = []
		inv_map = []

		for _ in range(n_p):
			order = list(range(N))
			random.shuffle(order)
			maps = dict(zip(range(N), order))
				
			mapping.append(maps)
			inv_map.append({v: k for k, v in maps.items()})
			G_c = nx.relabel_nodes(graph, mapping = maps, copy = True)
			G_igraph = nx_to_igraph(G_c)

			communities.append(G_igraph.community_fastgreedy(weights = 'weight').as_clustering())

		return communities


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Process some integers.')

	parser.add_argument('-f', metavar='filename', type=str, nargs = '?', help='file with edgelist')
	parser.add_argument('-np', metavar='n_p', type=int, nargs = '?', default=20, help='number of input partitions for the algorithm (Default value: 20)')
	parser.add_argument('-t', metavar='tau', type=float, nargs = '?', help='used for filtering weak edges')
	parser.add_argument('-d', metavar='del', type=float,  nargs = '?', default = 0.02, help='convergence parameter (default = 0.02). Converges when less than delta proportion of the edges are with wt = 1')
	parser.add_argument('--alg', metavar='alg', type=str, nargs = '?', default = 'louvain' , help='choose from \'louvain\' , \'cnm\' , \'lpm\' , \'infomap\' ')

	args = parser.parse_args()

	default_tau = {'louvain': 0.2, 'cnm': 0.7 ,'infomap': 0.6, 'lpm': 0.8}
	if (args.t == None):
		args.t = default_tau.get(args.alg, 0.2)
	
	if check_arguments(args) == False:

		quit()

	G = nx.read_edgelist(args.f, nodetype=int)

	output = fast_consensus(G, algorithm = args.alg, n_p = args.np, thresh = args.t, delta = args.d)

	if not os.path.exists('out_partitions'):
		os.makedirs('out_partitions')

	
	if(args.alg == 'louvain'):
		for i in range(len(output)):
			output[i] = group_to_partition(output[i])
		
	
	i = 0
	for partition in output:
		i += 1
		with open('out_partitions/' + str(i) , 'w') as f:
			for community in partition:
				print(*community, file = f)
