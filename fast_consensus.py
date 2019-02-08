import networkx as nx
from itertools import groupby
import numpy as np
import itertools
import matplotlib.pyplot as plt
import subprocess
import sys
import os
import random
import igraph as ig
import LFR
from networkx.algorithms import community
import networkx.algorithms.isolate
import time
import community as cm
import math
import random
import argparse
from collections import defaultdict

def check_consensus_graph(G, n_p = 20, delta = 0.02):
	count = 0
	
	for wt in nx.get_edge_attributes(G, 'weight').values():
		if wt != 0 and wt != n_p:
			count += 1

	if count > delta*G.number_of_edges():
		print('edges = ' + str(count/G.number_of_edges()))
		return False

	print('edges = ' + str(count/G.number_of_edges()))
	return True



def nx_to_igraph(Gnx):
	
	g = ig.Graph()
	g.add_vertices(sorted(Gnx.nodes()))
	g.add_edges(sorted(Gnx.edges()))
	g.es["weight"] = 1.0
	for edge in Gnx.edges():
		g[edge[0], edge[1]] = Gnx[edge[0]][edge[1]]['weight']
	return g



def write_partition(partition, filename):
	with open(filename, 'w') as f:
		

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
			print('Incorrect algorithm choose. Choose one from - louvain, cnm, infomap or lpm')
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

	parser.add_argument('-f', metavar='f', type=str, nargs = '?', help='file with edgelist')
	parser.add_argument('-np', metavar='n_p', type=int, nargs = '?', default=20, help='number of input partitions for the algorithm (Default value: 20)')
	parser.add_argument('-t', metavar='tau', type=float, nargs = '?', default = 0.25, help='used for filtering weak edges')
	parser.add_argument('-d', metavar='del', type=float,  nargs = '?', default = 0.02, help='convergence parameter (default = 0.02). Converges when less than delta proportion of the edges are with wt = 1')
	parser.add_argument('--alg', metavar='alg', type=str, nargs = '?', default = 'louvain' , help='choose from \'louvain\' , \'cnm\' , \'lpm\' , \'infomap\' ')

	args = parser.parse_args()

	
	#G = nx.read_edgelist(args.f)
	G = nx.karate_club_graph()

	output = fast_consensus(G, algorithm = args.alg, n_p = args.np, thresh = args.t, delta = args.d)


	if not os.path.exists('out_partitions'):
		os.makedirs('out_partitions')





