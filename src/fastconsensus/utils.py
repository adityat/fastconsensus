import igraph as ig
from typing import Dict, Any
import math

def calculate_modularity(graph: ig.Graph, partition: Dict[int, Any]) -> float:
    """
    Calculate modularity of a partition.

    :param graph: Graph object
    :param partition: Node to community mapping
    :return: Modularity score
    """
    return graph.modularity(list(partition.values()))

def compare_partitions(partition1: Dict[int, Any], partition2: Dict[int, Any]) -> float:
    """
    Compare partitions using Normalized Mutual Information (NMI).

    :param partition1: First partition
    :param partition2: Second partition
    :return: NMI score
    :raises ValueError: If partitions have different node sets
    """
    if set(partition1.keys()) != set(partition2.keys()):
        raise ValueError("Partitions must have the same set of nodes")
    
    n = len(partition1)
    
    # Convert all community IDs to strings to ensure they're hashable
    partition1 = {node: str(comm) for node, comm in partition1.items()}
    partition2 = {node: str(comm) for node, comm in partition2.items()}
    
    # Count occurrences of each community
    count1 = {}
    count2 = {}
    for node in partition1:
        count1[partition1[node]] = count1.get(partition1[node], 0) + 1
        count2[partition2[node]] = count2.get(partition2[node], 0) + 1
    
    # Calculate mutual information
    mi = 0
    for c1 in count1:
        for c2 in count2:
            n_ij = sum(1 for node in partition1 if partition1[node] == c1 and partition2[node] == c2)
            if n_ij > 0:
                mi += (n_ij / n) * math.log2((n * n_ij) / (count1[c1] * count2[c2]))
    
    # Calculate entropies
    h1 = sum(-(count / n) * math.log2(count / n) for count in count1.values())
    h2 = sum(-(count / n) * math.log2(count / n) for count in count2.values())
    
    # Calculate NMI
    return 2 * mi / (h1 + h2) if (h1 + h2) > 0 else 0