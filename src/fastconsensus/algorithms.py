from abc import ABC, abstractmethod
import igraph as ig
import community as cm

class CommunityDetectionAlgorithm(ABC):
    """
    Abstract base class for community detection algorithms.

    This class defines the interface for community detection algorithms
    to ensure consistent usage across different implementations.
    """

    @abstractmethod
    def detect_communities(self, graph: ig.Graph) -> dict:
        """
        Detect communities in the given graph.
        
        :param graph: igraph Graph object
        :return: A dictionary mapping node ids to community ids
        """
        pass

class LouvainAlgorithm(CommunityDetectionAlgorithm):
    """
    Implementation of the Louvain community detection algorithm.

    This algorithm optimizes modularity in a hierarchical manner.
    """
    def detect_communities(self, graph: ig.Graph, weight=None) -> dict:
        """
        Detect communities using the Louvain algorithm.

        Find communities by optimizing modularity in a multi-level approach, 0th level of the method.

        :param graph: igraph Graph object
        :param weight: Optional name of the edge attribute to be used as weight
        :return: A dictionary mapping node ids to community ids
        """
        partition = graph.community_multilevel(weights=weight, return_levels=True)[0]
        return {v: partition.membership[v] for v in range(graph.vcount())}

class LabelPropagationAlgorithm(CommunityDetectionAlgorithm):
    """
    Implementation of the Label Propagation community detection algorithm.

    This algorithm detects communities by propagating labels through the network.
    """
    def detect_communities(self, graph: ig.Graph) -> dict:

        """
        Detect communities using the Label Propagation algorithm.
        Use igraph's implementation of the Label Propagation algorithm.

        :param graph: igraph Graph object
        :param weight: Optional name of the edge attribute to be used as weight (ignored in this implementation)
        :return: A dictionary mapping node ids to community ids
        """
        partition = graph.community_label_propagation()
        return {v: partition.membership[v] for v in range(graph.vcount())}


class InfoMapAlgorithm(CommunityDetectionAlgorithm):
    """
    Implementation of the Infomap community detection algorithm.

    This algorithm finds communities by optimizing the map equation.
    """
    def detect_communities(self, graph: ig.Graph) -> dict:
        """
        Detect communities using the Infomap algorithm.
        Use igraph's implementation of the Infomap algorithm.

        :param graph: igraph Graph object
        :param weight: Optional name of the edge attribute to be used as weight
        :return: A dictionary mapping node ids to community ids
        """
        partition = graph.community_infomap()
        return {v: partition.membership[v] for v in range(graph.vcount())}

def get_algorithm(name: str) -> CommunityDetectionAlgorithm:
    """
    Function to get the specified community detection algorithm.

    :param name: Name of the algorithm to retrieve
    :return: An instance of the requested CommunityDetectionAlgorithm
    :raises ValueError: If the requested algorithm is not available
    """
    algorithms = {
        'louvain': LouvainAlgorithm(),
        'label_propagation': LabelPropagationAlgorithm(),
         'infomap': InfoMapAlgorithm(),  
    }
    return algorithms[name.lower()]  # This will raise a KeyError if the algorithm is not found