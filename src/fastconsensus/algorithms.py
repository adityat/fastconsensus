from abc import ABC, abstractmethod
import igraph as ig

class CommunityDetectionAlgorithm(ABC):
    @abstractmethod
    def detect_communities(self, graph: ig.Graph) -> dict:
        """
        Detect communities in the given graph.
        
        :param graph: igraph Graph object
        :return: A dictionary mapping node ids to community ids
        """
        pass

class LouvainAlgorithm(CommunityDetectionAlgorithm):
    def detect_communities(self, graph: ig.Graph) -> dict:
        partition = graph.community_multilevel()
        return {v: partition.membership[v] for v in range(graph.vcount())}

class LabelPropagationAlgorithm(CommunityDetectionAlgorithm):
    def detect_communities(self, graph: ig.Graph) -> dict:
        partition = graph.community_label_propagation()
        return {v: partition.membership[v] for v in range(graph.vcount())}


class InfoMapAlgorithm(CommunityDetectionAlgorithm):
    def detect_communities(self, graph: ig.Graph) -> dict:
        partition = graph.community_infomap()
        return {v: partition.membership[v] for v in range(graph.vcount())}

def get_algorithm(name: str) -> CommunityDetectionAlgorithm:
    algorithms = {
        'louvain': LouvainAlgorithm(),
        'label_propagation': LabelPropagationAlgorithm(),
         'infomap': InfoMapAlgorithm(),  
    }
    return algorithms[name.lower()]  # This will raise a KeyError if the algorithm is not found