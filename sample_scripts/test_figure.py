import networkx as nx
import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import logging
from networkx.generators.community import LFR_benchmark_graph
from fastconsensus.algorithms import get_algorithm
from fastconsensus.core import fast_consensus_clustering
from fastconsensus.utils import compare_partitions

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("experiment_results.log"),
                        logging.StreamHandler()
                    ])

def generate_lfr_graph(n, tau1, tau2, mu, average_degree, max_degree, min_community, max_community, max_attempts=10):
    for attempt in range(max_attempts):
        try:
            current_avg_degree = average_degree + random.uniform(-1, 1)
            
            G = LFR_benchmark_graph(
                n, tau1, tau2, mu, average_degree=current_avg_degree, max_degree=max_degree,
                min_community=min_community, max_community=max_community,
                tol=1e-3, max_iters=5000
            )
            
            true_communities = {node: frozenset(G.nodes[node]['community']) for node in G.nodes()}
            edges = list(G.edges())
            g = ig.Graph(n=G.number_of_nodes(), edges=edges)
            return g, true_communities
        
        except nx.ExceededMaxIterations:
            if attempt == max_attempts - 1:
                raise ValueError(f"Failed to generate LFR graph after {max_attempts} attempts")
            continue

def run_experiment(n_runs, n_nodes, mu_values):
    results = {
        'Louvain': [],
        'FastConsensus': []
    }
    
    for mu in tqdm(mu_values, desc="Processing μ values"):
        louvain_nmi = []
        fastconsensus_nmi = []
        
        for run in range(n_runs):
            try:
                g, true_communities = generate_lfr_graph(
                    n=n_nodes,
                    tau1=2,
                    tau2=2,
                    mu=mu,
                    average_degree=20,
                    max_degree=50,
                    min_community=20,
                    max_community=100
                )
                
                # Run Louvain
                louvain_alg = get_algorithm('louvain')
                louvain_partition = louvain_alg.detect_communities(g)
                louvain_nmi_value = compare_partitions(true_communities, louvain_partition)
                louvain_nmi.append(louvain_nmi_value)
                
                # Run FastConsensus
                fastconsensus_partition = fast_consensus_clustering(g, n_partitions=10, threshold=0.2, algorithm='louvain')
                fastconsensus_nmi_value = compare_partitions(true_communities, fastconsensus_partition)
                fastconsensus_nmi.append(fastconsensus_nmi_value)
                
                logging.info(f"μ={mu:.2f}, Run {run+1}/{n_runs}: Louvain NMI={louvain_nmi_value:.4f}, FastConsensus NMI={fastconsensus_nmi_value:.4f}")
            
            except ValueError as e:
                logging.error(f"Error generating graph for μ={mu}, run {run+1}: {str(e)}")
                continue
        
        if louvain_nmi and fastconsensus_nmi:
            avg_louvain_nmi = np.mean(louvain_nmi)
            avg_fastconsensus_nmi = np.mean(fastconsensus_nmi)
            results['Louvain'].append(avg_louvain_nmi)
            results['FastConsensus'].append(avg_fastconsensus_nmi)
            logging.info(f"Average NMI for μ={mu:.2f}: Louvain={avg_louvain_nmi:.4f}, FastConsensus={avg_fastconsensus_nmi:.4f}")
        else:
            results['Louvain'].append(None)
            results['FastConsensus'].append(None)
            logging.warning(f"No valid results for μ={mu:.2f}")
    
    return results

def plot_results(mu_values, results):
    plt.figure(figsize=(10, 6))
    
    for algorithm in ['Louvain', 'FastConsensus']:
        valid_results = [(mu, nmi) for mu, nmi in zip(mu_values, results[algorithm]) if nmi is not None]
        if valid_results:
            mu_vals, nmi_vals = zip(*valid_results)
            plt.plot(mu_vals, nmi_vals, '-o', label=algorithm)
    
    plt.xlabel('Mixing parameter μ')
    plt.ylabel('Normalized Mutual Information')
    plt.title('Louvain vs FastConsensus on LFR Benchmark (1000 nodes)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('louvain_vs_fastconsensus_lfr.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    n_runs = 2  # Number of runs for each μ value
    n_nodes = 100  # Number of nodes in the LFR graph
    mu_values = np.arange(0.1, 0.81, 0.05)  # μ values from 0.1 to 0.8 with 0.05 step
    
    logging.info(f"Starting experiment with {n_runs} runs for each of {len(mu_values)} μ values, on graphs with {n_nodes} nodes")
    results = run_experiment(n_runs, n_nodes, mu_values)
    logging.info("Experiment completed. Plotting results.")
    plot_results(mu_values, results)
    logging.info("Results plotted and saved as 'louvain_vs_fastconsensus_lfr.png'")