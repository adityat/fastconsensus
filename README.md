# Fastconsensus

Fastconsensus is a Python package that implements a fast consensus clustering algorithm for complex networks. It provides an efficient way to perform community detection on large-scale networks using the igraph library.

## Installation

### From source

To install fastconsensus from source, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fastconsensus.git
   cd fastconsensus
   ```

2. Create a conda environment (optional but recommended):
   ```bash
   conda env create -f environment.yml
   conda activate fastconsensus
   ```

3. Install the package:
   ```bash
   pip install -e .
   ```

## Usage

Here's a basic example of how to use fastconsensus:

```python
import igraph as ig
from fastconsensus import fast_consensus_clustering, read_graph_from_file

# Read a graph from a file
graph = read_graph_from_file("path/to/your/graph.gml", format="gml")

# Perform fast consensus clustering
partition = fast_consensus_clustering(graph, n_partitions=20, threshold=0.2)

# Print the resulting partition
print(partition)
```

For more detailed examples and usage scenarios, please refer to the Jupyter notebooks in the `notebooks/` directory.

## Running the Notebooks

To run the example notebooks:

1. Ensure you have Jupyter installed in your environment:
   ```bash
   conda install jupyter
   ```

2. Navigate to the `notebooks/` directory and start Jupyter:
   ```bash
   cd notebooks
   jupyter notebook
   ```

3. Open and run the notebook

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
