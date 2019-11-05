# Fast Consensus Clustering in Networks

Fast consensus is an implementation of the fast consensus clustering procedure laid out in - 

* Aditya Tandon, Aiiad Albeshri, Vijey Thayananthan, Wadee Alhalabi and Santo Fortunato: “[Fast consensus clustering in complex networks](https://arxiv.org/pdf/1902.04014.pdf)”, **Phys. Rev. E.**; 2019

If you use the script please cite this paper.
 
The procedure generates *median* or *consensus* partitions from multiple runs of a community detection algorithm. Tests on artificial benchmarks show that consensus partitions are more accurate than the ones obtained by the direct application of the clustering algorithm.

## Requirements

The script requires the following:

1. [Python 3.x](https://www.python.org/downloads/) 
2. [Numpy](http://www.numpy.org/)
3. [Networkx](https://networkx.github.io/)
4. [python-igraph](https://igraph.org/python/)
5. [python-louvain](https://github.com/taynaud/python-louvain)

## Usage

You can run the script with 

```
python fast_consensus.py -f path_to_file/edgelist.txt [--alg algorithm] [-np n_p] [-t tau] [-d delta]
```

with the following options -
```
-f filename.txt
```
(Required) where `filename.txt` is an edgelist of the network.

The file can be of the form 
```
0 1 0.5
0 4 1
1 3 0.3
.
.
.
```

where the first two numbers in each row are connected nodes and the third number is the edge weight. If only two numbers are provided the graph is treated as unweighted. The nodes must be integers starting from 0


```
--alg algorithm
```
(Optional) Here `algorithm` is the community detection method used on the network and it can be one of `louvain` ([Louvain algorithm](https://arxiv.org/abs/0803.0476)), `cnm` ([Fast greedy modularity maximization](https://arxiv.org/abs/cond-mat/0408187)), `lpm` ([Label Propagation Method](https://arxiv.org/abs/0709.2938)), `infomap` ([Infomap](http://www.mapequation.org/code.html)). If no algorithm is provided the script uses `louvain` for this purpose. 

```
-np n_p
```
(Optional) `n_p` is the number of partitions created by repeated application of the community detection algorithm. If no value is provided, `n_p = 20`

```
-t tau
```
(Optional) `tau` is a float between `0` and `1`. Elements of the consensus matrix with weight less than `tau` are set to zero in each step of the algorithm. If no value is provided, the code uses the value for which the chosen clustering algorithm gives the best performance on the [LFR benchmark graph](https://arxiv.org/abs/0805.4770) 

```
-d delta
```
(Optional) `delta` should be a float between `0.02` and `0.1`. The procedure ends when less than `delta` fraction of the edges have a weight not equal to 1. If no value is provided, `delta` is set to `0.02`


#### Example Usage

```
python fast_consensus.py -f examples/karate_club.txt --alg louvain -np 50 -t 0.2 -d 0.1
```

The file `examples/karate_club.txt` is provided. 


## Output
A folder `out_partitions` is created with `n_p` different files. Each file represents a partition; each line in the file lists all nodes belonging to a community.

For example, a run with `n_p = 2` will create two files `1` and `2`. Each file will be in the form:
```
0 1 2 5 7 8 9
3 4 6 10 11
```
This represents a partition with two communities : `{0, 1, 2, 5, 7, 8, 9}` and `{3, 4, 6, 10, 11}`
