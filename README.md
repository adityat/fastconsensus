# Fast Consensus

Fast consensus is an implementation of the fast consensus clustering procedure laid out in - 

The procedure is used to come up with a combined consensus, or *average*, result from multiple runs of a community detection algorithm that are shown to perform better than the original method. 

## Prerequisites

The script requires the following:

1. [Python 3.x](https://www.python.org/downloads/) 
2. [Numpy](http://www.numpy.org/)
3. [python-igraph](https://igraph.org/python/)
4. [python-louvain](https://github.com/taynaud/python-louvain)

## Usage

You can run the script with 

```
python fastconsensus.py -f path_to_file/edgelist.txt [--alg algorithm] [-np n_p] [-t tau] [-d delta]
```

with the following options -
```
-f filename.txt
```
(Required) Where `filename.txt` is an edgelist of the network (connected nodes separated by space on different lines of the file). 

```
--alg algorithm
```
(Optional) Here `algorithm` is the community detection method used on the network and it can be one of `louvain`, `cnm`, `lpm`, `infomap`. If no algorithm is provided the script uses `louvain` for this purpose. 

```
-np n_p
```
(Optional) `n_p` is the number of partitions created by repeated application of the community detection algorithm. If no value is provided, `n_p = 20`

```
-t tau
```
(Optional) `tau` is a float between `0` and `1`. Edges with weight less than `tau` are filtered out in each step of the algorithm. If no value is provided, an appropriate value is picked based on the algorithm.
```
-d delta
```
(Optional) `delta` should be a float between `0.02` and `0.1`. The procedure ends when less than delta fraction of the edges have a weight not equal to 1. If no value is provided, delta is set to 0.02

## Output
A folder `out_partitions` is created with `n_p` different files. Each file represents a partition with each line in the files representing a community 



