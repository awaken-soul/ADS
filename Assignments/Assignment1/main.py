import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
import csv
import sys

#We first get all the trees possible for the given number of nodes 
#The logic is to assign nodes to left branch and remaining to right branch. Thus by recursion we go down at each node find the possible ordering and then add all of this into a list

def get_tress(n):
    bsts = []
    if (n==0) : return [()]
    for i in range(n):
        j = n-1-i
        lopts = get_tress(i)
        ropts = get_tress(j)

        for left in lopts:
            for right in ropts:
                tree = (left,right) 
                bsts.append(tree)
    return bsts

#Function to obtain rotations for the tree that was inputted into it
def rotation(tree):
    neighbours = []

    if tree == (): return  []

    L,R = tree

    if L!=():
        LL, LR = L
        new_tree = (LL,(LR,R))
        neighbours.append(new_tree)

    if R!=():
        RL,RR = R
        new_tree = ((L,RL),RR)
        neighbours.append(new_tree)
    
    lopt = rotation(L)
    for rot in lopt:
        neighbours.append((rot,R))

    ropt = rotation(R)
    for rot in ropt:
        neighbours.append((L,rot))

    neighbours = list(set(neighbours))
    return neighbours

def bfs_rotation_distance(G, start):
    dist = {start: 0}
    for u, v in nx.bfs_edges(G, start):
        dist[v] = dist[u] + 1
    return dist

def deep_size(obj, seen=None):
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    size = sys.getsizeof(obj)

    if isinstance(obj, dict):
        size += sum(deep_size(k, seen) + deep_size(v, seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set)):
        size += sum(deep_size(i, seen) for i in obj)

    return size


#Test code to check get Trees: [added time measurement and size of all_trees]
n = int(input("Enter n "))
start = time.perf_counter()
all_trees = get_tress(n)
tree_time = time.perf_counter() - start
tree_count = len(all_trees)
tree_mem = deep_size(all_trees)

id = {tree : f"T{i}" for i , tree in enumerate(all_trees)}
trees = {v : k for k,v in id.items()}
for i in range(len(all_trees)):
    print(all_trees[i])

#graph construction [added time measurement for graph construction]
start = time.perf_counter()
G = nx.Graph()
for tree in all_trees:
    G.add_node(tree)
for T in all_trees:
    for N in rotation(T):
        G.add_edge(T,N)
graph_time = time.perf_counter() - start
graph_mem = deep_size(G)

#graph checking
for T in all_trees:
    print(f"{T} {rotation(T)}")
    print(G.degree(T))


#bfs traveseral and matrix calculation
start = time.perf_counter()
nodes = list(G.nodes)
index = {node: i for i, node in enumerate(nodes)}

n_nodes = len(nodes)
dist_matrix = np.zeros((n_nodes, n_nodes), dtype=int)

for u in nodes:
    dist = bfs_rotation_distance(G, u)
    i = index[u]
    for v, d in dist.items():
        j = index[v]
        dist_matrix[i, j] = d
matrix_time = time.perf_counter() - start
matrix_mem = deep_size(dist_matrix)

#matplotlib code for drawing the graph
pos = nx.spring_layout(G, seed=42)  # seed ensures reproducibility
nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=700)
nx.draw_networkx_edges(G, pos, width=1.5)
labels = {node: id[node] for node in G.nodes}
nx.draw_networkx_labels(G, pos, labels, font_size=8)
plt.axis('off')
plt.show()


#csv code to compute runtime benchmarking

csv_file = "execution_times.csv"

header = [
    "n",
    "number_of_trees",
    "tree_generation_time_sec",
    "graph_construction_time_sec",
    "distance_matrix_time_sec"
]

row = [
    n,
    tree_count,
    tree_time,
    graph_time,
    matrix_time
]

try:
    with open(csv_file, "x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(row)
except FileExistsError:
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


memory_csv = "memory_usage.csv"

header = [
    "n",
    "number_of_trees",
    "trees_memory_bytes",
    "graph_memory_bytes",
    "distance_matrix_memory_bytes"
]

row = [
    n,
    tree_count,
    tree_mem,
    graph_mem,
    matrix_mem
]

try:
    with open(memory_csv, "x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(row)
except FileExistsError:
    with open(memory_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)
