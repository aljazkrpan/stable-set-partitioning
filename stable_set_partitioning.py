import pickle
from pathlib import Path
import os
import networkx as nx
from metis import part_graph
from tqdm import tqdm
import math
import numpy as np
import dwave
from collections import defaultdict
from dimod import SampleSet

from dwave.system.composites import AutoEmbeddingComposite
from dwave.system.samplers import DWaveSampler

# Reads edge list from the Stable_set_data-main dataset
def read_edge_list(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    num_nodes, num_edges = map(int, lines[0].split())

    edges = []
    for line in lines[1:]:
        u, v, w = map(int, line.split())
        edges.append((u-1, v-1))
    
    return num_nodes, edges

# Creates graph from the edges we read
def create_networkx_graph(num_nodes, edges):
    G = nx.Graph()
    G.add_nodes_from(range(0, num_nodes))
    G.add_edges_from(edges)
    return G

def calculate_partition_with_halo(
        stable_set_graph,
        num_of_part
    ):
    clique_graph = nx.complement(stable_set_graph)
    objval, metis_partition_list = part_graph(clique_graph, nparts=num_of_part)
    # metis_partition_list (1 dimensional list) is mapping from vertex i to partition number metis_partition_list[i].
    # this means that number of partitions is max number in it + 1
    num_partitions = max(metis_partition_list)+1

    partition_with_halo_list = [None]*num_partitions
    partition_list = [None]*num_partitions
    # our partition format will be list of sets, where each set will contain vertice in partition
    # the below code coverts metis format to ours
    node_list = list(clique_graph.nodes)
    for i in range(num_partitions):
        partition_list[i] = set([node_list[j] for j, value in enumerate(metis_partition_list) if value == i])

    # to find the partition together with the halo, we go through neighbourhood of every vertex in the partition and we
    # make an union of each such neighbourhood with the current partition_list.
    for i in range(num_partitions):
        partition_with_halo_list[i] = set()
        partition_with_halo_list[i] = partition_with_halo_list[i].union(partition_list[i])
        for node in partition_list[i]:
            partition_with_halo_list[i] = partition_with_halo_list[i].union(set(clique_graph[node].keys()))
    
    return partition_with_halo_list

def print_best_partitions(
        stable_set_graph,
        max_num_of_part,
        min_num_of_part=2,
        sort='v'
    ):

    results = []
    # we calculate partition for every number of partition in desired range, we map the result so that we are
    # left with tuples (# of partitions, max # of vertices in partition, max # of edges in partition), and 
    # we sort the tuple according to desired criteria
    for num_of_part in tqdm(range(min_num_of_part, max_num_of_part+1)):
        partition_with_halo_list = calculate_partition_with_halo(stable_set_graph, num_of_part)
        results.append((num_of_part,
                    max(list(map(lambda x : len(x), partition_with_halo_list))),
                    max(list(map(lambda x : len(stable_set_graph.subgraph(x).edges()), partition_with_halo_list)))
        ))
    if(sort == 'v'):
        results.sort(key = lambda x : x[1])
    elif(sort == 'e'):
        results.sort(key = lambda x : x[2])
    
    # This is just a lot of effort to make the output aligned
    part_pad = max(list(map(lambda x : int(math.log(x[0], 10)), results)))+1
    vertex_pad = max(list(map(lambda x : int(math.log(x[1], 10)), results)))+1
    edge_pad = max(list(map(lambda x : int(math.log(x[2], 10)), results)))+1
    for i in reversed(results):
        a = f"{i[0]:>{part_pad}}"; b = f"{i[1]:>{vertex_pad}}"; c = f"{i[2]:>{edge_pad}}"
        print(f"Number of partitions: {a} || Max number of vertices: {b} || Max number of edges {c}")

def calculate_best_solution(
        stable_set_graph,
        sampler,
        beta=1,
        num_of_runs=1,
        num_of_part=1,
        output_file=None,
        no_output_file = False,
        console_output=True,
        partition_with_halo_list=None
    ):
    # if we want to produce a file but the directory does not exist or is not given, throw an Exception,
    # otherwise we might waste computational resources and be left without the result
    if(not no_output_file and (output_file is None or not os.path.exists(os.path.dirname(output_file)))):
        raise Exception("Output directory not given or does not exist! If you don't want to produce the output file, set no_output_file=True in function call.")
    if(False):
        print(f"Using {sampler} sampler")
    # this has to be done, otherwise the partition function with input num_of_part=1 throws an error
    if(num_of_part > 1 and partition_with_halo_list is None):
        partition_with_halo_list = calculate_partition_with_halo(stable_set_graph, num_of_part=num_of_part)
    elif(partition_with_halo_list is None):
        partition_with_halo_list = [set(stable_set_graph.nodes)]

    best_solution_nodes = []
    best_solution_energy = 0
    all_solutions_info = {}

    # we go through every partition, we create subgraph and Q matrix (which represents QUBO function)
    # we calculate the response, which is saved in a file (if desired), we save the best solution nodes
    # and energy, and we print the best result
    for k, partition in enumerate(partition_with_halo_list):
        if(len(partition_with_halo_list) > 1):
            subgraph = stable_set_graph.subgraph(list(partition))
        else:
            subgraph = stable_set_graph

        Q = defaultdict(int)
        for i in partition:
            Q[(i,i)]+= -1
        for i, j in subgraph.edges:
            Q[(i,j)]+= beta

        
        if(num_of_runs > 1):
            response = sampler.sample_qubo(Q, num_reads=num_of_runs)
        else:
            response = sampler.sample_qubo(Q)

        if(not no_output_file):
            if(len(partition_with_halo_list) > 1):
                with open(output_file+"_partition"+str(k).zfill(int(math.log10(len(partition_with_halo_list)-1)+1))+".pkl", 'wb') as file:
                    pickle.dump(response.to_serializable(), file)
            else:
                with open(output_file, 'wb') as file:
                    pickle.dump(response.to_serializable(), file)

        # this is how you get solution nodes and solution energy from the response
        solution_nodes = [h for h,v in response.first[0].items() if v == 1]
        solution_energy = response.first[1]

        all_solutions_info[k] = {
            "best_solution_nodes": solution_nodes,
            "best_solution_energy" : solution_energy,
            "sample_set": response
        }
        
        if(solution_energy < best_solution_energy):
            best_solution_nodes = solution_nodes
            best_solution_energy = solution_energy
        
        # this is to avoid double printing the result if we choose to calculate response without partitions
        if(len(partition_with_halo_list) > 1 and console_output):
            k_pad = f"{k+1:>{int(math.log(len(partition_with_halo_list),10))+1}}"
            print(f"Partition {k_pad}: beta={beta} || solution energy: {solution_energy} || number of edges in solution: {len(subgraph.subgraph(solution_nodes).edges)} || solution size: {len(solution_nodes)}")

    if(console_output):
        print(f"Best result: beta={beta} || best solution energy: {best_solution_energy} || number of edges in best solution: {len(subgraph.subgraph(best_solution_nodes).edges)} || best solution size: {len(best_solution_nodes)}\n")

    return all_solutions_info




all_dense_graph_files = [
    'Stable_set_data-main/Instances/c_fat_graphs/c_fat200_2_stable_set_edge_list.txt',
    'Stable_set_data-main/Instances/c_fat_graphs/c_fat500_5_stable_set_edge_list.txt'
]

all_graph_files = [
    'Stable_set_data-main/Instances/C_graphs/C125.9_stable_set_edge_list.txt',
    'Stable_set_data-main/Instances/dsjc_graphs/dsjc125.1_stable_set_edge_list.txt',
    'Stable_set_data-main/Instances/dsjc_graphs/dsjc125.5_stable_set_edge_list.txt',
    'Stable_set_data-main/Instances/hamming_graphs/hamming6_2_stable_set_edge_list.txt',
    'Stable_set_data-main/Instances/hamming_graphs/hamming6_4_stable_set_edge_list.txt',
    'Stable_set_data-main/Instances/johnson_graphs/johnson8_2_4_stable_set_edge_list.txt',
    'Stable_set_data-main/Instances/johnson_graphs/johnson8_4_4_stable_set_edge_list.txt',
    'Stable_set_data-main/Instances/johnson_graphs/johnson16_2_4_stable_set_edge_list.txt',
    'Stable_set_data-main/Instances/MANN_graphs/MANN_a9_stable_set_edge_list.txt',
    'Stable_set_data-main/Instances/paley_graphs/paley61_stable_set_edge_list.txt',
    'Stable_set_data-main/Instances/paley_graphs/paley73_stable_set_edge_list.txt',
    'Stable_set_data-main/Instances/paley_graphs/paley89_stable_set_edge_list.txt',
    'Stable_set_data-main/Instances/paley_graphs/paley97_stable_set_edge_list.txt',
    'Stable_set_data-main/Instances/paley_graphs/paley101_stable_set_edge_list.txt',
    'Stable_set_data-main/Instances/spin_graphs/spin5_stable_set_edge_list.txt',
    'Stable_set_data-main/Instances/torus_graphs/torus11_stable_set_edge_list.txt'
]


# example3 just outputs the the CH-partition information for each partition size between 2, ... , max_num_of_part that we get
# from the METIS algorithm. This works best on dense graphs (those with a lot of edges)
def example1():
    # for both graph we print out the best partitions sorted by cost, so that we can decide which one to use
    global all_dense_graph_files
    for edge_list_file_path in all_graph_files:
        num_nodes, edges = read_edge_list(edge_list_file_path)
        stable_set_graph = create_networkx_graph(num_nodes, edges)
        print_best_partitions(stable_set_graph, max_num_of_part=len(stable_set_graph.nodes))

# example4 uses suitable partition from the example3 to try to improve the solution.
def example2():
    # or if you don't have an account you can replace it with:
    # sampler = SimulatedAnnealingSampler()
    #sampler = AutoEmbeddingComposite(DWaveSampler(token="provide_you_token_here"))
    sampler = AutoEmbeddingComposite(DWaveSampler(token="token"))
    beta=200

    # for fat200_2, we'll use partition size 9 we got from example3()
    edge_list_file_path = 'Stable_set_data-main/Instances/c_fat_graphs/c_fat200_2_stable_set_edge_list.txt'
    num_nodes, edges = read_edge_list(edge_list_file_path)
    stable_set_graph = create_networkx_graph(num_nodes, edges)
    #output = "results/example_results/simplified_with_partitions/QPU/c_fat200_2_runs1000/c_fat200_2.pkl"
    output = "results/c_fat_200_2"
    solution = calculate_best_solution(stable_set_graph, sampler, beta=beta, output_file=output, num_of_part=9, num_of_runs=1000)

    # for fat500_5, we'll use partition size 8 we got from example3()
    edge_list_file_path = 'Stable_set_data-main/Instances/c_fat_graphs/c_fat500_5_stable_set_edge_list.txt'
    num_nodes, edges = read_edge_list(edge_list_file_path)
    stable_set_graph = create_networkx_graph(num_nodes, edges)
    #output = "results/example_results/simplified_with_partitions/QPU/c_fat500_5_runs1000/c_fat500_5.pkl"
    output = "results/c_fat_500_5"
    solution = calculate_best_solution(stable_set_graph, sampler, beta=beta, output_file=output, num_of_part=8, num_of_runs=1000)