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

# takes a .pkl file and returns the sample_set
def open_saved_sample_set(input_file):
    with open(input_file, 'rb') as file:
        sample_set = SampleSet.from_serializable(pickle.load(file))
    return sample_set

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
        partition_with_halo_list=None,
        generate_log_file=True
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

    if(generate_log_file):
        log_file = open(f"{os.path.dirname(output_file)}/log.txt", 'w')

    best_solution_nodes = []
    best_solution_energy = 0
    all_solutions_info = {}

    # we go through every partition, we create subgraph and Q matrix (which represents QUBO function)
    # we calculate the response, which is saved in a file (if desired), we save the best solution nodes
    # and energy, and we print the best result
    for k, partition in enumerate(partition_with_halo_list):
        if k+1 < 15:
            continue
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

        if(len(response.record) != 0):
            # this is how you get solution nodes and solution energy from the response
            solution_nodes = [h for h,v in response.first[0].items() if v == 1]
            solution_energy = response.first[1]
            all_solutions_info[k] = {
                "best_solution_nodes": solution_nodes,
                "best_solution_energy" : solution_energy,
                "sample_set": response
            }
        else:
            solution_nodes = []
            solution_energy = 0
            all_solutions_info[k] = {
                "best_solution_nodes": solution_nodes,
                "best_solution_energy" : solution_energy,
                "sample_set": response
            }
        
        if(solution_energy < best_solution_energy):
            best_solution_nodes = solution_nodes
            best_solution_energy = solution_energy
        
        # this is to avoid double printing the result if we choose to calculate response without partitions
        k_pad = f"{k+1:>{int(math.log(len(partition_with_halo_list),10))+1}}"
        if(generate_log_file):
            log_file.write(f"Partition {k_pad}: beta={beta//2} || solution energy: {solution_energy} || number of edges in solution: {len(subgraph.subgraph(solution_nodes).edges)} || solution size: {len(solution_nodes)}\n")
        if(len(partition_with_halo_list) > 1 and console_output):
            print(f"Partition {k_pad}: beta={beta//2} || solution energy: {solution_energy} || number of edges in solution: {len(subgraph.subgraph(solution_nodes).edges)} || solution size: {len(solution_nodes)}")

    if(generate_log_file):
        log_file.write(f"Best result: beta={beta//2} || best solution energy: {best_solution_energy} || number of edges in best solution: {len(subgraph.subgraph(best_solution_nodes).edges)} || best solution size: {len(best_solution_nodes)}\n")
        log_file.close()
    if(console_output):
        print(f"Best result: beta={beta//2} || best solution energy: {best_solution_energy} || number of edges in best solution: {len(subgraph.subgraph(best_solution_nodes).edges)} || best solution size: {len(best_solution_nodes)}\n")

    return all_solutions_info




