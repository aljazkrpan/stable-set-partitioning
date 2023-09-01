import networkx as nx
import os
from pathlib import Path
from stable_set_partitioning import calculate_best_solution
from stable_set_partitioning import print_best_partitions
from dwave.system.composites import AutoEmbeddingComposite
from dwave.system.samplers import DWaveSampler
from dwave.samplers import SimulatedAnnealingSampler


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

all_graph_files_with_best_part = [
    ('Stable_set_data-main/Instances/dsjc_graphs/dsjc125.5_stable_set_edge_list.txt', 42),
    ('Stable_set_data-main/Instances/hamming_graphs/hamming6_4_stable_set_edge_list.txt', 30),
    ('Stable_set_data-main/Instances/johnson_graphs/johnson8_2_4_stable_set_edge_list.txt', 12),
    ('Stable_set_data-main/Instances/johnson_graphs/johnson16_2_4_stable_set_edge_list.txt', 57),
    ('Stable_set_data-main/Instances/paley_graphs/paley61_stable_set_edge_list.txt', 27),
    ('Stable_set_data-main/Instances/paley_graphs/paley73_stable_set_edge_list.txt', 25),
    ('Stable_set_data-main/Instances/paley_graphs/paley89_stable_set_edge_list.txt', 40),
    ('Stable_set_data-main/Instances/paley_graphs/paley97_stable_set_edge_list.txt', 33),
    ('Stable_set_data-main/Instances/paley_graphs/paley101_stable_set_edge_list.txt', 34)
]

all_dense_graph_files_with_best_part = [
    ('Stable_set_data-main/Instances/c_fat_graphs/c_fat200_2_stable_set_edge_list.txt', 34),
    ('Stable_set_data-main/Instances/c_fat_graphs/c_fat500_5_stable_set_edge_list.txt', 57)
]

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


# Here we compute partitions of sizes 2...len(stable_set_graph.nodes), and we print them in order
def example1():
    for edge_list_file_path in all_dense_graph_files + all_graph_files:
        num_nodes, edges = read_edge_list(edge_list_file_path)
        stable_set_graph = create_networkx_graph(num_nodes, edges)
        print_best_partitions(stable_set_graph, max_num_of_part=len(stable_set_graph.nodes))
        input("Click enter for next graph...")

# example2 uses suitable partition from the example1 to try to improve the solution.
def example2():
    # or if you don't have an account you can replace it with:
    #sampler = SimulatedAnnealingSampler()
    sampler = AutoEmbeddingComposite(DWaveSampler(token="DEV-33b6ae807bdea6181a9c83c105cfc10c87b5ce73"))
    beta=1

    for edge_list_file_path in all_dense_graph_files_with_best_part + all_graph_files_with_best_part:
        num_nodes, edges = read_edge_list(edge_list_file_path[0])
        stable_set_graph = create_networkx_graph(num_nodes, edges)
        graph_name = Path(edge_list_file_path[0]).name.replace('_stable_set_edge_list.txt','')
        print(graph_name)
        if(not os.path.exists(f"results/QPU_{beta}/{graph_name}_runs100")):
            os.makedirs(f"results/QPU_{beta}/{graph_name}_runs100")
        output = f"results/QPU_{beta}/{graph_name}_runs100/{graph_name}"
        calculate_best_solution(stable_set_graph, sampler, beta=beta*2, output_file=output, num_of_part=edge_list_file_path[1], num_of_runs=100)
if __name__ == "__main__":
    example1()
    example2()