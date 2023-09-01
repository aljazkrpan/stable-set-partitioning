import networkx as nx
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
    for edge_list_file_path in all_graph_files:
        num_nodes, edges = read_edge_list(edge_list_file_path)
        stable_set_graph = create_networkx_graph(num_nodes, edges)
        print_best_partitions(stable_set_graph, max_num_of_part=len(stable_set_graph.nodes))

# example2 uses suitable partition from the example1 to try to improve the solution.
def example2():
    # or if you don't have an account you can replace it with:
    # sampler = SimulatedAnnealingSampler()
    sampler = AutoEmbeddingComposite(DWaveSampler(token="provide_you_token_here"))
    beta=2

    # for fat200_2, we'll use partition size 9 we got from example3()
    edge_list_file_path = 'Stable_set_data-main/Instances/c_fat_graphs/c_fat200_2_stable_set_edge_list.txt'
    num_nodes, edges = read_edge_list(edge_list_file_path)
    stable_set_graph = create_networkx_graph(num_nodes, edges)
    output = "./c_fat_200_2"
    solution = calculate_best_solution(stable_set_graph, sampler, beta=beta, output_file=output, num_of_part=9, num_of_runs=1000)

    # for fat500_5, we'll use partition size 8 we got from example3()
    edge_list_file_path = 'Stable_set_data-main/Instances/c_fat_graphs/c_fat500_5_stable_set_edge_list.txt'
    num_nodes, edges = read_edge_list(edge_list_file_path)
    stable_set_graph = create_networkx_graph(num_nodes, edges)
    output = "./c_fat_500_5"
    solution = calculate_best_solution(stable_set_graph, sampler, beta=beta, output_file=output, num_of_part=8, num_of_runs=1000)

if __name__ == "__main__":
    example2()