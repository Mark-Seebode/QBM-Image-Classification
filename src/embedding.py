import networkx as nx
import dwave_networkx as dnx
import dwave.embedding as dwave_embedding
from tqdm import tqdm
import pymetis
import matplotlib.pyplot as plt

import random


def get_qubit_list(embedding):
    out = []
    for a in embedding:
        out += embedding[a]
    return out


def rename_Nodes(bqm, problems):
    # graph = nx.Graph()
    G = nx.Graph()
    G.add_edges_from(bqm.quadratic)
    Target = nx.Graph()
    # Use bqm.quadratic to add edges
    for problem in range(problems):
        # for (u, v), weight in bqm.quadratic.items():
        #     graph.add_edge(str(e[0])+"_"+str(problem), str(e[1])+"_"+str(problem), weight=weight)
        for e in list(G.edges()):
            Target.add_edge(str(e[0]) + "_" + str(problem), str(e[1]) + "_" + str(problem))
    return Target


def deep_search(bqm, hardware_connectivity):
    for attempt in range(2):
        print(attempt)
        emb = dwave_embedding.minorminer.find_embedding(bqm, hardware_connectivity, tries=1000, max_no_improvement=1000,
                                                        chainlength_patience=1000, timeout=25600, threads=20)
        if emb != {}:
            return emb
    return {}


def exclude_subgraph_from_pegasus(target_graph, nodes_to_exclude):
    print("Target graph nodes: ", len(target_graph))
    modified_graph = target_graph.copy()
    modified_graph.remove_nodes_from([nodes_to_exclude[0]])
    print("Target graph nodes after exclusion: ", len(modified_graph))
    return modified_graph


def are_qubits_neighbors(new_embedding, saved_embeddings):
    pegasus_graph = dnx.pegasus_graph(16)
    invalid_qubits = set()
    for saved_embedding in saved_embeddings.values():
        for qubit_list in saved_embedding.values():
            for qubit in qubit_list:
                invalid_qubits.add(qubit)
                invalid_qubits.update(pegasus_graph.neighbors(qubit))

    for qubit_list in new_embedding.values():
        for qubit in qubit_list:
            if qubit in invalid_qubits:
                return False

    return True


def are_qubits_reused(new_embedding, saved_embeddings):
    used_qubits = set()
    for embedding in saved_embeddings.values():
        for qubit_list in embedding.values():
            used_qubits.update(qubit_list)

    for qubit_list in new_embedding.values():
        for qubit in qubit_list:
            if qubit in used_qubits:
                # print(qubit)
                # print(used_qubits)
                return False

    return True


def extract_separate_embeddings(saved_embeddings, embedding, problems):
    was_successful = True
    for problem in range(problems):
        qubo_embedding = {}
        for variable in list(embedding.keys()):
            s = variable.split("_")
            if problem == int(s[1]):
                qubo_embedding[int(s[0])] = embedding[variable]

        # Validate the new embedding
        if are_qubits_reused(qubo_embedding, saved_embeddings):
            saved_embeddings[problem] = qubo_embedding
        else:
            print("Invalid embedding found")
            was_successful = False
            break

    return saved_embeddings, was_successful


def lasthope(saved_embedding, problems):
    result = {}
    for problem in range(problems):
        clique_embedding = {}
        for variable in list(saved_embedding.keys()):
            s = variable.split("_")
            if problem == int(s[1]):
                clique_embedding[int(s[0])] = saved_embedding[variable]
        result[problem] = clique_embedding
    return result


def iterative_search(target_graph, bqm, sample_count):
    problems = 0
    separate_embeddings = {}
    while (True):
        problems += 1
        G = rename_Nodes(bqm, problems)
        emb = dwave_embedding.minorminer.find_embedding(G, target_graph, tries=2, max_no_improvement=2,
                                                        chainlength_patience=2, timeout=100, threads=1)
        if emb == {}:
            emb = dwave_embedding.minorminer.find_embedding(G, target_graph, tries=5, max_no_improvement=5,
                                                            chainlength_patience=5, timeout=200, threads=2)
            if emb == {}:
                emb = dwave_embedding.minorminer.find_embedding(G, target_graph, tries=10,
                                                                max_no_improvement=10,
                                                                chainlength_patience=10, timeout=400, threads=2)
                if emb == {}:
                    emb = dwave_embedding.minorminer.find_embedding(G, target_graph, tries=20,
                                                                    max_no_improvement=20, chainlength_patience=20,
                                                                    timeout=800, threads=2)
                    if emb == {}:
                        emb = dwave_embedding.minorminer.find_embedding(G, target_graph, tries=50,
                                                                        max_no_improvement=50, chainlength_patience=50,
                                                                        timeout=1600, threads=10)
                        if emb == {}:
                            emb = dwave_embedding.minorminer.find_embedding(G, target_graph, tries=100,
                                                                            max_no_improvement=100,
                                                                            chainlength_patience=100, timeout=3200,
                                                                            threads=10)
                            if emb == {}:
                                emb = dwave_embedding.minorminer.find_embedding(G, target_graph, tries=200,
                                                                                max_no_improvement=200,
                                                                                chainlength_patience=200, timeout=6400,
                                                                                threads=20)
                                if emb == {}:
                                    emb = dwave_embedding.minorminer.find_embedding(G, target_graph, tries=500,
                                                                                    max_no_improvement=500,
                                                                                    chainlength_patience=500,
                                                                                    timeout=12800, threads=20)
                                    if emb == {}:
                                        emb = dwave_embedding.minorminer.find_embedding(G, target_graph,
                                                                                        tries=1000,
                                                                                        max_no_improvement=1000,
                                                                                        chainlength_patience=1000,
                                                                                        timeout=25600, threads=20)
                                        if emb == {}:
                                            emb = deep_search(G, target_graph)
                                            if emb == {}:
                                                # raise Exception("Could not find embedding")
                                                embeddings = lasthope(saved_embedding, problems - 1)
                                                print("Found embeddings for ", problems, " problems.")
                                                assert len(embeddings) == problems - 1
                                                return embeddings
        saved_embedding = emb
        embeddings = lasthope(saved_embedding, problems)
        print("Found embeddings for ", problems, " problems.")
        assert len(embeddings) == problems
        #calculated_embedding = emb
        #separate_embeddings, was_successful = extract_separate_embeddings(separate_embeddings, calculated_embedding,
        #  #                                                                 problems)
        # if was_successful:
        #     print(f"Found embeddings for {problems} problems.")
        #     emb_qubits = get_qubit_list(calculated_embedding)
        #
        #     target_graph = exclude_subgraph_from_pegasus(target_graph, emb_qubits)
        #     problems += 1
        # if len(separate_embeddings) == sample_count:
        #     return separate_embeddings


def nichts(CLIQUE, problems):
    G = nx.Graph()
    cl = nx.complete_graph(CLIQUE)
    for problem in range(problems):
        for e in list(cl.edges()):
            G.add_edge(str(e[0]) + "_" + str(problem), str(e[1]) + "_" + str(problem))
    return G


def partition_graph(G, num_partitions):
    G = nx.convert_node_labels_to_integers(G)
    #adjacency_list = [list(graph.neighbors(i)) for i in graph.nodes]
    adjacency_list = [list(G.adj[i].keys()) for i in G.nodes]
    _, partitions = pymetis.part_graph(num_partitions, adjacency=adjacency_list)

    partition_sets = [{i for i, p in enumerate(partitions) if p == j} for j in range(num_partitions)]
    return partition_sets
def create_subgraphs(G, partition_sets):
    subgraphs = [G.subgraph(partition) for partition in partition_sets]
    return subgraphs


def create_subgraphs_with_buffer(G, partition_sets, thin_buffer=False):
    buffer_nodes = set()

    for u, v in G.edges:
        for i, partition_a in enumerate(partition_sets):
            for j, partition_b in enumerate(partition_sets):
                if i != j and u in partition_a and v in partition_b:
                    buffer_nodes.update([u, v])
    if thin_buffer:
        buffer_nodes = list(buffer_nodes)
        buffer_nodes = buffer_nodes[:len(buffer_nodes) // 4]
    for partition in partition_sets:
        partition.difference_update(buffer_nodes)

    subgraphs = [G.subgraph(partition) for partition in partition_sets]
    buffer_subgraph = G.subgraph(buffer_nodes)

    return subgraphs, buffer_subgraph



def plot_subgraphs(original_graph, subgraphs: list, speicherort=None):
    # Generate random colors for each subgraph
    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for _ in range(6)]) for _ in range(len(subgraphs))]
    unknown_color = "grey"

    node_colors = []
    for node in original_graph.nodes:
        assigned = False
        for i, subgraph in enumerate(subgraphs):
            if node in subgraph:
                node_colors.append("blue")
                assigned = True
                break
        if not assigned:
            node_colors.append(unknown_color)

    plt.figure(figsize=(10, 10))
    dnx.draw_pegasus(original_graph, node_color=node_colors, edge_color="grey", with_labels=False, node_size=20)
    #plt.title("Subgraphs in Pegasus Layout", fontsize=20, pad=-27)
    if speicherort:
        plt.savefig("src/" + speicherort + "subgraphs.png")
    plt.show()