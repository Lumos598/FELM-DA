import random
from typing import List
from attack import attacks
from collections import defaultdict, deque
import time


def make_centralized_matrix(nodes_n, byzantine_probs, attack):
    """
    Generate centralized matrix.
    
    Args:
        nodes_n: the number of nodes
        byzantine_porbs: the probs of one node become byzantine node
        attack: attack method
    """
    matrix = [[0] + [1] * (nodes_n-1)] + [
        [1] + [0] * (nodes_n-1)
        for i in range(nodes_n-1)
    ]
    attacks_matrix = [None]
    for i in range(1, nodes_n):
        if attack and random.random() < byzantine_probs:
            attacks_matrix.append(attacks[attack]())
        else:
            attacks_matrix.append(None)
    return matrix, attacks_matrix


def make_matrix(nodes_n, connect_probs, byzantine_probs, attack):
    """
    Generate matrix by number of nodes, connection probability and byzantine probability.

    Args:
        nodes_n (int): number of nodes
        connect_probs (float): connection probability
        byzantine_probs (float): byzantine probability

    Returns:
        matrix (List): adj matrix
        attacks (List): attacks
    """
    random.seed(int(time.time() * 1000))
    matrix = []
    attack_matrix = []
    non_byzantines = []
    for i in range(nodes_n):
        if attack and random.random() < byzantine_probs:
            attack_matrix.append(attacks[attack]())
        else:
            attack_matrix.append(None)
            non_byzantines.append(i)
    for i in range(nodes_n):
        adj_i = []
        has_benign = False
        for j in range(nodes_n):
            if i == j or random.random() > connect_probs:
                adj_i.append(0)
            else:
                adj_i.append(1)
                if j in non_byzantines:
                    has_benign = True
        if not has_benign:
            # make sure every node has a benign neighbor except self
            if len(non_byzantines) <= 1:
                continue
            benigh_n = random.choice(non_byzantines)
            while benigh_n == i:
                benigh_n = random.choice(non_byzantines)
            adj_i[benigh_n] = 1
        matrix.append(adj_i)
    matrix = ensure_connected(matrix, attack_matrix)
    return matrix, attack_matrix


def ensure_connected(matrix: List[List], attacks: List):
    # ensure matrix is connected
    # strongly connected graph
    sub_gs = []
    n = len(matrix)
    visited = {j: False for j in range(n)}
    while True:
        start = None
        for k, v in visited.items():
            if not v and attacks[k] is None:
                start = k
                break
        if start is None:
            break
        sub_g = [start]
        q = [start]
        while len(q) > 0:
            k = q.pop(0)
            visited[k] = True
            for v in matrix[k]:
                if v == 1 and not visited[v] and attacks[v] is None:
                    visited[v] = True
                    q.append(v)
                    sub_g.append(v)
        sub_gs.append(sub_g)
    for i in range(len(sub_gs) - 1):
        src = random.choice(sub_gs[i])
        dst = random.choice(sub_gs[i + 1])
        matrix[src][dst] = 1
    return matrix


def dfs(node, visited, adj_list, stack):
    visited[node] = True
    for neighbor in adj_list[node]:
        if not visited[neighbor]:
            dfs(neighbor, visited, adj_list, stack)
    stack.append(node)


def find_scc(adj_matrix):
    n = len(adj_matrix)
    visited = [False] * n
    adj_list = defaultdict(list)
    for i in range(n):
        for j in range(n):
            if adj_matrix[i][j] == 1:
                adj_list[i].append(j)

    stack = []
    for i in range(n):
        if not visited[i]:
            dfs(i, visited, adj_list, stack)

    # Create the transpose of the original graph
    transpose = defaultdict(list)
    for i in range(n):
        for j in adj_list[i]:
            transpose[j].append(i)

    visited = [False] * n
    scc_count = 0
    scc_nodes = []
    while stack:
        node = stack.pop()
        if not visited[node]:
            scc_count += 1
            component = []
            dfs(node, visited, transpose, component)
            scc_nodes.append(component)

    return scc_count, scc_nodes


def make_strongly_connected(adj_matrix):
    scc_count, scc_nodes = find_scc(adj_matrix)
    if scc_count == 1:
        return adj_matrix, 0  # Already strongly connected

    # Add edges to make the graph strongly connected
    added_edges = scc_count - 1
    for i in range(1, scc_count):
        for node in scc_nodes[i]:
            for prev_node in scc_nodes[i - 1]:
                adj_matrix[prev_node][node] = 1

    return adj_matrix, added_edges


if __name__ == "__main__":
    adj_matrix, attacks = make_matrix(10, 0.4, 0.5, "random")
    adj_matrix, edges_added = make_strongly_connected(adj_matrix)
    print(edges_added)
    print(adj_matrix)
    print(attacks)
