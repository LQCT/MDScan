"""
DESCRIPTION

Created on Sun Jul 25 19:25:18 2021
@author: Roy González Alemán
@contact: roy_gonzalez@fq.uh.cu, roy.gonzalez-aleman@u-psud.fr
"""
import heapq
import itertools as it

import numpy as np
import mdtraj as md
import networkx as nx

import mdpack.vantage as vnt
from mdpack.clusterize import get_node_side2, get_otree_topology2

# @profile
# def get_node_info(node, traj, k):
#     """
#     Get all the necessary information of a particular node.

#     Parameters
#     ----------
#     node : int
#         Node to analyze.
#     traj : mdtraj.Trajectory
#         Trajectory to analyze.
#     k : int
#         Number of nearest neighbors to calculate the CoreDistance(node).

#     Returns
#     -------
#     node_info : tuple
#         Tuple containing the necessary node information:
#             node_info[0]: CoreDistance(node) (inverted for a "max heap")
#             node_info[1]: node index
#             node_info[2]: iterator of the rmsd knn of node
#     """
#     # Get RMSD(node), Kd(node) and knn sorted partition -----------------------
#     # k += 1
#     node_rmsd = md.rmsd(traj, traj, node, precentered=True)
#     node_rmsd_part = np.argpartition(node_rmsd, k)[:k + 1]
#     argsort_indx = node_rmsd[node_rmsd_part].argsort()
#     ordered_indx = node_rmsd_part[argsort_indx]
#     node_knn = iter(ordered_indx)
#     next(node_knn)
#     # Get CoreDistance(A) as Kd -----------------------------------------------
#     node_Kd = node_rmsd[ordered_indx[-1]]
#     node_rmsd = None
#     node_info = (-node_Kd, node, node_knn)
#     return node_info


def get_node_info2(vptree, node, k):
    node_Kd, kheap = vptree.query_node(node, k)
    node_knn = (x[1] for x in sorted(kheap, key=lambda x: -x[0]))
    next(node_knn)
    node_info = (-node_Kd, node, node_knn)
    return node_info


def exhaust_neighborhoods(traj, k):
    """
    Exhaust knn of nodes searching for minimal mrd in a dual-heap approach.

    Parameters
    ----------
    traj : mdtraj.Trajectory
        Trajectory to analyze.
    k : int
        Number of nearest neighbors to calculate the CoreDistance(node).

    Returns
    -------
    Kd_arr : numpy.ndarray
        Array containing the CoreDistance(x) for each x of the trajectory.
    dist_arr : numpy.ndarray
        Array of Minimum Reachability Distances (MRD) between the nodes of the
        tree.
    nn_arr : numpy.ndarray
        Array of nearest neighbors forming edges. Indices correspond to each
        node in the graph and the value correspond to the other node forming
        a directed edge (index->value).
    exhausted : heapq
        Heap containing tuples of (-Kd, node) for every node that could not
        find a neighbor with a lower Kd.

    """
    N = traj.n_frames
    Kd_arr = np.zeros(N, dtype=np.float32)            # coreDistances array
    dist_arr = np.copy(Kd_arr)                        # MRDs array
    nn_arr = np.full(N, -1, dtype=np.int32)           # nearest neighbors array
    pool = []                                         # main heap
    exhausted = []                                    # auxiliary heap
    not_visited = set(range(N))                       # tracking unvisited
    # pool_lens = []
    # =========================================================================
    # Initialize the vantage tree datastructure
    # =========================================================================
    indices = np.arange(0, traj.n_frames, dtype=np.int)
    limit = round(int(N*0.1))
    sample_size = round(limit/4)
    indices = np.arange(0, traj.n_frames, dtype=np.int)
    vptree = vnt.vpTree(limit, sample_size, traj)
    vptree.getBothTrees(indices, traj)
    # =========================================================================
    # Find node 'A' whose neighborhood will be exhausted
    # =========================================================================
    while True:
        # pool_lens.append(len(pool))
        # get ( Kd(A), A, RMSD(A), and the sorted knn(A) partition ) ----------
        try:
            A_Kd, A, A_rmsd_knn = heapq.heappop(pool)
        # if pool is empty, check for a random not-yet-visited node -----------
        except IndexError:
            try:
                A = not_visited.pop()
                # A_Kd, A, A_rmsd_knn = get_node_info(A, traj, k)
                A_Kd, A, A_rmsd_knn = get_node_info2(vptree, A, k)
                Kd_arr[A] = -A_Kd
            # if all nodes visited, break & check exhausted heap --------------
            except KeyError:
                break
        # =====================================================================
        # Exhaust knn of A searching for a node 'B' for which: Kd(A) > Kd(B)
        # =====================================================================
        while True:
            try:
                # consume the knn(A) iterator (in rmsd ordering) --------------
                B = int(next(A_rmsd_knn))
            except StopIteration:
                # ____ if knn(A) exhausted, send A to exhausted heap then break
                heapq.heappush(exhausted, (A_Kd, A))
                break
            if B in not_visited:
                # B_info = get_node_info(B, traj, k)
                B_info = get_node_info2(vptree, B, k)
                heapq.heappush(pool, B_info)
                not_visited.remove(B)
                B_Kd = -B_info[0]
                Kd_arr[B] = B_Kd
            else:
                B_Kd = Kd_arr[B]
            # cases where Kd(A) > Kd(B) before exhaustion ---------------------
            if -A_Kd > B_Kd:
                nn_arr[A] = B
                dist_arr[A] = -A_Kd
                break
    # return Kd_arr, dist_arr, nn_arr, exhausted, pool_lens
    return Kd_arr, dist_arr, nn_arr, exhausted


def get_tree_side(root, nn_arr):
    """
    Parameters
    ----------
    root : int
        Node starting the expansion of one sub-tree (tree component).
    nn_arr : numpy.ndarray
        Array of nearest neighbors forming edges. Indices correspond to each
        node in the graph and the value correspond to the other node forming
        a directed edge (index->value).

    Returns
    -------
    root_side : set
        Set of nodes forming the sub-tree expanded from the desconnection of
        root node.
    """
    root_side = set()
    to_visit = [root]
    while True:
        try:
            x = to_visit.pop()
        except IndexError:
            return root_side
        expansion = np.where(nn_arr == x)[0]
        to_visit.extend(expansion)
        root_side.update(expansion)


#@profile
def join_exhausted(exhausted, Kd_arr, dist_arr, nn_arr, traj):
    # get disconnected components
    topo_forest = get_otree_topology2(nn_arr)
    components = np.zeros(Kd_arr.size, dtype=np.int)
    iKd = np.ndarray(Kd_arr.size, dtype=np.float)
    counter = it.count(1)
    for k, node in exhausted:
        component = get_node_side2(node, topo_forest)
        components[np.fromiter(component, np.int)] = next(counter)
    # join components
    while exhausted:
        kdneg, idx = heapq.heappop(exhausted)
        kd = -kdneg
        icomponent = components[idx]
        iforest = np.where(components == icomponent)[0]
        idx_rmsd = md.rmsd(traj, traj, idx, precentered=True)
        iKd.fill(kd)
        i_mdr = np.array([iKd, Kd_arr, idx_rmsd]).max(axis=0)
        i_mdr[iforest] = np.inf
        acceptor = i_mdr.argmin()
        distance = i_mdr[acceptor]
        if distance == np.inf:
            nn_arr[idx] = idx
            dist_arr[idx] = 0
            break
        else:
            components[iforest] = components[acceptor]
            dist_arr[idx] = distance
            nn_arr[idx] = acceptor
    return dist_arr, nn_arr


def check_tree(N, nn_arr, dist_arr):
    """
    Check that neighbors array represents a tree.

    Parameters
    ----------
    N : int
        Number of frames in trajectory.
    nn_arr : numpy.ndarray
        Array of nearest neighbors forming edges. Indices correspond to each
        node in the graph and the value correspond to the other node forming
        a directed edge (index->value).
    dist_arr : numpy.ndarray
        Array of Minimum Reachability Distances (MRD) between the nodes of the
        tree.

    Returns
    -------
    graph : nx.Graph()
        The graph represented by the neighbors array.

    """
    print('\n\nOnce upon a time, there was an approximate Graph that ...')
    graph = nx.Graph()
    graph.add_nodes_from(range(0, N))
    for i, x in enumerate(nn_arr):
        graph.add_edge(i, x, weight=dist_arr[i])
    try:
        nx.find_cycle(graph)
    except nx.exception.NetworkXNoCycle:
        nx.write_graphml(graph, 'tree.graphml')
        print('Contained No Cycles:   :O')
        if nx.is_connected(graph):
            print('Was connected:         ;)')
        if nx.is_tree(graph):
            print('Was a Tree:            :D')
    print('Had weight:            {:6.4f}'.format(dist_arr.sum()))
    return graph


def get_exact_MST(N, traj, Kd_arr, algorithm):
    """
    Get the exact MST from the complete graph of MDR (with networkx).

    Parameters
    ----------
    N : int
        Number of frames in trajectory.
    traj : mdtraj.Trajectory
        Trajectory to analyze.
    Kd_arr : numpy.ndarray
        Array containing the CoreDistance(x) for each x of the trajectory.
    algorithm : str
        Which algorithm to use for the 'exact' computation of MST.

    Returns
    -------
    total_weights : float
        Sum of edge's weights for the MST.

    """
    # Add nodes and lower triangle edges to graph -----------------------------
    graph = nx.Graph()
    graph.add_nodes_from(range(0, N))
    for i in range(N):
        node_rmsd = md.rmsd(traj, traj, i, precentered=True)[:i]
        for j, d in enumerate(node_rmsd):
            mdr_ij = max([Kd_arr[i], Kd_arr[j], d])
            graph.add_edge(i, j, weight=mdr_ij)
    # Get the minimum spanning tree and the weights ---------------------------
    t = nx.algorithms.tree.minimum_spanning_tree(graph, algorithm=algorithm)
    weights = []
    for x, y in t.edges():
        weights.append(t.get_edge_data(x, y)['weight'])
    total_weights = sum(weights)
    return total_weights
