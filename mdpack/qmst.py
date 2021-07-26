"""
DESCRIPTION

Created on Sun Jul 25 19:25:18 2021
@author: Roy González Alemán
@contact: roy_gonzalez@fq.uh.cu, roy.gonzalez-aleman@u-psud.fr
"""
import numpy as np
import heapq

import mdtraj as md
import networkx as nx


def get_node_info(node, traj, k):
    """
    Get all the necessary information of a particular node.

    Parameters
    ----------
    node : int
        Node to analyze.
    traj : mdtraj.Trajectory
        Trajectory to analyze.
    k : int
        Number of nearest neighbors to calculate the CoreDistance(node).

    Returns
    -------
    node_info : tuple
        Tuple containing the necessary node information:
            node_info[0]: CoreDistance(node) (inverted for a "max heap")
            node_info[1]: node index
            node_info[2]: iterator of the rmsd knn of node
    """
    # Get RMSD(node), Kd(node) and knn sorted partition -----------------------
    # k += 1
    node_rmsd = md.rmsd(traj, traj, node, precentered=True)
    node_rmsd_part = np.argpartition(node_rmsd, k)[:k + 1]
    argsort_indx = node_rmsd[node_rmsd_part].argsort()
    ordered_indx = node_rmsd_part[argsort_indx]
    node_knn = iter(ordered_indx)
    next(node_knn)
    # Get CoreDistance(A) as Kd -----------------------------------------------
    node_Kd = node_rmsd[ordered_indx[-1]]
    node_rmsd = None
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
                A_Kd, A, A_rmsd_knn = get_node_info(A, traj, k)
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
                B_info = get_node_info(B, traj, k)
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


def join_exhausted(exhausted, Kd_arr, dist_arr, nn_arr, traj):
    """
    Join components rooted at the exhausted nodes to the main tree.

    Parameters
    ----------
    exhausted : heapq
        Heap containing tuples of (-Kd, node) for every node that could not
        find a neighbor with a lower Kd.
    Kd_arr : numpy.ndarray
        Array containing the CoreDistance(x) for each x of the trajectory.
    dist_arr : numpy.ndarray
        Array of Minimum Reachability Distances (MRD) between the nodes of the
        tree.
    nn_arr : numpy.ndarray
        Array of nearest neighbors forming edges. Indices correspond to each
        node in the graph and the value correspond to the other node forming
        a directed edge (index->value).
    traj : mdtraj.Trajectory
        Trajectory to analyze.

    Returns
    -------
    dist_arr : numpy.ndarray
        Updated array of Minimum Reachability Distances (MRD) between the nodes
        of the tree.
    nn_arr : numpy.ndarray
        Updated array of nearest neighbors forming edges. Indices correspond to
        each node in the graph and the value correspond to the other node
        forming a directed edge (index->value).
    """
    N = Kd_arr.size
    minim_Kd, minim_A = heapq.nlargest(1, exhausted)[0]
    A_Kd_arr = np.zeros(N, np.float32)
    reordered_exhausted = sorted(exhausted, key=lambda x: x[0], reverse=False)
    for neg, A in reordered_exhausted:
        if neg < minim_Kd:
            node_side = get_tree_side(A, nn_arr)
            A_Kd_arr.fill(-neg)
            A_rms = md.rmsd(traj, traj, A, precentered=True)
            A_rms[A] = np.inf
            A_triple = np.array([A_Kd_arr, Kd_arr, A_rms])
            A_minim_mdr = A_triple.max(axis=0)
            candidates = A_minim_mdr.argsort()
            for x in candidates:
                if x not in node_side:
                    nn_arr[A] = x
                    dist_arr[A] = A_minim_mdr[x]
                    break
        else:
            A_rms = md.rmsd(traj, traj, A, precentered=True)
            nn_arr[A] = minim_A
            dist_arr[A] = max(A_rms[minim_A], Kd_arr[A], minim_Kd)
    dist_arr[minim_A] = 0
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
