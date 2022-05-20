"""
DESCRIPTION

Created on Sun Jul 25 19:25:18 2021
@author: Roy González Alemán
@contact: roy_gonzalez@fq.uh.cu, roy.gonzalez-aleman@u-psud.fr
"""
import heapq
import itertools as it
# from collections import deque
# from bitarray import bitarray as ba

import numpy as np
import mdtraj as md
# import networkx as nx
from numba import jit

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


@jit(nopython=True, fastmath=True)
def get_acceptor(Kd_arr, idx_rmsd, iforest):
    Kd_arr1 = Kd_arr.copy()
    Kd_arr1[iforest] = np.inf
    idx_rmsd[iforest] = np.inf
    min_found = np.inf
    for i in range(Kd_arr1.size):
            kd_val = Kd_arr1[i]
            rms_val = idx_rmsd[i]
            if (kd_val < min_found) and (rms_val < min_found):
                acceptor = i
                min_found = max(kd_val, rms_val)
    return acceptor, min_found


def get_node_info2(vptree, node, k):
    node_Kd, kheap = vptree.query_node(node, k)
    node_knn = (x[1] for x in sorted(kheap, key=lambda x: -x[0]))
    next(node_knn)
    node_info = (-node_Kd, node, node_knn)
    return node_info


def exhaust_neighborhoods(traj, k, nsplits):
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
    D1 = []
    # =========================================================================
    # Initialize the vantage tree datastructure
    # =========================================================================
    limit = N
    for i in range(nsplits):
        limit = int(limit / 2)
    sample_size = int(round(limit / 5))
    indices = np.arange(0, traj.n_frames, dtype=int)
    vptree = vnt.vpTree(nsplits, sample_size, traj)
    vptree.getBothTrees(indices, traj)

    # =========================================================================
    # Find node 'A' whose neighborhood will be exhausted
    # =========================================================================
    while True:
        D1.append((len(pool), len(exhausted)))
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
    vptree = None
    return Kd_arr, dist_arr, nn_arr, exhausted, D1


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
    # get disconnected components
    topo_forest = get_otree_topology2(nn_arr)
    components = np.zeros(Kd_arr.size, dtype=int)
    counter = it.count(1)
    for k, node in exhausted:
        component = get_node_side2(node, topo_forest)
        components[np.fromiter(component, int)] = next(counter)
    # join components
    exhausted_ordered = []
    while exhausted:
        kdneg, idx = heapq.heappop(exhausted)
        exhausted_ordered.append(idx)
        icomponent = components[idx]
        iforest = (components == icomponent).nonzero()[0]
        idx_rmsd = md.rmsd(traj, traj, idx, precentered=True)
        # iKd = np.full(Kd_arr.size, -kdneg)
        # i_mdr = np.array([iKd, Kd_arr, idx_rmsd]).max(axis=0)
        # i_mdr[iforest] = np.inf
        # acceptor = i_mdr.argmin()
        # distance = i_mdr[acceptor]
        acceptor, distance = get_acceptor(Kd_arr, idx_rmsd, iforest)
        # if acceptor1 != acceptor:
            # print(acceptor, acceptor1, distance, distance1)
        if distance == np.inf:
            nn_arr[idx] = idx
            dist_arr[idx] = 0
            break
        else:
            components[iforest] = components[acceptor]
            dist_arr[idx] = distance
            nn_arr[idx] = acceptor
    return dist_arr, nn_arr, exhausted_ordered


# def check_tree(N, nn_arr, dist_arr):
#     """
#     Check that neighbors array represents a tree.

#     Parameters
#     ----------
#     N : int
#         Number of frames in trajectory.
#     nn_arr : numpy.ndarray
#         Array of nearest neighbors forming edges. Indices correspond to each
#         node in the graph and the value correspond to the other node forming
#         a directed edge (index->value).
#     dist_arr : numpy.ndarray
#         Array of Minimum Reachability Distances (MRD) between the nodes of the
#         tree.

#     Returns
#     -------
#     graph : nx.Graph()
#         The graph represented by the neighbors array.

#     """
#     graph = nx.Graph()
#     graph.add_nodes_from(range(0, N))
#     for i, x in enumerate(nn_arr):
#         graph.add_edge(i, x, weight=dist_arr[i])
#     return graph


# def get_exact_MST(N, traj, Kd_arr):
#     """
#     Get the exact MST from the complete graph of MDR (with networkx).

#     Parameters
#     ----------
#     N : int
#         Number of frames in trajectory.
#     traj : mdtraj.Trajectory
#         Trajectory to analyze.
#     Kd_arr : numpy.ndarray
#         Array containing the CoreDistance(x) for each x of the trajectory.
#     algorithm : str
#         Which algorithm to use for the 'exact' computation of MST.

#     Returns
#     -------
#     total_weights : float
#         Sum of edge's weights for the MST.

#     """
#     # Add nodes and lower triangle edges to graph -----------------------------
#     graph = nx.Graph()
#     graph.add_nodes_from(range(0, N))
#     # vault = []
#     for i in range(N):
#         node_rmsd = md.rmsd(traj, traj, i, precentered=True)[:i]
#         for j, d in enumerate(node_rmsd):
#             if i == j:
#                 print('equals ', i, j)
#             # vault.append((i, j))
#             mdr_ij = max([Kd_arr[i], Kd_arr[j], d])
#             graph.add_edge(i, j, weight=mdr_ij)
#     # Get the minimum spanning tree and the weights ---------------------------
#     t = nx.algorithms.tree.minimum_spanning_tree(graph, algorithm='kruskal')
#     weights = []
#     for x, y in t.edges():
#         weights.append(t.get_edge_data(x, y)['weight'])
#     total_weights = sum(weights)
#     return t, total_weights


# def get_prim_mst(traj, Kd_arr):
#     # reserve the space for i, j, and w arrays
#     N = traj.n_frames
#     M = int(N * (N - 1) / 2)
#     i_array = np.zeros(M, dtype=np.int32)
#     j_array = np.zeros(M, dtype=np.int32)
#     w_array = np.zeros(M, dtype=np.float32)
#     # get mrd matrix
#     counter = it.count()
#     for i in range(N):
#         node_rmsd = md.rmsd(traj, traj, i, precentered=True)[:i]
#         for j, d in enumerate(node_rmsd):
#             c = next(counter)
#             i_array[c] = i
#             j_array[c] = j
#             w_array[c] = max([d, Kd_arr[i], Kd_arr[j]])
#     # order edges increasingly by weight
#     order = w_array.argsort()
#     w_array = w_array[order]
#     i_array = i_array[order]
#     j_array = j_array[order]
#     del(order)
#     # define auxiliar binary vectors
#     i_bit = np.zeros(M, bool)
#     j_bit = np.zeros(M, bool)
#     # Prim MST procedure
#     edges = []
#     last_added = i_array[0]
#     while True:
#         i_bit[(i_array == last_added)] = True
#         j_bit[(j_array == last_added)] = True
#         try:
#             last_added, insider, pos = find_01(i_bit, j_bit, i_array, j_array)
#         except TypeError:
#             break
#         edges.append((last_added, insider, w_array[pos]))
#     # Create the networkx graph
#     g = nx.Graph()
#     g.add_weighted_edges_from(edges)
#     weight = sum([x[2] for x in edges])
#     del edges
#     print('Are you a connected graph ?: {}'.format(nx.is_connected(g)))
#     try:
#         cycle = nx.find_cycle(g)
#     except:
#         cycle = 'Not cycles my Lord !'
#     print('May I see your cycles ?: {}'.format(cycle))
#     print('Are you a tree my dear ?: {}'.format(nx.is_tree(g)))
#     print('Ok, give me your weight !!!: {:3.5f}'.format(weight))
#     return g



# def get_prim_mst2(traj, Kd_arr):
#     # reserve the space for i, j, and w arrays
#     N = traj.n_frames
#     M = int(N * (N - 1) / 2)
#     i_array = np.zeros(M, dtype=np.int32)
#     j_array = np.zeros(M, dtype=np.int32)
#     w_array = np.zeros(M, dtype=np.float32)
#     # get mrd matrix
#     startCount = 0
#     for i in range(N):
#         node_rmsd = md.rmsd(traj, traj, i, precentered=True)[:i]
#         startCount = iterate_array(i, node_rmsd, Kd_arr, startCount,
#                                    i_array, j_array, w_array)
#     # order edges increasingly by weight
#     order = w_array.argsort()
#     w_array = w_array[order]
#     i_array = i_array[order]
#     j_array = j_array[order]
#     del(order)
#     # define auxiliar binary vectors
#     i_bit = np.zeros(M, bool)
#     j_bit = np.zeros(M, bool)
#     # Prim MST procedure
#     edges = []
#     last_added = i_array[0]
#     while True:
#         i_bit[(i_array == last_added)] = True
#         j_bit[(j_array == last_added)] = True
#         try:
#             last_added, insider, pos = find_01(i_bit, j_bit, i_array, j_array)
#         except TypeError:
#             break
#         edges.append((last_added, insider, w_array[pos]))
#     # Create the networkx graph
#     g = nx.Graph()
#     g.add_weighted_edges_from(edges)
#     weight = sum([x[2] for x in edges])
#     del edges
#     print('Are you a connected graph ?: {}'.format(nx.is_connected(g)))
#     try:
#         cycle = nx.find_cycle(g)
#     except:
#         cycle = 'Not cycles my Lord !'
#     print('May I see your cycles ?: {}'.format(cycle))
#     print('Are you a tree my dear ?: {}'.format(nx.is_tree(g)))
#     print('Ok, give me your weight !!!: {:3.5f}'.format(weight))
#     return g

#%
@jit(nopython=True, fastmath=True)
def find_01(i_bit, j_bit, i_array, j_array):
    for i in range(i_bit.size):
        I = i_bit[i]
        J = j_bit[i]
        if I ^ J:
            if I:
                return j_array[i], i_array[i], i
            elif J:
                return i_array[i], j_array[i], i


@jit(nopython=True, fastmath=True)
def find_01_set(i_array, j_array, w_array, auxiliar, mst):
    for i in range(i_array.size):
        if auxiliar[i]:
            i_val = i_array[i]
            j_val = j_array[i]
            I = i_val in mst
            J = j_val in mst
            if I ^ J:
                auxiliar[i] = False
                if I:
                    mst.add(j_val)
                    return i_val, j_val, w_array[i]
                elif J:
                    mst.add(i_val)
                    return i_val, j_val, w_array[i]


@jit(nopython=True, fastmath=True)
def find_01_set2(mix, auxiliar, mst):
    for i in range(mix.size):
        if auxiliar[i]:
            i_val = mix['i'][i]
            j_val = mix['j'][i]
            I = i_val in mst
            J = j_val in mst
            if I ^ J:
                auxiliar[i] = False
                if I:
                    mst.add(j_val)
                    return i_val, j_val, mix['w'][i]
                elif J:
                    mst.add(i_val)
                    return i_val, j_val, mix['w'][i]


@jit(nopython=True, fastmath=True)
def iterate_array2(i, node_rmsd, Kd_arr, startCount, mix):
    c = startCount
    for j in range(node_rmsd.size):
        d = node_rmsd[j]
        mix['i'][c] = i
        mix['j'][c] = j
        mix['w'][c] = max([d, Kd_arr[i], Kd_arr[j]])
        c += 1
    return c

#%
# def get_prim_set(traj, Kd_arr):
#     # reserve the space for i, j, and w arrays
#     print('Reserving the space')
#     N = traj.n_frames
#     M = int(N * (N - 1) / 2)
#     # i_array = np.zeros(M, dtype=np.int32)
#     # j_array = np.zeros(M, dtype=np.int32)
#     # w_array = np.zeros(M, dtype=np.float32)
#     mix = np.recarray(M, dtype=[('i', np.int32), ('j', np.int32), ('w', np.float32)])
#     # get mrd matrix
#     print('Calculating MRD matrix')
#     startCount = 0
#     for i in range(N):
#         node_rmsd = md.rmsd(traj, traj, i, precentered=True)[:i]
#         startCount = iterate_array2(i, node_rmsd, Kd_arr, startCount, mix)

#     # order edges increasingly by weight
#     print('Ordering edges before processing')
#     mix.sort(order='w')
#     # Prim MST procedure
#     print('Starting MST procedure')
#     edges = []
#     mst = set()
#     mst.add(mix['i'][0])
#     auxiliar = np.ones(M, dtype=bool)
#     while True:
#         try:
#             x, y, z = find_01_set2(mix, auxiliar, mst)
#             edges.append((x, y, z))
#         except TypeError:
#             break
#     del auxiliar
#     del mst
#     # Create the networkx graph
#     g = nx.Graph()
#     g.add_weighted_edges_from(edges)
#     weight = g.size(weight='weight')
#     del edges
#     print('Are you a connected graph ?: {}'.format(nx.is_connected(g)))
#     try:
#         cycle = nx.find_cycle(g)
#     except:
#         cycle = 'Not cycles my Lord !'
#     print('May I see your cycles ?: {}'.format(cycle))
#     print('Are you a tree my dear ?: {}'.format(nx.is_tree(g)))
#     print('Ok, give me your weight !!!: {:3.5f}'.format(weight))
#     return g


@jit(nopython=True, fastmath=True)
def iterate_array(i, node_rmsd, Kd_arr, startCount, i_array, j_array, w_array):
    c = startCount
    for j in range(node_rmsd.size):
        d = node_rmsd[j]
        i_array[c] = i
        j_array[c] = j
        w_array[c] = max([d, Kd_arr[i], Kd_arr[j]])
        c += 1
    return c


def get_mst(edges):
    # Get the minimum spanning tree and the weights ---------------------------
    graph = nx.Graph()
    graph.add_nodes_from(range(0, len(edges)))
    graph.add_weighted_edges_from(edges)
    t = nx.algorithms.tree.minimum_spanning_tree(graph, algorithm='prim')
    weights = []
    for x, y in t.edges():
        weights.append(t.get_edge_data(x, y)['weight'])
    total_weights = sum(weights)
    return t, total_weights
