#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 22:28:25 2020
@author: rga
@contact: roy_gonzalez@fq.uh.cu, roy.gonzalez-aleman@u-psud.fr
"""
import os
import heapq
import pickle
import argparse
import numpy as np
import pandas as pd
import itertools as it
from collections import defaultdict, deque

import mdtraj as md
import networkx as nx

valid_tops = set(['pdb', 'pdb.gz', 'h5', 'lh5', 'prmtop', 'parm7', 'prm7',
                  'psf', 'mol2', 'hoomdxml', 'gro', 'arc', 'hdf5', 'gsd'])
valid_trajs = set(['arc', 'dcd', 'binpos', 'xtc', 'trr', 'hdf5', 'h5', 'ncdf',
                   'netcdf', 'nc', 'pdb.gz', 'pdb', 'lh5', 'crd', 'mdcrd',
                   'inpcrd', 'restrt', 'rst7', 'ncrst', 'lammpstrj', 'dtr',
                   'stk', 'gro', 'xyz.gz', 'xyz', 'tng', 'xml', 'mol2',
                   'hoomdxml', 'gsd'])


def parse_arguments():
    """
    Parse all the arguments from the CLI.

    Returns
    -------
    user_inputs : parser.argparse
        namespace with user input arguments.

    """
    # Initializing argparse ---------------------------------------------------
    desc = '\nMDScan: An efficient approach to the RMSD-Based HDBSCAN \
        Clustering of long Molecular Dynamics'
    parser = argparse.ArgumentParser(prog='mdscan',
                                     description=desc,
                                     add_help=True,
                                     epilog='As simple as that ;)',
                                     allow_abbrev=False,
                                     usage='%(prog)s -traj trajectory [options]')
    # Arguments: loading trajectory -------------------------------------------
    traj = parser.add_argument_group(title='Trajectory options')
    traj.add_argument('-traj', dest='trajectory', action='store',
                      help='Path to trajectory file (pdb/dcd) \
                      [default: %(default)s]', type=str,
                      metavar='trajectory', required=True)
    traj.add_argument('-top', dest='topology', action='store',
                      help='Path to the topology file (psf/pdb)', type=str,
                      required=False, metavar='topology', default=None)
    traj.add_argument('-first', dest='first', action='store',
                      help='First frame to analyze (start counting from 0)\
                      [default: %(default)s]', type=int, required=False,
                      default=0, metavar='first_frame')
    traj.add_argument('-last', dest='last', action='store',
                      help='Last frame to analyze (start counting from 0)\
                      [default: last frame]', type=int, required=False,
                      default=None, metavar='last_frame')
    traj.add_argument('-stride', dest='stride', action='store',
                      help='Stride of frames to analyze\
                      [default: %(default)s]', type=int, required=False,
                      default=1, metavar='stride')
    traj.add_argument('-sel', dest='selection', action='store',
                      help='Atom selection (MDTraj syntax)\
                      [default: %(default)s]', type=str, required=False,
                      default='all', metavar='selection')
    # Arguments: clustering parameters ----------------------------------------
    clust = parser.add_argument_group(title='Clustering options')
    clust.add_argument('-min_samples', action='store', dest='k',
                       help='Number of k nearest neighbors to consider\
                       [default: %(default)s]',
                       type=int, required=False, default=10, metavar='k')
    clust.add_argument('-min_clust_size', action='store',
                       dest='min_clust_size',
                       help='Minimum number of points in agrupations to be\
                       considered as clusters [default: %(default)s]',
                       type=int, required=False, default=2, metavar='m')
    clust.add_argument('-clust_sel_met', action='store', dest='clust_sel_met',
                       help='Method used to select clusters from the condensed\
                       tree [default: %(default)s]', type=str, required=False,
                       default='eom', choices=['eom', 'leaf'])
    # Arguments: analysis -----------------------------------------------------
    out = parser.add_argument_group(title='Output options')
    out.add_argument('-odir', action='store', dest='outdir',
                     help='Output directory to store analysis\
                     [default: %(default)s]',
                     type=str, required=False, default='./', metavar='.')
    user_inputs = parser.parse_args()
    return user_inputs


def is_valid_traj(traj, valid_trajs):
    """
    Check if the trajectory extension is supported by MDTraj engine.

    Parameters
    ----------
    traj : str
        Path to the trajectory file.
    valid_trajs : set
        Set of supported trajectory extensions.

    Raises
    ------
    ValueError
        If trajectory extension is not supported by MDTraj.

    Returns
    -------
    bool
        True if trajectory extension is supported.

    """
    traj_ext = traj.split('.')[-1]
    if traj_ext not in valid_trajs:
        raise ValueError('The trajectory format "{}" '.format(traj_ext) +
                         'is not available. Valid trajectory formats '
                         'are: {}'.format(valid_trajs))
    return True


def traj_needs_top(traj):
    """
    Determine if trajectory extension does not contain topological information.

    Parameters
    ----------
    traj : str
        Path to the trajectory file.

    Returns
    -------
    bool
        True if trajectory needs topological information.

    """
    traj_ext = traj.split('.')[-1]
    if traj_ext in ['h5', 'lh5', 'pdb']:
        return False
    return True


def is_valid_top(topology, valid_tops):
    """
    Check if the topology extension is supported by MDTraj engine.

    Parameters
    ----------
    topology : str
        Path to the trajectory file.
    valid_tops : set
        Set of supported topology extensions.

    Raises
    ------
    ValueError
        If topology extension is not supported by MDTraj.

    Returns
    -------
    bool
        DESCRIPTION.

    """
    try:
        top_ext = topology.split('.')[-1]
    except AttributeError:
        raise ValueError('You should pass a topology object. '
                         'Valid topology formats are: {}'.format(valid_tops))

    if top_ext not in valid_tops:
        raise ValueError('The topology format "{}"'.format(top_ext) +
                         'is not available. Valid topology formats'
                         'are: {}'.format(valid_tops))
    return True


def load_raw_traj(traj, valid_trajs, topology=None):
    """
    Load the whole trajectory without any modifications.

    Parameters
    ----------
    traj : str
        Path to the trajectory file.
    valid_trajs : set
        Set of supported trajectory extensions.
    topology : str, optional
        Path to the trajectory file. The default is None.

    Returns
    -------
    mdtraj.Trajectory
        Raw trajectory.

    """
    if is_valid_traj(traj, valid_trajs) and traj_needs_top(traj):
        if is_valid_top(topology, valid_tops):
            return md.load(traj, top=topology)

    if is_valid_traj(traj, valid_trajs) and not traj_needs_top(traj):
        return md.load(traj)


def shrink_traj_selection(traj, selection):
    """
    Select a subset of atoms from the trajectory.

    Parameters
    ----------
    traj : mdtraj.Trajectory
        Trajectory object to which selection will be applied.
    selection : str
        Any MDTraj valid selection.

    Raises
    ------
    ValueError
        If specified selection is not valid.
        If specified selection corresponds to no atoms.

    Returns
    -------
    traj : mdtraj.Trajectory
        Trajectory containing the subset of specified atoms.

    """
    if selection != 'all':
        try:
            sel_indx = traj.topology.select(selection)
        except Exception:
            raise ValueError('Specified selection is invalid')
        if sel_indx.size == 0:
            raise ValueError('Specified selection corresponds to no atoms')
        traj = traj.atom_slice(sel_indx, inplace=True)
    return traj


def shrink_traj_range(first, last, stride, traj):
    """
    Select a subset of frames from the trajectory.

    Parameters
    ----------
    first : int
        First frame to consider (0-based indexing).
    last : TYPE
        Last frame to consider (0-based indexing).
    stride : TYPE
        Stride (step).
    traj : mdtraj.Trajectory
        Trajectory object to which slicing will be applied.

    Raises
    ------
    ValueError
        If first, last or stride are falling out of their valid ranges.

    Returns
    -------
    mdtraj.Trajectory
        Trajectory containing the subset of specified frames.

    """
    # Calculate range of available intervals ----------------------------------
    N = traj.n_frames
    first_range = range(0, N - 1)
    last_range = range(first + 1, N)
    try:
        delta = last - first
    except TypeError:
        delta = N - first
    stride_range = range(1, delta)
    # Raising if violations ---------------------------------------------------
    if first not in first_range:
        raise ValueError('"first" parameter should be in the interval [{},{}]'
                         .format(first_range.start, first_range.stop))
    if last and (last not in last_range):
        raise ValueError('"last" parameter should be in the interval [{},{}]'
                         .format(last_range.start, last_range.stop))
    if stride not in stride_range:
        raise ValueError('"stride" parameter should be in the interval [{},{}]'
                         .format(stride_range.start, stride_range.stop))
    # Slicing trajectory ------------------------------------------------------
    sliced = slice(first, last, stride)
    if sliced not in [slice(0, N, 1), slice(0, None, 1)]:
        return traj.slice(sliced)
    return traj


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
    k += 1
    node_rmsd = md.rmsd(traj, traj, node, precentered=True)
    node_rmsd_part = np.argpartition(node_rmsd, k)[:k]
    argsort_indx = node_rmsd[node_rmsd_part].argsort()
    ordered_indx = node_rmsd_part[argsort_indx]
    node_knn = iter(ordered_indx)
    next(node_knn)
    # Get CoreDistance(A) as Kd -----------------------------------------------
    node_Kd = node_rmsd[ordered_indx[-1]]
    node_rmsd = None
    node_info = (-node_Kd, node, node_knn)
    return node_info


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


def cut_tree(A, B, forest_top):
    """
    Cut one of the tree topologies into two sub-trees.

    Parameters
    ----------
    A : int
        Node index.
    B : int
        Node index.
    forest_top : dict
        Topology of the current forest from the original tree.

    Returns
    -------
    None.
    """
    forest_top[A].remove(B)
    forest_top[B].remove(A)


def get_node_side(X, forest_top):
    """
    Get the side of the tree from node X (inclusive).

    Parameters
    ----------
    X : int
        Node index.
    forest_top : dict
        Topology of the current forest from the original tree.

    Returns
    -------
    side_X : set
        Nodes composing the side of the tree rooted at X.
    """
    side_X = set([X])
    iterator = side_X.copy()
    while iterator:
        next_node = iterator.pop()
        next_family = forest_top[next_node] - side_X
        side_X.update(next_family)
        iterator.update(next_family)
    return side_X


def assert_node_side(X, forest_top, mpoints):
    """
    Check if the component rooted at X has a number of points >= mpoints.

    Parameters
    ----------
    X : int
        Node index.
    forest_top : dict
        Topology of the current forest from the original tree.

    Returns
    -------
    side_X : set
        Nodes composing the side of the tree rooted at X.
    """
    side_X = set([X])
    iterator = set([X])
    while iterator:
        next_node = iterator.pop()
        next_family = forest_top[next_node] - side_X
        side_X.update(next_family)
        iterator.update(next_family)
        if len(side_X) >= mpoints:
            break
    return len(side_X) >= mpoints


def get_master_topology(nn_arr, dist_arr, export=True):
    """
    Get the topological information of the directed tree represented by nn_arr.

    Parameters
    ----------
    nn_arr : numpy.ndarray
        Array of nearest neighbors forming edges. Indices correspond to each
        node in the graph and the value correspond to the other node forming
        a directed edge (index->value).
    dist_arr : numpy.ndarray
        Array of Minimum Reachability Distances (MRD) between the nodes of the
        tree.
    export : bool, optional
        Export the topology to graphml format?. The default is True.

    Returns
    -------
    forest_top : dict
        Dict of sets. Each key represents a node and each value (set) contains
        the nodes incident to the key node.

    """
    forest_top = defaultdict(set)
    if export:
        G = nx.Graph()
        for i, x in enumerate(nn_arr):
            forest_top[i].add(x)
            forest_top[x].add(i)
            G.add_node(i)
            G.add_edge(i, x, mdr=dist_arr[i])
        nx.write_graphml(G, 'forest_topology.graphml')
    else:
        for i, x in enumerate(nn_arr):
            forest_top[i].add(x)
            forest_top[x].add(i)
    return forest_top


def prune_qMST(nn_arr, dist_arr, forest_top, mpoints):
    """
    Prune the quasi-minimum spanning tree based on the algorithm of Campello
    et. al.

    Parameters
    ----------
    nn_arr : numpy.ndarray
        Array of nearest neighbors forming edges. Indices correspond to each
        node in the graph and the value correspond to the other node forming
        a directed edge (index->value).
    dist_arr : numpy.ndarray
        Array of Minimum Reachability Distances (MRD) between the nodes of the
        tree.
    forest_top : dict
        Topology of the current forest from the original tree.
    mpoints : int
        Minimum number of points in a set to be considered as a cluster.

    Returns
    -------
    clusters : dict
        DESCRIPTION.
    clust_array : numpy.ndarray
        Array containig the labels of the clusters.

    """
    # Clusters tracking machinery ---------------------------------------------
    N = dist_arr.size
    cluster_id = it.count(start=1, step=2)
    clust_array = np.zeros(N, dtype=np.int32)
    template = {'birth': 0, 'death': -1, 'parent': -1,
                'childA': -1, 'childB': -1, 'S': 0}
    clusters = {}
    clusters.update({0: template.copy()})
    rejected = set()
    components = dict()
    components.update({0: set(range(N))})
    # Cutting -----------------------------------------------------------------
    lamb = 1 / dist_arr
    order = iter(lamb.argsort())
    while True:
        try:
            A = next(order)
        except StopIteration:
            break
        B = nn_arr[A]
        if (A in rejected) or (B in rejected):
            continue
        lambda_dist = lamb[A]
        cut_tree(A, B, forest_top)
        # Retrieving sides with auxiliar dict of components -------------------
        for c in components:
            if A in components[c]:
                component_id = c
                break
        component = components[component_id]
        if assert_node_side(A, forest_top, mpoints):
            side_B = get_node_side(B, forest_top)
            side_A = component - side_B
        else:
            side_A = get_node_side(A, forest_top)
            side_B = component - side_A
        lenA = len(side_A)
        lenB = len(side_B)
        # =====================================================================
        # Both sides are big enough to form clusters: SPLITTING BRANCH
        # =====================================================================
        if (lenA >= mpoints) and (lenB >= mpoints):
            # create new labels for the splitted clusters
            id_A = next(cluster_id)
            id_B = id_A + 1
            del components[component_id]
            components.update({id_A: side_A, id_B: side_B})
            # determine the parent of theese clusters and their lambda birth
            sample = side_A.pop()
            parent = clust_array[sample]
            side_A.add(sample)
            template['birth'] = lambda_dist
            clusters[parent]['death'] = lambda_dist
            template['parent'] = parent
            # partition points into both clusters
            clust_array[np.fromiter(side_A, dtype=np.int32)] = id_A
            clusters.update({id_A: template.copy()})
            clust_array[np.fromiter(side_B, dtype=np.int32)] = id_B
            clusters.update({id_B: template.copy()})
            # set clusters as their parent children
            clusters[parent]['childA'] = id_A
            clusters[parent]['childB'] = id_B
            # compute the stability of the parent cluster
            split_stability = (lenA + lenB) * (lambda_dist - clusters[parent]['birth'])
            clusters[parent]['S'] += split_stability
        # =====================================================================
        # Both sides are too small to form clusters: CLOSING BRANCH
        # =====================================================================
        elif (lenA < mpoints) and (lenB < mpoints):
            # discard future cuts of both components
            rejected.update(side_A)
            rejected.update(side_B)
            del components[component_id]
            # determine the parent cluster
            sample = side_B.pop()
            parent = clust_array[sample]
            # update parent stability
            split_stability = (lenA + lenB) * (lambda_dist - clusters[parent]['birth'])
            clusters[parent]['S'] += split_stability
            clusters[parent]['death'] = lambda_dist
        # =====================================================================
        # One of the clusters is just loosing points: SHRINKING BRANCH
        # =====================================================================
        elif lenA < mpoints:
            # discard future cuts of side_A components
            rejected.update(side_A)
            components[component_id].difference_update(side_A)
            # determine the parent cluster of side_A
            sample = side_A.pop()
            side_A.add(sample)
            parent = clust_array[sample]
            # update parent stability
            split_stability = lenA * (lambda_dist - clusters[parent]['birth'])
            clusters[parent]['S'] += split_stability
        elif lenB < mpoints:
            # discard future cuts of side_B components
            rejected.update(side_B)
            components[component_id].difference_update(side_B)
            # determine the parent cluster of side_B
            sample = side_B.pop()
            side_B.add(sample)
            parent = clust_array[sample]
            # update parent stability
            split_stability = lenB * (lambda_dist - clusters[parent]['birth'])
            clusters[parent]['S'] += split_stability
    return clusters, clust_array


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


def assign_deltas_and_stabilities(clusters):
    """
    Assign deltas an stabilities of clusters according based on the algorithm
    by Campello et. al.

    Parameters
    ----------
    clusters : dict
        DESCRIPTION.

    Returns
    -------
    cinfo : pandas.DataFrame
        Dataframe containing all the clusters information.
    ctree : dict
        Dict of sets relating clusters parents and children.

    """
    # Creating the template for info storage ----------------------------------
    cinfo = pd.DataFrame(clusters).T.astype({'birth': float,
                                             'parent': np.int32,
                                             'childA': np.int32,
                                             'childB': np.int32,
                                             'S': float})
    cinfo['delta'] = 1
    cinfo.loc[0, 'delta'] = 0
    # top -> bottom pass ------------------------------------------------------
    top_bottom = deque()
    ctree = defaultdict(set)
    iterator = deque([0])
    childs_A = cinfo.childA
    childs_B = cinfo.childB
    while iterator:
        P = iterator.popleft()
        A = childs_A[P]
        if A == -1:
            continue
        B = childs_B[P]
        top_bottom.append([P, A, B])
        ctree[P].update([A, B])
        iterator.append(A)
        iterator.append(B)
    # bottom -> top pass ------------------------------------------------------
    top_bottom.popleft()
    while top_bottom:
        P, A, B = top_bottom.pop()
        sum_AB = cinfo.loc[A, 'S'] + cinfo.loc[B, 'S']
        if cinfo.loc[P, 'S'] < sum_AB:
            cinfo.loc[P, 'S'] = sum_AB
            cinfo.loc[P, 'delta'] = 0
        else:
            subtree = get_node_side(P, ctree)
            subtree.remove(P)
            cinfo.loc[list(subtree), 'delta'] = 0
    return cinfo, ctree


def get_final_clusters(selected, cinfo, ctree, orig_clust_array, clust_sel_met,
                       include_children=True):
    """
    Get the final labelling of the clustering job.

    Parameters
    ----------
    selected : iterable
        Selected clusters (those with delta == 1).
    cinfo : pandas.DataFrame
        Dataframe containing all the clusters information.
    ctree : dict
        Dict of sets relating clusters parents and children.
    orig_clust_array : numpy.ndarray
        Original array of clusters produced by the algorithm.
    clust_sel_met : str
        Cluster selection method (either by 'excess of mass' or 'leaf').
    include_children : bool, optional
        Include children clusters of selected clusters?. The default is True.

    Returns
    -------
    final_array : numpy.ndarray
        Final labeling of the selected clusters ordered by size (descending).

    """
    clust_array = orig_clust_array.copy()
    # restrict to leaf nodes ?
    if clust_sel_met == 'leaf':
        leaf_selected = []
        for s in selected:
            if cinfo.loc[s, 'childA'] == -1:
                leaf_selected.append(s)
        selected = leaf_selected
    # report children of selected clusters as valid members ?
    if include_children:
        for s in selected:
            if cinfo.loc[s, 'childA'] != -1:
                subtree = get_node_side(s, ctree)
                subtree.remove(s)
                for t in subtree:
                    clust_array[np.where(clust_array == t)[0]] = s
    # organize selected clusters by size
    N = clust_array.size
    sizes = []
    heapq.heapify(sizes)
    for s in selected:
        members = np.where(clust_array == s)[0]
        size = members.size
        heapq.heappush(sizes, (size, s, members))
    # assign final labels in descending order of cluster population
    final_id = it.count(start=len(sizes) - 1, step=-1)
    final_array = np.full(N, -1, dtype=np.int32)
    while sizes:
        size, num, members = heapq.heappop(sizes)
        id_ = next(final_id)
        final_array[members] = id_
    return final_array


def pickle_to_file(data, file_name):
    """
    Serialize data using pickle module.

    Parameters
    ----------
    data : obj
        any serializable object.
    file_name : str
        name of the pickle file to be created.

    Returns
    -------
    file_name : str
        File name.

    """
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)
    return file_name


def top_has_coords(topology):
    """
    Check if topology has cartesian coordinates information.

    Parameters
    ----------
    topology : str
        Path to the topology file.

    Returns
    -------
    int
        Number of cartesian frames if topology contains cartesians.
        False otherwise.
    """
    try:
        tt = md.load(topology)
    except OSError:
        return False
    return tt.xyz.shape[0]


def to_VMD(topology, first, N, last, stride, final_array):
    """
    Create a .log file for visualization of clusters in VMD through a
    third-party plugin.

    Parameters
    ----------
    topology : str
        Path to the topology file.
    first : int
        First frame to consider (0-based indexing).
    N : int
        default value when last == None.
    last : TYPE
        Last frame to consider (0-based indexing).
    stride : TYPE
        Stride (step).
    final_array : numpy.ndarray
        Final labeling of the selected clusters ordered by size (descending).

    Returns
    -------
    logname : str
        Log file to be used with VMD.

    """
    basename = os.path.basename(topology).split('.')[0]
    logname = '{}.log'.format(basename)
    vmd_offset = top_has_coords(topology)
    start = first
    if not last:
        stop = N
    else:
        stop = last
    slice_frames = np.arange(start, stop, stride, dtype=np.int32)
    nmr_offset = 1
    with open(logname, 'wt') as clq:
        for num in np.unique(final_array):
            clq.write('{}:\n'.format(num))
            cframes = np.where(final_array == num)[0]
            if vmd_offset:
                real_frames = slice_frames[cframes] + nmr_offset + vmd_offset
            else:
                real_frames = slice_frames[cframes] + nmr_offset
            str_frames = [str(x) for x in real_frames]
            members = ' '.join(str_frames)
            clq.write('Members: ' + members + '\n\n')
    return logname


if __name__ == '__main__':
    # ++++ Debugging ? ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    from argparse import Namespace
    folder = '/home/rga/BSProject/05-oldies/bitsuite/examples/'
    args = Namespace(
        topology=folder + 'aligned_tau.pdb',
        trajectory=folder + 'aligned_original_tau_6K.dcd',
        first=0, last=None, stride=1,
        selection='all',
        clust_sel_met='eom',
        min_clust_size=2,
        k=10,
        outdir='./')

    # ======================================================================= #
    # >>>> FIRST PART: qMST CONSTRUCTION                                      #
    # ======================================================================= #
    # ++++ Initializing
    # args = parse_arguments()
    np.seterr(divide='ignore', invalid='ignore')         # Avoid division error
    traj = load_raw_traj(args.trajectory, valid_trajs, args.topology)
    traj = shrink_traj_selection(traj, args.selection)
    N1 = traj.n_frames
    traj = shrink_traj_range(args.first, args.last, args.stride, traj)
    N2 = traj.n_frames
    traj.center_coordinates()
    # ++++ Exhausting neighborhoods
    Kd_arr, dist_arr, nn_arr, exhausted = exhaust_neighborhoods(traj, args.k)
    # ++++ Joining exhausted nodes
    dist_arr, nn_arr = join_exhausted(exhausted, Kd_arr, dist_arr, nn_arr, traj)
    # ++++ Checks
    # mdf.check_tree(N, nn_arr, dist_arr)
    # extc = mdf.get_exact_MST(N, traj, Kd_arr, 'prim')

    # ======================================================================= #
    # >>>> SECOND PART: CLUSTERS EXTRACTION                                   #
    # ======================================================================= #
    # ++++ Constructing the master topology of nodes
    forest_top = get_master_topology(nn_arr, dist_arr, export=False)
    # ++++ Pruning the qMST
    mcs = args.min_clust_size
    clusters, orig_clust_array = prune_qMST(nn_arr, dist_arr, forest_top, mcs)
    # ++++ Assigning deltas and stabilities in a two-pass approach
    cinfo, ctree = assign_deltas_and_stabilities(clusters)
    # ++++ Selecting flat clustering (deltas == 1)
    selected = cinfo[cinfo.delta == 1].index
    # ++++ Getting final clustering
    final_array = get_final_clusters(selected, cinfo, ctree, orig_clust_array,
                                     args.clust_sel_met, include_children=True)

    # ======================================================================= #
    # >>>> Third PART: Outputs & Reports                                      #
    # ======================================================================= #
    # saving python objects
    basename = os.path.basename(args.topology).split('.')[0]
    pickname = '{}_mdscan.pick'.format(basename)
    pickle_to_file(
        (Kd_arr, dist_arr, nn_arr, exhausted, selected, final_array), pickname)
    # saving VMD visualization script
    to_VMD(args.topology, args.first, args.last, N1, args.stride, final_array)