"""
DESCRIPTION

Created on Sun Jul 25 19:27:50 2021
@author: Roy González Alemán
@contact: roy_gonzalez@fq.uh.cu, roy.gonzalez-aleman@u-psud.fr
"""
import heapq
import itertools as it
from collections import defaultdict, deque

import numpy as np


def get_otree_topology2(nn_arr):
    """
    Get the oriented relationship of the spanning tree encoded in the nearest
    neighbors array.

    Parameters
    ----------
    nn_arr : numpy.ndarray
        Array of nearest neighbors forming edges. Indices correspond to each
        node in the graph and the value correspond to the other node forming
        a directed edge (index->value).

    Returns
    -------
    topo_forest : TYPE
        DESCRIPTION.

    """
    topo_forest = defaultdict(set)
    for i, x in enumerate(nn_arr):
        topo_forest[x].add(i)
    return topo_forest


def cut_tree(A, B, forest_top):
    """
    Cut a tree into two sub-trees. Destructive function on forest_top.

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
    if A in forest_top[B]:
        forest_top[B].remove(A)
    if B in forest_top[A]:
        forest_top[A].remove(B)


def get_node_side2(active_node, topo_forest):
    """
    Get from topo_forest the component of nodes having an oriented path to
    active_node.

    Parameters
    ----------
    active_node : int
        The node where all others point to.
    topo_forest : dict
        Encoding of the oriented tree topology. Dict keys correspond to nodes
        while dict values (sets) inform nodes pointing to key.

    Returns
    -------
    side_X : set
        Component of nodes pointing to active_node.

    """
    side_X = set([active_node])
    iterator = deque([active_node])
    while iterator:
        next_node = iterator.pop()
        next_family = topo_forest[next_node]
        side_X.update(next_family)
        iterator.extend(next_family)
    return side_X


def prune_qMST2(nn_arr, dist_arr, topo_forest, mpoints):
    """
    HDBSCAN procedure to prune the mutual reachability distances'
    Minimum Spanning Tree. See arXiv:1705.07321v2.

    Parameters
    ----------
    nn_arr : TYPE
        DESCRIPTION.
    dist_arr : TYPE
        DESCRIPTION.
    topo_forest : TYPE
        DESCRIPTION.
    mpoints : TYPE
        DESCRIPTION.

    Returns
    -------
    clusters : TYPE
        DESCRIPTION.
    clust_array : TYPE
        DESCRIPTION.

    """
    # Clusters tracking machinery
    N = dist_arr.size
    cluster_id = it.count(start=1, step=2)
    clust_array = np.zeros(N, dtype=np.int32)
    template = {'birth': 0, 'death': -1, 'parent': -1,
                'childA': -1, 'childB': -1, 'S': 0, 'delta': 1}
    clusters = {}
    clusters.update({0: template.copy()})
    rejected = set()
    components = dict()
    components.update({0: set(range(N))})
    mapping = np.zeros(N, dtype=np.int)
    # Cutting
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
        cut_tree(A, B, topo_forest)
        # Retrieving sides with auxiliar dict of components
        component_id = mapping[A]
        component = components[component_id]
        side_A = get_node_side2(A, topo_forest)
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
            # mapping[list(side_A)] = id_A
            # mapping[list(side_B)] = id_B
            mapping[np.fromiter(side_A, dtype=np.int)] = id_A
            mapping[np.fromiter(side_B, dtype=np.int)] = id_B
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


def assign_deltas_and_stabilities2(clusters):
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
    # cinfo = pd.DataFrame(clusters).T.astype({'birth': float,
    #                                           'parent': np.int32,
    #                                           'childA': np.int32,
    #                                           'childB': np.int32,
    #                                           'S': float})
    # cinfo['delta'] = 1
    # cinfo.loc[0, 'delta'] = 0
    clusters[0]['delta'] = 0
    # top -> bottom pass ------------------------------------------------------
    top_bottom = deque()
    ctree = defaultdict(set)
    iterator = deque([0])
    # childs_A = cinfo.childA
    # childs_B = cinfo.childB
    while iterator:
        P = iterator.popleft()
        A = clusters[P]['childA']
        if A == -1:
            continue
        B = clusters[P]['childB']
        top_bottom.append([P, A, B])
        ctree[P].update([A, B])
        iterator.append(A)
        iterator.append(B)
    # bottom -> top pass ------------------------------------------------------
    top_bottom.popleft()
    while top_bottom:
        P, A, B = top_bottom.pop()
        # sum_AB = cinfo.loc[A, 'S'] + cinfo.loc[B, 'S']
        sum_AB = clusters[A]['S'] + clusters[B]['S']
        if clusters[P]['S'] < sum_AB:
            clusters[P]['S'] = sum_AB
            clusters[P]['delta'] = 0
        else:
            subtree = get_node_side(P, ctree)
            subtree.remove(P)
            for x in subtree:
                clusters[x]['delta'] = 0
            # cinfo.loc[list(subtree), 'delta'] = 0
    return clusters, ctree


def get_final_clusters(selected, clusters, ctree, orig_clust_array,
                       clust_sel_met, include_children=True):
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
            # if cinfo.loc[s, 'childA'] == -1:
            if clusters[s]['childA'] == -1:
                leaf_selected.append(s)
        selected = leaf_selected
    # report children of selected clusters as valid members ?
    if include_children:
        for s in selected:
            # if cinfo.loc[s, 'childA'] != -1:
            if clusters[s]['childA'] != -1:
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
