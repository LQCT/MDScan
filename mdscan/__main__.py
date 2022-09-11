#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 22:28:25 2020
@author: rga
@contact: roy_gonzalez@fq.uh.cu, roy.gonzalez-aleman@u-psud.fr
"""
import os
import time

import numpy as np

from mdscan import qmst
from mdscan import trajload as trl
from mdscan import clusterize as clt
from mdscan import analysis as anl


def main():
    start = time.time()
    # ======================================================================= #
    # >>>> FIRST PART: qMST CONSTRUCTION                                      #
    # ======================================================================= #
    # ++++ Debugging ? ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # from argparse import Namespace
    # folder = '/home/roy.gonzalez-aleman/rprojects/BSProject/05-oldies/bitsuite/examples/'
    # args = Namespace(
    #     topology=folder + 'aligned_tau.pdb',
    #     trajectory=folder + 'aligned_original_tau_6K.dcd',
    #     # topology=folder + 'melvin.pdb',
    #     # trajectory=folder + 'melvin.dcd',
    #     first=0, last=None, stride=1,
    #     selection='all', clust_sel_met='eom',
    #     # selection='name CA', clust_sel_met='eom',
    #     nsplits=3, min_clust_size=5, k=5, outdir='./')

    # ++++ Initializing trajectory ++++++++++++++++++++++++++++++++++++++++++++
    args = trl.parse_arguments()
    np.seterr(divide='ignore', invalid='ignore')         # Avoid division error
    traj = trl.load_raw_traj(args.trajectory, trl.valid_trajs, args.topology)
    traj = trl.shrink_traj_selection(traj, args.selection)
    N1 = traj.n_frames
    traj = trl.shrink_traj_range(args.first, args.last, args.stride, traj)
    # N2 = traj.n_frames
    traj.center_coordinates()
    print('\n[1/4] Parsing of trajectory completed.')

    # # ++++ Exhausting neighborhoods +++++++++++++++++++++++++++++++++++++++++
    Kd_arr, dist_arr, nn_arr, exhausted, D1 = qmst.exhaust_neighborhoods(
        traj, args.k, args.nsplits)

    # ++++ Joining exhausted nodes ++++++++++++++++++++++++++++++++++++++++++++
    dist_arr, nn_arr, exh_ord = qmst.join_exhausted(
        exhausted, Kd_arr, dist_arr, nn_arr, traj)
    print('\n[2/4] Construction of the quasi-Minimum Spanning Tree completed.')

    # ++++ Checks +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # !!! to pass this check, you must set to -1 in nn_arr the argmin of dist_arr
    # tree = qmst.check_tree(N1, nn_arr, dist_arr)
    # cycl = nx.find_cycle(tree)
    # connected = nx.is_connected(tree)
    # network_exact = qmst.get_exact_MST(N1, traj, Kd_arr)
    # mst_exact2 = qmst.get_prim_set(traj, Kd_arr)

    # ======================================================================= #
    # >>>> SECOND PART: CLUSTERS EXTRACTION                                   #
    # ======================================================================= #
    # ++++ Constructing the master topology of nodes ++++++++++++++++++++++++++
    forest_top = clt.get_otree_topology2(nn_arr)

    # ++++ Pruning the qMST +++++++++++++++++++++++++++++++++++++++++++++++++++
    mcs = args.min_clust_size
    clusters, orig_clust_array = clt.prune_qMST2(nn_arr, dist_arr, forest_top, mcs)

    # ++++ Assigning deltas and stabilities in a two-pass approach ++++++++++++
    reclusters, ctree = clt.assign_deltas_and_stabilities2(clusters)
    selected = [x for x in reclusters if reclusters[x]['delta'] == 1]

    # ++++ Getting final clustering +++++++++++++++++++++++++++++++++++++++++++
    final_array = clt.get_final_clusters(selected, reclusters, ctree,
                                         orig_clust_array, args.clust_sel_met,
                                         include_children=True)
    print('\n[3/4] MDSCAN clustering completed.')

    # ======================================================================= #
    # >>>> Third PART: Outputs & Reports                                      #
    # ======================================================================= #
    # ++++ saving python objects as pickle ++++++++++++++++++++++++++++++++++++
    out_dir = os.path.abspath(args.outdir)
    # pickname = '{}.pick'.format(basename)
    # anl.pickle_to_file((Kd_arr, dist_arr, nn_arr, exhausted, selected,
    # final_array, D1, exh_ord), pickname)

    # ++++ saving VMD visualization script ++++++++++++++++++++++++++++++++++++
    os.makedirs(out_dir, exist_ok=True)
    anl.to_VMD(args.topology, args.first, args.last, N1, args.stride, final_array,
               args.outdir)
    print('\n[4/4] Output files writing completed.')

    print('\n\nMDSCAN normal termination. {} clusters found in {:3.2f} secs.'
          .format(final_array.max(), time.time() - start))


if __name__ == '__main__':
    main()
