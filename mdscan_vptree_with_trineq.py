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

from mdpack import qmst
from mdpack import trajload as trl
from mdpack import clusterize as clt
from mdpack import analysis as anl

start = time.time()
# =========================================================================== #
# >>>> FIRST PART: qMST CONSTRUCTION                                          #
# =========================================================================== #
# ++++ Debugging ? ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# from argparse import Namespace
# folder = '/home/rga/BSProject/05-oldies/bitsuite/examples/'
# folder = '/home/rga/BSProject/runners/trajs/trajs/'
# args = Namespace(
#     topology=folder + 'aligned_tau.pdb',
#     trajectory=folder + 'aligned_original_tau_6K.dcd',
#     # topology=folder + 'melvin.pdb',
#     # trajectory=folder + 'melvin.dcd',
#     first=0, last=None, stride=1,
#     selection='all', clust_sel_met='eom',
#     nsplits=0, min_clust_size=5, k=5, outdir='./')

# ++++ Initializing trajectory ++++++++++++++++++++++++++++++++++++++++++++++++
args = trl.parse_arguments()
np.seterr(divide='ignore', invalid='ignore')             # Avoid division error
traj = trl.load_raw_traj(args.trajectory, trl.valid_trajs, args.topology)
traj = trl.shrink_traj_selection(traj, args.selection)
N1 = traj.n_frames
traj = trl.shrink_traj_range(args.first, args.last, args.stride, traj)
N2 = traj.n_frames
traj.center_coordinates()

# ++++ Exhausting neighborhoods +++++++++++++++++++++++++++++++++++++++++++++++
Kd_arr, dist_arr, nn_arr, exhausted = qmst.exhaust_neighborhoods(traj, args.k,
                                                                 args.nsplits)

# ++++ Joining exhausted nodes ++++++++++++++++++++++++++++++++++++++++++++++++
dist_arr, nn_arr = qmst.join_exhausted(exhausted, Kd_arr, dist_arr,
                                       nn_arr, traj)

# ++++ Checks +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# !!! to pass this check, you must set to -1 in nn_arr the argmin of dist_arr
# tree = qmst.check_tree(N1, nn_arr, dist_arr)
# extc = qmst.get_exact_MST(N1, traj, Kd_arr, 'prim')


# =========================================================================== #
# >>>> SECOND PART: CLUSTERS EXTRACTION                                       #
# =========================================================================== #
# ++++ Constructing the master topology of nodes ++++++++++++++++++++++++++++++
# forest_top = get_master_topology(nn_arr, dist_arr, export=False)
forest_top = clt.get_otree_topology2(nn_arr)
# ++++ Pruning the qMST
mcs = args.min_clust_size
clusters, orig_clust_array = clt.prune_qMST2(nn_arr, dist_arr, forest_top, mcs)
# ++++ Assigning deltas and stabilities in a two-pass approach
reclusters, ctree = clt.assign_deltas_and_stabilities2(clusters)
# ++++ Selecting flat clustering (deltas == 1)
selected = [x for x in reclusters if reclusters[x]['delta'] == 1]
# ++++ Getting final clustering
final_array = clt.get_final_clusters(selected, reclusters, ctree,
                                     orig_clust_array, args.clust_sel_met,
                                     include_children=True)


# =========================================================================== #
# >>>> Third PART: Outputs & Reports                                          #
# =========================================================================== #
# saving python objects
basename = os.path.basename(args.topology).split('.')[0]
pickname = '{}_mdscan.pick'.format(basename)
anl.pickle_to_file((Kd_arr, dist_arr, nn_arr, exhausted, selected, final_array),
                   pickname)
# saving VMD visualization script
anl.to_VMD(args.topology, args.first, args.last, N1, args.stride, final_array)
print('\n\nMDSCAN normal termination. {} clusters found in {:3.2f} secs.'
      .format(final_array.max(), time.time() - start))
