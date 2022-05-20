"""
DESCRIPTION

Created on Sun Jul 25 19:28:36 2021
@author: Roy González Alemán
@contact: roy_gonzalez@fq.uh.cu, roy.gonzalez-aleman@u-psud.fr
"""
import os
import pickle

import numpy as np
import mdtraj as md


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


def to_VMD(topology, first, N, last, stride, final_array, odir):
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
    logname = os.path.join(odir, '{}.log'.format(basename))
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
            if num != -1:
                clq.write('{}:\n'.format(num))
                cframes = np.where(final_array == num)[0]
                if vmd_offset:
                    real_frames = slice_frames[cframes] + nmr_offset + vmd_offset
                else:
                    real_frames = slice_frames[cframes] + nmr_offset
                str_frames = [str(x) for x in real_frames]
                members = ' '.join(str_frames)
                clq.write('Members: ' + members + '\n\n')
        if -1 in np.unique(final_array):
            clq.write('{}:\n'.format(-1))
            cframes = np.where(final_array == -1)[0]
            if vmd_offset:
                real_frames = slice_frames[cframes] + nmr_offset + vmd_offset
            else:
                real_frames = slice_frames[cframes] + nmr_offset
            str_frames = [str(x) for x in real_frames]
            members = ' '.join(str_frames)
            clq.write('Members: ' + members + '\n\n')
    return logname


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
