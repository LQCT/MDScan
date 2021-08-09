"""
DESCRIPTION

Created on Sun Jul 25 19:26:29 2021
@author: Roy González Alemán
@contact: roy_gonzalez@fq.uh.cu, roy.gonzalez-aleman@u-psud.fr
"""
import argparse

import mdtraj as md


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
    desc = '\nMDScan: RMSD-Based HDBSCAN Clustering of Long Molecular Dynamics'
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
                       type=int, required=False, default=5, metavar='k')
    clust.add_argument('-min_clust_size', action='store',
                       dest='min_clust_size',
                       help='Minimum number of points in agrupations to be\
                       considered as clusters [default: %(default)s]',
                       type=int, required=False, default=5, metavar='m')
    clust.add_argument('-clust_sel_met', action='store', dest='clust_sel_met',
                       help='Method used to select clusters from the condensed\
                       tree [default: %(default)s]', type=str, required=False,
                       default='eom', choices=['eom', 'leaf'])
    clust.add_argument('-nsplits', action='store', dest='nsplits',
                       help='Number of binary splits to perform on the Vantage Point Tree\
                        [default: %(default)s]', type=int, required=False,
                       default=3)
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
