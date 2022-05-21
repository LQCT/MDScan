# MDSCAN
> RMSD-Based HDBSCAN Clustering of Long Molecular Dynamics

MDSCAN is a Python command-line interface (CLI) conceived to speed up and significantly lower the RAM memory needs of the HDBSCAN clustering of long Molecular Dynamics.


## Installation

There are some easy-to-install dependencies you should have before running MDSCAN. MDTraj (mandatory) will perform the heavy RMSD calculations, while VMD (optional) will help with visualization tasks. The rest of the dependencies (see the requirements.txt file) will be automatically managed while installing MDSCAN.


#### 1. **MDTraj**

It is recommended that you install __MDTraj__ using conda.

`conda install -c conda-forge mdtraj`

#### 2. **MDSCAN**

+ __Via **pip**__


After successfully installing __MDTraj__, you can easily install MDSCAN and the rest of its dependencies using pip.

`pip install mdscan`


+ __Via **GitHub**__

```
git clone https://github.com/LQCT/MDScan.git
cd mdscan/
python setup.py install
```
Then, you should be able to see MDSCAN help by typing in a console:

`mdscan -h`


#### 3. **VMD** and **VMD clustering plugin** (optional)

MDSCAN clusters can be visualized by loading a **.log**  file in VMD via a clustering plugin.
Please see this [VMD visualization tutorial](https://bitqt.readthedocs.io/en/latest/tutorial.html#visualizing-clusters-in-vmd).

The official site for VMD download and installation can be found [here](https://www.ks.uiuc.edu/Development/Download/download.cgi?PackageName=VMD>).

Instructions on how to install the clustering plugin of VMD are available [here](https://github.com/luisico/clustering).


## Basic Usage
You can display the primary usage of RCDPeaks by typing  `mdscan -h` in the command line.

```
$ mdscan -h

usage: mdscan -traj trajectory [options]

MDScan: RMSD-Based HDBSCAN Clustering of Long Molecular Dynamics

optional arguments:
  -h, --help            show this help message and exit

Trajectory options:
  -traj trajectory      Path to trajectory file (pdb/dcd) [default: None]
  -top topology         Path to the topology file (psf/pdb)
  -first first_frame    First frame to analyze (start counting from 0) [default: 0]
  -last last_frame      Last frame to analyze (start counting from 0) [default: last frame]
  -stride stride        Stride of frames to analyze [default: 1]
  -sel selection        Atom selection (MDTraj syntax) [default: all]

Clustering options:
  -min_samples k        Number of k nearest neighbors to consider [default: 5]
  -min_clust_size m     Minimum number of points in agrupations to be considered as clusters [default: 5]
  -clust_sel_met {eom,leaf}
                        Method used to select clusters from the condensed tree [default: eom]
  -nsplits NSPLITS      Number of binary splits to perform on the Vantage Point Tree [default: 3]

Output options:
  -odir .               Output directory to store analysis [default: ./]

```

In the examples folder, you can find a coordinate (pdb) and a trajectory (dcd) files to run an MDSCAN test.
Type the next command in the console and check if you can reproduce the content of the examples directory:

```mdscan -traj aligned_original_tau_6K.dcd -top aligned_tau.pdb -odir output_dir```


## Citation (work in-press)

If you make use of MDSCAN in your scientific work, [cite it ;)]()


## Licence

**MDSCAN** is licensed under GNU General Public License v3.0.

