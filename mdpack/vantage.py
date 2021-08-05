"""
DESCRIPTION

Created on Sun Jul 25 19:28:14 2021
@author: Roy González Alemán
@contact: roy_gonzalez@fq.uh.cu, roy.gonzalez-aleman@u-psud.fr
"""
import heapq
import itertools as it
from collections import deque

import numpy as np
import mdtraj as md
from numba import jit
import numpy_indexed as npi


@jit(nopython=True, fastmath=True)
def prunable_by_trineq(vant_matrix_sel, vector, tau):
    R, C = vant_matrix_sel.shape
    for row in range(R):
        atleast = False
        for i, x in enumerate(vant_matrix_sel[row]):
            diff = x - vector[i]
            absol = abs(diff)
            if absol > tau:
                atleast = True
        if not atleast:
            return False
    return True


class vpTree:
    """Vantage point tree datastructure to speed up RMSD computations"""

    def __init__(self, nsplits, sample_size, traj):
        self.bucketTree = dict()
        self.binTree = dict()
        self.traj = traj
        self.sample_size = sample_size
        self.niters = nsplits

    def splitTrajOri(self, real_indices, subtraj, vpoints):
        # ---- get a better-than-random vpoint to split subtraj
        N = subtraj.n_frames
        sample = np.arange(0, N, int(N / self.sample_size))
        P_idx = real_indices[sample]
        P = subtraj.slice(sample)
        P.center_coordinates()
        best_spread = 0
        best_p = None
        for p in range(P.n_frames):
            spread = np.var(md.rmsd(P, P, p, precentered=True))
            if (spread > best_spread) and (p not in vpoints):
                best_spread = spread
                best_p = p
        # ---- get one vs all from best_p
        real_p = P_idx[best_p]
        internal_p = npi.indices(real_indices, [real_p])
        p_dists = md.rmsd(subtraj, subtraj, internal_p, precentered=True)
        p_mu = np.mean(p_dists)
        boolean = p_dists >= p_mu
        # ---- get R traj
        R_realindx = real_indices[boolean]
        R_traj = subtraj.slice(boolean).center_coordinates()
        R_info = (R_realindx, R_traj)
        # ---- get L traj
        L_realindx = real_indices[~boolean]
        L_traj = subtraj.slice(~boolean).center_coordinates()
        L_info = (L_realindx, L_traj)
        return real_p, p_mu, R_info, L_info


    def splitTraj(self, real_indices, subtraj, vpoints):
        # ---- get a better-than-random vpoint to split subtraj
        N = real_indices.size
        sample = np.arange(0, N, int(N / self.sample_size))

        P_idx = real_indices[sample]  # real_indices
        P = subtraj.slice(P_idx).center_coordinates()

        best_spread = 0
        best_p = None
        for p in range(P.n_frames):
            spread = np.var(md.rmsd(P, P, p, precentered=True))
            if (spread > best_spread) and (p not in vpoints):
                best_spread = spread
                best_p = p
        # ---- get one vs all from best_p
        real_p = P_idx[best_p]
        # internal_p = npi.indices(real_indices, [real_p])
        p_dists = md.rmsd(subtraj, subtraj, real_p, precentered=True)[real_indices]
        p_mu = np.mean(p_dists)
        boolean = p_dists >= p_mu
        # ---- get R traj
        R_realindx = real_indices[boolean]
        # R_traj = subtraj.slice(boolean).center_coordinates()
        # R_info = (R_realindx, R_traj)
        # ---- get L traj
        L_realindx = real_indices[~boolean]
        # L_traj = subtraj.slice(~boolean).center_coordinates()
        # L_info = (L_realindx, L_traj)
        return real_p, p_mu, R_realindx, L_realindx

# %
    def getBothTrees(self, real_indices, subtraj):
        # ---- construct the binary Tree and the associated buckets
        niter = 0
        vpnts = []
        to_explore = deque([real_indices])

        while to_explore:
            realindx = to_explore.popleft()
            niter += 1

            vp, mu, R_info, L_info = self.splitTraj(realindx, subtraj, vpnts)
            vpnts.append(vp)
            self.binTree.update({vp: {'mu': mu}})

            if niter > self.niters:
                R_real_idx = R_info
                L_real_idx = L_info
                for i in R_real_idx:
                    if i not in vpnts:
                        RID = i
                        break
                for i in L_real_idx:
                    if i not in vpnts:
                        LID = i
                        break
                self.bucketTree.update({RID: {'P': vp, 'real_indx': R_real_idx,
                                              'side': 'R'}})
                self.binTree[vp].update({'R': RID})
                self.bucketTree.update({LID: {'P': vp, 'real_indx': L_real_idx,
                                              'side': 'L'}})
                self.binTree[vp].update({'L': LID})
                continue
            else:
                to_explore.append(L_info)
                to_explore.append(R_info)

        # ---- assign parent relationships
        parents = [-1]
        sides = ['C']
        for x in range(int((len(vpnts) - 1) / 2)):
            parents.extend(list(it.repeat(x, 2)))
            sides.extend(['L', 'R'])
        for i, x in enumerate(vpnts):
            if i != 0:
                self.binTree[x].update({'P': vpnts[parents[i]]})
                self.binTree[x].update({'side': sides[i]})
            else:
                self.binTree[x].update({'P': -1})
                self.binTree[x].update({'side': 'C'})
        # ---- assign children relationships
        lchildren = [x for x in range(1, len(vpnts)) if x % 2]
        rchildren = [x for x in range(1, len(vpnts)) if not x % 2]
        for i, child in enumerate(lchildren):
            self.binTree[vpnts[i]].update({'R': vpnts[rchildren[i]]})
            self.binTree[vpnts[i]].update({'L': vpnts[child]})
        # --- get & set bucket pointers
        pointers = real_indices.copy()
        for x in self.bucketTree:
            pointers[self.bucketTree[x]['real_indx']] = x
        self.bucketPointers = pointers
        # ---- get & set vantage points
        self.vpoints = vpnts
        # ---- get & set bucket points
        self.bucketpoints = [x for x in self.bucketTree]
        # ---- get & set vantage distances
        distances = dict()
        dist_matrix = np.ndarray((len(vpnts), self.traj.n_frames))
        for i, point in enumerate(self.vpoints):
            vector = md.rmsd(self.traj, self.traj, point, precentered=True)
            distances.update({point: vector})
            dist_matrix[i] = vector
        self.distances = distances
        self.vantage_matrix = dist_matrix
        # ---- set the sliced vantage matrix
        slices = dict()
        for x in self.bucketTree:
            idxs = self.bucketTree[x]['real_indx']
            sliced = dist_matrix[:, idxs].T
            slices.update({x: sliced})
        self.slices = slices

        # ---- set the buckets
        for x in self.bucketTree:
            mask = self.bucketTree[x]['real_indx']
            self.bucketTree[x].update({'bucket': subtraj[mask].center_coordinates()})

    def get_ancestors(self, vp):
        if vp in self.bucketTree:
            P = self.bucketTree[vp]['P']
            ancestors = deque()
            while P != -1:
                ancestors.append((P, self.binTree[P]['mu']))
                P = self.binTree[P]['P']
            return ancestors
        else:
            print(vp, ' is not a Vantage Point on the generated Tree !')
            return None

    def getBothTreesOri(self, real_indices, subtraj):
        # ---- construct the binary Tree and the associated buckets
        niter = 0
        vpnts = []
        to_explore = deque([[real_indices, subtraj]])
        while to_explore:
            subindx, subtraj = to_explore.popleft()
            niter += 1
            vp, mu, R_info, L_info = self.splitTraj(subindx, subtraj, vpnts)
            subtraj = None
            vpnts.append(vp)
            self.binTree.update({vp: {'mu': mu}})
            if niter > self.niters:
                R_real_idx, R_subtraj = R_info
                L_real_idx, L_subtraj = L_info
                for i in R_real_idx:
                    if i not in vpnts:
                        RID = i
                        break
                for i in L_real_idx:
                    if i not in vpnts:
                        LID = i
                        break
                self.bucketTree.update({RID: {'P': vp, 'real_indx': R_real_idx,
                                              'side': 'R', 'bucket': R_subtraj}})
                self.binTree[vp].update({'R': RID})
                self.bucketTree.update({LID: {'P': vp, 'real_indx': L_real_idx,
                                              'side': 'L', 'bucket': L_subtraj}})
                self.binTree[vp].update({'L': LID})
                continue
            else:
                to_explore.append(L_info)
                to_explore.append(R_info)
        # ---- assign parent relationships
        parents = [-1]
        sides = ['C']
        for x in range(int((len(vpnts) - 1) / 2)):
            parents.extend(list(it.repeat(x, 2)))
            sides.extend(['L', 'R'])
        for i, x in enumerate(vpnts):
            if i != 0:
                self.binTree[x].update({'P': vpnts[parents[i]]})
                self.binTree[x].update({'side': sides[i]})
            else:
                self.binTree[x].update({'P': -1})
                self.binTree[x].update({'side': 'C'})
        # ---- assign children relationships
        lchildren = [x for x in range(1, len(vpnts)) if x % 2]
        rchildren = [x for x in range(1, len(vpnts)) if not x % 2]
        for i, child in enumerate(lchildren):
            self.binTree[vpnts[i]].update({'R': vpnts[rchildren[i]]})
            self.binTree[vpnts[i]].update({'L': vpnts[child]})
        # --- get & set bucket pointers
        pointers = real_indices.copy()
        for x in self.bucketTree:
            pointers[self.bucketTree[x]['real_indx']] = x
        self.bucketPointers = pointers
        # ---- get & set vantage points
        self.vpoints = vpnts
        # ---- get & set bucket points
        self.bucketpoints = [x for x in self.bucketTree]
        # ---- get & set vantage distances
        distances = dict()
        dist_matrix = np.ndarray((len(vpnts), self.traj.n_frames))
        for i, point in enumerate(self.vpoints):
            vector = md.rmsd(self.traj, self.traj, point, precentered=True)
            distances.update({point: vector})
            dist_matrix[i] = vector
        self.distances = distances
        self.vantage_matrix = dist_matrix
        # ---- set the sliced vantage matrix
        slices = dict()
        for x in self.bucketTree:
            idxs = self.bucketTree[x]['real_indx']
            sliced = dist_matrix[:, idxs].T
            slices.update({x: sliced})
        self.slices = slices

    def get_buckets2xplor(self, vp, vp_side):
        xplor = deque([self.binTree[vp][vp_side]])
        buckets = []
        while xplor:
            subroot = xplor.popleft()
            if subroot in self.bucketTree:
                buckets.append(subroot)
                continue
            xplor.append(self.binTree[subroot]['L'])
            xplor.append(self.binTree[subroot]['R'])
        return buckets

    # @profile
    def query_node(self, q, k):
        # initial q info
        bucket_idx = self.bucketPointers[q]
        q_indices = self.bucketTree[bucket_idx]['real_indx']
        q_idx = np.where(q_indices == q)[0][0]
        q_side = self.bucketTree[bucket_idx]['side']
        q_ancestors = self.get_ancestors(bucket_idx)
        # initial bucket info
        q_traj = self.bucketTree[bucket_idx]['bucket']
        q_vec = md.rmsd(q_traj, q_traj, q_idx, precentered=True)
        eta = q_vec.argpartition(k)[:k + 1]
        dists = q_vec[eta]
        kheap = [*zip(-dists, q_indices[eta])]
        heapq.heapify(kheap)
        tau = -kheap[0][0]
        # calc real tau for comparison
        # real_vec = md.rmsd(self.traj, self.traj, q, precentered=True)
        # real_tau = real_vec[real_vec.argpartition(k)[k]]
        for q_parent, mu_parent in q_ancestors:
            q_dist2vp = self.distances[q_parent][q]
            if (q_side == 'L'):
                if (q_dist2vp <= mu_parent - tau):
                    q_side = self.binTree[q_parent]['side']
                    continue
                else:
                    trajs2xplore = self.get_buckets2xplor(q_parent, 'R')
                    for traj in trajs2xplore:
                        subtraj = self.bucketTree[traj]['bucket']
                        real_idx = self.bucketTree[traj]['real_indx']
                        vant_matrix_sel = self.slices[traj]
                        vector = self.vantage_matrix[:, q]
                        if not prunable_by_trineq(vant_matrix_sel, vector, tau):
                            q_vec2 = md.rmsd(subtraj, q_traj, q_idx, precentered=True)
                            eta2 = q_vec2.argpartition(k)[:k + 1]
                            dists2 = q_vec2[eta2]
                            boolean = dists2 < tau
                            candidates = [*zip(-dists2[boolean], real_idx[eta2][boolean])]
                            for c in candidates:
                                if c[0] > kheap[0][0]:
                                    # heapq.heapreplace(kheap, c)
                                    heapq.heappushpop(kheap, c)
                                    tau = -kheap[0][0]
                    q_side = self.binTree[q_parent]['side']
            elif (q_side == 'R'):
                if (q_dist2vp >= mu_parent + tau):
                    q_side = self.binTree[q_parent]['side']
                    continue
                else:
                    trajs2xplore = self.get_buckets2xplor(q_parent, 'L')
                    for traj in trajs2xplore:
                        subtraj = self.bucketTree[traj]['bucket']
                        real_idx = self.bucketTree[traj]['real_indx']
                        vant_matrix_sel = self.slices[traj]
                        vector = self.vantage_matrix[:, q]
                        if not prunable_by_trineq(vant_matrix_sel, vector, tau):
                            q_vec2 = md.rmsd(subtraj, q_traj, q_idx, precentered=True)
                            eta2 = q_vec2.argpartition(k)[:k + 1]
                            dists2 = q_vec2[eta2]
                            boolean = dists2 < tau
                            candidates = [*zip(-dists2[boolean], real_idx[eta2][boolean])]
                            for c in candidates:
                                if c[0] > kheap[0][0]:
                                    # heapq.heapreplace(kheap, c)
                                    heapq.heappushpop(kheap, c)
                                    tau = -kheap[0][0]
                    q_side = self.binTree[q_parent]['side']
        return tau, kheap
