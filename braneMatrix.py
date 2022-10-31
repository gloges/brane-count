"""Methods/class for constraint matrices."""

import numpy as np
import braneCount
import braneCreate
from itertools import combinations, product


# Partition numbers p(n) for n = 0,...,800
PARTITIONS = np.load('partitions.npy', allow_pickle=True)


class matrix:
    """Constraint matrix with corresponding A- and B-brane data.

    Parameters
    ----------
    M : ndarray, optional
        3x4 constraint matrix in (I)nteger (R)educed (R)ow-(E)chelon (F)orm.
    As : ndarray, optional
        kx3x4 array of A branes compatible with `M`, containing XIpos, XIneg, YI data.
    Bs : ndarray, optional
        kx2x4 array of B branes compatible with `M`, containing XI, YI data.

    Attributes
    ----------
    M : ndarray
        3x4 constraint matrix in (I)nteger (R)educed (R)ow-(E)chelon (F)orm.
    rankM : {0, 1, 2, 3}
        Matrix rank of `M`.
    XIpos_A : ndarray
        kx4 array of positive tadpoles for A branes.
    XIneg_A : ndarray
        kx4 array of (absolute values of) negative tadpoles for A branes.
    YI_A : ndarray
        kx4 array of contraints for A branes.
    XI_B : ndarray
        kx4 array of tadpoles for B branes.
    YI_B : ndarray
        kx4 array of constraints for B branes.
    counted: bool
        Whether the number of solutions has been counted.

    """

    def __init__(self, M=None, As=None, Bs=None):

        self.M = np.zeros([3, 4], dtype='int')
        self.rankM = 0
        self.rmin = 1

        self.XIpos_A = np.empty([0, 4], dtype='int')
        self.XIneg_A = np.empty([0, 4], dtype='int')
        self.YI_A    = np.empty([0, 4], dtype='int')

        self.XI_B = np.empty([0, 4], dtype='int')
        self.YI_B = np.empty([0, 4], dtype='int')

        if M is not None:
            self.M = M
            self.rankM = np.linalg.matrix_rank(M)
            self.rmin = braneCount.minModulusRatio(self.M)

        if As is not None:
            self.addAs(As)

        if Bs is not None:
            self.addBs(Bs)

        self.counted = False

    def addAs(self, As):
        """Add A branes to XIpos, XIneg, YI arrays."""

        # Add data to existing arrays
        As = np.array(As)
        self.XIpos_A = np.append(self.XIpos_A, As[:, 0], axis=0)
        self.XIneg_A = np.append(self.XIneg_A, As[:, 1], axis=0)
        self.YI_A    = np.append(self.YI_A,    As[:, 2], axis=0)

        # Remove any duplicates
        self.removeRepeats()

    def addBs(self, Bs):
        """Add A branes to XIpos, XIneg, YI arrays."""

        # Add data to existing arrays
        Bs = np.array(Bs)
        self.XI_B = np.append(self.XI_B, Bs[:, 0], axis=0)
        self.YI_B = np.append(self.YI_B, Bs[:, 1], axis=0)

        # Remove any duplicates
        self.removeRepeats()

    def getABmatrices(self, T, bix2):
        """Construct all constraint matrices which result from adding B branes to existing A branes.

        This method should only be called for rank-1 or rank-2 matrices with only A branes.

        * `bix2` specifies which tori are tilted.
        * Only those matrices which allow for 0 < U0 ≤ U1 ≤ U2 ≤ U3 are returned.

        Parameters
        ----------
        T : int
            Tadpole (positive integer).
        bix2 : array_like
            Tori tilts. `bix2` should have three entries, each either 0 (untilted) or 1 (tilted).

        Returns
        -------
        matrices_AB_rank2 : ndarray
            The kx3x4 array of rank-2 matrices in (I)nteger (R)educed (R)ow-(E)chelon (F)orm.
            This will be nonempty only if self.rankM is 1.
        matrices_AB_rank3 : ndarray
            The kx3x4 array of rank-3 matrices in (I)nteger (R)educed (R)ow-(E)chelon (F)orm.
        """

        # Prepare arrays for matrices of different rank
        matrices_AB_rank2 = np.empty([0], dtype='object')
        matrices_AB_rank3 = np.empty([0], dtype='object')

        # Collect A-brane data
        As = [[XIpos, XIneg, YI] for XIpos, XIneg, YI in zip(self.XIpos_A, self.XIneg_A, self.YI_A)]


        if self.rankM == 1:
            # M is rank 1 and corresponds to a lone A brane

            # Get info about lone A brane
            TI_A = self.XIpos_A[0] - self.XIneg_A[0]
            YI_A = self.YI_A[0]
            Na_max = T // max(self.XIpos_A[0])

            # Construct complete list of possible A-tadpoles
            TI_A_list = np.array([Na*TI_A for Na in range(1, Na_max + 1)])

            # Get complete list of B branes which...
            #     1) satisfy the tadpole bounds for some value of Na
            #     2) lead to a rank-2 constraint matrix which intersects the modulus cone

            # All six types of B branes
            Jpairs = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]

            # Prepare array for B branes
            Bs = np.empty([6], dtype='object')

            for ii, [J1, J2] in enumerate(Jpairs):

                # Get tadpole bounds for B branes (one will be larger than T
                # because the A brane has a negative tadpole, the other will be less than T)
                XI_B_max = T - np.min(TI_A_list[:, [J1, J2]], axis=0)

                if min(XI_B_max) <= 0:
                    # No B branes of this type are possible
                    Bs[ii] = np.empty([0, 2, 4], dtype='int')
                else:
                    # Get all B branes of this type satisfying naive tadpole bounds
                    Bs[ii] = braneCreate.typeBIJ(J1, J2, XI_B_max, bix2)

                # Restrict to those B branes which lead to an admissible rank-2 constraint matrix
                Ms = np.array([braneCount.IRREF3(YI_A, b[1]) for b in Bs[ii]])
                cone = [braneCount.intersectsModulusCone(M) for M in Ms]
                Ms = Ms[cone]
                Bs[ii] = Bs[ii][cone]

                # Prepare array to keep track of which B branes can actually
                # be combined with a stack of the A brane
                possible = np.zeros(len(Bs[ii]), dtype='bool')

                # Scan through B branes
                for jj, [b, M] in enumerate(zip(Bs[ii], Ms)):

                    # Check that there is a stack size Na of the A brane which actually works
                    for TI in TI_A_list:
                        if max(TI + b[0]) <= T:
                            possible[jj] = True
                            break

                    if possible[jj]:
                        # Create the matrix and add to list of rank-2 constraint matrices
                        if braneCount.gaugeFixed(M):
                            newMatrix = matrix(M, As=As, Bs=[b])
                            matrices_AB_rank2 = np.append(matrices_AB_rank2, [newMatrix])

                # Restrict to B branes which can be combined with the A brane in some way
                Bs[ii] = Bs[ii][possible]

            # Scan through pairs of types of B branes (e.g. B[01] branes and B[13] branes)
            for ii, jj in combinations(range(6), 2):

                if len(Bs[ii]) == 0 or len(Bs[jj]) == 0:
                    continue

                J1, J2 = Jpairs[ii]
                J3, J4 = Jpairs[jj]

                # Scan through B branes of the first type
                for b1 in Bs[ii]:
                    MAb1 = braneCount.IRREF3(YI_A, b1[1])
                    if not braneCount.intersectsModulusCone(MAb1):
                        continue

                    # Scan through stack sizes for the A brane
                    for TI in TI_A_list:
                        TI_A_b1 = TI + b1[0]

                        # Find all B branes of the second type which satisfy the tadpole bounds
                        whr_b2 = np.where(np.max(TI_A_b1 + Bs[jj][:, 0], axis=1) <= T)[0]
                        tad = np.max(TI_A_b1 + Bs[jj][whr_b2, 0], axis=1) <= T
                        whr_b2 = whr_b2[tad]

                        # Restrict to those for which M[A,b1,b2] is admissible
                        Ms = np.array([braneCount.IRREF3(MAb1, b2[1]) for b2 in Bs[jj][whr_b2]])
                        cone = [braneCount.intersectsModulusCone(M) for M in Ms]
                        whr_b2 = whr_b2[cone]
                        Ms = Ms[cone]

                        # For those triples (A,b1,b2) which work, create matrix object and save
                        for b2, M in zip(Bs[jj][whr_b2], Ms):
                            newMatrix = matrix(M, As=As, Bs=[b1, b2])
                            if newMatrix.rankM == 3:
                                matrices_AB_rank3 = np.append(matrices_AB_rank3, [newMatrix])


        elif self.rankM == 2:
            # M is rank 2 and corresponds to one or more A branes

            # Get tadpoles, matrices, etc from combining A branes in all possible ways
            # while satisfying the tadpole bounds
            G_TI, G_KI, G_M, G_nums, G_rnkAB = self.getG_A(T)

            # Restrict to those which actually have the correct constraint matrix
            whr = [np.array_equal(M, self.M) for M in G_M]
            G_TI = G_TI[whr]
            G_M = G_M[whr]

            # Scan over all six types of B branes
            for J1, J2 in combinations(range(4), 2):

                # Get naive tadpole bounds on these B branes
                XI_B_max = T - np.min(G_TI[:, [J1, J2]], axis=0)

                if min(XI_B_max) <= 0:
                    continue

                # Create all B branes satisfying the naive tadpole bounds
                Bs = braneCreate.typeBIJ(J1, J2, XI_B_max, bix2)

                # Get all constraints (in reduced form)
                YIs = [YI // np.gcd.reduce(YI) for YI in Bs[:, 1]]

                # Get unique B brane constraints
                YIunique, YIinverse = np.unique(YIs, axis=0, return_inverse=True)

                # Scan through unique B brane constraints
                for ii, YI_B in enumerate(YIunique):

                    # Get constraint matrix from adding in the B brane constraint
                    Mnew = braneCount.IRREF3(self.M, YI_B)

                    if np.array_equal(Mnew, self.M):
                        # If M is unchanged, add the corresponding B branes to self
                        self.addBs(Bs[YIinverse == ii])

                    elif braneCount.intersectsModulusCone(Mnew):
                        # This leads to an admissible rank-3 constraint matrix
                        # Get B branes with this constraint and check it there is a combination
                        # of A branes which allows for the tadpole bounds to actually be satisfied
                        Bs_subset = Bs[YIinverse == ii]
                        possible = False
                        for b, TI_A in product(Bs_subset, G_TI):
                            if max(b[0] + TI_A) <= T:
                                possible = True
                                break

                        if possible:
                            # Create matrix object and save
                            newMatrix = matrix(Mnew, As=As, Bs=Bs_subset)
                            matrices_AB_rank3 = np.append(matrices_AB_rank3, [newMatrix])

        # Combine identical matrices
        matrices_AB_rank2 = braneCount.combineRepeats(matrices_AB_rank2)
        matrices_AB_rank3 = braneCount.combineRepeats(matrices_AB_rank3)

        return matrices_AB_rank2, matrices_AB_rank3

    def getG_A(self, T):
        """Find all combinations of A branes which satisfy the tadpole bounds."""

        # Prepare arrays for info about A brane combinations
        # Initialized with the 'empty configuration'
        G_TIpos = np.zeros([1, 4], dtype='int')
        G_TIneg = np.zeros([1, 4], dtype='int')
        G_KI    = np.zeros([1, 4], dtype='int')
        G_M     = np.zeros([1, 3, 4], dtype='int')
        G_nums  = np.ones([1, 2], dtype='object')
        G_rnkAB = np.empty([1, 4], dtype='object')

        # The 'hack' ensures it remains a numpy array of 'objects'
        G_rnkAB[0] = [np.array([0]), np.array([1]), np.array([1]), 'hack']

        # Scan through A branes
        for ii, [XIpos, XIneg, YI] in enumerate(zip(self.XIpos_A, self.XIneg_A, self.YI_A)):

            # Prepare arrays of data to be added to G_*
            toAdd_TIpos = np.empty([0, 4], dtype='int')
            toAdd_TIneg = np.empty([0, 4], dtype='int')
            toAdd_KI    = np.empty([0, 4], dtype='int')
            toAdd_M     = np.empty([0, 3, 4], dtype='int')
            toAdd_nums  = np.empty([0, 2], dtype='object')
            toAdd_rnkAB = np.empty([0, 4], dtype='object')

            # Scan through existing A brane combinations in 'G'
            for TIpos, TIneg, KI, M, nums, rnkAB in \
              zip(G_TIpos, G_TIneg, G_KI, G_M, G_nums, G_rnkAB):

                # Determine largest stack size using crude (but exact) bounds on T2+,T2-,T3+,T3-
                Na_max = np.inf
                if XIpos[2] > 0:
                    Na_max = min(Na_max, (2*T - TIpos[2]) // XIpos[2])
                if XIpos[3] > 0:
                    Na_max = min(Na_max, (3*T//2 - TIpos[3]) // XIpos[3])
                if XIneg[2] > 0:
                    Na_max = min(Na_max, (T - TIneg[2]) // XIneg[2])
                if XIneg[3] > 0:
                    Na_max = min(Na_max, (T//2 - TIneg[3]) // XIneg[3])

                # New constraint matrix is guaranteed to be admissible (since self.M is)
                M_new = braneCount.IRREF3(M, YI)

                # Scan through stack sizes for A brane being added, compute all data and save
                for Na in range(1, Na_max + 1):
                    TIpos_new = TIpos + Na*XIpos
                    TIneg_new = TIneg + Na*XIneg

                    KI_new = (KI + Na*YI) % 2

                    nums_new = nums.copy()
                    nums_new[1] *= PARTITIONS[Na]

                    rnkAB_new = np.empty([4], dtype='object')
                    rnkAB_new[0] = rnkAB[0] + Na
                    rnkAB_new[1] = rnkAB[1]
                    rnkAB_new[2] = rnkAB[2] * PARTITIONS[Na]
                    rnkAB_new[3] = 'hack'

                    toAdd_TIpos = np.append(toAdd_TIpos, [TIpos_new], axis=0)
                    toAdd_TIneg = np.append(toAdd_TIneg, [TIneg_new], axis=0)
                    toAdd_KI    = np.append(toAdd_KI,    [KI_new],    axis=0)
                    toAdd_M     = np.append(toAdd_M,     [M_new],     axis=0)
                    toAdd_nums  = np.append(toAdd_nums,  [nums_new],  axis=0)
                    toAdd_rnkAB = np.append(toAdd_rnkAB, [rnkAB_new], axis=0)

            # Add in all new combinations from adding this A brane to all existing configurations
            G_TIpos = np.append(G_TIpos, toAdd_TIpos, axis=0)
            G_TIneg = np.append(G_TIneg, toAdd_TIneg, axis=0)
            G_KI    = np.append(G_KI,    toAdd_KI,    axis=0)
            G_M     = np.append(G_M,     toAdd_M,     axis=0)
            G_nums  = np.append(G_nums,  toAdd_nums,  axis=0)
            G_rnkAB = np.append(G_rnkAB, toAdd_rnkAB, axis=0)

        # Get net tadpoles
        G_TI = G_TIpos - G_TIneg

        # Delete all combinations which oversaturate one or more tadpoles
        toDelete = np.where(np.max(G_TI, axis=1) > T)[0]

        G_TI    = np.delete(G_TI,    toDelete, axis=0)
        G_KI    = np.delete(G_KI,    toDelete, axis=0)
        G_M     = np.delete(G_M,     toDelete, axis=0)
        G_nums  = np.delete(G_nums,  toDelete, axis=0)
        G_rnkAB = np.delete(G_rnkAB, toDelete, axis=0)

        # Collect all identical entries (i.e. those with the same TI, KI, M)
        GG = combineG(G_TI, G_KI, G_M, G_nums, G_rnkAB)
        G_TI, G_KI, G_M, G_nums, G_rnkAB = GG

        # Is this check necessary?
        if np.min(G_nums) < 0:
            print('Overflow')

        return G_TI, G_KI, G_M, G_nums, G_rnkAB

    def countSolutions(self, T, bix2):
        """Count the total number of brane configurations corresponding to the constraint matrix.

        Parameters
        ----------
        T : int
            Tadpole (positive integer).
        bix2 : array_like
            Tori tilts. `bix2` should have three entries, each either 0 (untilted) or 1 (tilted).
        """

        # Get all combinations of A branes with TI <= T
        G_TI, G_KI, G_M, G_nums, G_rnkAB = self.getG_A(T)

        # Make sure list of B branes is complete
        XI_B_max = T - np.min(G_TI, axis=0)
        Bs_toAdd = np.empty([0, 2, 4], dtype='int')
        for J1, J2 in combinations(range(4), 2):
            Bs = braneCreate.getCompatibleBIJ(J1, J2, self.M, XI_B_max[[J1, J2]], bix2)
            for b in Bs:
                for TI_A in G_TI:
                    if max(TI_A + b[0]) <= T:
                        Bs_toAdd = np.append(Bs_toAdd, [b], axis=0)
                        break
        self.addBs(Bs_toAdd)

        # Add B branes to [G] one-by-one
        for ii, [XI_b, YI_b] in enumerate(zip(self.XI_B, self.YI_B)):
            toAdd_TI    = np.empty([0, 4], dtype='int')
            toAdd_KI    = np.empty([0, 4], dtype='int')
            toAdd_M     = np.empty([0, 3, 4], dtype='int')
            toAdd_nums  = np.empty([0, 2], dtype='object')
            toAdd_rnkAB = np.empty([0, 4], dtype='object')

            for TI, KI, M, nums, rnkAB in zip(G_TI, G_KI, G_M, G_nums, G_rnkAB):
                M_new = braneCount.IRREF3(M, YI_b)

                if max(TI + XI_b) > T:
                    continue

                Nb = 1
                TI_new = TI + Nb*XI_b
                while max(TI_new) <= T:
                    KI_new = (KI + Nb*YI_b) % 2

                    nums_new = nums.copy()
                    nums_new[1] *= PARTITIONS[Nb]

                    rnkAB_new = np.empty([4], dtype='object')
                    rnkAB_new[0] = rnkAB[0] + Nb
                    rnkAB_new[1] = rnkAB[1]
                    rnkAB_new[2] = rnkAB[2] * PARTITIONS[Nb]
                    rnkAB_new[3] = 'hack'

                    toAdd_TI    = np.append(toAdd_TI,    [TI_new],    axis=0)
                    toAdd_KI    = np.append(toAdd_KI,    [KI_new],    axis=0)
                    toAdd_M     = np.append(toAdd_M,     [M_new],     axis=0)
                    toAdd_nums  = np.append(toAdd_nums,  [nums_new],  axis=0)
                    toAdd_rnkAB = np.append(toAdd_rnkAB, [rnkAB_new], axis=0)

                    Nb += 1
                    TI_new = TI + Nb*XI_b

            G_TI    = np.append(G_TI,    toAdd_TI,    axis=0)
            G_KI    = np.append(G_KI,    toAdd_KI,    axis=0)
            G_M     = np.append(G_M,     toAdd_M,     axis=0)
            G_nums  = np.append(G_nums,  toAdd_nums,  axis=0)
            G_rnkAB = np.append(G_rnkAB, toAdd_rnkAB, axis=0)

            GG = combineG(G_TI, G_KI, G_M, G_nums, G_rnkAB)
            G_TI, G_KI, G_M, G_nums, G_rnkAB = GG


        # Find those with the correct constraint matrix
        whr = np.where([np.array_equal(M, self.M) for M in G_M])[0]

        # Further restict to those which can cancel tadpoles exactly with inclusion on C branes
        numTilts = sum(bix2)
        tadC = [np.array_equal((T - TI) % (2**numTilts), [0, 0, 0, 0]) for TI in G_TI[whr]]
        whr = whr[tadC]

        # Compute number of C branes and multiplicative factor when counting vacua
        numCbranes = (T - G_TI) // (2**numTilts)
        Cbranefactor = np.array([PARTITIONS[C0] * PARTITIONS[C1] * PARTITIONS[C2] * PARTITIONS[C3]
                                 for C0, C1, C2, C3 in numCbranes])
        G_nums[whr, 1] = G_nums[whr, 1] * Cbranefactor[whr]

        # Record number of solutions and which A branes are used when K-theory is not imposed
        self.numSoln_noK = np.sum(G_nums[whr],  axis=0)

        # Get rank of gauge group including C branes and combine tallies
        G_rnkABC = np.empty([0, 4], dtype='object')
        for g, numCs in zip(G_rnkAB, numCbranes):
            factor = PARTITIONS[numCs[0]] * PARTITIONS[numCs[1]] * PARTITIONS[numCs[2]] * PARTITIONS[numCs[3]]
            G_rnkABC = np.append(G_rnkABC, [[g[0] + sum(numCs), g[1], factor*g[2], 'hack']], axis=0)

        if len(whr) > 0:
            self.rnkAB_noK  = combineRankCounts(G_rnkAB[whr])
            self.rnkABC_noK = combineRankCounts(G_rnkABC[whr])
        else:
            self.rnkAB_noK  = [*np.empty([3, 0], dtype='int'), 'hack']
            self.rnkABC_noK = [*np.empty([3, 0], dtype='int'), 'hack']


        # Now restrict to those which cancel the K-theory charges
        Ktheory = [np.array_equal(KI, [0, 0, 0, 0]) for KI in G_KI[whr]]
        whr = whr[Ktheory]

        # Record data as before
        self.numSoln = np.sum(G_nums[whr],  axis=0)
        if len(whr) > 0:
            self.rnkAB  = combineRankCounts(G_rnkAB[whr])
            self.rnkABC = combineRankCounts(G_rnkABC[whr])
        else:
            self.rnkAB  = [*np.empty([3, 0], dtype='int'), 'hack']
            self.rnkABC = [*np.empty([3, 0], dtype='int'), 'hack']

        self.counted = True

    def combineWith(self, matrices):
        """Combine with one or more matrix objects which have the same constraint matrix."""

        for matrix in matrices:

            if not np.array_equal(self.M, matrix.M):
                print('Constraint matrices not equal.')
                return

            self.XIpos_A = np.append(self.XIpos_A, matrix.XIpos_A, axis=0)
            self.XIneg_A = np.append(self.XIneg_A, matrix.XIneg_A, axis=0)
            self.YI_A    = np.append(self.YI_A,    matrix.YI_A,    axis=0)

            self.XI_B = np.append(self.XI_B, matrix.XI_B, axis=0)
            self.YI_B = np.append(self.YI_B, matrix.YI_B, axis=0)

        self.removeRepeats()

    def removeRepeats(self):
        """Prune duplicate A and B brane XI,YI data."""

        if len(self.XIpos_A) > 0:
            Adata = np.array([[XIpos, XIneg, YI] for XIpos, XIneg, YI in zip(self.XIpos_A, self.XIneg_A, self.YI_A)])
            Adata = np.unique(Adata, axis=0)

            self.XIpos_A = Adata[:, 0]
            self.XIneg_A = Adata[:, 1]
            self.YI_A    = Adata[:, 2]

        if len(self.XI_B) > 0:
            Bdata = np.array([[XI, YI] for XI, YI in zip(self.XI_B, self.YI_B)])
            Bdata = np.unique(Bdata, axis=0)

            self.XI_B = Bdata[:, 0]
            self.YI_B = Bdata[:, 1]

    def getAJcounts(self):
        """Returns the number of A branes of each type."""

        if len(self.XIneg_A) == 0:
            return [0, 0, 0, 0]

        numAJs = [XIneg // max(XIneg) for XIneg in self.XIneg_A]
        return np.sum(numAJs, axis=0)

    def getBIJcounts(self):
        """Returns the number of B branes of each type."""

        if len(self.XI_B) == 0:
            return [0, 0, 0, 0, 0, 0]

        numBIJs = np.zeros(6, dtype='int')
        for ii, [J1, J2] in enumerate(combinations(range(4), 2)):
            numBIJs[ii] = sum((self.XI_B[:, J1] > 0) * (self.XI_B[:, J2] > 0))
        return numBIJs

    def display(self):
        """Display the constraint matrix, A and B brane data, and number of vacua."""

        print(100*'-')
        print(4*' '  + '{:6}{:6}{:6}{:6}'.format(*self.M[0]))
        print('M = ' + '{:6}{:6}{:6}{:6}'.format(*self.M[1]))
        print(4*' '  + '{:6}{:6}{:6}{:6}'.format(*self.M[2]))

        if len(self.XIpos_A) > 0:
            print('\nType A branes:')
            for XIpos, XIneg, YI in zip(self.XIpos_A, self.XIneg_A, self.YI_A):
                print(5*' ' + '{:4}{:4}{:4}{:4}'.format(*(XIpos - XIneg)), end='')
                print(5*' ' + '{:4}{:4}{:4}{:4}'.format(*YI))

        if len(self.XI_B) > 0:
            print('\nType B branes:')
            for XI, YI in zip(self.XI_B, self.YI_B):
                print(5*' ' + '{:4}{:4}{:4}{:4}'.format(*XI), end='')
                print(5*' ' + '{:4}{:4}{:4}{:4}'.format(*YI))

        if self.counted:
            print('\nSolutions:' + 10*' ' + '|      c(n)=1   c(n)=p(n)')
            print(14*' ' + 32*'-')
            print(15*' ' + 'no K |{:>12}{:>12}'.format(*self.numSoln_noK))
            print(15*' ' + '   K |{:>12}{:>12}'.format(*self.numSoln))
        else:
            print('\nSolutions: not counted yet')

    def getSaveData(self):
        """Returns a string of constraint matrix and A and B brane XI,YI data."""

        data = [*self.M.ravel(), len(self.YI_A), len(self.YI_B),
                *self.XIpos_A.ravel(), *self.XIneg_A.ravel(), *self.YI_A.ravel(),
                *self.XI_B.ravel(), *self.YI_B.ravel()]

        toSave = ' '.join([str(d) for d in data])

        return toSave

    def getCountData(self):
        """Returns a string of data summarizing the results of counting vacua."""

        data = [*self.M.ravel(), round(self.rmin, 4),
                *self.numSoln_noK, *self.numSoln,
                len(self.rnkAB_noK[0]), *self.rnkAB_noK[0], *self.rnkAB_noK[1], *self.rnkAB_noK[2],
                len(self.rnkABC_noK[0]), *self.rnkABC_noK[0], *self.rnkABC_noK[1], *self.rnkABC_noK[2],
                len(self.rnkAB[0]), *self.rnkAB[0], *self.rnkAB[1], *self.rnkAB[2],
                len(self.rnkABC[0]), *self.rnkABC[0], *self.rnkABC[1], *self.rnkABC[2]]

        toSave = ' '.join([str(d) for d in data])

        return toSave

    def restoreFromSave(self, saveData):
        """Initializes matrix using a string of data in the format provided by getSaveData()."""

        data = np.array([int(d) for d in saveData.split(' ')])

        self.M = data[:12].reshape(3, 4)
        self.rankM = np.linalg.matrix_rank(self.M)
        self.rmin = braneCount.minModulusRatio(self.M)

        numA, numB = data[12:14]

        self.XIpos_A = data[(14         ):(14 +  4*numA)].reshape(numA, 4)
        self.XIneg_A = data[(14 + 4*numA):(14 +  8*numA)].reshape(numA, 4)
        self.YI_A    = data[(14 + 8*numA):(14 + 12*numA)].reshape(numA, 4)

        iB = 14 + 12*numA

        self.XI_B = data[(iB         ):(iB + 4*numB)].reshape(numB, 4)
        self.YI_B = data[(iB + 4*numB):(iB + 8*numB)].reshape(numB, 4)

def combineG(G_TI, G_KI, G_M, G_nums, G_rnkAB):
    """Combines entries for which TI,KI,M are all identical."""

    flattened = np.empty([0, 20], dtype='int')
    for TI, KI, M in zip(G_TI, G_KI, G_M):
        flattened = np.append(flattened, [[*TI, *KI, *np.ravel(M)]], axis=0)

    unique, inverse, tallies = np.unique(flattened, axis=0, return_inverse=True, return_counts=True)

    toDelete = np.empty([0], dtype='int')
    whrRepeated = np.where(tallies > 1)[0]
    for ii in whrRepeated:
        inds = np.where(inverse == ii)[0]
        toDelete = np.append(toDelete, inds[1:])

        G_nums[inds[0]]  = np.sum(G_nums[inds],  axis=0)
        G_rnkAB[inds[0]] = combineRankCounts(G_rnkAB[inds])

    G_TI    = np.delete(G_TI,    toDelete, axis=0)
    G_KI    = np.delete(G_KI,    toDelete, axis=0)
    G_M     = np.delete(G_M,     toDelete, axis=0)
    G_nums  = np.delete(G_nums,  toDelete, axis=0)
    G_rnkAB = np.delete(G_rnkAB, toDelete, axis=0)

    return G_TI, G_KI, G_M, G_nums, G_rnkAB

def combineRankCounts(rnkarray):
    """Combines redundant entries in rnkarray."""

    ranks_combined = np.concatenate(rnkarray[:, 0])
    tally_1_combined = np.concatenate(rnkarray[:, 1])
    tally_2_combined = np.concatenate(rnkarray[:, 2])

    ranks_unique, inverse = np.unique(ranks_combined, return_inverse=True)
    tally_1_new = np.zeros(len(ranks_unique), dtype='object')
    tally_2_new = np.zeros(len(ranks_unique), dtype='object')

    for ii in range(len(ranks_unique)):
        subset = np.where(inverse == ii)[0]

        # Loop and int() are to avoid overflow errors
        for jj in subset:
            tally_1_new[ii] += int(tally_1_combined[jj])
            tally_2_new[ii] += int(tally_2_combined[jj])

    return [ranks_unique, tally_1_new, tally_2_new, 'hack']
