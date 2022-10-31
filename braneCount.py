"""Methods to count the number of vacua for the T6/Z2xZ2 orientifold."""


import numpy as np
from scipy.optimize import linprog
import braneCreate
import braneMatrix
import matplotlib.pyplot as plt
from itertools import product, combinations, permutations
import time


def getTIbounds_AJ12_2(J1, J2, T, bix2):

    if [J1, J2] == [0, 1]:
        if T == 1: return [10, 10]
        if T == 2: return [10, 10]
        if T == 3: return [12, 10]
        if T == 4: return [46, 30]
        if T == 5:
            if np.array_equal(bix2, [0, 0, 0]): return [80, 35]
            if np.array_equal(bix2, [0, 0, 1]): return [80, 35]
            if np.array_equal(bix2, [0, 1, 0]): return [80, 35]
            if np.array_equal(bix2, [1, 0, 0]): return [80, 35]
            if np.array_equal(bix2, [0, 1, 1]): return [50, 20]
            if np.array_equal(bix2, [1, 0, 1]): return [80, 35]
            if np.array_equal(bix2, [1, 1, 0]): return [80, 35]
            if np.array_equal(bix2, [1, 1, 1]): return [50, 20]
        if T == 6:
            if np.array_equal(bix2, [0, 0, 0]): return [170, 90]
            if np.array_equal(bix2, [0, 0, 1]): return [155, 55]
            if np.array_equal(bix2, [0, 1, 0]): return [155, 55]
            if np.array_equal(bix2, [1, 0, 0]): return [170, 90]
            if np.array_equal(bix2, [0, 1, 1]): return [65, 25]
            if np.array_equal(bix2, [1, 0, 1]): return [155, 55]
            if np.array_equal(bix2, [1, 1, 0]): return [155, 55]
            if np.array_equal(bix2, [1, 1, 1]): return [65, 25]
        if T == 7:
            if np.array_equal(bix2, [0, 0, 0]): return [570, 115]
            if np.array_equal(bix2, [0, 0, 1]): return [320, 115]
            if np.array_equal(bix2, [0, 1, 0]): return [320, 115]
            if np.array_equal(bix2, [1, 0, 0]): return [570, 115]
            if np.array_equal(bix2, [0, 1, 1]): return [125, 70]
            if np.array_equal(bix2, [1, 0, 1]): return [320, 115]
            if np.array_equal(bix2, [1, 1, 0]): return [320, 115]
            if np.array_equal(bix2, [1, 1, 1]): return [125, 70]
        if T == 8:
            if np.array_equal(bix2, [0, 0, 0]): return [800, 225]
            if np.array_equal(bix2, [0, 0, 1]): return [800, 130]
            if np.array_equal(bix2, [0, 1, 0]): return [800, 130]
            if np.array_equal(bix2, [1, 0, 0]): return [800, 150]
            if np.array_equal(bix2, [0, 1, 1]): return [205, 80]
            if np.array_equal(bix2, [1, 0, 1]): return [800, 130]
            if np.array_equal(bix2, [1, 1, 0]): return [800, 130]
            if np.array_equal(bix2, [1, 1, 1]): return [205, 65]

    elif [J1, J2] == [0, 2]:
        if T == 1: return [4, 2]
        if T == 2: return [4, 2]
        if T == 3: return [5, 3]
        if T == 4: return [5, 3]
        if T == 5: return [15, 5]
        if T == 6: return [25, 6]
        if T == 7: return [35, 7]
        if T == 8: return [60, 8]

    elif [J1, J2] == [0, 3]:
        if T == 1: return [4, 2]
        if T == 2: return [4, 2]
        if T == 3: return [5, 3]
        if T == 4: return [5, 3]
        if T == 5: return [8, 4]
        if T == 6: return [8, 4]
        if T == 7: return [10, 4]
        if T == 8: return [25, 4]

    elif [J1, J2] == [1, 2]:
        return [T, T]

    elif [J1, J2] == [1, 3]:
        return [T, T//2]

    elif [J1, J2] == [2, 3]:
        return [T, T//2]

    else:
        print('Unexpected J1/J2:', J1, J2)
        return

def getTIbounds_AJ12_3(J1, J2, T, bix2):

    if [J1, J2] == [0, 1]:
        if T == 1: return [5, 3]
        if T == 2: return [5, 3]
        if T == 3: return [15, 10]
        if T == 4: return [15, 10]
        if T == 5:
            if np.array_equal(bix2, [0, 0, 0]): return [27, 12]
            if np.array_equal(bix2, [0, 0, 1]): return [17, 9]
            if np.array_equal(bix2, [0, 1, 0]): return [17, 9]
            if np.array_equal(bix2, [1, 0, 0]): return [27, 12]
            if np.array_equal(bix2, [0, 1, 1]): return [15, 8]
            if np.array_equal(bix2, [1, 0, 1]): return [15, 8]
            if np.array_equal(bix2, [1, 1, 0]): return [15, 8]
            if np.array_equal(bix2, [1, 1, 1]): return [15, 8]
        if T == 6:
            if np.array_equal(bix2, [0, 0, 0]): return [100, 35]
            if np.array_equal(bix2, [0, 0, 1]): return [55, 22]
            if np.array_equal(bix2, [0, 1, 0]): return [55, 22]
            if np.array_equal(bix2, [1, 0, 0]): return [100, 35]
            if np.array_equal(bix2, [0, 1, 1]): return [16, 10]
            if np.array_equal(bix2, [1, 0, 1]): return [55, 22]
            if np.array_equal(bix2, [1, 1, 0]): return [55, 22]
            if np.array_equal(bix2, [1, 1, 1]): return [20, 10]
        if T == 7:
            if np.array_equal(bix2, [0, 0, 0]): return [110, 40]
            if np.array_equal(bix2, [0, 0, 1]): return [70, 40]
            if np.array_equal(bix2, [0, 1, 0]): return [70, 40]
            if np.array_equal(bix2, [1, 0, 0]): return [110, 40]
            if np.array_equal(bix2, [0, 1, 1]): return [40, 20]
            if np.array_equal(bix2, [1, 0, 1]): return [70, 40]
            if np.array_equal(bix2, [1, 1, 0]): return [70, 40]
            if np.array_equal(bix2, [1, 1, 1]): return [40, 20]
        if T == 8:
            if np.array_equal(bix2, [0, 0, 0]): return [195, 65]
            if np.array_equal(bix2, [0, 0, 1]): return [140, 50]
            if np.array_equal(bix2, [0, 1, 0]): return [130, 45]
            if np.array_equal(bix2, [1, 0, 0]): return [195, 65]
            if np.array_equal(bix2, [0, 1, 1]): return [75, 30]
            if np.array_equal(bix2, [1, 0, 1]): return [140, 45]
            if np.array_equal(bix2, [1, 1, 0]): return [140, 45]
            if np.array_equal(bix2, [1, 1, 1]): return [75, 30]

    elif [J1, J2] == [0, 2]:
        if T == 1: return [4, 3]
        if T == 2: return [4, 2]
        if T == 3: return [3, 3]
        if T == 4: return [4, 4]
        if T == 5: return [5, 5]
        if T == 6: return [15, 6]
        if T == 7: return [25, 7]
        if T == 8: return [35, 8]

    elif [J1, J2] == [0, 3]:
        if T == 1: return [4, 0]
        if T == 2: return [4, 1]
        if T == 3: return [3, 1]
        if T == 4: return [4, 2]
        if T == 5: return [5, 2]
        if T == 6: return [6, 3]
        if T == 7: return [10, 3]
        if T == 8: return [20, 4]

    elif [J1, J2] == [1, 2]:
        return [T, T]

    elif [J1, J2] == [1, 3]:
        return [T, T//2]

    elif [J1, J2] == [2, 3]:
        return [T, T//2]

    else:
        print('Unexpected J1/J2:', J1, J2)
        return

def getTIbounds_AJ123(J1, J2, J3, T, bix2):

    if [J1, J2, J3] == [0, 1, 2]:
        if T == 1: return [5, 5, 1]
        if T == 2: return [8, 6, 2]
        if T == 3: return [8, 6, 3]
        if T == 4: return [8, 6, 4]
        if T == 5: return [8, 6, 5]
        if T == 6: return [8, 6, 6]
        if T == 7: return [14, 4, 3]
        if T == 8: return [30, 5, 4]

    elif [J1, J2, J3] == [0, 1, 3]:
        return [T, T, T//2]

    elif [J1, J2, J3] == [0, 2, 3]:
        return [T, T, T//2]

    elif [J1, J2, J3] == [1, 2, 3]:
        return [T, T, T//2]

    else:
        print('Unexpected J1/J2/J3:', J1, J2, J3)
        return

def getMatrices_1xJ(J, T, bix2, save=False):
    """Find all constraint matrices which result from combining only A[`J`] branes.

    Parameters
    ----------
    J : {0, 1, 2, 3}
        Which type of A brane to use.
    T : int
        Tadpole (positive integer).
    bix2 : array_like
        Tori tilts. `bix2` should have three entries, each either 0 (untilted) or 1 (tilted).
    save : bool, optional
        Whether results are saved to a file or returned.
    """

    # Since, XI > |XJ| for I < J, the negative tadpole is larger than T only for A[0] branes
    TJneg_max = T
    if J == 0:
        TJneg_max = T**3

    print('\n' + 100*'=')
    print('Building all constraint matrices for T={} and 2*bi={}{}{}'.format(T, *bix2))
    print('using only A[{}] branes'.format(J), end='')
    print(' subject to T{}- <= {}'.format(J, TJneg_max))
    print(100*'-' + '\n')

    # Build tadpole bounds
    TIpos_max = T * np.ones(4, dtype='int')
    TIneg_max = np.zeros(4, dtype='int')
    XImax = T * np.ones(4, dtype='int')
    XImax[J] = TJneg_max
    TIneg_max[J] = TJneg_max


    # Get all A[J] branes given XI bounds
    print('Getting A[{}] branes...'.format(J), end=' ')
    As = braneCreate.typeAJ(J, XImax, bix2)
    print('{} found'.format(len(As)))

    # Prepare arrays to store matrix objects
    matrices_AJ_rank1 = np.empty([0], dtype='object')
    matrices_AJ_rank2 = np.empty([0], dtype='object')
    matrices_AJ_rank3 = np.empty([0], dtype='object')

    if len(As) > 0:

        # Make matrix objects for lone A[J] branes
        for a in As:
            M1 = IRREF3(a[2])
            if gaugeFixed(M1):
                newMatrix = braneMatrix.matrix(M1, As=[a])
                matrices_AJ_rank1 = np.append(matrices_AJ_rank1, [newMatrix])

        print('\nFinding compatible A[{}]/A[{}] pairs...'.format(J, J))

        # Get (upper triangular) compatibility matrix then restrict
        # to those which appear in one or more pairs
        compatPairs = buildCompatibilityMatrix(As, As, T, TIpos_max, TIneg_max, [J], remove_diag=True)
        used = np.max(compatPairs, axis=0)
        compatPairs = compatPairs[used]
        compatPairs = compatPairs[:, used]
        compatPairs = np.triu(compatPairs)
        As = As[used]


        ii_list, jj_list = np.where(compatPairs)
        pairs = zip(ii_list, jj_list)

        if len(ii_list) == 0:
            print('\nNo compatible A[{}]/A[{}] pairs found.'.format(J, J))

        else:
            print('\nScanning through {} compatible A[{}]/A[{}] pairs...'.format(len(ii_list), J, J))

            t0 = time.time()
            for index in range(len(ii_list)):

                # Display progress (roughly every 10%)
                if (10*index) // len(ii_list) != (10*(index+1)) // len(ii_list):
                    print('{} / {} = {:.2f} %\t{:.2} min'.format(index, len(ii_list), (100*index)/len(ii_list), (time.time() - t0)/60))

                # Get next pair of compatible A branes
                ii, jj = next(pairs)
                a1 = As[ii]
                a2 = As[jj]

                # Get combined constraints
                M12 = IRREF3(a1[2], a2[2])

                # This pair gives an admissible matrix which will always be rank-2
                if gaugeFixed(M12):
                    matrices_AJ_rank2 = np.append(matrices_AJ_rank2, [braneMatrix.matrix(M12, As=[a1, a2])])

                # Get other A[J] branes which are compatible with BOTH a1 and a2
                whr_1 = np.where(compatPairs[ii])[0]
                whr_2 = np.where(compatPairs[jj])[0]
                kk_list = np.intersect1d(whr_1, whr_2)
                a3s = As[kk_list]

                if len(a3s) == 0:
                    continue

                # Get list of constraint matrices obtained by adding in these a3 branes
                M123s = np.array([IRREF3(M12, YI) for YI in a3s[:, 2]])
                M123s_unique, inverse = np.unique(M123s, axis=0, return_inverse=True)

                # a3 branes which do not increase the constraint matrix's rank are potentially
                # compatible with all triples
                whr_rank2 = np.where([np.array_equal(M123, M12) for M123 in M123s_unique])[0]

                # Scan through all M123 constraint matrices
                for mm, M123 in enumerate(M123s_unique):

                    if not intersectsModulusCone(M123):
                        continue

                    # Get all branes compatible with the matrix M123
                    a3s_subset = a3s[inverse == mm]
                    if len(whr_rank2) > 0:
                        a3s_subset = np.append(a3s_subset, a3s[inverse == whr_rank2[0]], axis=0)

                    # Determine if there is a linear combination which satisfies all tadpole bounds
                    possible, TIneg = findLinearCombination_2_more(a1, a2, a3s_subset, T)

                    # If so, create matrix object and add to list based on rank
                    if possible:
                        newMatrix = braneMatrix.matrix(M123, As=[a1, a2, *a3s_subset])
                        if newMatrix.rankM == 2:
                            if gaugeFixed(M123):
                                matrices_AJ_rank2 = np.append(matrices_AJ_rank2, newMatrix)
                        else:
                            matrices_AJ_rank3 = np.append(matrices_AJ_rank3, newMatrix)

    # Combine repeated constraint matrices, collecting all compatible
    # A[J] branes into single matrix object
    matrices_AJ_rank1 = combineRepeats(matrices_AJ_rank1)
    matrices_AJ_rank2 = combineRepeats(matrices_AJ_rank2)
    matrices_AJ_rank3 = combineRepeats(matrices_AJ_rank3)

    # Print summary:
    print('\nTotal constraint matrices found:')
    print('  Rank 1: {:5}'.format(len(matrices_AJ_rank1)))
    print('  Rank 2: {:5}'.format(len(matrices_AJ_rank2)))
    print('  Rank 3: {:5}'.format(len(matrices_AJ_rank3)))

    if save:
        filename_rank1 = 'matrixdata/T={}/2bi={}{}{}/matrices_A[{}]_rank1.txt'.format(T, *bix2, J)
        filename_rank2 = 'matrixdata/T={}/2bi={}{}{}/matrices_A[{}]_rank2.txt'.format(T, *bix2, J)
        filename_rank3 = 'matrixdata/T={}/2bi={}{}{}/matrices_A[{}]_rank3.txt'.format(T, *bix2, J)

        saveToFile(filename_rank1, matrices_AJ_rank1)
        saveToFile(filename_rank2, matrices_AJ_rank2)
        saveToFile(filename_rank3, matrices_AJ_rank3)

        return

    else:
        return matrices_AJ_rank1, matrices_AJ_rank2, matrices_AJ_rank3

def getMatrices_2xJ_2xStack(Js, T, bix2, TJ12neg_max=None, save=False):
    """Find all constraint matrices which result from combining exactly one A[J1] brane stack
    with exactly one A[J2] brane stack.

    Parameters
    ----------
    Js : array_like
        Array for values J1, J2 (0,1,2,3), the types of A branes to combine.
    T : int
        Tadpole (positive integer).
    bix2 : array_like
        Tori tilts. `bix2` should have three entries, each either 0 (untilted) or 1 (tilted).
    TJ12neg_max : array_like, optional
        Two bounds on the negative tadpoles. If not given, values are set by getTIbounds_AJ12_2().
    save : bool, optional
        Whether results are saved to a file or returned.
    """

    # Unpack values of J1, J2
    J1, J2 = Js

    if TJ12neg_max is None:
        TJ12neg_max = getTIbounds_AJ12_2(J1, J2, T, bix2)

    print('\n' + 100*'=')
    print('Building all constraint matrices for T={} and 2*bi={}{}{}'.format(T, *bix2), end='')
    print('using exactly one A[{}] brane and one A[{}] brane'.format(J1, J2), end='')
    print(' subject to T{}- <= {}'.format(J1, TJ12neg_max[0]), end='')
    print(' and T{}- <= {}'.format(J2, TJ12neg_max[1]))
    print(100*'-' + '\n')

    # Build tadpole bounds
    TIneg_max = np.zeros(4, dtype='int')
    TIneg_max[[J1, J2]] = TJ12neg_max

    TIpos_max = T + TIneg_max

    XI1max = TIpos_max.copy()
    XI1max[J1] = TIneg_max[J1]

    XI2max = TIpos_max.copy()
    XI2max[J2] = TIneg_max[J2]

    # Slightly improve bounds since will be looking for solutions with 2+ stacks
    for J in range(4):
        if J not in [J1, J2]:
            XI1max[J] -= 1
            XI2max[J] -= 1


    # Get all A[J1] and A[J2] branes compatible with bounds
    print('Getting A[{}] branes...'.format(J1), end=' ')
    A1s = braneCreate.typeAJ(J1, XI1max, bix2)
    print('{} found'.format(len(A1s)))

    print('Getting A[{}] branes...'.format(J2), end=' ')
    A2s = braneCreate.typeAJ(J2, XI2max, bix2)
    print('{} found'.format(len(A2s)))


    # Prepare arrays to store matrix objects
    matrices_AJ12_2_rank2 = np.empty([0], dtype='object')


    print('\nFinding compatible A[{}]/A[{}] pairs...'.format(J1, J2))

    # Keep track of TIneg
    TIneg_counts = np.zeros([TJ12neg_max[0] + 1, TJ12neg_max[1] + 1], dtype='int')

    t0 = time.time()
    # Loop through all A[J1] branes
    for index, a1 in enumerate(A1s):

        if (10*index) // len(A1s) != (10*(index+1)) // len(A1s):
            print('{} / {} = {:.2f} %\t{:.2} min'.format(index, len(A1s), (100*index)/len(A1s), (time.time() - t0)/60))


        # Get all compatible A[J2] branes
        whr_pos_2 = np.where(np.max(a1[0] + A2s[:, 0] - TIpos_max, axis=1) <= 0)[0]
        whr_neg_2 = np.where(np.max(a1[1] + A2s[whr_pos_2, 1] - TIneg_max, axis=1) <= 0)[0]
        whr_2 = whr_pos_2[whr_neg_2]

        improved = [satisfiesImprovedTIbounds(a1[0] + a2[0], a1[1] + a2[1],
                                              TIpos_max, TIneg_max, T, J_addable=[J1, J2])
                    for a2 in A2s[whr_2]]

        whr_2 = whr_2[improved]

        M12s = np.array([IRREF3(a1[2], a2[2]) for a2 in A2s[whr_2]])
        cone = np.where([intersectsModulusCone(M) for M in M12s])[0]
        whr_2 = whr_2[cone]
        M12s = M12s[cone]

        for a2, M12 in zip(A2s[whr_2], M12s):

            possible, TIneg = findLinearCombination_2(a1, a2, T)

            if possible and (TIneg <= TIneg_max).all():
                TIneg_counts[TIneg[J1], TIneg[J2]] += 1
                if gaugeFixed(M12):
                    newMatrix = braneMatrix.matrix(M12, As=[a1, a2])
                    matrices_AJ12_2_rank2 = np.append(matrices_AJ12_2_rank2, newMatrix)

    matrices_AJ12_2_rank2 = combineRepeats(matrices_AJ12_2_rank2)

    # Print summary:
    print('\nTotal constraint matrices found:')
    print('  Rank 2: {:8}'.format(len(matrices_AJ12_2_rank2)))



    whr = np.where(TIneg_counts > 0)

    if len(whr[0]) > 0:
        print('\nT{}- bound: {}\tT{}- bound: {}'.format(J1, TJ12neg_max[0], J2, TJ12neg_max[1]))
        print('Max found: {}\tMax found: {}'.format(max(whr[0]), max(whr[1])))

        toPlot = TIneg_counts.T
        toPlot[toPlot != 0] += np.max(toPlot)//2

        plt.imshow(toPlot, origin='lower', cmap='magma', aspect='auto')
        plt.title('T={}, 2bi={}{}{}, A[{}{}]$_2$'.format(T, *bix2, J1, J2))
        plt.xlabel('$T_-^{}$'.format(J1))
        plt.ylabel('$T_-^{}$'.format(J2))
        plt.tight_layout()

        if save:
            plt.savefig('matrixdata/T={}/2bi={}{}{}/A[{}{}]2.png'.format(T, *bix2, J1, J2), dpi=300)
        else:
            plt.show()

    if save:

        filename_rank2 = 'matrixdata/T={}/2bi={}{}{}/matrices_A[{}{}]_2_rank2.txt'.format(T, *bix2, J1, J2)

        saveToFile(filename_rank2, matrices_AJ12_2_rank2)

        return

    else:
        return matrices_AJ12_2_rank2

def getMatrices_2xJ_3xStack(Js, T, bix2, TJ12neg_max=None, save=False):

    J1, J2 = Js

    if TJ12neg_max is None:
        TJ12neg_max = getTIbounds_AJ12_3(J1, J2, T, bix2)

    print('\n' + 100*'=')
    print('Building all constraint matrices for T={} and 2*bi={}{}{}'.format(T, *bix2), end='')
    print('using at least one A[{}] brane and at least one A[{}] brane'.format(J1, J2))
    print('(three or more in total) subject to T{}- <= {}'.format(J1, TJ12neg_max[0]), end='')
    print(' and T{}- <= {}'.format(J2, TJ12neg_max[1]))
    print(100*'-' + '\n')

    # Build tadpole bounds
    TIneg_max = np.zeros(4, dtype='int')
    TIneg_max[[J1, J2]] = TJ12neg_max

    TIpos_max = T + TIneg_max

    XI1max = TIpos_max.copy()
    XI1max[J1] = TIneg_max[J1]

    XI2max = TIpos_max.copy()
    XI2max[J2] = TIneg_max[J2]

    # Slightly improve bounds since will be looking for solutions with 2+ stacks
    for J in range(4):
        if J not in [J1, J2]:
            XI1max[J] -= 2
            XI2max[J] -= 2


    # Get all A[J1] and A[J2] branes compatible with bounds
    print('Getting A[{}] branes...'.format(J1), end=' ')
    A1s = braneCreate.typeAJ(J1, XI1max, bix2)
    print('{} found'.format(len(A1s)))

    print('Getting A[{}] branes...'.format(J2), end=' ')
    A2s = braneCreate.typeAJ(J2, XI2max, bix2)
    print('{} found'.format(len(A2s)))


    # Prepare arrays to store matrix objects
    matrices_AJ12_3_rank2 = np.empty([0], dtype='object')
    matrices_AJ12_3_rank3 = np.empty([0], dtype='object')

    # Keep track of TIneg
    TIneg_counts = np.zeros([TJ12neg_max[0] + 1, TJ12neg_max[1] + 1], dtype='int')


    # Build three compatibility matrices. Solutions must containt at least one A[J1] brane
    # and one A[J2] branes, so compatPairs_12 will form the basis of candidate pairs/triples/etc.

    print('\nFinding compatible A[{}]/A[{}] pairs...'.format(J1, J1))
    compatPairs_11 = buildCompatibilityMatrix(A1s, A1s, T, TIpos_max, TIneg_max, [J1, J2], remove_diag=True)

    print('\nFinding compatible A[{}]/A[{}] pairs...'.format(J2, J2))
    compatPairs_22 = buildCompatibilityMatrix(A2s, A2s, T, TIpos_max, TIneg_max, [J1, J2], remove_diag=True)

    print('\nFinding compatible A[{}]/A[{}] pairs...'.format(J1, J2))
    compatPairs_12 = buildCompatibilityMatrix(A1s, A2s, T, TIpos_max, TIneg_max, [J1, J2])

    ii_list, jj_list = np.where(compatPairs_12)
    pairs_12 = zip(ii_list, jj_list)

    print('\nScanning through {} compatible A[{}]/A[{}] pairs...'.format(len(ii_list), J1, J2))

    t0 = time.time()
    # Loop through all compatible A[J1]--A[J2] pairs
    for index in range(len(ii_list)):
        if (10*index) // len(ii_list) != (10*(index+1)) // len(ii_list):
            print('{} / {} = {:.2f} %\t{:.2} min'.format(index, len(ii_list), (100*index)/len(ii_list), (time.time() - t0)/60))


        ii, jj = next(pairs_12)

        a1 = A1s[ii]
        a2 = A2s[jj]

        M12 = IRREF3(a1[2], a2[2])

        # Get other A1s/A2s compatible with a1/a2, respectively
        kk1_list = np.where(compatPairs_11[ii])[0]
        kk2_list = np.where(compatPairs_22[jj])[0]

        # Restrict to those which are compatible with both a1/a2
        kk1_list = kk1_list[compatPairs_12[kk1_list, jj]]
        kk2_list = kk2_list[compatPairs_12[ii, kk2_list]]

        a3s = np.concatenate([A1s[kk1_list], A2s[kk2_list]])

        if len(a3s) == 0:
            continue

        M123s = [IRREF3(M12, YI) for YI in a3s[:, 2]]
        M123s_unique, inverse = np.unique(M123s, axis=0, return_inverse=True)

        whr_rank2 = np.where([np.array_equal(M123, M12) for M123 in M123s_unique])[0]

        for mm, M123 in enumerate(M123s_unique):

            if not intersectsModulusCone(M123):
                continue

            a3s_subset = a3s[inverse == mm]

            if len(whr_rank2) > 0:
                a3s_subset = np.append(a3s_subset, a3s[inverse == whr_rank2[0]], axis=0)

            possible, TIneg = findLinearCombination_2_more(a1, a2, a3s_subset, T)

            if possible and (TIneg <= TIneg_max).all():
                TIneg_counts[TIneg[J1], TIneg[J2]] += 1
                newMatrix = braneMatrix.matrix(M123, As=[a1, a2, *a3s_subset])
                if newMatrix.rankM == 2:
                    if gaugeFixed(M123):
                        matrices_AJ12_3_rank2 = np.append(matrices_AJ12_3_rank2, newMatrix)
                else:
                    matrices_AJ12_3_rank3 = np.append(matrices_AJ12_3_rank3, newMatrix)


    matrices_AJ12_3_rank2 = combineRepeats(matrices_AJ12_3_rank2)
    matrices_AJ12_3_rank3 = combineRepeats(matrices_AJ12_3_rank3)

    # Print summary:
    print('\nTotal constraint matrices found:')
    print('  Rank 2: {:8}'.format(len(matrices_AJ12_3_rank2)))
    print('  Rank 3: {:8}'.format(len(matrices_AJ12_3_rank3)))


    whr = np.where(TIneg_counts > 0)

    if len(whr[0]) > 0:
        print('\nT{}- bound: {}\tT{}- bound: {}'.format(J1, TJ12neg_max[0], J2, TJ12neg_max[1]))
        print('Max found: {}\tMax found: {}'.format(max(whr[0]), max(whr[1])))

        toPlot = TIneg_counts.T
        toPlot[toPlot != 0] += np.max(toPlot)//2
        plt.imshow(toPlot, origin='lower', cmap='magma', aspect='auto')
        plt.title('T={}, 2bi={}{}{}, A[{}{}]$_3$'.format(T, *bix2, J1, J2))
        plt.xlabel('$T_-^{}$'.format(J1))
        plt.ylabel('$T_-^{}$'.format(J2))
        plt.tight_layout()

        if save:
            plt.savefig('matrixdata/T={}/2bi={}{}{}/A[{}{}]3.png'.format(T, *bix2, J1, J2), dpi=300)
        else:
            plt.show()

    if save:

        filename_rank2 = 'matrixdata/T={}/2bi={}{}{}/matrices_A[{}{}]_3_rank2.txt'.format(T, *bix2, J1, J2)
        filename_rank3 = 'matrixdata/T={}/2bi={}{}{}/matrices_A[{}{}]_3_rank3.txt'.format(T, *bix2, J1, J2)

        saveToFile(filename_rank2, matrices_AJ12_3_rank2)
        saveToFile(filename_rank3, matrices_AJ12_3_rank3)

        return

    else:
        return matrices_AJ12_3_rank2, matrices_AJ12_3_rank3

def getMatrices_3xJ(Js, T, bix2, TJ123neg_max=None, save=False):

    J1, J2, J3 = Js

    if TJ123neg_max is None:
        TJ123neg_max = getTIbounds_AJ123(J1, J2, J3, T, bix2)

    print('\n' + 100*'=')
    print('Building all constraint matrices for T={} and 2*bi={}{}{}'.format(T, *bix2), end='')
    print('using at least one A[{}] brane, at least one A[{}] brane and at least one A[{}] brane'.format(J1, J2, J3))
    print('subject to T{}- <= {}'.format(J1, TJ123neg_max[0]), end='')
    print(', T{}- <= {} and T{}- <= {}'.format(J2, TJ123neg_max[1], J3, TJ123neg_max[2]))
    print(100*'-' + '\n')

    # Build tadpole bounds
    TIneg_max = np.zeros(4, dtype='int')
    TIneg_max[[J1, J2, J3]] = TJ123neg_max

    TIpos_max = T + TIneg_max

    XI1max = TIpos_max.copy()
    XI1max[J1] = TIneg_max[J1]

    XI2max = TIpos_max.copy()
    XI2max[J2] = TIneg_max[J2]

    XI3max = TIpos_max.copy()
    XI3max[J3] = TIneg_max[J3]

    # Slightly improve bounds since will be looking for solutions with 3+ stacks
    for J in range(4):
        if J not in [J1, J2, J3]:
            XI1max[J] -= 2
            XI2max[J] -= 2
            XI3max[J] -= 2

    # Get all A[J1], A[J2] and A[J3] branes compatible with bounds
    print('Getting A[{}] branes...'.format(J1), end=' ')
    A1s = braneCreate.typeAJ(J1, XI1max, bix2)
    print('{} found'.format(len(A1s)))

    print('Getting A[{}] branes...'.format(J2), end=' ')
    A2s = braneCreate.typeAJ(J2, XI2max, bix2)
    print('{} found'.format(len(A2s)))

    print('Getting A[{}] branes...'.format(J3), end=' ')
    A3s = braneCreate.typeAJ(J3, XI3max, bix2)
    print('{} found'.format(len(A3s)))


    # Prepare arrays to store matrix objects
    matrices_AJ123_rank2 = np.empty([0], dtype='object')
    matrices_AJ123_rank3 = np.empty([0], dtype='object')

    # Keep track of TIneg
    TIneg_counts = np.zeros([TJ123neg_max[0] + 1, TJ123neg_max[1] + 1, TJ123neg_max[2] + 1], dtype='int')


    # Build six compatibility matrices. Solutions must containt at least one A[J1], one A[J2]
    # and one A[J3] brane

    print('\nFinding compatible A[{}]/A[{}] pairs...'.format(J1, J1))
    compatPairs_11 = buildCompatibilityMatrix(A1s, A1s, T, TIpos_max, TIneg_max, [J1, J2], remove_diag=True)

    print('\nFinding compatible A[{}]/A[{}] pairs...'.format(J2, J2))
    compatPairs_22 = buildCompatibilityMatrix(A2s, A2s, T, TIpos_max, TIneg_max, [J1, J2], remove_diag=True)

    print('\nFinding compatible A[{}]/A[{}] pairs...'.format(J3, J3))
    compatPairs_33 = buildCompatibilityMatrix(A3s, A3s, T, TIpos_max, TIneg_max, [J1, J2], remove_diag=True)

    print('\nFinding compatible A[{}]/A[{}] pairs...'.format(J1, J2))
    compatPairs_12 = buildCompatibilityMatrix(A1s, A2s, T, TIpos_max, TIneg_max, [J1, J2])

    print('\nFinding compatible A[{}]/A[{}] pairs...'.format(J1, J3))
    compatPairs_13 = buildCompatibilityMatrix(A1s, A3s, T, TIpos_max, TIneg_max, [J1, J2])

    print('\nFinding compatible A[{}]/A[{}] pairs...'.format(J2, J3))
    compatPairs_23 = buildCompatibilityMatrix(A2s, A3s, T, TIpos_max, TIneg_max, [J1, J2])


    ii2_list, ii3_list = np.where(compatPairs_23)
    pairs_23 = zip(ii2_list, ii3_list)

    print('\nScanning through {} compatible A[{}]/A[{}] pairs...'.format(len(ii2_list), J2, J3))

    t0 = time.time()
    # Loop through all compatible A[J2]--A[J3] pairs
    for index in range(len(ii2_list)):
        if (10*index) // len(ii2_list) != (10*(index+1)) // len(ii2_list):
            print('{} / {} = {:.2f} %\t{:.2} min'.format(index, len(ii2_list), (100*index)/len(ii2_list), (time.time() - t0)/60))

        ii2, ii3 = next(pairs_23)

        a2 = A2s[ii2]
        a3 = A3s[ii3]

        M23 = IRREF3(a2[2], a3[2])

        # Get A1s compatible with both a2/a3
        ii1_list = np.where(compatPairs_12[:, ii2] * compatPairs_13[:, ii3])[0]

        whr_pos = np.where(np.max(a2[0] + a3[0] + A1s[ii1_list, 0] - TIpos_max, axis=1) <= 0)[0]
        whr_neg = np.where(np.max(a2[1] + a3[1] + A1s[ii1_list, 1] - TIneg_max, axis=1) <= 0)[0]
        whr_tad = np.intersect1d(whr_pos, whr_neg)
        ii1_list = ii1_list[whr_tad]

        M123s = np.array([IRREF3(M23, a1[2]) for a1 in A1s[ii1_list]])
        cone = [intersectsModulusCone(M) for M in M123s]
        M123s = M123s[cone]
        ii1_list = ii1_list[cone]

        for ii1, M123 in zip(ii1_list, M123s):
            a1 = A1s[ii1]

            jj1_list = np.where(compatPairs_11[ii1])[0]
            jj2_list = np.where(compatPairs_22[ii2])[0]
            jj3_list = np.where(compatPairs_33[ii3])[0]

            a_subset = np.concatenate([A1s[jj1_list], A2s[jj2_list], A3s[jj3_list]])
            M1234s = np.array([IRREF3(M123, a[2]) for a in a_subset])
            cone = [intersectsModulusCone(M) for M in M1234s]
            a_subset = a_subset[cone]
            M1234s = M1234s[cone]

            #Just use a1/a2/a3
            possible, TIneg = findLinearCombination_3(a1, a2, a3, T)

            if possible and (TIneg <= TIneg_max).all():

                TIneg_counts[TIneg[J1], TIneg[J2], TIneg[J3]] += 1
                newMatrix = braneMatrix.matrix(M123, As=[a1, a2, a3])
                if newMatrix.rankM == 2:
                    if gaugeFixed(M123):
                        matrices_AJ123_rank2 = np.append(matrices_AJ123_rank2, newMatrix)
                else:
                    matrices_AJ123_rank3 = np.append(matrices_AJ123_rank3, newMatrix)

            if len(a_subset) > 0:
                M1234s_unique, inverse = np.unique(M1234s, axis=0, return_inverse=True)

                # a branes which do not increase the constraint matrix's rank are potentially
                # compatible with all triples
                whr_unchanged = np.where([np.array_equal(M1234, M123) for M1234 in M1234s_unique])[0]

                # Scan through all M123 constraint matrices
                for mm, M1234 in enumerate(M1234s_unique):

                    # Get all branes compatible with the matrix M123
                    a4s_subset = a_subset[inverse == mm]
                    if len(whr_unchanged) > 0:
                        a4s_subset = np.append(a4s_subset, a_subset[inverse == whr_unchanged[0]], axis=0)

                    # Determine if there is a linear combination which satisfies all tadpole bounds
                    possible, TIneg = findLinearCombination_3_more(a1, a2, a3, a4s_subset, T)


                    if possible and (TIneg <= TIneg_max).all():
                        TIneg_counts[TIneg[J1], TIneg[J2], TIneg[J3]] += 1
                        newMatrix = braneMatrix.matrix(M1234, As=[a1, a2, a3, *a4s_subset])
                        if newMatrix.rankM == 2:
                            if gaugeFixed(M1234):
                                matrices_AJ123_rank2 = np.append(matrices_AJ123_rank2, newMatrix)
                        else:
                            matrices_AJ123_rank3 = np.append(matrices_AJ123_rank3, newMatrix)


    matrices_AJ123_rank2 = combineRepeats(matrices_AJ123_rank2)
    matrices_AJ123_rank3 = combineRepeats(matrices_AJ123_rank3)

    # Print summary:
    print('\nTotal constraint matrices found:')
    print('  Rank 2: {:8}'.format(len(matrices_AJ123_rank2)))
    print('  Rank 3: {:8}'.format(len(matrices_AJ123_rank3)))

    whr = np.where(TIneg_counts > 0)

    if len(whr[0]) > 0:
        print('\nT{}- bound: {}\tT{}- bound: {}\tT{}- bound: {}'.format(J1, TJ123neg_max[0], J2, TJ123neg_max[1], J3, TJ123neg_max[2]))
        print('Max found: {}\tMax found: {}\tMax found: {}'.format(max(whr[0]), max(whr[1]), max(whr[2])))

        toPlot = np.sum(TIneg_counts, axis=2).T
        toPlot[toPlot != 0] += np.max(toPlot)//2
        plt.imshow(toPlot, origin='lower', cmap='magma', aspect='auto')
        plt.title('T={}, 2bi={}{}{}, A[{}{}{}]$_3$'.format(T, *bix2, J1, J2, J3))
        plt.xlabel('$T_-^{}$'.format(J1))
        plt.ylabel('$T_-^{}$'.format(J2))
        plt.tight_layout()

        if save:
            plt.savefig('matrixdata/T={}/2bi={}{}{}/A[{}{}{}]3.png'.format(T, *bix2, J1, J2, J3), dpi=300)
        else:
            plt.show()

    if save:

        filename_rank2 = 'matrixdata/T={}/2bi={}{}{}/matrices_A[{}{}{}]_rank2.txt'.format(T, *bix2, J1, J2, J3)
        filename_rank3 = 'matrixdata/T={}/2bi={}{}{}/matrices_A[{}{}{}]_rank3.txt'.format(T, *bix2, J1, J2, J3)

        saveToFile(filename_rank2, matrices_AJ123_rank2)
        saveToFile(filename_rank3, matrices_AJ123_rank3)

        return

    else:
        return matrices_AJ123_rank2, matrices_AJ123_rank3

def getMatrices_Bonly(T, bix2, save=False):
    """Find all constraint matrices which result from combining only B branes.

    Parameters
    ----------
    T : int
        Tadpole (positive integer).
    bix2 : array_like
        Tori tilts. `bix2` should have three entries, each either 0 (untilted) or 1 (tilted).
    save : bool, optional
        Whether results are saved to a file or returned.
    """

    print('\n' + 100*'=')
    print('Building all constraint matrices using only B branes for T={}'.format(T), end='')
    print(' and 2*bi={}{}{}'.format(*bix2))
    print(100*'-' + '\n')

    # Prepare arrays for matrix objects
    matrices_B_rank1 = np.empty([0], dtype='object')
    matrices_B_rank2 = np.empty([0], dtype='object')
    matrices_B_rank3 = np.empty([0], dtype='object')

    # Get complete lists of all six B brane types
    Bs = np.empty([0, 2, 4], dtype='int')
    for J1, J2 in combinations(range(4), 2):
        print('Getting B[{}{}] branes...'.format(J1, J2), end=' ')
        BIJs = braneCreate.typeBIJ(J1, J2, [T, T], bix2)
        print('{} found'.format(len(BIJs)))

        Bs = np.append(Bs, BIJs, axis=0)

    # Get constraints in reduced form and find list of unique constraints
    YIs = [YI // np.gcd.reduce(YI) for YI in Bs[:, 1]]
    YIunique, YIinverse = np.unique(YIs, axis=0, return_inverse=True)

    # Build rank-1 matrices and save
    for ii, YI in enumerate(YIunique):
        whr = np.where(YIinverse == ii)[0]
        M1 = IRREF3(YI)
        if gaugeFixed(M1):
            newMatrix = braneMatrix.matrix(M1, Bs=Bs[whr])
            matrices_B_rank1 = np.append(matrices_B_rank1, [newMatrix])

    # Find all pairs of unique YI which lead to admissible constraint matrix and can satisfy tadpole
    print('\nFinding all compatible B/B pairs...')

    # Array for recording compatible pairs
    compatible = np.zeros([len(YIunique), len(YIunique)], dtype='bool')

    # Scan through all pairs
    numPairs = len(YIunique)*(len(YIunique) - 1) // 2
    indexPairs = combinations(range(len(YIunique)), 2)
    t0 = time.time()
    for index in range(numPairs):

        # Display progress (roughly every 10%)
        if (10*index) // numPairs != (10*(index+1)) // numPairs:
            print('{} / {} = {:.2f} %\t{:.2} min'.format(index, numPairs, (100*index)/numPairs, (time.time() - t0)/60))

        # Get next pair and corresponding constraints
        ii, jj = next(indexPairs)
        YaI = YIunique[ii]
        YbI = YIunique[jj]

        # The combined constraint matrix
        Mab = IRREF3(YaI, YbI)

        if intersectsModulusCone(Mab):

            # Get all branes which have one of these constraints
            whr_a = np.where(YIinverse == ii)[0]
            whr_b = np.where(YIinverse == jj)[0]

            XaI_list = Bs[whr_a, 0]
            XbI_list = Bs[whr_b, 0]

            # Look at pairs of these branes to see if the tadpole bounds can be satisfied
            for XaI, XbI in product(XaI_list, XbI_list):
                if max(XaI + XbI) <= T:
                    compatible[ii, jj] = True
                    break

    pairs = np.array(np.where(compatible)).T

    # Triples of B branes (to give rank-3 constraint matrices) can be found
    # by looking for mutually-compatible pairs
    print('\nScanning through {} compatible B/B pairs...'.format(len(pairs)))

    t0 = time.time()
    for index in range(len(pairs)):

        # Display progress (roughly every 10%)
        if (10*index) // len(pairs) != (10*(index+1)) // len(pairs):
            print('{} / {} = {:.2f} %\t{:.2} min'.format(index, len(pairs), (100*index)/len(pairs), (time.time() - t0)/60))

        # Get the next compatible pair and the constraints
        ii, jj = pairs[index]
        YaI = YIunique[ii]
        YbI = YIunique[jj]

        # Find B branes with these constraints
        whr_a = np.where(YIinverse == ii)[0]
        whr_b = np.where(YIinverse == jj)[0]

        # Combined constraint matrix
        Mab = IRREF3(YaI, YbI)

        # We already know this pair is admissible - create matrix object and save
        if gaugeFixed(Mab):
            newMatrix = braneMatrix.matrix(Mab, Bs=[*Bs[whr_a], *Bs[whr_b]])
            matrices_B_rank2 = np.append(matrices_B_rank2, [newMatrix])

        # Find all constraints which are compatible with BOTH of those in the pair being considered
        kk_list = np.where(compatible[ii] * compatible[jj])[0]

        # Restrict to those which give an admissible rank-3 matrix
        Ms = np.array([IRREF3(Mab, YI) for YI in YIunique[kk_list]])
        cone = [intersectsModulusCone(M) for M in Ms]
        kk_list = kk_list[cone]
        Ms = Ms[cone]

        for kk, Mabc in zip(kk_list, Ms):

            # Find B branes with this third constraint
            whr_c = np.where(YIinverse == kk)[0]

            # Get lists of B branes with these three constraints
            XaI_list = Bs[whr_a, 0]
            XbI_list = Bs[whr_b, 0]
            XcI_list = Bs[whr_c, 0]

            # Loop through triples and look for a combination which satisfies the tadpole bounds
            for XaI, XbI, XcI in product(XaI_list, XbI_list, XcI_list):
                if max(XaI + XbI + XcI) <= T:
                    newMatrix = braneMatrix.matrix(Mabc, Bs=[*Bs[whr_a], *Bs[whr_b], *Bs[whr_c]])

                    # Only need to save if rank-3 (rank-2 would have already been saved)
                    if newMatrix.rankM == 3:
                        matrices_B_rank3 = np.append(matrices_B_rank3, [newMatrix])

                    break

    # Merge duplicate constraint matrices
    matrices_B_rank1 = combineRepeats(matrices_B_rank1)
    matrices_B_rank2 = combineRepeats(matrices_B_rank2)
    matrices_B_rank3 = combineRepeats(matrices_B_rank3)

    # Print summary
    print('\nTotal constraint matrices found:')
    print('  Rank 1: {:5}'.format(len(matrices_B_rank1)))
    print('  Rank 2: {:5}'.format(len(matrices_B_rank2)))
    print('  Rank 3: {:5}'.format(len(matrices_B_rank3)))


    if save:

        filename_rank1 = 'matrixdata/T={}/2bi={}{}{}/matrices_B_rank1.txt'.format(T, *bix2)
        filename_rank2 = 'matrixdata/T={}/2bi={}{}{}/matrices_B_rank2.txt'.format(T, *bix2)
        filename_rank3 = 'matrixdata/T={}/2bi={}{}{}/matrices_B_rank3.txt'.format(T, *bix2)

        saveToFile(filename_rank1, matrices_B_rank1)
        saveToFile(filename_rank2, matrices_B_rank2)
        saveToFile(filename_rank3, matrices_B_rank3)

        return

    else:
        return matrices_B_rank1, matrices_B_rank2, matrices_B_rank3

def buildCompatibilityMatrix(A1s, A2s, T, TIpos_max, TIneg_max, J_addable, remove_diag=False):

    compatPairs = np.zeros([len(A1s), len(A2s)], dtype='bool')

    t0 = time.time()
    for ii, a1 in enumerate(A1s):

        if (10*ii) // len(A1s) != (10*(ii+1)) // len(A1s):
            print('{} / {} = {:.2f} %\t{:.2} min'.format(ii, len(A1s), (100*ii)/len(A1s), (time.time() - t0)/60))


        a1 = A1s[ii]

        whr_pos_2 = np.where(np.max(a1[0] + A2s[:, 0] - TIpos_max, axis=1) <= 0)[0]
        whr_neg_2 = np.where(np.max(a1[1] + A2s[whr_pos_2, 1] - TIneg_max, axis=1) <= 0)[0]
        whr_2 = whr_pos_2[whr_neg_2]

        if remove_diag:
            whr_2 = whr_2[whr_2 != ii]

        improved = [satisfiesImprovedTIbounds(a1[0] + a2[0], a1[1] + a2[1],
                                              TIpos_max, TIneg_max, T, J_addable=J_addable)
                    for a2 in A2s[whr_2]]

        whr_2 = whr_2[improved]

        M12s = [IRREF3(a1[2], a2[2]) for a2 in A2s[whr_2]]
        cone = np.where([intersectsModulusCone(M) for M in M12s])[0]
        whr_2 = whr_2[cone]
        compatPairs[ii, whr_2] = True

    return compatPairs

def findLinearCombination_2(a1, a2, T):
    """Determine if all tadpole bounds can be satisfied for given stacks.
    a1, a2 are the two branes whose stacks must be nonempty.

    """

    J1 = np.where(a1[1] > 0)[0][0]
    J2 = np.where(a2[1] > 0)[0][0]
    Jnot12 = np.setdiff1d(range(4), [J1, J2])

    N1_max = min(T // a1[0, Jnot12])
    for N1 in range(1, N1_max + 1):
        N2_max = min((T - N1*a1[0, Jnot12]) // a2[0, Jnot12])
        for N2 in range(1, N2_max + 1):
            if max(N1*(a1[0] - a1[1]) + N2*(a2[0] - a2[1])) <= T:
                return True, N1*a1[1] + N2*a2[1]

    return False, [0, 0, 0, 0]

def findLinearCombination_3(a1, a2, a3, T):
    """Determine if all tadpole bounds can be satisfied for given stacks.
    a1, a2, a3 are the three branes whose stacks must be nonempty.

    """

    J1 = np.where(a1[1] > 0)[0][0]
    J2 = np.where(a2[1] > 0)[0][0]
    J3 = np.where(a3[1] > 0)[0][0]
    Jnot123 = np.setdiff1d(range(4), [J1, J2, J3])

    N1_max = min(T // a1[0, Jnot123])
    for N1 in range(1, N1_max + 1):
        N2_max = min((T - N1*a1[0, Jnot123]) // a2[0, Jnot123])
        for N2 in range(1, N2_max + 1):
            N3_max = min((T - N1*a1[0, Jnot123] - N2*a2[0, Jnot123]) // a3[0, Jnot123])
            for N3 in range(1, N3_max + 1):
                if max(N1*(a1[0] - a1[1]) + N2*(a2[0] - a2[1]) + N3*(a3[0] - a3[1])) <= T:
                    return True, N1*a1[1] + N2*a2[1] + N3*a3[1]

    return False, [0, 0, 0, 0]

def findLinearCombination_2_more(a1, a2, a3s, T):
    """Determine if all tadpole bounds can be satisfied for given stacks.
    a1, a2 are the two branes whose stacks must be nonempty.
    a3s gives a list of branes, not all of whose stacks can be empty.

    """

    possible = False


    J1 = np.where(a1[1] > 0)[0][0]
    J2 = np.where(a2[1] > 0)[0][0]
    J3s = [np.where(a3[1] > 0)[0][0] for a3 in a3s]

    Js = [J1, J2, *J3s]
    Js_notPresent = np.setdiff1d(range(4), Js)

    N1_max = max((T - a2[0, Js_notPresent] - np.min(a3s[:, 0, Js_notPresent], axis=0)) // a1[0, Js_notPresent])

    G_12_only = np.empty([0, 2, 4], dtype='int')
    for N1 in range(1, N1_max + 1):
        N2_max = max((T - N1*a1[0, Js_notPresent] - np.min(a3s[:, 0, Js_notPresent], axis=0)) // a2[0, Js_notPresent])

        for N2 in range(1, N2_max + 1):
            TIpos = N1*a1[0] + N2*a2[0]
            TIneg = N1*a1[1] + N2*a2[1]

            if max(TIpos[Js_notPresent] - TIneg[Js_notPresent]) <= T:
                G_12_only = np.append(G_12_only, [[TIpos, TIneg]], axis=0)

    G_123 = np.empty([0, 2, 4], dtype='int')

    for a3 in a3s:
        X3Ipos = a3[0]
        X3Ineg = a3[1]
        X3I = X3Ipos - X3Ineg

        toAdd = np.empty([0, 2, 4], dtype='int')

        for G in np.concatenate([G_12_only, G_123]):
            N3_max = min((T - (G[0, Js_notPresent] - G[1, Js_notPresent])) // X3I[Js_notPresent])

            for N3 in range(1, N3_max + 1):
                toAdd = np.append(toAdd, [[N3*X3Ipos + G[0], N3*X3Ineg + G[1]]], axis=0)

        G_123 = np.append(G_123, toAdd, axis=0)

    whr = np.where(np.max(G_123[:, 0] - G_123[:, 1], axis=1) <= T)[0]
    G_123 = G_123[whr]

    if len(G_123) > 0:
        TIneg_max = np.max(G_123[:, 1], axis=0)
        return True, TIneg_max
    else:
        return False, [0, 0, 0, 0]

def findLinearCombination_3_more(a1, a2, a3, a4s, T):
    """Determine if all tadpole bounds can be satisfied for given stacks.
    a1, a2, a3 are the three branes whose stacks must be nonempty.
    a4s gives a list of branes, not all of whose stacks can be empty.

    """

    possible = False


    J1 = np.where(a1[1] > 0)[0][0]
    J2 = np.where(a2[1] > 0)[0][0]
    J3 = np.where(a3[1] > 0)[0][0]
    J4s = [np.where(a4[1] > 0)[0][0] for a4 in a4s]

    Js = [J1, J2, J3, *J4s]
    Js_notPresent = np.setdiff1d(range(4), Js)

    N1_max = max((T - a2[0, Js_notPresent] - a3[0, Js_notPresent] - np.min(a4s[:, 0, Js_notPresent], axis=0)) // a1[0, Js_notPresent])

    G_123_only = np.empty([0, 2, 4], dtype='int')
    for N1 in range(1, N1_max + 1):
        N2_max = max((T - N1*a1[0, Js_notPresent] - a3[0, Js_notPresent] - np.min(a4s[:, 0, Js_notPresent], axis=0)) // a2[0, Js_notPresent])

        for N2 in range(1, N2_max + 1):
            N3_max = max((T - N1*a1[0, Js_notPresent] - N2*a2[0, Js_notPresent] - np.min(a4s[:, 0, Js_notPresent], axis=0)) // a3[0, Js_notPresent])

            for N3 in range(1, N3_max + 1):

                TIpos = N1*a1[0] + N2*a2[0] + N3*a3[0]
                TIneg = N1*a1[1] + N2*a2[1] + N3*a3[1]

                if max(TIpos[Js_notPresent] - TIneg[Js_notPresent]) <= T:
                    G_123_only = np.append(G_123_only, [[TIpos, TIneg]], axis=0)

    G_1234 = np.empty([0, 2, 4], dtype='int')

    for a4 in a4s:
        X4Ipos = a4[0]
        X4Ineg = a4[1]
        X4I = X4Ipos - X4Ineg

        toAdd = np.empty([0, 2, 4], dtype='int')

        for G in np.concatenate([G_123_only, G_1234]):
            N4_max = min((T - (G[0, Js_notPresent] - G[1, Js_notPresent])) // X4I[Js_notPresent])

            for N4 in range(1, N4_max + 1):
                toAdd = np.append(toAdd, [[N4*X4Ipos + G[0], N4*X4Ineg + G[1]]], axis=0)

        G_1234 = np.append(G_1234, toAdd, axis=0)

    whr = np.where(np.max(G_1234[:, 0] - G_1234[:, 1], axis=1) <= T)[0]
    G_1234 = G_1234[whr]

    if len(G_1234) > 0:
        TIneg_max = np.max(G_1234[:, 1], axis=0)
        return True, TIneg_max
    else:
        return False, [0, 0, 0, 0]

def combineRepeats(matrices):

    if len(matrices) == 0:
        return matrices

    Ms = [matrix.M for matrix in matrices]

    Ms_unique, inverse, counts = np.unique(Ms, axis=0, return_inverse=True, return_counts=True)

    whr_repeats = np.where(counts > 1)[0]

    toDelete = np.empty([0], dtype='int')

    for whr in whr_repeats:
        inds = np.where(inverse == whr)[0]
        toDelete = np.append(toDelete, inds[1:])

        matrices[inds[0]].combineWith(matrices[inds[1:]])

    return np.delete(matrices, toDelete)

def improvedTI(Tpos, Tneg, T, J_addable=[0,1,2,3]):
    """Augment TI+ and TI- by adding minimal contributions required to bring tadpoles below T.

    This relies on the tadpoles for A[J] branes satisfying XI > |XJ| for I<J when 0 < U1 ≤ U2 ≤ U3.

    Parameters
    ----------
    Tpos : ndarray
        Integer array of four TI+ values.
    Tneg : ndarray
        Integer array of four TI- values.
    T : int
        Tadpole (positive integer).
    J_addable : array_like, optional
        Types of A branes which might be added in the future.

    Returns
    -------
    Tpos_improved, Tneg_improved : ndarrays
        Integer arrays of four TI+ and four TI- values which any consistent configuration must
        match or surpass given the current TI+, TI- values.
    """

    Tpos_improved = Tpos.copy()
    Tneg_improved = Tneg.copy()
    Tpos_imp_last = Tpos_improved.copy()
    Tneg_imp_last = Tneg_improved.copy()

    # Using the bounds XI > |XJ|, each type of A brane may be added at most once
    # since the negative tadpoles may be larger than these bounds suggest.
    used = [False, False, False, False]

    # Go until no further changes (at most four times through the loop)
    fixedPoint = False
    while not fixedPoint:

        # For each A brane type, see if the tadpole is oversaturated. If so, at least 'surplus'
        # of this type of brane must eventually be added in order to get a consistent configuration.

        if 0 in J_addable and not used[0]:
            surplus0 = Tpos_improved[0] - Tneg_improved[0] - T
            if surplus0 > 0:
                used[0] = True

                Tpos_improved[1] += 1
                Tpos_improved[2] += 1
                Tpos_improved[3] += 1

                Tneg_improved[0] += surplus0

        if 1 in J_addable and not used[1]:
            surplus1 = Tpos_improved[1] - Tneg_improved[1] - T
            if surplus1 > 0:
                used[1] = True

                Tpos_improved[0] += 1 + surplus1
                Tpos_improved[2] += 1
                Tpos_improved[3] += 1

                Tneg_improved[1] += surplus1

        if 2 in J_addable and not used[2]:
            surplus2 = Tpos_improved[2] - Tneg_improved[2] - T
            if surplus2 > 0:
                used[2] = True

                Tpos_improved[0] += 1 + surplus2
                Tpos_improved[1] += 1 + surplus2
                Tpos_improved[3] += 1

                Tneg_improved[2] += surplus2

        if 3 in J_addable and not used[3]:
            surplus3 = Tpos_improved[3] - Tneg_improved[3] - T
            if surplus3 > 0:
                used[3] = True

                Tpos_improved[0] += 1 + surplus3
                Tpos_improved[1] += 1 + surplus3
                Tpos_improved[2] += 1 + surplus3

                Tneg_improved[3] += surplus3

        if np.array_equal(Tpos_improved, Tpos_imp_last):
            fixedPoint = True
        else:
            Tpos_imp_last = Tpos_improved.copy()
            Tneg_imp_last = Tneg_improved.copy()

    return Tpos_improved, Tneg_improved

def satisfiesImprovedTIbounds(TIpos, TIneg, TIpos_max, TIneg_max, T, J_addable=[0,1,2,3]):
    """Check that the improved TI+,TI- satisfy the tadpole bounds."""

    TIpos_improved, TIneg_improved = improvedTI(TIpos, TIneg, T, J_addable=J_addable)

    return max(TIpos_improved - TIpos_max) <= 0 and max(TIneg_improved - TIneg_max) <= 0

def IRREF3(M1, M2=None):
    """Compute (I)nteger (R)educed (R)ow-(E)chelon (F)orm of 1-3 SUSY-Y constraints.

    Parameters
    ----------
    M1 : ndarray
        kx4 integer array of k constraints on the moduli.
    M2 : ndarray, optional
        kx4 integer array of k constraints on the moduli.

    Returns
    -------
    M : ndarray
        3x4 integer array of constraints on the moduli in (I)nteger (R)educed (R)ow-(E)chelon (F)orm.
    """

    # Combine M1, M2
    if M2 is None and len(np.shape(M1)) == 1:
        M = np.pad([M1], ((0, 2), (0, 0)), mode='constant')
    elif M2 is None:
        M = M1.copy()
    else:
        M = np.block([[M1], [M2]])

    # Pad with zeros to atleast three rows
    if len(M) < 3:
        M = np.pad(M, ((0, 3-len(M)), (0, 0)), mode='constant')


    # Perform Gaussian elimination avoiding floating-point arithmetic

    pivotRow = 0
    pivotCol = 0
    while pivotCol < 4:

        for row in range(pivotRow + 1, len(M)):
            if M[pivotRow, pivotCol] == 0 and M[row, pivotCol] != 0:
                M[[pivotRow, row]] = M[[row, pivotRow]]
                break

        if M[pivotRow, pivotCol] != 0:
            for row in range(len(M)):
                if row == pivotRow:
                    continue
                if M[row, pivotCol] != 0:
                    lcm = np.lcm(M[pivotRow, pivotCol], M[row, pivotCol])
                    M[pivotRow] = M[pivotRow] * lcm // M[pivotRow, pivotCol]
                    M[row] = M[row] * lcm // M[row, pivotCol]

                    M[row] -= M[pivotRow]

                    M[pivotRow] = M[pivotRow] // np.gcd.reduce(M[pivotRow])
                    gcd = np.gcd.reduce(M[row])
                    if gcd != 0:
                        M[row] = M[row] // gcd

            pivotRow += 1
        pivotCol += 1

    # Restrict to first three rows
    M = M[:3]

    # Bring each row to reduced form (gcd=1 and first nonzero entry positive)
    for row in range(3):
        whr = np.where(M[row] != 0)[0]
        if len(whr) > 0:
            M[row] = np.sign(M[row, whr[0]]) * M[row] // np.gcd.reduce(M[row])

    return M

def intersectsModulusCone(M):
    """Determine if the null-space of M intersects the region where 0 < U0 ≤ U1 ≤ U2 ≤ U3.

    The constraints are imposed on the moduli as [M].[1/UI] = 0. Working in a normalization
    where U0 = 1, M describes an affine subspace in the 3D space of u1 = 1/U1, u2 = 1/U2,
    and u3 = 1/U3. The region of interest is then the tetrahedron
        [u1, u2, u3] ∈ ConvexHull{[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]}
    The faces ui = 0 should be discarded since the moduli should be finite.
    There are several cases to check based on rank(M):
     - If M describes a plane, one or more vertex must lie on each side of the plane (inclusive).
     - If M describes a line, there are several subcases.
     - If M describes a point, the inequalities are checked directly.

    Parameters
    ----------
    M : ndarray [int]
        1x4 or 3x4 integer array of constraints on the moduli in integer reduced row-echelon form.

    Returns
    -------
    bool
        Boolean indicating whether the null-space of M intersects the desired region.
    """

    if np.array_equal(M, np.zeros([3, 4])):
        return True

    # If M = [a, b, c, d] is a single constraint, augment with zeros
    if len(np.shape(M)) == 1:
        M = np.array([M, [0, 0, 0, 0], [0, 0, 0, 0]])

    # Vertices of tetrahedron
    v1 = [1, 0, 0, 0]
    v2 = [1, 1, 0, 0]
    v3 = [1, 1, 1, 0]
    v4 = [1, 1, 1, 1]
    vertices = np.array([v1, v2, v3, v4]).T

    # Check that the hyperplane described by each row intersects the region of interest.
    # A row m of M is a linear map which is positive on one side of the plane mv=0 and negative
    # on the other. If m applied to all four vertices gives the same sign then the tetrahedron
    # lies entirely on one side of the plane described by m.
    Mv = np.matmul(M, vertices)

    # Each row must contain both nonnegative and nonpositive values
    for rr in range(3):
        row = Mv[rr]

        if np.array_equal(row, [0, 0, 0, 0]):
            continue

        minVal = min(row)
        if minVal > 0:
            return False

        if minVal == 0 and row[-1] > 0:
            return False

    rank = np.linalg.matrix_rank(M)

    if rank == 1:
        # The above check on 'Mv' is sufficient
        return True

    elif rank == 2:
        # While two planes may individually intersect the tetrahedron their intersection (a line)
        # may not. Since M is in IRREF we can break into relatively few subcases.
        #   M = [[a b c d]
        #        [0 e f g]
        #        [0 0 0 0]]

        a, b, c, d = M[0]
        e, f, g = M[1, 1:]

        if a > 0 and e > 0:
            # M = [[a 0 c d]
            #      [0 e f g]
            #      [0 0 0 0]]

            det = c*g - d*f

            if det != 0:
                # One can find moduli in terms of u1:
                # u0 = 1, u1, u2 = (-ag+de*u1)/(cg-df), u3 = (af-ce*u1)/(cg-df)
                # The bounds u0 ≥ u1 ≥ u2 ≥ u3 > 0 then give an allowed range for u1.

                lowerBound = 0
                upperBound = 1

                if (1 - d*e/det) > 0:
                    lowerBound = max(lowerBound, a*g/(d*e-det))
                elif (1 - d*e/det) < 0:
                    upperBound = min(upperBound, a*g/(d*e-det))
                elif (-a*g/det) > 0:
                    return False

                if e*(c+d)/det > 0:
                    lowerBound = max(lowerBound, a*(f+g) / ((c+d)*e))
                elif e*(c+d)/det < 0:
                    upperBound = min(upperBound, a*(f+g) / ((c+d)*e))
                elif a*(f+g)/det > 0:
                    return False

                if -c*e/det > 0:
                    lowerBound = max(lowerBound, a*f/(c*e))
                elif -c*e/det < 0:
                    upperBound = min(upperBound, a*f/(c*e))
                elif -a*f/det > 0:
                    return False

                return (lowerBound <= upperBound) and (upperBound > 0) \
                  and ((a*f - c*e*lowerBound)/det > 0 or (a*f - c*e*upperBound)/det > 0)

            elif c == 0:
                # M = [[a 0 0 d]
                #      [0 e 0 g]
                #      [0 0 0 0]]

                return (e+g <= 0) and (a+d <= 0) and (d*e <= a*g)

            elif d == 0:
                # M = [[a 0 c 0]
                #      [0 e f 0]
                #      [0 0 0 0]]

                return (e+f <= 0) and (a+c <= 0) and (c*e <= a*f)

            else:
                # M = [[a 0 c d]   <=>   [[a*g -d*e 0 0]
                #      [0 e f g]          [  0    e f g]
                #      [0 0 0 0]]         [  0    0 0 0]
                # The value of u1 is fixed to u1 = a*g/(d*e) * u0.

                if a*g/(d*e) > 1:
                    return False

                # Else write u3 = -(a/d)-(f/g)*u2 and find allowed range for u2
                lowerBound = 0
                upperBound = a*g/(d*e)

                if (1 + f/g) > 0:
                    lowerBound = max(lowerBound, -a / (c + d))
                elif (1 + f/g) < 0:
                    upperBound = min(upperBound, -a / (c + d))
                elif (-a/d) > 0:
                    return False

                if (-f/g) > 0:
                    lowerBound = max(lowerBound, -a/c)
                elif (-f/g) < 0:
                    upperBound = min(upperBound, -a/c)

                return (lowerBound <= upperBound) and (upperBound > 0) \
                  and (-(a/d)-(f/g)*lowerBound > 0 or -(a/d)-(f/g)*upperBound > 0)

        elif a > 0 and e == 0:
            # M = [[a b 0 d]
            #      [0 0 f g]
            #      [0 0 0 0]]
            # Find u2 = g*(a+b*u1)/(df) and u3 = -(a+b*u1)/d

            if d == 0:
                return (a+b <= 0) and (f+g <= 0)

            lowerBound = 0
            upperBound = 1

            if (1-b*g/(d*f)) > 0:
                lowerBound = max(lowerBound, a*g/(d*f - b*g))
            elif (1-b*g/(d*f)) < 0:
                upperBound = min(upperBound, a*g/(d*f - b*g))
            elif a*g/(d*f) > 0:
                return False

            if (-b/d) > 0:
                lowerBound = max(lowerBound, -a/b)
            elif (-b/d) < 0:
                upperBound = min(upperBound, -a/b)
            elif (a/d) > 0:
                return False

            return (lowerBound <= upperBound) and (upperBound > 0) \
              and (-(a+b*lowerBound)/d > 0 or -(a+b*upperBound)/d > 0)


        else:
            # M = [[0 b 0 d]
            #      [0 0 f g]
            #      [0 0 0 0]]
            # with b,f>0 and d,g<0. Simply check that U1 = (-b/d)*U3 ≤ (-f/g)*U3 = U2

            return (b*g >= d*f)

    elif rank == 3:
        # M = [[a 0 0 b]
        #      [0 c 0 d]
        #      [0 0 e f]]
        # with a,c,e>0 and b,d,f<0.

        # Get numerators and denominators of moduli U0, U1, U2 (U3=1)
        nums = [M[i, i] for i in range(3)]
        dens = -M[:, 3]

        # Check correct order
        U0lessthanU1 = (nums[0]*dens[1] <= dens[0]*nums[1])
        U1lessthanU2 = (nums[1]*dens[2] <= dens[1]*nums[2])
        U2lessthanU3 = (nums[2]*(1)     <= dens[2]*(1)    )

        return U0lessthanU1 and U1lessthanU2 and U2lessthanU3

def gaugeFixed(M):
    """For rank-1 and rank-2 matrices, check that fully gauged fixed."""

    for perm in permutations(range(4)):
        Mperm = IRREF3(M[:, perm])
        cone = intersectsModulusCone(Mperm)
        if cone:
            for w1, w2 in zip(M.ravel(), Mperm.ravel()):
                if w2 < w1:
                    break
                elif w2 > w1:
                    return False

    return True

def minModulusRatio(M):
    """Returns the smallest value of U3/U0 compatible with a constraint matrix."""

    rank = np.linalg.matrix_rank(M)

    if rank == 0:
        # No constraints imposed by M
        return 1

    elif rank == 3:
        # U3/U0 is completely fixed by M
        return -M[0, 3] /  M[0, 0]

    else:
        # M is rank 1 or rank 2. Use linear programming to minimize U3/U0 given constraints.
        #
        # Wlg, fix u0 = 1/U0 = 1 and define ui = 1/Ui = [1/U1, 1/U2, 1/U3].
        # Would like to minimize c.ui = -u3 = -1/U3
        # A_eq, b_eq implement the three equalities M.u = 0
        # A_ub, b_ub implement the three inequalities 1 = u0 ≥ u1 ≥ u2 ≥ u3

        c = [0, 0, -1]
        A_eq = M[:rank, 1:]
        b_eq = -M[:rank, 0]

        A_ub = [[1, 0, 0], [-1, 1, 0], [0, -1, 1]]
        b_ub = [1, 0, 0]

        soln = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=(0, 1), method='revised simplex')
        # soln = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=(0, 1))

        if soln.success:
            return -1/soln.fun
        else:
            print(M[:rank], '\n')
            print(soln, '\n\n')
            return 0

def myUnique(array):
    """Returns unique elements along axis=0."""

    if len(array) == 0:
        return array
    else:
        return np.unique(array, axis=0)


def combineTallies(array):
    """Combines redundant entries in array.

    The first row contains the values and all other row contain various corresponding tallies.
    """

    if len(array) == 0:
        return array

    array = np.array(array, dtype='object')

    unique, inverse = np.unique(array[0], return_inverse=True)
    tallies_new = np.zeros([len(array) - 1, len(unique)], dtype='object')

    for ii in range(len(unique)):
        subset = np.where(inverse == ii)[0]

        # This is not great: loops and int() are to avoid overflow issues
        for jj in range(1, len(array)):
            for kk in subset:
                tallies_new[jj-1, ii] += int(array[jj, kk])

    return np.array([unique, *tallies_new], dtype='object')


def saveToFile(filename, matrices):
    """Save array of matrix objects to file."""

    toSave = [mat.getSaveData() for mat in matrices]

    with open(filename, 'w') as f:
        for line in toSave:
            f.write(line)
            f.write('\n')

def loadFromFile(filename):
    """Load data from file and return array of matrix objects."""

    with open(filename, 'r') as f:
        lines = f.readlines()

    matrices = np.empty([0], dtype='object')

    for line in lines:
        newMatrix = braneMatrix.matrix()
        newMatrix.restoreFromSave(line)
        matrices = np.append(matrices, [newMatrix])

    return matrices
