import numpy as np
import braneCount
import braneMatrix
import sys
import time


T = int(sys.argv[1])
bix2 = [int(d) for d in sys.argv[2]]
rank = int(sys.argv[3])
loadfile = sys.argv[4]
savefile = sys.argv[5]

print('\nT =', T)
print('2bi = {}{}{}'.format(*bix2))
print('rank =', rank)


# Load data from file
t0 = time.time()
with open(loadfile, 'r') as f:
    mat_data = f.readlines()
t1 = time.time()

print('\n{} lines loaded in {:.4f} s'.format(len(mat_data), t1 - t0))


# Arrays for summary data
rmincounts_noK = np.empty([4, 0], dtype='object')
rmincounts     = np.empty([4, 0], dtype='object')
rankcounts_noK = np.empty([3, 0], dtype='object')
rankcounts     = np.empty([3, 0], dtype='object')


# Loop through matrices counting solutions
t0 = time.time()
with open(savefile, 'w') as f:

    for ii, mat in enumerate(mat_data[:339]):

        matrix = braneMatrix.matrix()
        matrix.restoreFromSave(mat)
        matrix.countSolutions(T, bix2)

        f.write(matrix.getCountData())
        f.write('\n')

        rbinindex = int(12 * np.log10(matrix.rmin))

        if min(matrix.numSoln_noK) > 0:
            rmincounts_noK = np.append(rmincounts_noK, np.array([[rbinindex, 1, *matrix.numSoln_noK]], dtype='object').T, axis=1)

        if min(matrix.numSoln) > 0:
            rmincounts = np.append(rmincounts, np.array([[rbinindex, 1, *matrix.numSoln]], dtype='object').T, axis=1)

        if len(matrix.rnkABC_noK[0]) > 0:
            if min(matrix.rnkABC_noK[2]) < 0:
                print('Negative rank noK:', matrix.rnkABC_noK[:3])
        if len(matrix.rnkABC[0]) > 0:
            if min(matrix.rnkABC[2]) < 0:
                print('negative rank:', matrix.rnkABC[:3])

        rankcounts_noK = np.append(rankcounts_noK, np.array(matrix.rnkABC_noK[:3], dtype='object'), axis=1)
        rankcounts     = np.append(rankcounts,     np.array(matrix.rnkABC[:3], dtype='object'),     axis=1)

        rmincounts_noK = braneCount.combineTallies(rmincounts_noK)
        rmincounts     = braneCount.combineTallies(rmincounts    )
        rankcounts_noK = braneCount.combineTallies(rankcounts_noK)
        rankcounts     = braneCount.combineTallies(rankcounts    )

        if len(rankcounts_noK[0]) > 0:
            if min(rankcounts_noK[2]) < 0:
                print('Negative rankcount noK:', rankcounts_noK)
        if len(rankcounts[0]) > 0:
            if min(rankcounts[2]) < 0:
                print('Negative rankcount:', rankcounts)

t1 = time.time()


print('\nTotal time: {:.2f} s = {:.2f} min = {:.2f} hr'.format(t1 - t0, (t1 - t0)/60, (t1 - t0)/3600))
print('Average time: {:.4f} s\n'.format((t1 - t0) / len(mat_data)))



# Create and save summary data
summaryfile = 'T{}_2bi{}{}{}_'.format(T, *bix2) + savefile + '_summary.txt'
print('\nSaving summary data to...', summaryfile)

with open(summaryfile, 'w') as f:
    for data in [*rmincounts_noK, *rmincounts, *rankcounts_noK, *rankcounts]:
        f.write(' '.join([str(d) for d in data]))
        f.write('\n')
