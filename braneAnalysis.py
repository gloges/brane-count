import numpy as np
import braneCount
from zipfile import ZipFile
from tqdm.autonotebook import tqdm


# Partition numbers p(n) for n = 0,...,800
PARTITIONS = np.load('partitions.npy', allow_pickle=True)

def getSummaryData(T, bix2, files):

    CbraneTad = np.product([(1 + int(d)) for d in bix2])
    if T % CbraneTad == 0:
        numCs = T // CbraneTad
        rmincounts_noK = np.array([[0], [1], [1], [PARTITIONS[numCs]**4]], dtype='int')
        rmincounts     = np.array([[0], [1], [1], [PARTITIONS[numCs]**4]], dtype='int')
        rankcounts_noK = np.array([[4*T], [1], [PARTITIONS[numCs]**4]], dtype='int')
        rankcounts     = np.array([[4*T], [1], [PARTITIONS[numCs]**4]], dtype='int')
    else:
        rmincounts_noK = np.empty([4, 0], dtype='int')
        rmincounts     = np.empty([4, 0], dtype='int')
        rankcounts_noK = np.empty([3, 0], dtype='int')
        rankcounts     = np.empty([3, 0], dtype='int')


    for file in files:
        with open(file, 'r') as f:
            data = f.readlines()

            if len(data[0]) > 1:
                data_rminbins_noK   = [int(d) for d in data[0].split(' ')]
                data_rmintally1_noK = [int(d) for d in data[1].split(' ')]
                data_rmintally2_noK = [int(d) for d in data[2].split(' ')]
                data_rmintally3_noK = [int(d) for d in data[3].split(' ')]
            else:
                data_rminbins_noK   = np.empty([0], dtype='int')
                data_rmintally1_noK = np.empty([0], dtype='int')
                data_rmintally2_noK = np.empty([0], dtype='int')
                data_rmintally3_noK = np.empty([0], dtype='int')

            if len(data[4]) > 1:
                data_rminbins   = [int(d) for d in data[4].split(' ')]
                data_rmintally1 = [int(d) for d in data[5].split(' ')]
                data_rmintally2 = [int(d) for d in data[6].split(' ')]
                data_rmintally3 = [int(d) for d in data[7].split(' ')]
            else:
                data_rminbins   = np.empty([0], dtype='int')
                data_rmintally1 = np.empty([0], dtype='int')
                data_rmintally2 = np.empty([0], dtype='int')
                data_rmintally3 = np.empty([0], dtype='int')

            if len(data[8]) > 1:
                data_ranklist_noK   = [int(d) for d in data[8].split(' ')]
                data_ranktally1_noK = [int(d) for d in data[9].split(' ')]
                data_ranktally2_noK = [int(d) for d in data[10].split(' ')]
            else:
                data_ranklist_noK   = np.empty([0], dtype='int')
                data_ranktally1_noK = np.empty([0], dtype='int')
                data_ranktally2_noK = np.empty([0], dtype='int')

            if len(data[11]) > 1:
                data_ranklist   = [int(d) for d in data[11].split(' ')]
                data_ranktally1 = [int(d) for d in data[12].split(' ')]
                data_ranktally2 = [int(d) for d in data[13].split(' ')]
            else:
                data_ranklist   = np.empty([0], dtype='int')
                data_ranktally1 = np.empty([0], dtype='int')
                data_ranktally2 = np.empty([0], dtype='int')

            data_rmincounts_noK = np.array([data_rminbins_noK, data_rmintally1_noK, data_rmintally2_noK, data_rmintally3_noK])
            data_rmincounts     = np.array([data_rminbins,     data_rmintally1,     data_rmintally2,     data_rmintally3])
            data_rankcounts_noK = np.array([data_ranklist_noK, data_ranktally1_noK, data_ranktally2_noK])
            data_rankcounts     = np.array([data_ranklist,     data_ranktally1,     data_ranktally2])

            rmincounts_noK = np.append(rmincounts_noK, data_rmincounts_noK, axis=1)
            rmincounts     = np.append(rmincounts,     data_rmincounts,     axis=1)
            rankcounts_noK = np.append(rankcounts_noK, data_rankcounts_noK, axis=1)
            rankcounts     = np.append(rankcounts,     data_rankcounts,     axis=1)

    rmincounts_noK = braneCount.combineTallies(rmincounts_noK)
    rmincounts     = braneCount.combineTallies(rmincounts    )
    rankcounts_noK = braneCount.combineTallies(rankcounts_noK)
    rankcounts     = braneCount.combineTallies(rankcounts    )

    if len(rmincounts_noK[0]) > 0:
        toAdd = np.empty([4, 0], dtype='int')
        for ii in range(-1, int(max(rmincounts_noK[0])) + 2):
            if ii not in rmincounts_noK[0]:
                toAdd = np.append(toAdd, [[ii], [0], [0], [0]], axis=1)
        rmincounts_noK = np.append(rmincounts_noK, toAdd, axis=1)
        rmincounts_noK = rmincounts_noK[:, np.argsort(rmincounts_noK[0])]

    if len(rmincounts[0]) > 0:
        toAdd = np.empty([4, 0], dtype='int')
        for ii in range(-1, int(max(rmincounts[0])) + 2):
            if ii not in rmincounts[0]:
                toAdd = np.append(toAdd, [[ii], [0], [0], [0]], axis=1)
        rmincounts = np.append(rmincounts, toAdd, axis=1)
        rmincounts = rmincounts[:, np.argsort(rmincounts[0])]

    if len(rankcounts_noK[0]) > 0:
        toAdd = np.empty([3, 0], dtype='int')
        for ii in range(int(max(rankcounts_noK[0])) + 2):
            if ii not in rankcounts_noK[0]:
                toAdd = np.append(toAdd, [[ii], [0], [0]], axis=1)
        rankcounts_noK = np.append(rankcounts_noK, toAdd, axis=1)
        rankcounts_noK = rankcounts_noK[:, np.argsort(rankcounts_noK[0])]

    if len(rankcounts[0]) > 0:
        toAdd = np.empty([3, 0], dtype='int')
        for ii in range(int(max(rankcounts[0])) + 2):
            if ii not in rankcounts[0]:
                toAdd = np.append(toAdd, [[ii], [0], [0]], axis=1)
        rankcounts = np.append(rankcounts, toAdd, axis=1)
        rankcounts = rankcounts[:, np.argsort(rankcounts[0])]

    print('c=1, noK : \t{:,}\t{:,}'.format(int(sum(rmincounts_noK[2])), int(sum(rankcounts_noK[1]))))
    print('c=1,   K : \t{:,}\t{:,}'.format(int(sum(rmincounts[2])), int(sum(rankcounts[1]))))
    print('c=p, noK : \t{:,}\t{:,}'.format(int(sum(rmincounts_noK[3])), int(sum(rankcounts_noK[2]))))
    print('c=p,   K : \t{:,}\t{:,}'.format(int(sum(rmincounts[3])), int(sum(rankcounts[2]))))

    return rmincounts_noK, rmincounts, rankcounts_noK, rankcounts


def getData(T, bix2, Mrank, IDs=None):

    # Build filepath to zip archive
    folder = 'matrixdata_counted/T={}/2bi={}/'.format(T, bix2)
    zipfile = folder + 'T{}_2bi{}_counted_rank{}.zip'.format(T, bix2, Mrank)


    # Prepare arrays for data
    Ms = np.empty([0, 3, 4], dtype='int')
    rmin = np.empty([0], dtype='float')

    numSoln_noK = np.empty([0, 2], dtype='int')
    numSoln     = np.empty([0, 2], dtype='int')


    # Crack open the zip archive
    with ZipFile(zipfile, 'r') as zf:

        # Get/create list of files in zip to be analyzed
        if IDs is None:
            filenames = zf.namelist()
        else:
            filenames = ['counted_rank{}_{}'.format(Mrank, ID) for ID in IDs]

        # Loop through files to be analyzed
        for files in tqdm(filenames):
            with zf.open(files, 'r') as f:

                # Scan through lines in file
                for line in tqdm(f, total=100000):

                    # Parse data and save to arrays
                    linedata = parseLine(line)

                    Ms = np.append(Ms, [linedata[0]], axis=0)
                    rmin = np.append(rmin, [linedata[1]])

                    numSoln_noK = np.append(numSoln_noK, [linedata[2]], axis=0)
                    numSoln     = np.append(numSoln,     [linedata[5]], axis=0)

    return Ms, rmin, numSoln_noK, numSoln


def parseLine(line):

    # Remove newline character and split by spaces
    numbers = line[:-1].split(b' ')

    # The first 3x4=12 numbers comprise the matrix M
    M = numbers[:12]
    M = np.array([int(d) for d in M]).reshape(3, 4)

    rmin = float(numbers[12])

    numSoln_noK = [int(numbers[13]), int(numbers[14])]
    usedA_noK = int(numbers[15])
    usedB_noK = int(numbers[16])

    numSoln = [int(numbers[17]), int(numbers[18])]
    usedA = int(numbers[19])
    usedB = int(numbers[20])

    return M, rmin, \
            numSoln_noK, usedA_noK, usedB_noK, \
            numSoln, usedA, usedB
