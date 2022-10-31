"""Methods for generating type A and B branes."""


import numpy as np
import braneCount


def getXI(w, bix2):
    """Compute XI from winding numbers."""

    n = [w[0], w[2], w[4]]
    mhat = [w[1] + bix2[0]*(w[0] + w[1]),
            w[3] + bix2[1]*(w[2] + w[3]),
            w[5] + bix2[2]*(w[4] + w[5])]

    X0 = +   n[0] *    n[1] *    n[2]
    X1 = -   n[0] * mhat[1] * mhat[2]
    X2 = -mhat[0] *    n[1] * mhat[2]
    X3 = -mhat[0] * mhat[1] *    n[2]

    return np.array([X0, X1, X2, X3])

def getYI(w, bix2):
    """Compute YI from winding numbers."""

    n = [w[0], w[2], w[4]]
    mhat = [w[1] + bix2[0]*(w[0] + w[1]),
            w[3] + bix2[1]*(w[2] + w[3]),
            w[5] + bix2[2]*(w[4] + w[5])]

    Y0 = +mhat[0] * mhat[1] * mhat[2]
    Y1 = -mhat[0] *    n[1] *    n[2]
    Y2 = -   n[0] * mhat[1] *    n[2]
    Y3 = -   n[0] *    n[1] * mhat[2]

    return np.array([Y0, Y1, Y2, Y3])

def getXYI(w, bix2):
    """Compute XI and YI from winding numbers."""
    return [getXI(w, bix2), getYI(w, bix2)]

def typeAJ(J, XImax, bix2):
    """Construct type A[`J`] branes.

    * Tadpoles are bounded |XI| ≤ `XImax`.
    * `bix2` specifies which tori are tilted.
    * Only those A[`J`] branes which allow for 0 < U0 ≤ U1 ≤ U2 ≤ U3 are returned.

    Parameters
    ----------
    J : {0, 1, 2, 3}
        Type of A brane. `J` indicates which XI is negative.
    XImax : array_like
        Four (integer) bounds on tadpoles.
    bix2 : array_like
        Tori tilts. `bix2` should have three entries, each either 0 (untilted) or 1 (tilted).

    Returns
    -------
    XYaI : ndarray
        The kx3x4 integer array of XIpos, XIneg, YI for the constructed A[`J`] branes.
    """

    # Prepare array of winding numbers
    ws = np.empty([0, 6], dtype='int')

    # Loop through all six winding numbers, using tadpole bounds
    # and checking that the (n,m) pairs are coprime.

    # n1 is bounded through both X0 and X1
    n1max = min(XImax[0], XImax[1])
    for n1 in range(1, n1max + 1):

        # m1hat is bounded through both X2 and X3 -> translate to range for m1
        m1hatmax = min(XImax[2], XImax[3])
        m1min = int(np.ceil((1 - bix2[0]*n1) / (1 + bix2[0])))
        m1max = (m1hatmax - bix2[0]*n1) // (1 + bix2[0])
        for m1 in range(m1min, m1max + 1):

            # Check coprimality
            if np.gcd(n1, m1) != 1:
                continue

            m1hat = m1 + bix2[0]*(n1 + m1)

            # n2 is bounded through both X0 and X2
            n2max = min(XImax[0]//n1, XImax[2]//m1hat)
            for n2 in range(1, n2max + 1):

                # m2hat is bounded through both X1 and X3 -> translate to range for m2
                m2hatmax = min(XImax[1]//n1, XImax[3]//m1hat)
                m2min = int(np.ceil((1 - bix2[1]*n2) / (1 + bix2[1])))
                m2max = (m2hatmax - bix2[1]*n2) // (1 + bix2[1])
                for m2 in range(m2min, m2max + 1):

                    # Check coprimality
                    if np.gcd(n2, m2) != 1:
                        continue

                    m2hat = m2 + bix2[1]*(n2 + m2)

                    # n3 is bounded through both X0 and X3
                    n3max = min(XImax[0]//(n1*n2), XImax[3]//(m1hat*m2hat))
                    for n3 in range(1, n3max + 1):

                        # m3hat is bounded through both X1 and X2 -> translate to range for m3
                        m3hatmax = min(XImax[1]//(n1*m2hat), XImax[2]//(m1hat*n2))
                        m3min = int(np.ceil((1 - bix2[2]*n3) / (1 + bix2[2])))
                        m3max = (m3hatmax - bix2[2]*n3) // (1 + bix2[2])
                        for m3 in range(m3min, m3max + 1):

                            # Check coprimality
                            if np.gcd(n3, m3) == 1:
                                # Save to winding number array
                                ws = np.append(ws, [[n1, m1, n2, m2, n3, m3]], axis=0)

    # Assign signs to winding numbers to make type A[J]
    if J == 0:
        ws[:, 4] *= -1
        ws[:, 5] -= bix2[2]*ws[:, 4]
    elif J == 1:
        ws[:, 3] = -ws[:, 3] - bix2[1]*ws[:, 2]
    elif J == 2:
        ws[:, 1] = -ws[:, 1] - bix2[0]*ws[:, 0]

    ws[:, 5] = -ws[:, 5] - bix2[2]*ws[:, 4]

    # Compute all XI, YI
    XYaI = np.array([getXYI(w, bix2) for w in ws])

    # Extract XIpos, XIneg, YI
    if len(XYaI) == 0:
        AJs = np.empty([0, 3, 4], dtype='int')
    else:
        AJs = np.zeros([len(XYaI), 3, 4], dtype='int')
        AJs[:, 0] = (abs(XYaI[:, 0]) + XYaI[:, 0]) / 2
        AJs[:, 1] = (abs(XYaI[:, 0]) - XYaI[:, 0]) / 2
        AJs[:, 2] = XYaI[:, 1]

    # Restrict to those which allow for moduli in the correct order
    Ms = np.array([braneCount.IRREF3(YI) for YI in AJs[:, 2]])
    cone = [braneCount.intersectsModulusCone(M) for M in Ms]
    AJs = AJs[cone]

    return AJs

def typeBIJ(J1, J2, XImax, bix2):
    """Construct type B[`J1`,`J2`] branes.

    * Tadpoles are bounded |XI| ≤ `XImax`.
    * `bix2` specifies which tori are tilted.
    * Only those B[`J1`,`J2`] branes which allow for 0 < U0 ≤ U1 ≤ U2 ≤ U3 are returned.

    Parameters
    ----------
    J1, J2 : {0, 1, 2, 3}
        `J1`, `J2` with `J1` < `J2` indicate which XI are nonzero.
    XImax : array_like
        Two (integer) bounds on the nonzero tadpoles.
    bix2 : array_like
        Tori tilts. `bix2` should have three entries, each either 0 (untilted) or 1 (tilted).

    Returns
    -------
    XYaI: ndarray
        The kx2x4 integer array of XI, YI for the constructed B[`J1`,`J2`] branes.
    """

    # Check J1, J2
    if J1 >= J2:
        print('J1 >= J2')
        return

    # Prepare array of winding numbers
    ws = np.empty([0, 6], dtype='int')

    # Loop through four nontrivial winding numbers, using tadpole bounds
    # and checking that the (n,m) pairs are coprime.
    if J1 == 0 and J2 == 1:
        # n1, m1 are fixed to be either (1, 0) or (2, -1)
        n1 = 1 + bix2[0]
        m1 = -bix2[0]

        # n2 is bounded through X0
        n2max = XImax[0]//n1
        for n2 in range(1, n2max + 1):

            # m2hat is bounded through X1 -> translate to range for m2
            m2hatmax = XImax[1]//n1
            m2min = int(np.ceil((1 - bix2[1]*n2) / (1 + bix2[1])))
            m2max = (m2hatmax - bix2[1]*n2) // (1 + bix2[1])
            for m2 in range(m2min, m2max + 1):

                # Check coprimality
                if np.gcd(n2, m2) != 1:
                    continue

                m2hat = m2 + bix2[1]*(n2+m2)

                # n3 is bounded through X0
                n3max = XImax[0]//(n1*n2)
                for n3 in range(1, n3max + 1):

                    # m3hat is bounded through X1 -> translate to range for m3
                    m3hatmax = XImax[1]//(n1*m2hat)
                    m3min = int(np.ceil((1 - bix2[2]*n3) / (1 + bix2[2])))
                    m3max = (m3hatmax - bix2[2]*n3) // (1 + bix2[2])
                    for m3 in range(m3min, m3max + 1):

                        # Check coprimality
                        if np.gcd(n3, m3) == 1:
                            # Save to winding number array (flipping sign of m2hat)
                            ws = np.append(ws, [[n1, m1, n2, -m2-bix2[1]*n2, n3, m3]], axis=0)

    elif J1 == 0 and J2 == 2:
        # n2, m2 are fixed to be either (1, 0) or (2, -1)
        n2 = 1 + bix2[1]
        m2 = -bix2[1]

        # n1 is bounded through X0
        n1max = XImax[0]//n2
        for n1 in range(1, n1max + 1):

            # m1hat is bounded through X2 -> translate to range for m1
            m1hatmax = XImax[1]//n2
            m1min = int(np.ceil((1 - bix2[0]*n1) / (1 + bix2[0])))
            m1max = (m1hatmax - bix2[0]*n1) // (1 + bix2[0])
            for m1 in range(m1min, m1max + 1):

                # Check coprimality
                if np.gcd(n1, m1) != 1:
                    continue

                m1hat = m1 + bix2[0]*(n1+m1)

                # n3 is bounded through X0
                n3max = XImax[0]//(n1*n2)
                for n3 in range(1, n3max + 1):

                    # m3hat is bounded through X2 -> translate to range for m3
                    m3hatmax = XImax[1]//(m1hat*n2)
                    m3min = int(np.ceil((1 - bix2[2]*n3) / (1 + bix2[2])))
                    m3max = (m3hatmax - bix2[2]*n3) // (1 + bix2[2])
                    for m3 in range(m3min, m3max + 1):

                        # Check coprimality
                        if np.gcd(n3, m3) == 1:
                            # Save to winding number array (flipping sign of m1hat)
                            ws = np.append(ws, [[n1, -m1-bix2[0]*n1, n2, m2, n3, m3]], axis=0)

    elif J1 == 0 and J2 == 3:
        # n3, m3 are fixed to be either (1, 0) or (2, -1)
        n3 = 1 + bix2[2]
        m3 = -bix2[2]

        # n1 is bounded through X0
        n1max = XImax[0]//n3
        for n1 in range(1, n1max + 1):

            # m1hat is bounded through X3 -> translate to range for m1
            m1hatmax = XImax[1]//n3
            m1min = int(np.ceil((1 - bix2[0]*n1) / (1 + bix2[0])))
            m1max = (m1hatmax - bix2[0]*n1) // (1 + bix2[0])
            for m1 in range(m1min, m1max + 1):

                # Check coprimality
                if np.gcd(n1, m1) != 1:
                    continue

                m1hat = m1 + bix2[0]*(n1+m1)

                # n2 is bounded through X0
                n2max = XImax[0]//(n1*n3)
                for n2 in range(1, n2max + 1):

                    # m2hat is bounded through X3 -> translate to range for m2
                    m2hatmax = XImax[1]//(m1hat*n3)
                    m2min = int(np.ceil((1 - bix2[1]*n2) / (1 + bix2[1])))
                    m2max = (m2hatmax - bix2[1]*n2) // (1 + bix2[1])
                    for m2 in range(m2min, m2max + 1):

                        # Check coprimality
                        if np.gcd(n2, m2) == 1:
                            # Save to winding number array (flipping sign of m1hat)
                            ws = np.append(ws, [[n1, -m1-bix2[0]*n1, n2, m2, n3, m3]], axis=0)

    elif J1 == 1 and J2 == 2:
        # n3, m3 are fixed to be (0, 1)
        n3, m3 = 0, 1
        m3hat = 1 + bix2[2]

        # n1 is bounded through X1
        n1max = XImax[0]//m3hat
        for n1 in range(1, n1max + 1):

            # m1hat is bounded through X2 -> translate to range for m1
            m1hatmax = XImax[1]//m3hat
            m1min = int(np.ceil((1 - bix2[0]*n1) / (1 + bix2[0])))
            m1max = (m1hatmax - bix2[0]*n1) // (1 + bix2[0])
            for m1 in range(m1min, m1max + 1):

                # Check coprimality
                if np.gcd(n1, m1) != 1:
                    continue

                m1hat = m1 + bix2[0]*(n1+m1)

                # n2 is bounded through X2
                n2max = XImax[1]//(m1hat*m3hat)
                for n2 in range(1, n2max + 1):

                    # m2hat is bounded through X1 -> translate to range for m2
                    m2hatmax = XImax[0]//(n1*m3hat)
                    m2min = int(np.ceil((1 - bix2[1]*n2) / (1 + bix2[1])))
                    m2max = (m2hatmax - bix2[1]*n2) // (1 + bix2[1])
                    for m2 in range(m2min, m2max + 1):

                        # Check coprimality
                        if np.gcd(n2, m2) == 1:
                            # Save to winding number array (flipping signs of m1hat, m2hat)
                            ws = np.append(ws, [[n1, -m1-bix2[0]*n1, n2, -m2-bix2[1]*n2, n3, m3]], axis=0)

    elif J1 == 1 and J2 == 3:
        # n2, m2 are fixed to be (0, 1)
        n2, m2 = 0, 1
        m2hat = 1 + bix2[1]

        # n1 is bounded through X1
        n1max = XImax[0]//m2hat
        for n1 in range(1, n1max + 1):

            # m1hat is bounded through X3 -> translate to range for m1
            m1hatmax = XImax[1]//m2hat
            m1min = int(np.ceil((1 - bix2[0]*n1) / (1 + bix2[0])))
            m1max = (m1hatmax - bix2[0]*n1) // (1 + bix2[0])
            for m1 in range(m1min, m1max + 1):

                # Check coprimality
                if np.gcd(n1, m1) != 1:
                    continue

                m1hat = m1 + bix2[0]*(n1+m1)

                # n3 is bounded through X3
                n3max = XImax[1]//(m1hat*m2hat)
                for n3 in range(1, n3max + 1):

                    # m3hat is bounded through X1 -> translate to range for m3
                    m3hatmax = XImax[0]//(n1*m2hat)
                    m3min = int(np.ceil((1 - bix2[2]*n3) / (1 + bix2[2])))
                    m3max = (m3hatmax - bix2[2]*n3) // (1 + bix2[2])
                    for m3 in range(m3min, m3max + 1):

                        # Check coprimality
                        if np.gcd(n3, m3) == 1:
                            # Save to winding number array (flipping signs of m1hat, m3hat)
                            ws = np.append(ws, [[n1, -m1-bix2[0]*n1, n2, m2, n3, -m3-bix2[2]*n3]], axis=0)

    elif J1 == 2 and J2 == 3:
        # n1, m1 are fixed to be (0, 1)
        n1, m1 = 0, 1
        m1hat = 1 + bix2[0]

        # n2 is bounded through X2
        n2max = XImax[0]//m1hat
        for n2 in range(1, n2max + 1):

            # m2hat is bounded through X3 -> translate to range for m2
            m2hatmax = XImax[1]//m1hat
            m2min = int(np.ceil((1 - bix2[1]*n2) / (1 + bix2[1])))
            m2max = (m2hatmax - bix2[1]*n2) // (1 + bix2[1])
            for m2 in range(m2min, m2max + 1):

                # Check coprimality
                if np.gcd(n2, m2) != 1:
                    continue

                m2hat = m2 + bix2[1]*(n2+m2)

                # n3 is bounded through X3
                n3max = XImax[1]//(m1hat*m2hat)
                for n3 in range(1, n3max + 1):

                    # m3hat is bounded through X2 -> translate to range for m3
                    m3hatmax = XImax[0]//(m1hat*n2)
                    m3min = int(np.ceil((1 - bix2[2]*n3) / (1 + bix2[2])))
                    m3max = (m3hatmax - bix2[2]*n3) // (1 + bix2[2])
                    for m3 in range(m3min, m3max + 1):

                        # Check coprimality
                        if np.gcd(n3, m3) == 1:
                            # Save to winding number array (flipping signs of m2hat, m3hat)
                            ws = np.append(ws, [[n1, m1, n2, -m2-bix2[1]*n2, n3, -m3-bix2[2]*n3]], axis=0)

    # Compute all XI,YI
    if len(ws) > 0:
        XYaI = np.array([getXYI(w, bix2) for w in ws])
    else:
        XYaI = np.empty([0, 2, 4], dtype='int')

    # Restrict to those which allow for moduli in the correct order
    Ms = np.array([braneCount.IRREF3(YI) for YI in XYaI[:, 1]])
    cone = [braneCount.intersectsModulusCone(M) for M in Ms]
    XYaI = XYaI[cone]

    return XYaI

def getCompatibleBIJ(J1, J2, M, XImax, bix2):
    """Construct type B[`J1`,`J2`] branes compatible with the constraint matrix `M`.

    * Tadpoles are bounded |XI| ≤ `XImax`.
    * `bix2` specifies which tori are tilted.

    Parameters
    ----------
    J1, J2 : {0, 1, 2, 3}
        `J1`, `J2` with `J1` < `J2` indicate which XI are nonzero.
    M : ndarray
        3x4 matrix in (I)nteger (R)educed (R)ow-(E)chelon (F)orm.
    XImax : array_like
        Two (integer) bounds on the nonzero tadpoles.
    bix2 : array_like
        Tori tilts. `bix2` should have three entries, each either 0 (untilted) or 1 (tilted).

    Returns
    -------
    XYaI: ndarray
        The kx2x4 integer array of XI, YI for the constructed B[`J1`,`J2`] branes.
    """

    # Check J1, J2
    if J1 >= J2:
        print('J1 >= J2')
        return

    # Get which YI are nonzero
    J3, J4 = np.setdiff1d(range(4), [J1, J2])

    # Determine the ratio of nonzero YI required by the matrix M
    Mrank = np.linalg.matrix_rank(M)

    if Mrank == 0:
        # No B[J1,J2] branes can be compatible with M
        return []

    elif Mrank == 1:
        # The B-brane's YI must be proportional to the only nontrivial row in M
        if M[0, J1] == 0 and M[0, J2] == 0 and M[0, J3] != 0 and M[0, J4] != 0:
            Yrequired = M[0, [J3, J4]]
        else:
            # No B[J1,J2] branes can be compatible with M
            return []

    elif Mrank == 2:
        # For type B[J1,J2] branes to be compatible with M there must be a linear combination
        # of the rows of M for which the nonzero entries are exactly in columns J3,J4

        # subM is the 2x2 'submatrix' whose rows must be linearly dependent
        # (so that there exists a linear combination which is all zeros)
        subM = M[:2, [J1, J2]]
        det = subM[0, 0] * subM[1, 1] - subM[0, 1] * subM[1, 0]

        if det != 0:
            # No B[J1, J2] branes can be compatible with M
            return []

        # Construct the linear combination of rows of M to which
        # the YI of B[J1,J2] branes must be proportional
        col = 0
        if subM[0, col] == 0 and subM[1, col] == 0:
            col = 1
        Yrequired = subM[1, col] * M[0] - subM[0, col] * M[1]

        # Extract the nonzero YI
        Yrequired = Yrequired[[J3, J4]]

    elif Mrank == 3:
        # The moduli are completely fixed by M and extracting the YI ratio is straightforward
        Yrequired = M[J3, [J3, 3]]
        if J4 != 3:
            Yrequired *= M[J4, [3, J4]]
            Yrequired *= [-1, 1]

    # Make sure reduced (coprime) and signs [+,-]
    Yrequired //= np.gcd(*Yrequired)
    Yrequired //= np.sign(Yrequired[0])

    # Prepare array of winding numbers
    ws = np.empty([0, 6], dtype='int')

    # Loop through four nontrivial winding numbers, using tadpole bounds,
    # required YI ratio, and checking that the (n,m) pairs are coprime.
    if J1 == 0 and J2 == 1:
        # n1, m1 are fixed to be either (1, 0) or (2, -1)
        n1 = 1 + bix2[0]
        m1 = -bix2[1]

        # n2 is bounded through X0
        n2max = XImax[0]//n1
        for n2 in range(1, n2max + 1):

            # m2hat is bounded through X1 -> translate to range for m2
            m2hatmax = XImax[1]//n1
            m2min = int(np.ceil((1 - bix2[1]*n2) / (1 + bix2[1])))
            m2max = (m2hatmax - bix2[1]*n2) // (1 + bix2[1])
            for m2 in range(m2min, m2max + 1):

                # Check coprimality
                if np.gcd(n2, m2) != 1:
                    continue

                m2hat = m2 + bix2[1]*(n2+m2)

                # m3hat will be determined through the YI ratio:
                #     Y2/Y3 = (m2hat*n3) / (n2*m3hat)
                # This means that n3 is bounded through both X0 and X1:
                #     X1 = -n1*m2hat*m3hat = -(n1 * m2hat^2 * n3*Y3) / (n2*Y2)
                # From n2*m3hat*Y2 = m2hat*n3*Y3 one finds Y2 | m2hat*n3 and Y2/gcd(Y2,m2hat) | n3
                n3max = min(XImax[0]//(n1*n2), -(XImax[1] * n2 * Yrequired[0])//(n1 * m2hat**2 * Yrequired[1]))
                n3incr = Yrequired[0] // np.gcd(Yrequired[0], m2hat)
                for n3 in range(n3incr, n3max + 1, n3incr):

                    # m3hat = (m2hat*n3*Y3) / (n2*Y2) should be integer
                    if (m2hat*n3*Yrequired[1]) % (n2*Yrequired[0]) != 0:
                        continue
                    m3hat = -(m2hat*n3*Yrequired[1]) // (n2*Yrequired[0])

                    # m3 = (m3hat - 2b3*n3) / (1 + 2b3) should be integer
                    if bix2[2] == 1 and (m3hat - n3) % 2 != 0:
                        continue
                    m3 = (m3hat - bix2[2]*n3) // (1 + bix2[2])

                    # Check coprimality
                    if np.gcd(n3, m3) == 1:
                        # Save to winding number array (flipping sign of m2hat)
                        ws = np.append(ws, [[n1, m1, n2, -m2-bix2[1]*n2, n3, m3]], axis=0)

    elif J1 == 0 and J2 == 2:
        # n2, m2 are fixed to be either (1, 0) or (2, -1)
        n2 = 1 + bix2[1]
        m2 = -bix2[1]

        # n1 is bounded through X0
        n1max = XImax[0]//n2
        for n1 in range(1, n1max + 1):

            # m1hat is bounded through X2 -> translate to range for m1
            m1hatmax = XImax[1]//n2
            m1min = int(np.ceil((1 - bix2[0]*n1) / (1 + bix2[0])))
            m1max = (m1hatmax - bix2[0]*n1) // (1 + bix2[0])
            for m1 in range(m1min, m1max + 1):

                # Check coprimality
                if np.gcd(n1, m1) != 1:
                    continue

                m1hat = m1 + bix2[0]*(n1+m1)

                # m3hat will be determined through the YI ratio:
                #     Y1/Y3 = (m1hat*n3) / (n1*m3hat)
                # This means that n3 is bounded through both X0 and X2:
                #     X2 = -m1hat*n2*m3hat = -(m1hat^2 *n2*n3*Y3) / (n1*Y1)
                # From n1*m3hat*Y1 = m1hat*n3*Y3 one finds Y1 | m1hat*n3 and Y1/gcd(Y1,m1hat) | n3
                n3max = min(XImax[0]//(n1*n2), -(XImax[1] * n1 * Yrequired[0])//(n2 * m1hat**2 * Yrequired[1]))
                n3incr = Yrequired[0] // np.gcd(Yrequired[0], m1hat)
                for n3 in range(n3incr, n3max + 1, n3incr):

                    # m3hat = (m1hat*n3*Y3) / (n1*Y1) should be integer
                    if (m1hat*n3*Yrequired[1]) % (n1*Yrequired[0]) != 0:
                        continue
                    m3hat = -(m1hat*n3*Yrequired[1]) // (n1*Yrequired[0])

                    # m3 = (m3hat - 2b3*n3) / (1 + 2b3) should be integer
                    if bix2[2] == 1 and (m3hat - n3) % 2 != 0:
                        continue
                    m3 = (m3hat - bix2[2]*n3) // (1 + bix2[2])

                    # Check coprimality
                    if np.gcd(n3, m3) == 1:
                        # Save to winding number array (flipping sign of m1hat)
                        ws = np.append(ws, [[n1, -m1-bix2[0]*n1, n2, m2, n3, m3]], axis=0)

    elif J1 == 0 and J2 == 3:
        # n3, m3 are fixed to be either (1, 0) or (2, -1)
        n3 = 1 + bix2[2]
        m3 = -bix2[2]

        # n1 is bounded through X0
        n1max = XImax[0]//n3
        for n1 in range(1, n1max + 1):

            # m1hat is bounded through X3 -> translate to range for m1
            m1hatmax = XImax[1]//n3
            m1min = int(np.ceil((1 - bix2[0]*n1) / (1 + bix2[0])))
            m1max = (m1hatmax - bix2[0]*n1) // (1 + bix2[0])
            for m1 in range(1, XImax[1] + 1):

                # Check coprimality
                if np.gcd(n1, m1) != 1:
                    continue

                m1hat = m1 + bix2[0]*(n1+m1)

                # m2hat will be determined through the YI ratio:
                #     Y1/Y2 = (m1hat*n2) / (n1*m2hat)
                # This means that n2 is bounded through both X0 and X3:
                #     X3 = -m1hat*m2hat*n3 = -(m1hat^2 * n2*n3*Y2) / (n1*Y1)
                # From n1*m2hat*Y1 = m1hat*n2*Y2 one finds Y1 | m1hat*n2 and Y1/gcd(Y1,m1hat) | n2
                n2max = min(XImax[0]//(n1*n3), -(XImax[1] * n1 * Yrequired[0])//(n3 * m1hat**2 * Yrequired[1]))
                n2incr = Yrequired[0] // np.gcd(Yrequired[0], m1hat)
                for n2 in range(n2incr, n2max + 1, n2incr):

                    # m2hat = (m1hat*n2*Y2) / (n1*Y1) should be integer
                    if (m1hat*n2*Yrequired[1]) % (n1*Yrequired[0]) != 0:
                        continue
                    m2hat = -(m1hat*n2*Yrequired[1]) // (n1*Yrequired[0])

                    # m2 = (m2hat - 2b2*n2) / (1 + 2b2) should be integer
                    if bix2[1] == 1 and (m2hat - n2) % 2 != 0:
                        continue
                    m2 = (m2hat - bix2[1]*n2) // (1 + bix2[1])

                    # Check coprimality
                    if np.gcd(n2, m2) == 1:
                        # Save to winding number array (flipping sign of m1hat)
                        ws = np.append(ws, [[n1, -m1-bix2[0]*n1, n2, m2, n3, m3]], axis=0)

    elif J1 == 1 and J2 == 2:
        # n3, m3 are fixed to be (0, 1)
        n3, m3 = 0, 1
        m3hat = 1 + bix2[2]

        # n1 is bounded through X1
        n1max = XImax[0]//m3hat
        for n1 in range(1, n1max + 1):

            # m1hat is bounded through X2 -> translate to range for m1
            m1hatmax = XImax[1]//m3hat
            m1min = int(np.ceil((1 - bix2[0]*n1) / (1 + bix2[0])))
            m1max = (m1hatmax - bix2[0]*n1) // (1 + bix2[0])
            for m1 in range(m1min, m1max + 1):

                # Check coprimality
                if np.gcd(n1, m1) != 1:
                    continue

                m1hat = m1 + bix2[0]*(n1+m1)

                # m2hat will be determined through the YI ratio:
                #     Y0/Y3 = -(m1hat*m2hat) / (n1*n2)
                # This means that n2 is bounded through both X1 and X2:
                #     X1 = -n1*m2hat*m3hat = -(n1^2 *n2*m3hat*Y0) / (m1hat*Y3)
                # From m1hat*m2hat*Y3 = n1*n2*Y0 one finds Y3 | n1*n2 and Y3/gcd(Y3,n1) | n2
                n2max = min(XImax[1]//(m1hat*m3hat), -(XImax[0] * m1hat * Yrequired[1])//(n1**2 * m3hat * Yrequired[0]))
                n2incr = -Yrequired[1] // np.gcd(Yrequired[1], n1)
                for n2 in range(n2incr, n2max + 1, n2incr):

                    # m2hat = -(n1*n2*Y0) / (m1hat*Y3) should be integer
                    if (n1*n2*Yrequired[0]) % (m1hat*Yrequired[1]) != 0:
                        continue
                    m2hat = -(n1*n2*Yrequired[0]) // (m1hat*Yrequired[1])

                    # m2 = (m2hat - 2b2*n2) / (1 + 2b2) should be integer
                    if bix2[1] == 1 and (m2hat - n2) % 2 != 0:
                        continue
                    m2 = (m2hat - bix2[1]*n2) // (1 + bix2[1])

                    # Check coprimality
                    if np.gcd(n2, m2) == 1:
                        # Save to winding number array (flipping signs of m1hat, m2hat)
                        ws = np.append(ws, [[n1, -m1-bix2[0]*n1, n2, -m2-bix2[1]*n2, n3, m3]], axis=0)

    elif J1 == 1 and J2 == 3:
        # n2, m3 are fixed to be (0, 1)
        n2, m2 = 0, 1
        m2hat = 1 + bix2[1]

        # n1 is bounded through X1
        n1max = XImax[0]//m2hat
        for n1 in range(1, n1max + 1):

            # m1hat is bounded through X3 -> translate to range for m1
            m1hatmax = XImax[1]//m2hat
            m1min = int(np.ceil((1 - bix2[0]*n1) / (1 + bix2[0])))
            m1max = (m1hatmax - bix2[0]*n1) // (1 + bix2[0])
            for m1 in range(m1min, m1max + 1):

                # Check coprimality
                if np.gcd(n1, m1) != 1:
                    continue

                m1hat = m1 + bix2[0]*(n1+m1)

                # m3hat will be determined through the YI ratio:
                #     Y0/Y2 = -(m1hat*m3hat) / (n1*n3)
                # This means that n3 is bounded through both X1 and X3:
                #     X1 = -n1*m2hat*m3hat = -(n1^2 *m2hat*n3*Y0) / (m1hat*Y2)
                # From m1hat*m3hat*Y2 = n1*n3*Y0 one finds Y2 | n1*n3 and Y2/gcd(Y2,n1) | n3
                n3max = min(XImax[1]//(m1hat*m2hat), -(XImax[0] * m1hat * Yrequired[1])//(n1**2 * m2hat * Yrequired[0]))
                n3incr = -Yrequired[1] // np.gcd(Yrequired[1], n1)
                for n3 in range(n3incr, n3max + 1, n3incr):

                    # m3hat = -(n1*n3*Y0) / (m1hat*Y2) should be integer
                    if (n1*n3*Yrequired[0]) % (m1hat*Yrequired[1]) != 0:
                        continue
                    m3hat = -(n1*n3*Yrequired[0]) // (m1hat*Yrequired[1])

                    # m3 = (m3hat - 2b3*n3) / (1 + 2b3) should be integer
                    if bix2[2] == 1 and (m3hat - n3) % 2 != 0:
                        continue
                    m3 = (m3hat - bix2[2]*n3) // (1 + bix2[2])

                    # Check coprimality
                    if np.gcd(n3, m3) == 1:
                        # Save to winding number array (flipping signs of m1hat, m3hat)
                        ws = np.append(ws, [[n1, -m1-bix2[0]*n1, n2, m2, n3, -m3-bix2[2]*n3]], axis=0)

    elif J1 == 2 and J2 == 3:
        # n1, m1 are fixed to be (0, 1)
        n1, m1 = 0, 1
        m1hat = 1 + bix2[0]

        # n2 is bounded through X2
        n2max = XImax[0]//m1hat
        for n2 in range(1, n2max + 1):

            # m2hat is bounded through X3 -> translate to range for m2
            m2hatmax = XImax[1]//m1hat
            m2min = int(np.ceil((1 - bix2[1]*n2) / (1 + bix2[1])))
            m2max = (m2hatmax - bix2[1]*n2) // (1 + bix2[1])
            for m2 in range(m2min, m2max + 1):

                # Check coprimality
                if np.gcd(n2, m2) != 1:
                    continue

                m2hat = m2 + bix2[1]*(n2+m2)

                # m3hat will be determined through the YI ratio:
                #     Y0/Y1 = -(m2hat*m3hat) / (n2*n3)
                # This means that n3 is bounded through both X2 and X3:
                #     X2 = -m1hat*n2*m3hat = (m1hat* n2^2 *n3*Y0)/(m2hat*Y1)
                # From m2hat*m3hat*Y1 = n2*n3*Y0 one finds Y1 | n2*n3 and Y1/gcd(Y1,n2) | n3
                n3max = min(XImax[1]//(m1hat*m2hat), -(XImax[0] * m2hat * Yrequired[1])//(n2**2 * m1hat * Yrequired[0]))
                n3incr = -Yrequired[1] // np.gcd(Yrequired[1], n2)
                for n3 in range(n3incr, n3max + 1, n3incr):

                    # m3hat = -(n2*n3*Y0) / (m2hat*Y1) should be integer
                    if (n2*n3*Yrequired[0]) % (m2hat*Yrequired[1]) != 0:
                        continue
                    m3hat = -(n2*n3*Yrequired[0]) // (m2hat*Yrequired[1])

                    # m3 = (m3hat - 2b3*n3) / (1 + 2b3) should be integer
                    if bix2[2] == 1 and (m3hat - n3) % 2 != 0:
                        continue
                    m3 = (m3hat - bix2[2]*n3) // (1 + bix2[2])

                    # Check coprimality
                    if np.gcd(n3, m3) == 1:
                        # Save to winding number array (flipping signs of m2hat, m3hat)
                        ws = np.append(ws, [[n1, m1, n2, -m2-bix2[1]*n2, n3, -m3-bix2[2]*n3]], axis=0)

    # Compute all XI,YI
    XYaI = np.array([getXYI(w, bix2) for w in ws])

    return XYaI
