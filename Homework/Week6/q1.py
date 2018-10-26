import numpy as np


# using alternating least squares method for collaborative filtering:
# http://www.quuxlabs.com/blog/2010/09/matrix-factorization-a-simple-tutorial-and-implementation-in-python/
def get_error(Q, X, Y, W):
    return np.sum((W * (Q - np.dot(X, Y)))**2)


def matrix_factorization(R, P, Q, K, steps=10000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i, :], Q[:, j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = np.dot(P, Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i, :], Q[:, j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))
        if e < 0.0001:
            break
    return P, Q.T


def initialiseVariables():
    R = np.array(
        [[0, 1, np.nan],
         [1, np.nan, 1],
         [np.nan, 1, 2]]
    )
    N = len(R)
    M = len(R[0])
    K = 1  # number of rows

    P = np.random.rand(N, K)
    Q = np.random.rand(M, K)
    return M, N, K, P, Q, R


def checkAgainst(_nR):
    desired_matrix = np.array(
        [[0, 1, np.nan],
         [1, np.nan, 1],
         [np.nan, 1, 2]]
    )
    for i in range(len(desired_matrix)):
        for j in range(len(desired_matrix[0])):
            # skip these numbers
            if i == 0 and j == 2 or i == 1 and j == 1 or i == 2 and j == 0:
                continue
            else:
                if round(_nR[i][j]) != desired_matrix[i][j]:
                    return False
    return True


if __name__ == '__main__':
    M, N, K, P, Q, R = initialiseVariables()

    nP, nQ = matrix_factorization(R, P, Q, K)
    nR = np.dot(nP, nQ.T)
    print(nR)
    print(nP)
    print(nQ)
    print(checkAgainst(nR))
