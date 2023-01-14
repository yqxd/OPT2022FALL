from numpy import *
from numpy.random import *
from matplotlib.pyplot import *
import pandas as pd
import scipy.linalg as la
from copy import deepcopy
from datetime import *

eps = 1e-16


class MEP:
    def __init__(self, blk_mat, blk_vec):
        self.blk_mat = deepcopy(blk_mat)
        self.nrow = self.blk_mat.shape[0]
        self.ncol = self.blk_mat.shape[1]
        self.size = self.blk_mat.shape[2]
        self.gs = blk_vec[0]
        self.rs = blk_vec[1]
        self.ss = blk_vec[2]
        for idx in range(1, self.nrow):
            # self.blk_mat[1, 0] = self.blk_mat[0, 0] - matmul(self.gs[0], self.rs[0].T)
            # self.blk_mat[1, 2] = self.blk_mat[0, 2] - matmul(self.gs[0], self.ss[0].T)
            self.blk_mat[idx, 0] = self.blk_mat[idx-1, 0] - matmul(self.gs[idx-1], self.rs[idx-1].T)
            self.blk_mat[idx, idx + 1] = self.blk_mat[idx-1, idx + 1] - matmul(self.gs[idx-1], self.ss[idx-1].T)

    def Ti(self, lambdas, i):
        assert lambdas.__len__() == self.ncol - 1
        crt_sum = -deepcopy(self.blk_mat[i, 0])
        for index in range(1, self.ncol):
            crt_sum += self.blk_mat[i, index] * lambdas[index - 1]
        return crt_sum

    def Tix(self, lambdas, i, x):
        return matmul(self.Ti(lambdas, i), x)

    def fi(self, i, x):
        return (matmul(self.rs[i].T, x) / matmul(self.ss[i].T, x))[0][0]

    def E(self, lambdas, x):
        assert lambdas.__len__() == self.ncol - 1
        crt_sum = -deepcopy(self.blk_mat[0, 0])
        crt_sum += self.blk_mat[0, 1] * lambdas[0]
        for index in range(2, self.ncol):
            crt_sum += self.blk_mat[0, index] * self.fi(index-2, x)
        return la.norm(matmul(crt_sum, x), 2)



def matmul2(x, A, y):
    return matmul(matmul(x, A), y)[0][0]


def is_symmetric(S, m, n):
    S = S / mean(abs(S))
    S = S.reshape(*[n for _ in range(m)])
    S_ = deepcopy(S)
    perm = tuple([_ for _ in range(1, m)] + [0])
    for _ in range(m):
        S_ = S_.transpose(*perm)
        if mean(abs(S - S_)) > 1e-2:
            return False
    return True


def from_txt(M_path, V_path, blk_nrow, blk_ncol):
    assert blk_nrow == blk_ncol - 1
    with open(M_path, "r") as f:
        lmatrix = f.readlines()
    lmatrix = [_.strip().split(",") for _ in lmatrix]
    lmatrix = array(lmatrix, dtype=complex128)

    with open(V_path, "r") as f:
        lvector = f.readlines()
    lvector= [_.strip().split(",") for _ in lvector]
    lvector = array(lvector, dtype=complex128)

    blk_size = int(lmatrix.__len__() / blk_nrow)
    return MEP(array([[lmatrix[blk_size * i: blk_size * (i + 1), blk_size * j: blk_size * (j + 1)]
                       for j in range(blk_ncol)] for i in range(blk_nrow)]),
               array(
                   [
                       [lvector[:blk_size, [i]] for i in range(blk_ncol - 2)],
                       [lvector[blk_size:2*blk_size, [i]] for i in range(blk_ncol - 2)],
                       [lvector[2*blk_size:3*blk_size, [i]] for i in range(blk_ncol - 2)]
                   ]
               ))

def from_random(n, blk_nrow, blk_ncol):
    assert blk_nrow == blk_ncol - 1
    lmatrix = zeros((blk_nrow, blk_ncol, n, n)).astype(complex128)
    lmatrix[0, :, :, :] = 10 * randn(blk_ncol, n, n)
    for i in range(1, blk_nrow):
        lmatrix[i, :, :, :] = deepcopy(lmatrix[0, :, :, :])
    g = randn(blk_ncol - 2, n, 1).astype(complex128)
    r = randn(blk_ncol - 2, n, 1).astype(complex128)
    s = randn(blk_ncol - 2, n, 1).astype(complex128)
    lvector = array([g, r, s])
    return MEP(lmatrix, lvector)

def kron_n(blk_mat):
    blk_mat = deepcopy(blk_mat)
    assert blk_mat.shape[0] == blk_mat.shape[1]
    if blk_mat.shape[0] == 1:
        return blk_mat[0, 0]
    else:
        crt_sum = 0
        for j in range(blk_mat.shape[1]):
            c_matrix = concatenate((blk_mat[1:, :j], blk_mat[1:, j + 1:]), axis=1)
            kron_n_mat = kron_n(c_matrix)
            crt_sum += (-1) ** j * kron(blk_mat[0, j], kron_n_mat)
        return crt_sum


def deltai(mep, index):
    blk_mat = deepcopy(mep.blk_mat)
    blk_mat[:, index] = blk_mat[:, 0]
    return kron_n(blk_mat[:, 1:])


def eigop(mep):
    delta0 = deltai(mep, 0)
    delta1 = deltai(mep, 1)
    D, V = la.eig(delta1, delta0)
    n = mep.size
    m = mep.nrow
    symmind = []
    for index in range(n ** m):
        S = deepcopy(V[:, index])
        if is_symmetric(S, m, n):
            symmind.append(index)
    return V, D, array(symmind)


def s2m(s):
    s = s.replace("I", "j").strip().split("\n")
    s = [_.strip()[:-1].split("j") for _ in s]
    s = [[eval(j.strip() + "j") for j in i] for i in s]
    s = array(s, dtype=complex128)
    return s

def closest(list, element):
    return list[argmin((list - element).__abs__())]

def kronn(a, n):
    if n == 1:
        return a
    else:
        return kron(kronn(a, n-1), a)