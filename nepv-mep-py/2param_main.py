import matplotlib.pyplot as plt

from utils import *
from alg import *

PRE = "/home/zyxu/_projects_/python-project/"

# self-defined
n = 40 # 20
seed_num = 106 # 104 106
max_iter = 100 # 20 100
show_E = 1 # error
show_L = 1 # lambda
blk_nrow = 2
blk_ncol = 3 # blk_nrow + 1

seed(seed_num)
# data preparation
# mep = from_txt(PRE + "data/rnd100", PRE + "data/rnd100v", 2, 3)
mep = from_random(n, blk_nrow, blk_ncol)
V, D, symmind = eigop(mep)

## residual inverse algorithm
x0 = V[:, [2]][:mep.size, :] + 1e-2 * randn(mep.size, 1)
lambda0 = D[2] + 1e-2 * randn(1) + 1j * 1e-2 * randn(1)
lambdas0 = [lambda0] + [mep.fi(_, x0) for _ in range(blk_nrow - 1)]
vs = [randn(mep.size, 1)]
v = vs[0]
tol = 1e-16

# origin_resinv
# origin_resinv_rqi
# reduced_resinv_quasi
# reduced_resinv_quasi_rqi
# reduced_resinv
# reduced_resinv_rqi

x_origin_resinv, lambdas_origin_resinv, es_origin_resinv = origin_resinv(mep, lambdas0, v, x0, tol, max_iter)
x_origin_resinv_rqi, lambdas_origin_resinv_rqi, es_origin_resinv_rqi = origin_resinv_rqi(mep, lambdas0, v, x0, tol, max_iter)
x_reduced_resinv_quasi, lambdas_reduced_resinv_quasi, es_reduced_resinv_quasi = reduced_resinv_quasi(mep, lambdas0, v, x0, tol, max_iter)
x_reduced_resinv_quasi_rqi, lambdas_reduced_resinv_quasi_rqi, es_reduced_resinv_quasi_rqi = reduced_resinv_quasi_rqi(mep, lambdas0, v, x0, tol, max_iter)
x_reduced_resinv, lambdas_reduced_resinv, es_reduced_resinv = reduced_resinv(mep, lambdas0, v, x0, tol, max_iter)
x_reduced_resinv_rqi, lambdas_reduced_resinv_rqi, es_reduced_resinv_rqi = reduced_resinv_rqi(mep, lambdas0, v, x0, tol, max_iter)

if (show_E):
 e = abs(es_origin_resinv)
 l1, = semilogy(arange(max_iter + 1), e / e[0], marker="*", linestyle="-", color='dodgerblue')
 e = abs(es_origin_resinv_rqi)
 l2, = semilogy(arange(max_iter + 1), e / e[0], marker="*", linestyle="--", color = 'dodgerblue')
 e = abs(es_reduced_resinv_quasi)
 l3, = semilogy(arange(max_iter + 1), e / e[0], marker="3", linestyle="-", color='darkorange')
 e = abs(es_reduced_resinv_quasi_rqi)
 l4, = semilogy(arange(max_iter + 1), e / e[0], marker="3", linestyle="--", color = 'darkorange')
 e = abs(es_reduced_resinv)
 l5, = semilogy(arange(max_iter + 1), e / e[0], marker="+", linestyle="-", color='red')
 e = abs(es_reduced_resinv_rqi)
 l6, = semilogy(arange(max_iter + 1), e / e[0], marker="+", linestyle="--", color = 'red')

 legend([l1, l2, l3, l4, l5, l6],
        ["origin_resinv",
         "origin_resinv_rqi",
         "reduced_resinv_quasi",
         "reduced_resinv_quasi_rqi",
         "reduced_resinv",
         "reduced_resinv_rqi"])
 title("residual error $\Vert T(\lambda, x)x\Vert_2$ of NEPv")
 # plt.savefig(PRE + "figure/f" + str(seed_num) + ".pdf", transparent=True)
 show()
 # close()

if (show_L):
 e = abs(lambdas_reduced_resinv[:, 0] - closest(D, lambdas_reduced_resinv[-1, 0]))
 l1, = semilogy(arange(max_iter + 1), e / e[0])
 e = abs(lambdas_origin_resinv_rqi[:, 0] - closest(D, lambdas_origin_resinv_rqi[-1, 0]))
 l2, = semilogy(arange(max_iter + 1), e / e[0])
 e = abs(lambdas_reduced_resinv_quasi[:, 0] - closest(D, lambdas_reduced_resinv_quasi[-1, 0]))
 l3, = semilogy(arange(max_iter + 1), e / e[0])
 e = abs(lambdas_reduced_resinv_quasi_rqi[:, 0] - closest(D, lambdas_reduced_resinv_quasi_rqi[-1, 0]))
 l4, = semilogy(arange(max_iter + 1), e / e[0])
 e = abs(lambdas_reduced_resinv[:, 0] - closest(D, lambdas_reduced_resinv[-1, 0]))
 l5, = semilogy(arange(max_iter + 1), e / e[0])
 e = abs(lambdas_reduced_resinv_rqi[:, 0] - closest(D, lambdas_reduced_resinv_rqi[-1, 0]))
 l6, = semilogy(arange(max_iter + 1), e / e[0])

 legend([l1, l2, l3, l4, l5, l6],
        ["origin_resinv",
         "origin_resinv_rqi",
         "reduced_resinv_quasi",
         "reduced_resinv_quasi_rqi",
         "reduced_resinv",
         "reduced_resinv_rqi"])

 # plt.savefig(PRE + "figure/f" + str(seed_num) + ".pdf", transparent=True)
 title("error of lambda")
 show()
 # close()

## inverse iteration
