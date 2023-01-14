import matplotlib.pyplot as plt

from utils import *
from alg import *

PRE = "/home/zyxu/_projects_/python-project/"
SAVETO = "/home/zyxu/_projects_/latex/PJ1128/graph/"

# self-defined
n = 10 # 20 100
seed_num = 100 # 100 101
max_iter = 20 # 20 100
show_E = 1 # error
show_L = 1 # lambda
blk_nrow = 2
blk_ncol = 3 # blk_nrow + 1

seed(seed_num)
# data preparation
mep = from_random(n, blk_nrow, blk_ncol)
V, D, symmind = eigop(mep)

# algorithm
# origin_resinv
# origin_resinv_rqi
# Newton_quasi
# Newton_quasi_rqi
# Newton
# Newton_rqi

x0 = V[:, [2]][:mep.size, :] + 1e-2 * randn(mep.size, 1)
lambda0 = D[2] + 1e-1 * randn(1) + 1j * 1e-2 * randn(1)
lambdas0 = [lambda0] + [mep.fi(_, x0) for _ in range(blk_nrow - 1)]
vs = [randn(mep.size, 1)]
v = vs[0]
tol = 1e-16

x_origin_resinv, lambdas_origin_resinv, es_origin_resinv = origin_resinv(mep, lambdas0, v, x0, tol, max_iter)
x_origin_resinv_rqi, lambdas_origin_resinv_rqi, es_origin_resinv_rqi = origin_resinv_rqi(mep, lambdas0, v, x0, tol, max_iter)
x_Newton_quasi, lambdas_Newton_quasi, es_Newton_quasi = Newton_quasi(mep, lambdas0, v, x0, tol, max_iter)
x_Newton_quasi_rqi, lambdas_Newton_quasi_rqi, es_Newton_quasi_rqi = Newton_quasi_rqi(mep, lambdas0, v, x0, tol, max_iter)
x_Newton, lambdas_Newton, es_Newton = Newton(mep, lambdas0, v, x0, tol, max_iter)
x_Newton_rqi, lambdas_Newton_rqi, es_Newton_rqi = Newton_rqi(mep, lambdas0, v, x0, tol, max_iter)


## plot 1.1
# e = abs(es_origin_resinv)
# l1, = semilogy(arange(max_iter + 1), e / e[0], marker="*", linestyle="-", color = 'dodgerblue')
# e = abs(es_origin_resinv_rqi)
# l2, = semilogy(arange(max_iter + 1), e / e[0], marker="*", linestyle="--", color = 'dodgerblue')
e = abs(es_Newton)
l3, = semilogy(arange(max_iter + 1), e / e[0], marker="o", linestyle="-", color = 'darkorange')
e = abs(es_Newton_rqi)
l4, = semilogy(arange(max_iter + 1), e / e[0], marker="o", linestyle="--", color = 'darkorange')
e = abs(es_Newton_quasi)
l5, = semilogy(arange(max_iter + 1), e / e[0], marker="+", linestyle="-", color = 'red')
e = abs(es_Newton_quasi_rqi)
l6, = semilogy(arange(max_iter + 1), e / e[0], marker="+", linestyle="--", color = 'red')
xticks(range(0, max_iter, int(max_iter / 8)))
xlabel("iteration num")
ylabel("error")
legend([l3, l4, l5, l6],
        ["dynamic $P_1(t)$",
         "dynamic $P_1(t)$ and Rayleigh quotient",
         "fixed $P_1(t)$",
         "fixed $P_1(t)$ and Rayleigh quotient"])
title("residual error $\Vert T(\lambda, x)x\Vert_2$ of NEPv")
grid(linestyle="--")
savefig(SAVETO + "f_3_3_residual_error_100.pdf", transparent=True)
show()