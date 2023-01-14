import matplotlib.pyplot as plt

from utils import *
from alg import *

PRE = "/home/zyxu/_projects_/python-project/"
SAVETO = "/home/zyxu/_projects_/latex/PJ1128/graph/"

# self-defined
n = 20 # 20 100
seed_num = 99 # 100 101
max_iter = 100 # 20 100
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
e = abs(es_origin_resinv)
l1, = semilogy(arange(max_iter + 1), e / e[0], marker="*", linestyle="-", color = 'dodgerblue')
# e = abs(es_origin_resinv_rqi)
# l2, = semilogy(arange(max_iter + 1), e / e[0], marker="*", linestyle="--", color = 'dodgerblue')
e = abs(es_Newton)
l3, = semilogy(arange(max_iter + 1), e / e[0], marker="3", linestyle="-", color = 'darkorange')
# e = abs(es_Newton_quasi_rqi)
# l4, = semilogy(arange(max_iter + 1), e / e[0], marker="o", linestyle="--", color = 'darkorange')
e = abs(es_Newton_quasi)
l5, = semilogy(arange(max_iter + 1), e / e[0], marker="+", linestyle="-", color = 'red')
# e = abs(es_Newton_rqi)
# l6, = semilogy(arange(max_iter + 1), e / e[0], marker="+", linestyle="--", color = 'red')
xticks(range(0, max_iter, int(max_iter / 8)))
xlabel("iteration num")
ylabel("error")
legend([l1, l3, l5],
        ["residual iteration",
         "quasi-Newton method with dynamic $P_1(t)$",
         "quasi-Newton method with fixed $P_1(t)$"])
title("residual error $\Vert T(\lambda, x)x\Vert_2$ of NEPv")
grid(linestyle="--")
# savefig(SAVETO + "f_3_1_residual_error_100.pdf", transparent=True)
show()

## plot 1.2
e = abs(es_origin_resinv)
l1, = semilogy(arange(max_iter + 1), e / e[0], marker="*", linestyle="-", color = 'dodgerblue')
# e = abs(es_origin_resinv_rqi)
# l2, = semilogy(arange(max_iter + 1), e / e[0], marker="*", linestyle="--", color = 'dodgerblue')
e = abs(es_Newton)
l3, = semilogy(arange(max_iter + 1), e / e[0], marker="3", linestyle="-", color = 'darkorange')
# e = abs(es_Newton_quasi_rqi)
# l4, = semilogy(arange(max_iter + 1), e / e[0], marker="o", linestyle="--", color = 'darkorange')
e = abs(es_Newton_quasi)
l5, = semilogy(arange(max_iter + 1), e / e[0], marker="+", linestyle="-", color = 'red')
# e = abs(es_Newton_rqi)
# l6, = semilogy(arange(max_iter + 1), e / e[0], marker="+", linestyle="--", color = 'red')
xlim(0, 20)
xticks(range(0, 22, 2))
xlabel("iteration num")
ylabel("error")
legend([l1, l3, l5],
        ["residual iteration",
         "quasi-Newton method with dynamic $P_1(t)$",
         "quasi-Newton method with fixed $P_1(t)$"])
title("residual error $\Vert T(\lambda, x)x\Vert_2$ of NEPv")
grid(linestyle="--")
# plt.savefig(SAVETO + "f_3_1_residual_error_20.pdf", transparent=True)
show()

## plot 2
n = 5
seed_num = 100 # 100 101
seed(seed_num)
max_iter = 20
# data preparation
mep = from_random(n, blk_nrow, blk_ncol)
V, D, symmind = eigop(mep)
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

e = abs(es_origin_resinv)
l1, = semilogy(arange(max_iter + 1), e / e[0], marker="*", linestyle="-", color = 'dodgerblue')
# e = abs(es_origin_resinv_rqi)
# l2, = semilogy(arange(max_iter + 1), e / e[0], marker="*", linestyle="--", color = 'dodgerblue')
e = abs(es_Newton)
l3, = semilogy(arange(max_iter + 1), e / e[0], marker="o", linestyle="-", color = 'darkorange')
# e = abs(es_Newton_quasi_rqi)
# l4, = semilogy(arange(max_iter + 1), e / e[0], marker="o", linestyle="--", color = 'darkorange')
e = abs(es_Newton_quasi)
l5, = semilogy(arange(max_iter + 1), e / e[0], marker="+", linestyle="-", color = 'red')
# e = abs(es_Newton_rqi)
# l6, = semilogy(arange(max_iter + 1), e / e[0], marker="+", linestyle="--", color = 'red')
xticks(range(0, max_iter + 1, int(max_iter / 8)))
xlim(0, max_iter)
xlabel("iteration num")
ylabel("error")
legend([l1, l3, l5],
        ["residual iteration",
         "quasi-Newton method with dynamic $P_1(t)$",
         "quasi-Newton method with fixed $P_1(t)$"])
title("residual error $\Vert T(\lambda, x)x\Vert_2$ of NEPv")
grid(linestyle="--")
# savefig(SAVETO + "f_3_2_residual_error_20.pdf", transparent=True)
show()

## table
repeat = 5
max_iter = 100
blk_nrow = 3
blk_ncol = 4

def time_count(func, n, max_iter, mep, x0):
    # data preparation
    lambda0 = D[2] + 1e-1 * randn(1) + 1j * 1e-2 * randn(1)
    lambdas0 = [lambda0] + [mep.fi(_, x0) for _ in range(blk_nrow - 1)]
    vs = [randn(mep.size, 1)]
    v = vs[0]
    func(mep, lambdas0, v, x0, 0, max_iter)
    begin = datetime.now()
    for _ in range(repeat):
        func(mep, lambdas0, v, x0, 0, max_iter)
    end = datetime.now()
    return (end - begin).total_seconds() / repeat

seed(2)
ns = [16, 32, 64, 128, 256]
funcs = [origin_resinv, Newton, Newton_quasi]
time_matrix = zeros((len(ns), len(funcs)))
for idx1 in range(len(ns)):
    n = ns[idx1]
    mep = from_random(n, blk_nrow, blk_ncol)
    x0 = randn(n, 1)
    for idx2 in range(len(funcs)):
        time_matrix[idx1, idx2] = time_count(funcs[idx2], ns[idx1], max_iter, mep, x0)
        print(idx1, idx2)

time_matrix.tofile("./tmp")
for idx1 in range(len(ns)):
    print("$N=" + str(ns[idx1]) + "$", end=" & ")
    for idx2 in range(len(funcs)):
        print("$", time_matrix[idx1, idx2], end="$")
        if idx2 != len(ns) - 1:
            print(" & ", end="")
    if idx1 != len(ns) - 1:
        print(r" \\")








#
#
# e = abs(lambdas_Newton[:, 0] - closest(D, lambdas_Newton[-1, 0]))
# l1, = semilogy(arange(max_iter + 1), e / e[0])
# e = abs(lambdas_origin_resinv_rqi[:, 0] - closest(D, lambdas_origin_resinv_rqi[-1, 0]))
# l2, = semilogy(arange(max_iter + 1), e / e[0])
# e = abs(lambdas_Newton_quasi[:, 0] - closest(D, lambdas_Newton_quasi[-1, 0]))
# l3, = semilogy(arange(max_iter + 1), e / e[0])
# e = abs(lambdas_Newton_quasi_rqi[:, 0] - closest(D, lambdas_Newton_quasi_rqi[-1, 0]))
# l4, = semilogy(arange(max_iter + 1), e / e[0])
# e = abs(lambdas_Newton[:, 0] - closest(D, lambdas_Newton[-1, 0]))
# l5, = semilogy(arange(max_iter + 1), e / e[0])
# e = abs(lambdas_Newton_rqi[:, 0] - closest(D, lambdas_Newton_rqi[-1, 0]))
# l6, = semilogy(arange(max_iter + 1), e / e[0])
# legend([l1, l2, l3, l4, l5, l6],
#         ["origin_resinv",
#          "origin_resinv_rqi",
#          "Newton_quasi",
#          "Newton_quasi_rqi",
#          "Newton",
#          "Newton_rqi"])
#
# # plt.savefig(PRE + "figure/f" + str(seed_num) + ".pdf", transparent=True)
# title("error of lambda")
# show()
# # close()
# ## inverse iteration
