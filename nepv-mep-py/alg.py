from utils import *

"""
origin_resinv
origin_resinv_rqi
reduced_resinv_quasi
reduced_resinv_quasi_rqi
reduced_resinv
reduced_resinv_rqi
"""


# residual inverse iteration
def origin_resinv(mep, lambdas0, v, x0, tol, max_iter):
    # def F(lambdas, x):
    #     n = mep.size
    #     crt = zeros((n * 2 + 2, 1), dtype=complex128)
    #     crt[:n, [0]] = mep.Tix(lambdas, 0, x)
    #     crt[n:2 * n, [0]] = mep.Tix(lambdas, 1, x)
    #     crt[2 * n, 0] = matmul(v.T, x) - 1
    #     crt[2 * n + 1, 0] = matmul(v.T, x) - 1
    #     return crt
    def F(lambdas, x):
        n = mep.size
        m = mep.nrow
        crt = zeros((n * m + m, 1), dtype=complex128)
        for idx in range(m):
            crt[idx * n:(idx + 1) * n, [0]] = mep.Tix(lambdas, idx, x)
            crt[m * n + idx] = matmul(v.T, x) - 1
        return crt

    # def F_(lambdas, x):
    #     n = mep.size
    #     J = zeros((n * 2 + 2, n * 2 + 2), dtype=complex128)
    #     J[:n, :n] = mep.Ti(lambdas, 0)
    #     J[n:(2 * n), n:(2 * n)] = mep.Ti(lambdas, 1)
    #     J[:n, [2 * n]] = matmul(mep.blk_mat[0, 1], x)
    #     J[:n, [2 * n + 1]] = matmul(mep.blk_mat[0, 2], x)
    #     J[n:2 * n, [2 * n]] = matmul(mep.blk_mat[1, 1], x)
    #     J[n:2 * n, [2 * n + 1]] = matmul(mep.blk_mat[1, 2], x)
    #     J[[2 * n], :n] = v.T
    #     J[[2 * n + 1], n:2 * n] = v.T
    #     return J
    def F_(lambdas, x):
        n = mep.size
        m = mep.nrow
        J = zeros((n * m + m, n * m + m), dtype=complex128)
        for idx in range(m):
            J[idx * n:(idx + 1) * n, idx * n:(idx + 1) * n] = mep.Ti(lambdas, idx)
            J[[m * n + idx], idx * n:(idx + 1) * n] = v.T
            for idxx in range(m):
                J[idx * n:(idx + 1) * n, [m * n + idxx]] = matmul(mep.blk_mat[idx, idxx + 1], x)
        return J

    n = mep.size
    m = mep.nrow
    lambdas = lambdas0
    all_lambdas = [deepcopy(lambdas)]
    x = x0
    all = [x] * mep.nrow + [_.reshape(-1, 1) for _ in lambdas]
    all = concatenate(all)
    iter = 0
    es = [mep.E(lambdas, x)]
    while iter < max_iter and es[-1] > tol:
        all = all - linalg.solve(F_(lambdas, x), F(lambdas, x))
        x = all[:n, [0]]
        x = x / matmul(v.T, x)
        lambdas = [all[m * n + _] for _ in range(m)]
        all = [x] * mep.nrow + [_.reshape(-1, 1) for _ in lambdas]
        all = concatenate(all)
        iter += 1
        # es.append(la.norm(F(lambdas, x)[:n*mep.nrow]))
        es.append(mep.E(lambdas, x))
        all_lambdas.append(deepcopy(lambdas))
    return x, array(all_lambdas), array(es)


def origin_resinv_rqi(mep, lambdas0, v, x0, tol, max_iter):
    def F(lambdas, x):
        n = mep.size
        m = mep.nrow
        crt = zeros((n * m + m, 1), dtype=complex128)
        for idx in range(m):
            crt[idx * n:(idx + 1) * n, [0]] = mep.Tix(lambdas, idx, x)
            crt[m * n + idx] = matmul(v.T, x) - 1
        return crt

    def F_(lambdas, x):
        n = mep.size
        m = mep.nrow
        J = zeros((n * m + m, n * m + m), dtype=complex128)
        for idx in range(m):
            J[idx * n:(idx + 1) * n, idx * n:(idx + 1) * n] = mep.Ti(lambdas, idx)
            J[[m * n + idx], idx * n:(idx + 1) * n] = v.T
            for idxx in range(m):
                J[idx * n:(idx + 1) * n, [m * n + idxx]] = matmul(mep.blk_mat[idx, idxx + 1], x)
        return J

    n = mep.size
    m = mep.nrow
    lambdas = lambdas0
    all_lambdas = [deepcopy(lambdas)]
    x = x0
    all = [x] * mep.nrow + [_.reshape(-1, 1) for _ in lambdas]
    all = concatenate(all)
    iter = 0
    es = [mep.E(lambdas, x)]
    while iter < max_iter and es[-1] > tol:
        all = all - linalg.solve(F_(lambdas, x), F(lambdas, x))
        x = all[:n, [0]]
        x = x / matmul(v.T, x)

        # use rq to solve lambda
        rq_A = zeros((m, m), dtype=complex128)
        rq_b = zeros((m, 1), dtype=complex128)
        for idx in range(m):
            rq_b[idx, 0] = matmul2(x.T, mep.blk_mat[idx, 0], x)
            for idxx in range(m):
                rq_A[idx, idxx] = matmul2(x.T, mep.blk_mat[idx, idxx + 1], x)
        rq_l = linalg.solve(rq_A, rq_b)
        lambdas = [rq_l[_, 0] for _ in range(m)]

        all = [x] * mep.nrow + [_.reshape(-1, 1) for _ in lambdas]
        all = concatenate(all)
        iter += 1
        # es.append(la.norm(F(lambdas, x)[:n*mep.nrow]))
        es.append(mep.E(lambdas, x))
        all_lambdas.append(deepcopy(lambdas))
    return x, array(all_lambdas), array(es)


def reduced_resinv_quasi(mep, lambdas0, v, x0, tol, max_iter):
    n = mep.size
    m = mep.nrow
    lambdas = deepcopy(lambdas0)
    all_lambdas = [deepcopy(lambdas)]
    x = deepcopy(x0)
    iter = 0
    es = [mep.E(lambdas, x)]

    T0v = linalg.inv(mep.Ti(lambdas0, 0))

    while iter < max_iter and es[-1] > tol:
        b = zeros((m, 1), dtype=complex128)
        b[0] = -matmul(v.T, x)
        b[1] = -matmul(v.T, x)
        A = zeros((m, m), dtype=complex128)
        for i in range(m):
            for j in range(m):
                A[i, j] = matmul(v.T, la.solve(mep.Ti(lambdas, i), matmul(mep.blk_mat[i, j + 1], x)))[0][0]
        dlambdas = la.solve(A, b)
        blambdas = deepcopy(lambdas)
        for idx in range(m):
            lambdas[idx] += dlambdas[idx]
        x = x - matmul(T0v, mep.Tix(lambdas, 0, x))
        x = x / matmul(v.T, x)
        iter += 1
        es.append(mep.E(lambdas, x))
        all_lambdas.append(deepcopy(lambdas))
    return x, array(all_lambdas), array(es)


def reduced_resinv_quasi_rqi(mep, lambdas0, v, x0, tol, max_iter):
    n = mep.size
    m = mep.nrow
    lambdas = deepcopy(lambdas0)
    all_lambdas = [deepcopy(lambdas)]
    x = deepcopy(x0)
    iter = 0
    es = [mep.E(lambdas, x)]
    while iter < max_iter and es[-1] > tol:
        b = zeros((m, 1), dtype=complex128)
        b[0] = -matmul(v.T, x)
        b[1] = -matmul(v.T, x)
        A = zeros((m, m), dtype=complex128)
        for i in range(m):
            for j in range(m):
                A[i, j] = matmul(v.T, la.solve(mep.Ti(lambdas, i), matmul(mep.blk_mat[i, j + 1], x)))[0][0]
        dlambdas = la.solve(A, b)
        blambdas = deepcopy(lambdas)
        for idx in range(m):
            lambdas[idx] += dlambdas[idx]
        x = x - la.solve(mep.Ti(lambdas0, 0), mep.Tix(lambdas, 0, x))
        x = x / matmul(v.T, x)

        # use rq to solve lambda
        rq_A = zeros((m, m), dtype=complex128)
        rq_b = zeros((m, 1), dtype=complex128)
        for idx in range(m):
            rq_b[idx, 0] = matmul2(x.T, mep.blk_mat[idx, 0], x)
            for idxx in range(m):
                rq_A[idx, idxx] = matmul2(x.T, mep.blk_mat[idx, idxx + 1], x)
        rq_l = linalg.solve(rq_A, rq_b)
        lambdas = [rq_l[_, 0] for _ in range(m)]

        iter += 1
        es.append(mep.E(lambdas, x))
        all_lambdas.append(deepcopy(lambdas))
    return x, array(all_lambdas), array(es)


def reduced_resinv(mep, lambdas0, v, x0, tol, max_iter):
    n = mep.size
    m = mep.nrow
    lambdas = deepcopy(lambdas0)
    all_lambdas = [deepcopy(lambdas)]
    x = deepcopy(x0)
    iter = 0
    es = [mep.E(lambdas, x)]
    while iter < max_iter and es[-1] > tol:
        b = zeros((m, 1), dtype=complex128)
        b[0] = -matmul(v.T, x)
        b[1] = -matmul(v.T, x)
        A = zeros((m, m), dtype=complex128)
        for i in range(m):
            for j in range(m):
                A[i, j] = matmul(v.T, la.solve(mep.Ti(lambdas, i), matmul(mep.blk_mat[i, j + 1], x)))[0][0]
        dlambdas = la.solve(A, b)
        blambdas = deepcopy(lambdas)
        for idx in range(m):
            lambdas[idx] += dlambdas[idx]
        x = x - la.solve(mep.Ti(blambdas, 0), mep.Tix(lambdas, 0, x))
        x = x / matmul(v.T, x)
        iter += 1
        es.append(mep.E(lambdas, x))
        all_lambdas.append(deepcopy(lambdas))
    return x, array(all_lambdas), array(es)


def reduced_resinv_rqi(mep, lambdas0, v, x0, tol, max_iter):
    n = mep.size
    m = mep.nrow
    lambdas = deepcopy(lambdas0)
    all_lambdas = [deepcopy(lambdas)]
    x = deepcopy(x0)
    iter = 0
    es = [mep.E(lambdas, x)]
    while iter < max_iter and es[-1] > tol:
        b = zeros((m, 1), dtype=complex128)
        b[0] = -matmul(v.T, x)
        b[1] = -matmul(v.T, x)
        A = zeros((m, m), dtype=complex128)
        for i in range(m):
            for j in range(m):
                A[i, j] = matmul(v.T, la.solve(mep.Ti(lambdas, i), matmul(mep.blk_mat[i, j + 1], x)))[0][0]
        dlambdas = la.solve(A, b)
        blambdas = deepcopy(lambdas)
        for idx in range(m):
            lambdas[idx] += dlambdas[idx]
        x = x - la.solve(mep.Ti(blambdas, 0), mep.Tix(lambdas, 0, x))
        x = x / matmul(v.T, x)

        # use rq to solve lambda
        rq_A = zeros((m, m), dtype=complex128)
        rq_b = zeros((m, 1), dtype=complex128)
        for idx in range(m):
            rq_b[idx, 0] = matmul2(x.T, mep.blk_mat[idx, 0], x)
            for idxx in range(m):
                rq_A[idx, idxx] = matmul2(x.T, mep.blk_mat[idx, idxx + 1], x)
        rq_l = linalg.solve(rq_A, rq_b)
        lambdas = [rq_l[_, 0] for _ in range(m)]

        iter += 1
        es.append(mep.E(lambdas, x))
        all_lambdas.append(deepcopy(lambdas))
    return x, array(all_lambdas), array(es)

def inverse(mep, lambdas0, sigma, x0, tol, max_iter):
    n = mep.size
    m = mep.nrow
    delta0 = deltai(mep, 0)
    delta1 = deltai(mep, 1)

    lambdas = lambdas0
    all_lambdas = [lambdas]
    x = deepcopy(x0) / linalg.norm(x0)
    iter = 0

    z = kronn(x, mep.nrow)
    es = [mep.E(lambdas, x)]

    while iter < max_iter and es[-1] > tol:
        z = linalg.solve(delta1 - sigma *delta0, matmul(delta0, z))
        z = z / linalg.norm(z)
        lambdas[0] = matmul2(z.T, delta1, z) / matmul2(z.T, delta0, z)
        idx = (abs(z).reshape(5, -1)**2).sum(axis=0).argmax()
        x = z[idx*n:(idx+1)*n, [0]]
        x = x / linalg.norm(x)
        for idx in range(1, m):
            lambdas[idx] = mep.fi(idx - 1, x)
        all_lambdas.append(lambdas)
        es.append(linalg.norm(matmul(delta1 - delta0 * sigma, z)))
        iter += 1
    return x, array(all_lambdas), array(es)

def inverse_rqi(mep, lambdas0, sigma, x0, tol, max_iter):
    n = mep.size
    m = mep.nrow
    delta0 = deltai(mep, 0)
    delta1 = deltai(mep, 1)

    lambdas = lambdas0
    all_lambdas = [lambdas]
    x = deepcopy(x0) / linalg.norm(x0)
    iter = 0

    z = kronn(x, mep.nrow)
    es = [mep.E(lambdas, x)]

    while iter < max_iter and es[-1] > tol:
        z = linalg.solve(delta1 - sigma *delta0, matmul(delta0, z))
        z = z / linalg.norm(z)
        lambdas[0] = matmul2(z.T, delta1, z) / matmul2(z.T, delta0, z)
        idx = (abs(z).reshape(5, -1)**2).sum(axis=0).argmax()
        x = z[idx*n:(idx+1)*n, [0]]
        x = x / linalg.norm(x)
        for idx in range(1, m):
            lambdas[idx] = mep.fi(idx - 1, x)
        sigma = lambdas[0]
        all_lambdas.append(lambdas)
        es.append(linalg.norm(matmul(delta1 - delta0 * sigma, z)))
        iter += 1
    return x, array(all_lambdas), array(es)

Newton = reduced_resinv
Newton_rqi = reduced_resinv_rqi
Newton_quasi = reduced_resinv_quasi
Newton_quasi_rqi = reduced_resinv_quasi_rqi