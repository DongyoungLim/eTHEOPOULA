import numpy as np

np.random.seed(111)

def generate_path(asset_model, p, n, time_step, r_f):
    if asset_model == 'BS':
        R_f = np.exp(r_f * time_step)
        if p == 5:
            lamb = 0.1 * np.ones(p)
            lamb[2:] = 0.2
            sig = 0.01 * np.ones([p, p])
            np.fill_diagonal(sig, 0.15)
        elif p == 50:
            lamb = 0.01 * np.ones(p)
            lamb[25:] = 0.05
            sig = 0.005 * np.ones([p, p])
            np.fill_diagonal(sig, 0.15)
        elif p == 100:
            lamb = 0.01 * np.ones(p)
            lamb[50:] = 0.05
            sig = 0.0025 * np.ones([p, p])
            np.fill_diagonal(sig, 0.15)

        K = int(1 / time_step)
        rs = np.zeros([n, K, p])

        randomness = np.random.multivariate_normal(np.zeros(p), np.identity(p), size=(n, K))
        tmp1 = (r_f + np.matmul(sig, lamb) - 0.5 * np.diag(np.matmul(sig, np.transpose(sig)))) * time_step
        rs[:, :, :] = tmp1

        tmp2 = np.sqrt(time_step) * np.matmul(randomness, sig)
        rs = np.exp(rs + tmp2) - R_f

        # for k in range(K):
        #     randomness = np.random.multivariate_normal(np.zeros(p), np.identity(p), size=n)
        #
        #     tmp1 = (r_f + np.matmul(sig, lamb) - 0.5 * np.diag(np.matmul(sig, np.transpose(sig)))) * time_step
        #     tmp1 = tmp1.reshape(-1, 1)
        #     tmp1 = np.repeat(tmp1, n, axis=1)
        #     tmp2 = np.sqrt(time_step) * np.matmul(sig, np.transpose(randomness))
        #
        #     r_k = np.exp(tmp1 + tmp2) - R_f
        #     rs[:, k, :] = np.transpose(r_k)


        # for m in range(n):
        #     randomness = np.random.multivariate_normal(np.zeros(p), np.identity(p), size=K)
        #     tmp1 = (r_f + np.matmul(sig, lamb) - 0.5 * np.diag(np.matmul(sig, np.transpose(sig)))) * time_step
        #     tmp1 = tmp1.reshape(-1, 1)
        #     tmp1 = np.repeat(tmp1, K, axis=1)
        #     tmp2 = np.sqrt(time_step) * np.matmul(sig, np.transpose(randomness))
        #
        #     r_k = np.exp(tmp1 + tmp2) - R_f
        #     rs[m, :, :] = np.transpose(r_k)

    elif asset_model == 'AR':
        name = 'case1'

        if name == 'case1':
            alpha_value = 0.015
            sig_ii = 0.0238
            sig_ij = 0.0027
        elif name == 'case2':
            alpha_value = 0.005
            sig_ii = 0.005
            sig_ij = 0.002

        K = int(1 / time_step)
        alpha = alpha_value * np.ones(p)
        alpha = np.transpose(np.repeat(alpha.reshape(-1,1), n, axis=1))
        A = -0.15 * np.identity(p)

        sig = sig_ij * np.ones([p, p])
        np.fill_diagonal(sig, sig_ii)

        R_initial = sig_ii/1.15 * np.ones(p)

        rs = np.zeros([n, K+1, p])
        rs[:, 0, :] = R_initial

        for k in range(1, K+1):
            randomness = np.random.multivariate_normal(np.zeros(p), sig, size=n)
            #tmp1 = np.transpose(np.repeat(alpha.reshape(-1,1), n, axis=1))

            rs[:, k, :] = alpha + np.transpose(np.matmul(A, np.transpose(rs[:, k-1, :]))) + randomness



    return rs


        #file_name = '{}_p{}_n{}.pkl'.format(asset_model, p, n)
        #pkl.dump(rs, open(os.path.join('./data', file_name), 'wb'))








