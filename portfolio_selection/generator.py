import numpy as np

np.random.seed(111)

def generate_path(asset_model, p, n, time_step, R_f):
    if asset_model == 'BS':
        r_f = 0.03

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

        randomness = np.random.randn(n, K, p)

        tmp1 = (r_f + np.matmul(sig, lamb) - 0.5 * np.diag(np.matmul(sig, np.transpose(sig)))) * time_step
        tmp2 = np.sqrt(time_step) * np.matmul(randomness, sig)
        rs = np.exp(tmp1 + tmp2) - R_f


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








