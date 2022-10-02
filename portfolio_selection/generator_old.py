import numpy as np
import pickle as pkl
import os
import time

def generate_path(asset_model, p, n, time_step, r_f):

    print('Generate asset_returns under the {} model with the time step {} and the interest rate {}'.format(asset_model, time_step, r_f))

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
        # for m in range(n):
        #     if m%10000 == 0:
        #         print(m)
        #     for k in range(K):
        #         randomness = np.random.multivariate_normal(np.zeros(p), np.identity(p))
        #         r_k = np.exp((r_f + np.matmul(sig, lamb) - 0.5 * np.diag(np.matmul(sig, np.transpose(sig)))) * time_step + np.sqrt(time_step) * np.matmul(sig, randomness)) - R_f
        #         rs[m, k, :] = r_k
        start = time.time()
        for m in range(n):
            if m%10000 == 0:
                print(m, time.time()-start)

            randomness = np.random.multivariate_normal(np.zeros(p), np.identity(p), size=K)
            #print('randomness shape..', randomness.shape)

            tmp1 = (r_f + np.matmul(sig, lamb) - 0.5 * np.diag(np.matmul(sig, np.transpose(sig)))) * time_step
            #print('tmp1 shape...', tmp1.shape)
            tmp1 = tmp1.reshape(-1, 1)
            #print('tmp1 shape...', tmp1.shape)
            tmp1 = np.repeat(tmp1, K, axis=1)
            #print('tmp1 shape...', tmp1.shape)


            tmp2 = np.sqrt(time_step) * np.matmul(sig, np.transpose(randomness))
            #print('tmp2 shape...', tmp2.shape)

            r_k = np.exp(tmp1 + tmp2)- R_f
            #print('r_k shape...', r_k.shape)
            rs[m, :, :] = np.transpose(r_k)


        file_name = '{}_p{}_n{}.pkl'.format(asset_model, p, n)
        pkl.dump(rs, open(os.path.join('./data', file_name), 'wb'), protocol=4)








