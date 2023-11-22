#  Langevin dynamics based algorithm e-THEO POULA for stochastic optimization problems with discontinuous stochastic gradient

This repository is the official implementation of "Langevin dynamics based algorithm e-THEO POULA for stochastic optimization problems with discontinuous stochastic gradient". 

Authors: Dong-Young Lim, Ariel Neufeld, Sotirios Sabanis, and Ying Zhang

Abstract: We introduce a new Langevin dynamics based algorithm, called e-THEO POULA, to solve optimization problems with discontinuous stochastic gradients which naturally appear in real-world applications such as quantile estimation, vector quantization, CVaR minimization, and regularized optimization problems involving ReLU neural networks. We demonstrate both theoretically and numerically the applicability of the e-THEO POULA algorithm. More precisely, under the conditions that the stochastic gradient is locally Lipschitz \textit{in average} and satisfies a certain convexity at infinity condition, we establish non-asymptotic error bounds for e-TH$\varepsilon$O POULA in Wasserstein distances and provide a non-asymptotic estimate for the expected excess risk, which can be controlled to be arbitrarily small. Three key applications in finance and insurance are provided, namely, multi-period portfolio optimization, transfer learning in multi-period portfolio optimization, and insurance claim prediction, which involve neural networks with (Leaky)-ReLU activation functions. Numerical experiments conducted using real-world datasets illustrate the superior empirical performance of e-TH$\varepsilon$O POULA compared to SGLD, TUSLA, ADAM, and AMSGrad in terms of model accuracy.

## Dependencies
-Python 3.6
-Pytorch_cuda 1.8.0 

## Guide
This repository contains three applications in finance: the multi-period portfolio optimization, transfer learning in the multi-period portfolio optimization, and the insurance claim prediction. 

### Multi-period portfolio optimization (Section 3.1)
Please refer to the folder name portfolio_selection. Execute run_BS and run_AR for training models. The numerical results are summarized in plot_Results.ipynb.  



### Transfer learning (Section 3.2)
Please refer to the folder name, portfolio_selection_transferlearning. Excute file_name.sh files for training models. The numerical results are summarized in the outputs folder. 

### Insurance claim prediction (Section 3.3)
Please refer to the folder name nonlinear regression. Excute main.py file for training models. The numerical results are summarized in plot_Results.ipynb.

## Data
For the multi-period portfolio optimization, data is automatically generated.
If you are interested in the insurance claims data, please email me, dlim@unist.ac.kr.








