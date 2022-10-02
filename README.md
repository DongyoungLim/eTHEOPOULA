# e-THEO POULA with discontinuous updating

This repository is the official implementation of "e-THEO POULA with discontinuous updating". 

Abstract: In this paper, we focus on optimization problems with discontinuous stochastic gradient. We solve the aforementioned problems by extending TH$\varepsilon$O POULA, which is recently developed in \citeauthor{lim2021polygonal} (\citeyear{lim2021polygonal}) based on the advances of polygonal Euler approximations. We demonstrate both theoretically and numerically the applicability of the extended TH$\varepsilon$O POULA algorithm. In particular, under the conditions that the stochastic gradient is locally Lipschitz in average and satisfies a certain convexity at infinity condition, we establish non-asymptotic error bounds for the extended TH$\varepsilon$O POULA in Wasserstein distances, and provide a non-asymptotic estimate for the expected excess risk. Three key applications in finance are provided, namely, the multi-period portfolio optimization, transfer learning in the multi-period portfolio optimization, and the insurance claim prediction, which involve neural networks with (Leaky)-ReLU activation functions. Numerical experiments conducted using real-world datasets illustrate the superior empirical performance of the extended TH$\varepsilon$O POULA compared to SGLD, ADAM, and AMSGrad in terms of model's accuracy.

## Dependencies
-Python 3.6
-Pytorch_cuda 1.8.0 

## Guide
This repository contains three applications in finance: the multi-period portfolio optimization, transfer learning in the multi-period portfolio optimization, and the insurance claim prediction. 

### Multi-period portfolio optimization (Section 3.1)
Please refer to the folder name portfolio_selection. Execute run_BS and run_AR for training models. The numerical results are summarized in plot_Results.ipynb.  



### Transfer learning
Please refer to the folder name, portfolio_selection_transferlearning. Excute file_name.sh files for training models. The numerical results are summarized in the outputs folder. 

### Insurance claim prediction
Please refer to the folder name nonlinear regression. Excute main.py file for training models. The numerical results are summarized in plot_Results.ipynb.

## Data
For the multi-period portfolio optimization, data is automatically generated.
If you are interested in the insurance claims data, please email me, dlim@unist.ac.kr.








