import math
import torch
import numpy as np
from torch.optim.optimizer import Optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class THEOPOULA(Optimizer):

    def __init__(self, params, lr=1e-1, eta=0, beta=1e14, r=3, eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, beta=beta, eta=eta, r=r, eps=eps, weight_decay=weight_decay)
        super(THEOPOULA, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(THEOPOULA, self).__setstate__(state)


    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()


        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)


                if len(state) == 0:
                    state['step'] = 0

                eta, beta, lr, eps = group['eta'], group['beta'], group['lr'], group['eps']
                g_abs = torch.abs(grad)
                numer = g_abs.clone().detach()
                denom = g_abs.clone().detach()

                #numer = grad * ( 1 + math.sqrt(lr)/(eps + g_abs))
                #denom = 1 + math.sqrt(lr) * g_abs

                numer.add_(eps).reciprocal_().mul_(math.sqrt(lr)).add_(1).mul_(grad)
                denom.mul_(math.sqrt(lr)).add_(1)

                noise = math.sqrt(2 * lr / beta) * torch.randn(size=p.size(), device=device)


                p.data.addcdiv_(value=-lr, tensor1=numer, tensor2=denom).add_(noise)


                if eta > 0:
                    concat_params = torch.cat([p.view(-1) for p in group['params']])
                    r = group['r']
                    total_norm = torch.pow(torch.norm(concat_params), 2 * r)

                    reg_num = eta * p * total_norm
                    reg_denom = 1 + math.sqrt(lr) * total_norm
                    reg = reg_num/reg_denom
                    p.data.add_(reg)

        return loss

