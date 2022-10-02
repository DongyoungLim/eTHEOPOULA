import math
import torch
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
            pnorm = 0
            eta = group['eta']
            if eta > 0:
                for p in group['params']:
                    pnorm += torch.sum(torch.pow(p.data, exponent=2))

                r = group['r']
                total_norm = torch.pow(pnorm, r)

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)


                if len(state) == 0:
                    state['step'] = 0

                eta, beta, lr = group['eta'], group['beta'], group['lr']

                noise = math.sqrt(2 * lr / beta) * torch.randn(size=p.size(), device=device)
                numer = grad * ( 1 + math.sqrt(lr)/ (group['eps'] + torch.abs(grad)))
                denom = 1 + math.sqrt(lr) * torch.abs(grad)

                p.data.addcdiv_(value=-lr, tensor1=numer, tensor2=denom).add_(noise)

                if eta > 0:
                    reg_num = eta * p * total_norm
                    reg_denom = 1 + math.sqrt(lr) * total_norm
                    reg = reg_num/reg_denom
                    p.data.add_(reg)




        return loss

