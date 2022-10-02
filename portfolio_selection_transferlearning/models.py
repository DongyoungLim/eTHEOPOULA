import torch.nn as nn
import torch
import numpy as np
eps = 1e-8
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1)
class Subnet(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, D, act_fn):
        super(Subnet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.act_fn = act_fn
        self.l = D[0] # lower bound
        self.u = D[1] # upper bound

        self.hidden_layer1 = nn.Linear(input_size, self.hidden_size)
        self.hidden_layer2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, output_size)

    def forward(self, x):
        activation = {
            'sigmoid': nn.Sigmoid(),
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'tanh': nn.Tanh()
        }[self.act_fn]
        x = x.view(-1, self.input_size)
        x = activation(self.hidden_layer1(x))
        x = activation(self.hidden_layer2(x))
        x = nn.Tanh()(self.output_layer(x))

        b = (self.u - self.l) / 2
        a = (self.u + self.l) / (self.u - self.l)
        out = (x +a) * b # the range of out => D
        return out


class Fullnet(nn.Module):

    def __init__(self, state_size, num_asset, num_step, r_f, u_gamma, D=[0, 1],  asset_model='BS', hidden_size=50, act_fn='relu'):
        super(Fullnet, self).__init__()
        self.state_size = state_size
        self.num_asset = num_asset
        self.num_step = num_step
        self.r_f = r_f
        self.u_gamma = u_gamma
        self.D = D
        self.hidden_size = hidden_size
        self.act_fn = act_fn
        self.asset_model = asset_model

        # define subnets
        self.subnets = nn.ModuleList([Subnet(state_size, num_asset, hidden_size, D, act_fn) for k in range(num_step)])

    def forward(self, x):

        if self.asset_model == 'BS':   # state_variable = W
            W_k = torch.ones([x.shape[0], 1], device=device, requires_grad=True)
            x.requires_grad = False

            for k in range(self.num_step):
                r_k = x[:, k, :].clone().detach()
                g_k = self.subnets[k](W_k)
                W_k = W_k * (torch.sum(r_k * g_k, dim=1).view(-1,1) + self.r_f)

        elif self.asset_model == 'AR': # state_variable = (W, R_K)

            W_k = torch.ones([x.shape[0], 1], device=device, requires_grad=True)
            x.requires_grad = False

 #          wealth = torch.ones(x.shape[0], x.shape[1])
            for k in range(1, self.num_step+1):
                r_k_previous = x[:, k-1, :].clone().detach()
                s_k = torch.cat((W_k / self.u_gamma, r_k_previous), dim=1)  ## W_k normalization 했다
                g_k = self.subnets[k-1](s_k)
                r_k = x[:, k, :].clone().detach()
                W_k = W_k * (torch.sum(r_k * g_k, dim=1).view(-1, 1) + self.r_f)
#               wealth[:, k] = W_k.view(-1).clone().detach()

        utility = torch.pow((W_k - self.u_gamma / 2), 2)





        return utility

class Subnet_slnn(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, D, act_fn):
        super(Subnet_slnn, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.act_fn = act_fn
        self.l = D[0] # lower bound
        self.u = D[1] # upper bound

        self.hidden_layer1 = nn.Linear(input_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, output_size)

    def forward(self, x):
        activation = {
            'sigmoid': nn.Sigmoid(),
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'tanh': nn.Tanh()
        }[self.act_fn]
        x = x.view(-1, self.input_size)
        x = activation(self.hidden_layer1(x))
        x = nn.Tanh()(self.output_layer(x))

        b = (self.u - self.l) / 2
        a = (self.u + self.l) / (self.u - self.l)
        out = (x +a) * b # the range of out => D
        return out



class Childnet(nn.Module):

    def __init__(self, state_size, num_asset, num_step, r_f, u_gamma, D=[0, 1],  asset_model='BS', hidden_size=50, act_fn='relu'):
        super(Childnet, self).__init__()
        self.state_size = state_size
        self.num_asset = num_asset
        self.num_step = num_step
        self.r_f = r_f
        self.u_gamma = u_gamma
        self.D = D
        self.hidden_size = hidden_size
        self.act_fn = act_fn
        self.asset_model = asset_model

        # define a subnet
        self.subnet = Subnet_slnn(state_size, num_asset, hidden_size, D, act_fn)


    def forward(self, x):

        if self.asset_model == 'BS':   # state_variable = W
            W_0 = torch.ones([x.shape[0], 1], device=device, requires_grad=True)
            x.requires_grad = False

            #r_0 = x[:, 0, :].clone().detach()
            #g_0 = self.subnet(W_0)
            W_1 = W_0 * (torch.sum(x[:, 0, :] * self.subnet(W_0), dim=1).view(-1,1) + self.r_f)

            #r_k = x[:, 1:, :].clone().detach()


        elif self.asset_model == 'AR': # state_variable = (W, R_K)

            W_0 = torch.ones([x.shape[0], 1], device=device, requires_grad=True)
            x.requires_grad = False

            s_0 = torch.cat((W_0 / self.u_gamma, x[:, 0, :]), dim=1)
            #s_0 = torch.cat((W_0 / self.u_gamma, r_0_previous), dim=1)                ## W_k normalization 했다
            g_0 = self.subnet(s_0)
            #r_0 = x[:, 1, :].clone().detach()
            W_1 = W_0 * (torch.sum(x[:, 1, :] * g_0, dim=1).view(-1, 1) + self.r_f)
            #W_1 = W_0 * (torch.sum(r_0 * g_0, dim=1).view(-1, 1) + self.r_f)

            #r_k = x[:, 1:, :].clone().detach()

        return W_1


# #
# class Childnet(nn.Module):
#
#     def __init__(self, parent_net, state_size, num_asset, num_step, r_f, u_gamma, D=[0, 1],  asset_model='BS', hidden_size=50, act_fn='relu'):
#         super(Childnet, self).__init__()
#         self.parent_net = parent_net
#         self.state_size = state_size
#         self.num_asset = num_asset
#         self.num_step = num_step
#         self.r_f = r_f
#         self.u_gamma = u_gamma
#         self.D = D
#         self.hidden_size = hidden_size
#         self.act_fn = act_fn
#         self.asset_model = asset_model
#
#         # define a subnet
#         self.subnet = Subnet_slnn(state_size, num_asset, hidden_size, D, act_fn)
#
#
#     def forward(self, x):
#
#         if self.asset_model == 'BS':   # state_variable = W
#             W_0 = torch.ones([x.shape[0], 1], device=device, requires_grad=True)
#             x.requires_grad = False
#
#             #r_0 = x[:, 0, :].clone().detach()
#             #g_0 = self.subnet(W_0)
#             W_1 = W_0 * (torch.sum(x[:, 0, :] * self.subnet(W_0), dim=1).view(-1,1) + self.r_f)
#             print('===========================W_1===========================')
#             print(W_1)
#             #r_k = x[:, 1:, :].clone().detach()
#
#
#         elif self.asset_model == 'AR': # state_variable = (W, R_K)
#
#             W_0 = torch.ones([x.shape[0], 1], device=device, requires_grad=True)
#             x.requires_grad = False
#
#             s_0 = torch.cat((W_0 / self.u_gamma, x[:, 0, :]), dim=1)
#             #s_0 = torch.cat((W_0 / self.u_gamma, r_0_previous), dim=1)                ## W_k normalization 했다
#             g_0 = self.subnet(s_0)
#             #r_0 = x[:, 1, :].clone().detach()
#             W_1 = W_0 * (torch.sum(x[:, 1, :] * g_0, dim=1).view(-1, 1) + self.r_f)
#             #W_1 = W_0 * (torch.sum(r_0 * g_0, dim=1).view(-1, 1) + self.r_f)
#
#             #r_k = x[:, 1:, :].clone().detach()
#
#         #print(W_1)
#         utility = self.parent_net(x[:, 1:, :], W_1)
#
#
#         return utility
#



class Parentnet(nn.Module):

    def __init__(self, state_size, num_asset, num_step, r_f, u_gamma, D=[0, 1],  asset_model='BS', hidden_size=50, act_fn='relu'):
        super(Parentnet, self).__init__()
        self.state_size = state_size
        self.num_asset = num_asset
        self.num_step = num_step
        self.r_f = r_f
        self.u_gamma = u_gamma
        self.D = D
        self.hidden_size = hidden_size
        self.act_fn = act_fn
        self.asset_model = asset_model

        # define subnets
        self.subnets = nn.ModuleList([Subnet(state_size, num_asset, hidden_size, D, act_fn) for k in range(num_step)])

    def forward(self, x, W0):

        if self.asset_model == 'BS':   # state_variable = W
            W_k = torch.ones([x.shape[0], 1], device=device, requires_grad=True)
            x.requires_grad = False


            for k in range(self.num_step):
                W_k = W_k * (torch.sum(x[:, k, :] * self.subnets[k](W_k), dim=1).view(-1, 1) + self.r_f)


        elif self.asset_model == 'AR': # state_variable = (W, R_K)

            W_k = torch.ones([x.shape[0], 1], device=device, requires_grad=True) * W0
            x.requires_grad = False

 #          wealth = torch.ones(x.shape[0], x.shape[1])
            for k in range(1, self.num_step+1):
                r_k_previous = x[:, k-1, :].clone().detach()
                s_k = torch.cat((W_k / self.u_gamma, r_k_previous), dim=1)  ## W_k normalization 했다
                g_k = self.subnets[k-1](s_k)
                r_k = x[:, k, :].clone().detach()
                W_k = W_k * (torch.sum(r_k * g_k, dim=1).view(-1, 1) + self.r_f)
#               wealth[:, k] = W_k.view(-1).clone().detach()
        W_k = W_k * W0
        utility = torch.pow((W_k - self.u_gamma / 2), 2)


        return utility








##

