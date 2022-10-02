## import packages
import torch
import torch.optim as optim
import os
import argparse
import numpy as np
import pickle as pkl
import pandas as pd
#import matplotlib.pyplot as plt
from generator import *
from optimizers import *
from utils import ReturnDataset
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from models import Parentnet, Childnet
import time

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1)

## parsing
parser = argparse.ArgumentParser('portfolio selection')
parser.add_argument('--seed', default=111, type=int)
parser.add_argument('--batch_size', default=64, type=int, help='batch_size')
parser.add_argument('--epochs', default=100, type=int, help='# of epochs')
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--lr_trs', default=0.1, type=float)
parser.add_argument('--act_fn', default='relu', type=str)
parser.add_argument('--hidden_size', default=20, type=int, help='number of neurons')
parser.add_argument('--optimizer', default='theopoula', type=str)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--r', default=0, type=float)
parser.add_argument('--eta', default=0, type=float)
parser.add_argument('--beta', default='1e12', type=float)
parser.add_argument('--eps', default=1e-1, type=float)
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--patience', default=5, type=int)
parser.add_argument('--lr_gamma', default=0.1, type=float)
parser.add_argument('--steplr', default=50, type=int)
parser.add_argument('--T_max', default=100, type=int)
parser.add_argument('--eta_min', default=0.01, type=float)
parser.add_argument('--log_dir', default='./logs/', type=str)
parser.add_argument('--ckpt_dir', default='./ckpt/', type=str)
parser.add_argument('--when', nargs="+", type=int, default=[-1])

# model parameters
parser.add_argument('--asset_model', default='AR', choices=['AR', 'CCC-GARCH', 'BS'], type=str)
parser.add_argument('--num_asset', default=30, type=int)
parser.add_argument('--num_path', default=80000, type=int)
parser.add_argument('--r_f', default=1.03, type=float)
parser.add_argument('--u_gamma', default=7, type=float, help='parameter for the utility function')
parser.add_argument('--num_step', default=10, type=int)
parser.add_argument('--scheduler_type', default='step', type=str)
parser.add_argument('--scheduler', action='store_true')
parser.add_argument('--beta_annealing', action='store_true')
parser.add_argument('--beta_gamma', default=10, type=float)
parser.add_argument('--beta_step', default=25, type=int)


args = parser.parse_args()
torch.manual_seed(args.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)
start = time.time()
# set the bounding constraints
if args.asset_model == 'BS':
    if args.num_asset == 5:
        D = [0, 1.5]
        time_step = 1/40
    elif args.num_asset == 50:
        D = [0, 0.5]
        time_step = 1 / 40
    elif args.num_asset == 100:
        D = [0, 0.5]
        time_step = 1 / 30
    else:
        # print('no explicit bounding constraints ==> set [0, 1]^p')
        D = [0, 1]

    state_size = 1
    num_step = int(1 / time_step)
    args.r_f = np.exp(0.03 * time_step)

elif (args.asset_model == 'AR') | (args.asset_model == 'CCC-GARCH'):
    state_size = 1 + args.num_asset
    D = [0, 1]
    num_step = args.num_step
    time_step = 1/num_step

num_batch = int(np.ceil(args.num_path / args.batch_size))

def get_ckpt_name(model='BS', num_asset=5, seed=111, optimizer='sgd', lr=0.1, momentum=0.9,
                  beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=5e-4, lr_gamma=0.1, steplr=100,
                  beta=1e12, scheduler=False, hidden_size=5, T_max=100, eta_min=0.001, num_step=10,
                  batch_size=64, scheduler_type='step', epochs=100, patience=5,
                  beta_annealing=False, beta_gamma=10, beta_step=25):
    name = {
        'sgld': 'seed{}-lr{}-beta{:.1e}--wdecay{}'.format(seed, lr, beta, weight_decay),
        'adam': 'seed{}-lr{}-betas{}-{}-wdecay{}'.format(seed, lr, beta1, beta2, weight_decay),
        'amsgrad': 'seed{}-lr{}-betas{}-{}-wdecay{}'.format(seed, lr, beta1, beta2, weight_decay),
        'theopoula': 'seed{}-lr{}-eps{}-wdecay{}-beta{:.1e}'.format(seed, lr, eps, weight_decay, beta),
    }[optimizer]
    if scheduler:
        name = name + '-scheduler{}-steplr{}-lrgamma{}'.format(scheduler_type, steplr, lr_gamma)

    if beta_annealing:
        name = name + '-beta_annealing{}-betagamma{}-betastep{}'.format(beta_annealing, beta_gamma, beta_step)

    return 'TransfL-{}-p{}-K{}-hs{}-bs{}-{}-{}-epochs{}'.format(model, num_asset, num_step, hidden_size, batch_size, optimizer, name, epochs)

save = get_ckpt_name(model=args.asset_model, seed=args.seed, num_asset=args.num_asset, optimizer=args.optimizer, lr=args.lr,
                     eps=args.eps, num_step=num_step, weight_decay=args.weight_decay, lr_gamma=args.lr_gamma, steplr=args.steplr,
                     beta=args.beta, scheduler=args.scheduler, hidden_size=args.hidden_size, T_max=args.T_max, eta_min=args.eta_min,
                     batch_size=args.batch_size, scheduler_type=args.scheduler_type, epochs=args.epochs, patience=args.patience,
                     beta_annealing=args.beta_annealing, beta_gamma=args.beta_gamma, beta_step=args.beta_step, momentum=args.momentum,
                     ) + 'r{}-eta{}'.format(args.r, args.eta)


## loading a model

# print('\n=> Building model.. on {%s}'%device)

parent_net = Parentnet(state_size=state_size,
              num_asset=args.num_asset,
              num_step=num_step,
              r_f=args.r_f,
              u_gamma=args.u_gamma,
              D=D,
              asset_model=args.asset_model,
              hidden_size=args.hidden_size,
              act_fn=args.act_fn)
parent_net.to(device)

name = 'seed{}-lr{}-eps{}-wdecay{}-beta{:.1e}'.format(args.seed, args.lr, args.eps, args.weight_decay, args.beta) \
       + '-scheduler{}-steplr{}-lrgamma{}'.format(args.scheduler_type, args.steplr, args.lr_gamma)
state_path = './ckpt/{}-p{}-K{}-hs{}-bs{}-{}-{}-epochs{}'.format(args.asset_model, args.num_asset, num_step, args.hidden_size, args.batch_size, args.optimizer, name, args.epochs)
print(state_path)
state = torch.load(state_path)

parent_net.load_state_dict(state['net'])
parent_net.eval()

print('========transferring===============')
## Define a child net

child_net = Childnet(state_size=state_size,
              num_asset=args.num_asset,
              num_step=num_step,
              r_f=args.r_f,
              u_gamma=args.u_gamma,
              D=D,
              asset_model=args.asset_model,
              hidden_size=args.hidden_size,
              act_fn='relu',
              )
child_net.to(device)
child_net.subnet.hidden_layer1.weight.requires_grad = False  #fixed input weight matrix
#child_net.hidden_layer1.weight.requires_grad = False
## setting an optimizer

optimizer = { 'sgld': SGLD(child_net.parameters(), lr=args.lr_trs, beta=args.beta, weight_decay=args.weight_decay),
              'adam': optim.Adam(child_net.parameters(), lr=args.lr_trs, weight_decay=args.weight_decay),
              'amsgrad': optim.Adam(child_net.parameters(), lr=args.lr_trs, amsgrad=True, weight_decay=args.weight_decay),
              'theopoula': THEOPOULA(child_net.parameters(), lr=args.lr_trs, eta=args.eta, beta=args.beta, eps=args.eps, weight_decay=args.weight_decay, r=args.r)
}[args.optimizer]




## Training
# print('\n==> Start training ')

history = {'train_score': [],
           'test_score': [],
           }

best_score = 99999
training_time1 = 0
training_time2 = 0
training_time3 = 0
time_to_opt = 0

state = {}
train_data = generate_path(args.asset_model, args.num_asset, args.num_path, time_step, args.r_f)


def adjust_learning_rate(optimizer, epoch, steplr=50, lr_gamma=0.1):
    if epoch % steplr == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_gamma
        

def train(epoch, net):
    global training_time1, training_time2, training_time3
    # print('\n Epoch: %d'%epoch)
    net.train()
    train_score = []

    train_data = generate_path(args.asset_model, args.num_asset, args.num_path, time_step, args.r_f)
    start_time = time.time()
    for batch_idx in range(num_batch):
        start = batch_idx * args.batch_size
        end = np.minimum((batch_idx + 1) * args.batch_size, args.num_path)

        samples = torch.FloatTensor(train_data[start:end]).to(device)
        optimizer.zero_grad()
        W1 = net(samples)
        output = W1 * parent_net(samples[:, 1:, :], torch.ones_like(W1))
        score = torch.mean(output)



        start_time2 = time.time()
        score.backward()
        optimizer.step()
        training_time3 += time.time() - start_time2

        train_score += [score.item()]



    history['train_score'].append(np.mean(train_score))
    training_time1 += time.time() - start_time
    training_time2 += time.time() - start_time






def test(epoch, net):
    global best_score, state, time_to_opt, start_time_opt
    net.eval()
    test_score =[]

    with torch.no_grad():
        num_batch_test = int(np.ceil(num_path_test / args.batch_size))

        for batch_idx in range(num_batch_test):
            start = batch_idx * args.batch_size
            end = np.minimum((batch_idx + 1) * args.batch_size, num_path_test)

            samples = torch.FloatTensor(test_data[start:end]).to(device)
            W1 = net(samples)
            output = parent_net(samples[:, 1:, :], W1)

            score = torch.mean(output)

            test_score += [score.item()]

        history['test_score'].append(np.mean(test_score))



    if np.mean(test_score) < best_score:
        state = {
            'net': net.state_dict(),
            'score': np.mean(test_score),
            'epoch': epoch,
            'optim': optimizer.state_dict()
        }
        best_score = np.mean(test_score)
        time_to_opt = time.time() - time_to_opt

    return np.mean(test_score)

num_path_test = 50000
test_data = generate_path(args.asset_model, args.num_asset, num_path_test, time_step, args.r_f)

start_time_opt = time.time()

for epoch in range(1, args.epochs+1):
    train(epoch, child_net)
    start_time = time.time()
    test_score = test(epoch, child_net)

    adjust_learning_rate(optimizer, epoch, args.steplr, args.lr_gamma)

    training_time1 += time.time() - start_time




##save results

if not os.path.isdir(args.log_dir):
    os.mkdir(args.log_dir)
pkl.dump(history, open(os.path.join(args.log_dir, save), 'wb'))

if not os.path.isdir(args.ckpt_dir):
    os.mkdir(args.ckpt_dir)
torch.save(state, os.path.join(args.ckpt_dir, save))


print(save)
print('best score...', best_score)
print('running time1:', training_time1)
print('running time2:', training_time2)
print('running time3:', training_time3)
print('time to opt:', time_to_opt)