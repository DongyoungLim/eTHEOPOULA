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
from optimizers import ADAM, THEOPOULA, TUSLA, SGLD
from utils import ReturnDataset
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from models import Fullnet
import time

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


## parsing
parser = argparse.ArgumentParser('portfolio selection')
parser.add_argument('--seed', default=111, type=int)
parser.add_argument('--batch_size', default=64, type=int, help='batch_size')
parser.add_argument('--epochs', default=10, type=int, help='# of epochs')
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--act_fn', default='relu', type=str)
parser.add_argument('--hidden_size', default=20, type=int, help='number of neurons')
parser.add_argument('--optimizer', default='adam', type=str)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--eta', default=0, type=float)
parser.add_argument('--beta', default='1e12', type=float)
parser.add_argument('--eps', default=1e-1, type=float)
parser.add_argument('--TUSLA_r', default=0.5, type=float)
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
parser.add_argument('--R_f', default=1.03, type=float)
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
        time_step = 1/args.num_step#1/40
    elif args.num_asset == 50:
        D = [0, 0.5]
        time_step = 1/args.num_step#1 / 40
    elif args.num_asset == 100:
        D = [0, 0.5]
        time_step = 1/args.num_step#1 / 30
    else:
        print('no explicit bounding constraints ==> set [0, 1]^p')
        D = [0, 1]

    state_size = 1
    num_step = int(1 / time_step)
    print(time_step)
    R_f = np.exp(0.03 * time_step)

elif (args.asset_model == 'AR') | (args.asset_model == 'CCC-GARCH'):
    state_size = 1 + args.num_asset
    D = [0, 1]
    num_step = args.num_step
    time_step = 1/num_step
    R_f = args.R_f

num_batch = int(np.ceil(args.num_path / args.batch_size))


def get_ckpt_name(model='BS', num_asset=5, seed=111, optimizer='sgd', lr=0.1, momentum=0.9,
                  beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=5e-4, lr_gamma=0.1, steplr=100,
                  beta=1e12, scheduler=False, hidden_size=5, T_max=100, eta_min=0.001, num_step=10,
                  batch_size=64, scheduler_type='step', epochs=100, patience=5,
                  beta_annealing=False, beta_gamma=10, beta_step=25, r=0.5, num_path=20000):
    name = {
        'sgld': 'seed{}-lr{}-beta{:.1e}--wdecay{}'.format(seed, lr, beta, weight_decay),
        'adam': 'seed{}-lr{}-betas{}-{}-wdecay{}'.format(seed, lr, beta1, beta2, weight_decay),
        'amsgrad': 'seed{}-lr{}-betas{}-{}-wdecay{}'.format(seed, lr, beta1, beta2, weight_decay),
        'theopoula': 'seed{}-lr{}-eps{}-wdecay{}-beta{:.1e}'.format(seed, lr, eps, weight_decay, beta),
        'tusla': 'seed{}-lr{}-r{}-wdecay{}-beta{:.1e}'.format(seed, lr, r, weight_decay, beta),
    }[optimizer]
    if scheduler:
        name = name + '-scheduler{}-steplr{}-lrgamma{}'.format(scheduler_type, steplr, lr_gamma)

    if beta_annealing:
        name = name + '-beta_annealing{}-betagamma{}-betastep{}'.format(beta_annealing, beta_gamma, beta_step)

    return '{}-p{}-K{}-hs{}-bs{}-{}-{}-epochs{}-n_paths{}'.format(model, num_asset, num_step, hidden_size, batch_size, optimizer, name, epochs, num_path)

save = get_ckpt_name(model=args.asset_model, seed=args.seed, num_asset=args.num_asset, optimizer=args.optimizer, lr=args.lr,
                     eps=args.eps, num_step=num_step, weight_decay=args.weight_decay, lr_gamma=args.lr_gamma, steplr=args.steplr,
                     beta=args.beta, scheduler=args.scheduler, hidden_size=args.hidden_size, T_max=args.T_max, eta_min=args.eta_min,
                     batch_size=args.batch_size, scheduler_type=args.scheduler_type, epochs=args.epochs, patience=args.patience,
                     beta_annealing=args.beta_annealing, beta_gamma=args.beta_gamma, beta_step=args.beta_step, momentum=args.momentum, r=args.TUSLA_r,
                     num_path = args.num_path
                     )


## Building model

print('\n=> Building model.. on {%s}'%device)

net = Fullnet(state_size=state_size,
              num_asset=args.num_asset,
              num_step=num_step,
              R_f=R_f,
              u_gamma=args.u_gamma,
              D=D,
              asset_model=args.asset_model,
              hidden_size=args.hidden_size,
              act_fn=args.act_fn)
net.to(device)

print('\n==> Setting optimizer.. use {%s}'%args.optimizer)
print('setting: {}'.format(save))


optimizer = { 'sgld': SGLD(net.parameters(), lr=args.lr, beta=args.beta, weight_decay=args.weight_decay),
              'adam': ADAM(net.parameters(), lr=args.lr, weight_decay=args.weight_decay),
              'amsgrad': ADAM(net.parameters(), lr=args.lr, amsgrad=True, weight_decay=args.weight_decay),
              'theopoula': THEOPOULA(net.parameters(), lr=args.lr, eta=args.eta, beta=args.beta, eps=args.eps, weight_decay=args.weight_decay),
              'tusla': TUSLA(net.parameters(), lr=args.lr, r=args.TUSLA_r, beta=args.beta, weight_decay=args.weight_decay)
}[args.optimizer]




## Training
print('\n==> Start training ')

history = {'train_score': [],
           'test_score': [],
           'running_time': 0,
           'training_time': 0,
           'best_epoch': 0,
           }
#state = {}
best_score = 99999
training_time1 = 0

train_data = generate_path(args.asset_model, args.num_asset, args.num_path, time_step, R_f)


print(args.num_step, time_step, train_data.shape, np.mean(train_data), np.max(train_data), np.min(train_data), np.std(train_data), R_f)

def adjust_learning_rate(optimizer, epoch, steplr=50, lr_gamma=0.1):
    if epoch % steplr == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_gamma
        
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, verbose=True, min_lr=args.lr*0.001)
def train(epoch, net):
    global training_time1
    print('\n Epoch: %d'%epoch)
    net.train()
    train_score = 0

    train_data = generate_path(args.asset_model, args.num_asset, args.num_path, time_step, R_f)

    start_time = time.time()

    for batch_idx in range(num_batch):
        start = batch_idx * args.batch_size
        end = np.minimum((batch_idx + 1) * args.batch_size, args.num_path)

        samples = torch.FloatTensor(train_data[start:end]).to(device)
        optimizer.zero_grad()
        output = net(samples)
        score = torch.mean(output)

        score.backward()
        optimizer.step()

        train_score += score.item() * len(samples)


    history['train_score'].append(train_score/args.num_path)
    training_time1 += time.time() - start_time
    print(train_score/args.num_path)


def test(epoch, net):
    global best_score
    net.eval()
    test_score = 0

    with torch.no_grad():
        num_batch_test = int(np.ceil(num_path_test / batch_size_test))

        for batch_idx in range(num_batch_test):
            start = batch_idx * batch_size_test
            end = np.minimum((batch_idx + 1) * batch_size_test, num_path_test)

            samples = torch.FloatTensor(test_data[start:end]).to(device)

            output = net(samples)
            score = torch.mean(output)

            test_score += score.item() * len(samples)

        history['test_score'].append(test_score/num_path_test)
        print(test_score/num_path_test)

    if test_score/num_path_test < best_score:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'score': test_score/num_path_test,
            'epoch': epoch,
            'optim': optimizer.state_dict()
        }
        best_score = np.mean(test_score)

        if not os.path.isdir(args.ckpt_dir):
            os.mkdir(args.ckpt_dir)
        torch.save(state, os.path.join(args.ckpt_dir, save))
    return np.mean(test_score)


num_path_test = 100000#1000000
batch_size_test = 5000
test_data = generate_path(args.asset_model, args.num_asset, num_path_test, time_step, R_f)

for epoch in range(1, args.epochs+1):
    train(epoch, net)
    test_score = test(epoch, net)

    if args.scheduler:
        if args.scheduler_type == 'auto':
            scheduler.step(np.mean(test_score))
        elif args.scheduler_type == 'step':
            adjust_learning_rate(optimizer, epoch, args.steplr, args.lr_gamma)

    if (args.beta_annealing) & (epoch % args.beta_step == 0):
        for param_group in optimizer.param_groups:
            param_group['beta'] *= args.beta_gamma
#







##save results
print(save)
print('best score...', best_score)
print('running time:', time.time()-start)
print('training time:', training_time1)

history['best_score'] = best_score
history['running_time'] = time.time()-start
history['training_time'] = training_time1


state = torch.load(open(os.path.join(args.ckpt_dir, save), 'rb'))
print('best_epoch', state['epoch'])

history['best_epoch'] = state['epoch']


if not os.path.isdir(args.log_dir):
    os.mkdir(args.log_dir)
pkl.dump(history, open(os.path.join(args.log_dir, save), 'wb'))

