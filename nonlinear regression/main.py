## package
import torch
import torch.optim as optim
import os
import argparse
import numpy as np
import pickle as pkl
import pandas as pd
from utils import get_nll_S
from models import S_net
import matplotlib.pyplot as plt
from optimizers import ADAM, SGLD, TUSLA, THEOPOULA
from utils import FS_Dataset
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import time

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

## parsing
parser = argparse.ArgumentParser('Nonlinear gamma regression')
parser.add_argument('--s_dist', default='Gamma', type=str, help='severity distribution')
parser.add_argument('--seed', default=111, type=int)
parser.add_argument('--batch_size', default=128, type=int, help='batch_size')
parser.add_argument('--epochs', default=100, type=int, help='# of epochs')
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--act_fn', default='leaky_relu', type=str)
parser.add_argument('--hidden_size', default=50, type=int, help='number of neurons')
parser.add_argument('--optimizer_name', default='adam', type=str)
parser.add_argument('--with_BN', default=False)
parser.add_argument('--with_DO', default=False)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--p_dropout', default=0.25, type=float)
parser.add_argument('--eta', default='0', type=float)
parser.add_argument('--beta', default='1e12', type=float)
parser.add_argument('--eps', default=1e-2, type=float)
parser.add_argument('--lr_gamma', default=0.1, type=float)
parser.add_argument('--eps_gamma', default=1, type=float)
parser.add_argument('--log_dir', default='./logs/', type=str)
parser.add_argument('--ckpt_dir', default='./ckpt/', type=str)
parser.add_argument('--weight_decay', default=5e-5, type=str)
parser.add_argument('--when', default=[25], type=int)

args = parser.parse_args()
torch.manual_seed(args.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

## data preparation
print(args.with_BN, args.with_DO)
file = open('./data/refined_data_category.pkl', 'rb')
data = pkl.load(file)

num_features = (data.shape[1] - 3)

train_data, test_data = train_test_split(data, random_state=0, test_size=0.3)

train_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)

trainloader = torch.utils.data.DataLoader(FS_Dataset(train_data), batch_size=args.batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(FS_Dataset(test_data), batch_size=args.batch_size, shuffle=False)



num_data = len(train_data)
num_batch = np.ceil(num_data / args.batch_size)

def get_ckpt_name(seed=111, optimizer='sgld', lr=0.1, momentum=0.9,
                  beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=5e-4,
                  beta=1e12, r=0.5, epochs=50):
    name = {
        'sgld': 'seed{}-lr{}-wdecay{}'.format(seed, lr,weight_decay),
        'adam': 'seed{}-lr{}-betas{}-{}-wdecay{}'.format(seed, lr, beta1, beta2,weight_decay),
        'amsgrad': 'seed{}-lr{}-betas{}-{}-wdecay{}'.format(seed, lr, beta1, beta2, weight_decay),
        'theopoula': 'seed{}-lr{}-eps{}-wdecay{}-beta{:.1e}'.format(seed, lr, eps, weight_decay, beta),
        'tusla': 'seed{}-lr{}-r{}-wdecay{}-beta{:.1e}'.format(seed, lr, r, weight_decay, beta),
    }[optimizer]
    return '{}-{}-epoch{}'.format(optimizer, name, epochs)
save = get_ckpt_name(seed=args.seed, optimizer=args.optimizer_name, lr=args.lr, eps=args.eps,
                          weight_decay=args.weight_decay, beta=args.beta, epochs=args.epochs)
print(save)
start = time.time()
print('==> Start severity model..')

## Preparing data and dataloader
print('==> Preparing data..')

train_data = train_data.loc[train_data['freq']>0].reset_index(drop=True)
test_data = test_data.loc[test_data['freq']>0].reset_index(drop=True)
print('==> train data..shape: ', train_data.shape)
print('==> test data..shape: ', test_data.shape)

print(train_data.shape, train_data['freq'].value_counts())
print(test_data.shape, test_data['freq'].value_counts())



trainloader = torch.utils.data.DataLoader(FS_Dataset(train_data), batch_size=args.batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(FS_Dataset(test_data), batch_size=args.batch_size, shuffle=False)

best_nll = 999
state = []
best_epoch = 0
num_data = len(train_data)
num_batch = np.ceil(num_data/args.batch_size)


## Severity model, optimizer

print('==> Building model.. on {%s}'%device)

S_model = S_net(input_size=num_features,
                hidden_size=args.hidden_size,
                act_fn=args.act_fn,
                with_BN=args.with_BN,
                with_DO=args.with_DO,
                p_dropout=args.p_dropout
                )
S_model.to(device)

print('==> Set optimizer.. use {%s}'%args.optimizer_name)

optimizer = { 'sgld': SGLD(S_model.parameters(), lr=args.lr, beta=args.beta, weight_decay=args.weight_decay),
              'adam': optim.Adam(S_model.parameters(), lr=args.lr, weight_decay=args.weight_decay),
              'amsgrad': optim.Adam(S_model.parameters(), lr=args.lr, amsgrad=True, weight_decay=args.weight_decay),
              'theopoula': THEOPOULA(S_model.parameters(), lr=args.lr, eta=args.eta, beta=args.beta, eps=args.eps, weight_decay=args.weight_decay),
              'tusla': TUSLA(S_model.parameters(), lr=args.lr, beta=args.beta, weight_decay=args.weight_decay)
}[args.optimizer_name]

history = {'train_nll': [],
           'test_nll': [],
           'running_time': [],
           'best_epoch': 0,
           }

## Training - severity model
print('==> Start training - Severity model')
loss_mse = torch.nn.MSELoss()
hist_train_nll = []
hist_test_nll = []
state = {}

def adjust_learning_rate(optimizer, epoch, when=[150], lr_gamma=0.1, eps_gamma=0.1):
    for param_group in optimizer.param_groups:
        if epoch in when:
            param_group['lr'] *= lr_gamma
            if args.optimizer_name == 'theopoula':
                param_group['eps'] *= eps_gamma
        current_lr = param_group['lr']
        print('current_lr: ', current_lr)


def S_train(epoch, net):
    global hist_train_nll
    print('\n Epoch: %d'%epoch)
    net.train()
    train_nll = []
    for batch_idx, samples in enumerate(trainloader):
        samples = samples.to(device)
        optimizer.zero_grad()

        cov = samples[:, :num_features]
        y = samples[:, -1]
        output = net(cov)

        nll = get_nll_S(args.s_dist, y, output)
        nll.backward()
        optimizer.step()

        train_nll += [nll.item()]
        if batch_idx % 200 ==0:
            print('TRAIN:BATCH %04d/%04d | NLL %.4f | Dispersion %.4f'
              %(batch_idx, num_batch, np.mean(train_nll), output[1].cpu().data))

    print('TRAIN: EPOCH %04d/%04d | NLL %.4f | Dispersion %.4f'
          % (epoch, args.epochs, np.mean(train_nll), output[1].cpu().data))

    hist_train_nll += [np.mean(train_nll)]
    history['train_nll'].append(np.mean(train_nll))

def S_test(epoch, net):
    global state, best_nll, hist_test_nll, best_epoch
    net.eval()
    test_nll =[]

    with torch.no_grad():
        for batch_idx, samples in enumerate(testloader):
            samples = samples.to(device)

            cov = samples[:, :num_features]
            y = samples[:, -1]

            output = net(cov)
            nll = get_nll_S(args.s_dist, y, output)

            test_nll += [nll.item()]

        print('TEST: NLL %.4f '% (np.mean(test_nll)))


    if np.mean(test_nll) < best_nll:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': np.mean(test_nll),
            'epoch': epoch,
            'optim': optimizer.state_dict()
        }
        best_nll = np.mean(test_nll)

    hist_test_nll += [np.mean(test_nll)]
    history['test_nll'].append(np.mean(test_nll))

for epoch in range(1, args.epochs+1):
    adjust_learning_rate(optimizer, epoch, when=args.when, lr_gamma=args.lr_gamma,
                         eps_gamma=args.eps_gamma)
    S_train(epoch, S_model)
    S_test(epoch, S_model)

# plt.figure(1)
# plt.title('train_nll')
# plt.plot(range(1, args.epochs+1), hist_train_nll, label='train')
# plt.xlabel('epochs')
# plt.ylabel('nll')
# plt.figure(2)
# plt.title('test_nll')
# plt.plot(range(1, args.epochs+1), hist_test_nll, label='test')
# plt.xlabel('epochs')
# plt.ylabel('nll')
# plt.legend()

history['running_time'] = time.time()-start
history['best_epoch'] = state['epoch']

#save result
if not os.path.isdir(args.log_dir):
    os.mkdir(args.log_dir)
pkl.dump(history, open(os.path.join(args.log_dir, save), 'wb'))

# save model
if not os.path.isdir(args.ckpt_dir):
    os.mkdir(args.ckpt_dir)
torch.save(state, os.path.join(args.ckpt_dir, save))


# plt.show()

