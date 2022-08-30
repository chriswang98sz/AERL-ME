from __future__ import absolute_import, division, print_function

import gym
import argparse

import torch

import warnings

from model import ES

from train.AERL_ME import train_loop_aerl_me
from train.cesan import train_loop_ces_an
from train.cesmeps import train_loop_ces_meps
from train.mepsan import train_loop_meps_an
from train.es import train_loop_es
from train.iw_es import train_loop_iw_es
parser = argparse.ArgumentParser(description='ES')
parser.add_argument('--env-name', default='CartPole-v1',
                    metavar='ENV', help='environment')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--sigma', type=float, default=0.25, metavar='SD',
                    help='noise standard deviation')
parser.add_argument('--inisigma', type=float, default=0.05, metavar='SD',
                    help='noise standard deviation')
parser.add_argument('--tem', type=float, default=0.1, metavar='TEM',
                    help='temperature')
parser.add_argument('--n', type=int, default=40, metavar='N',
                    help='batch size, must be even')
parser.add_argument('--max-episode-length', type=int, default=10000,
                    metavar='MEL', help='maximum length of an episode')
parser.add_argument('--max-gradient-updates', type=int, default=100,
                    metavar='MGU', help='maximum number of updates')
parser.add_argument('--capacity', type=int, default=10000,
                    help='memory_capacity')
parser.add_argument('--delta', type=float, default=0.005,
                    help='delta')
parser.add_argument('--alpha', type=float, default=1.01,
                    help='alpha')
parser.add_argument('--trainmodel', type=int, default=1,
                    help='0:AERL-ME 1:ES 2:MEPS+AN 3:CES+AN 4:CES+MEPS')
parser.add_argument('--k', type=int, default=1)
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    args = parser.parse_args()
    assert args.n % 2 == 0
    args.inisigma = args.sigma
    env = gym.make(args.env_name)
    synced_model = ES(env.observation_space.shape[0],
                      env.action_space)
    for param in synced_model.parameters():
        param.requires_grad = False
    if args.trainmodel == 0:
        train_loop_aerl_me(args, synced_model, env)
    elif args.trainmodel == 1:
        train_loop_es(args, synced_model, env)
    elif args.trainmodel == 2:
        train_loop_meps_an(args, synced_model, env)
    elif args.trainmodel == 3:
        train_loop_ces_an(args, synced_model, env)
    elif args.trainmodel == 4:
        train_loop_ces_meps(args, synced_model, env)
    elif args.trainmodel == 5:
        train_loop_iw_es(args, synced_model, env)