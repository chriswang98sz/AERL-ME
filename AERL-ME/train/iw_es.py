from __future__ import absolute_import, division, print_function
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch


import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.autograd import Variable

from model import ES
import time


def do_rollouts(args, models, random_seeds, return_queue, env):
    """
    For each model, do a rollout. Supports multiple models per thread but
    don't do it -- it's inefficient (it's mostly a relic of when I would run
    both a perturbation and its antithesis on the same thread).
    """
    all_returns = []
    all_num_frames = []
    for model in models:
        state = env.reset()
        state = torch.from_numpy(state)
        this_model_return = 0
        this_model_num_frames = 0
        # Rollout
        for step in range(args.max_episode_length):
            state = state.float()
            state = state.view(1, env.observation_space.shape[0])
            with torch.no_grad():
                state = Variable(state)
            logit = model(state)
            prob = F.softmax(logit, dim=1)
            action = prob.max(1)[1].data.numpy()
            next_state, reward, done, _ = env.step(action[0])
            state = next_state
            this_model_return += reward
            this_model_num_frames += 1
            if done:
                break
            state = torch.from_numpy(state)
        all_returns.append(this_model_return)
        all_num_frames.append(this_model_num_frames)
    return_queue.put((random_seeds, all_returns, all_num_frames))


def perturb_model(args, model, random_seed, env):
    """
    Modifies the given model with a pertubation of its parameters,
    as well as the negative perturbation, and returns both perturbed
    models.
    """
    new_model = ES(env.observation_space.shape[0],
                   env.action_space)
    new_model.load_state_dict(model.state_dict())
    np.random.seed(random_seed)
    for (k, v)in new_model.es_params():
        eps = np.random.normal(0, 1, v.size())
        v += torch.from_numpy(args.sigma * eps).float()
    
    return [new_model]

def gradient_update(args, synced_model, returns,random_seeds,
                    num_eps, num_frames, unperturbed_results):

    def fitness_shaping():
        """
        A rank transformation on the rewards, which reduces the chances
        of falling into local optima early in training.
        """
        sorted_returns_backwards = sorted(returns)[::-1]
        lamb = len(returns)
        shaped_returns = []
        denom = sum([max(0, math.log(lamb / 2 + 1, 2) -
                         math.log(sorted_returns_backwards.index(r) + 1, 2))
                     for r in returns])
        for r in returns:
            num = max(0, math.log(lamb / 2 + 1, 2) -
                      math.log(sorted_returns_backwards.index(r) + 1, 2))
            shaped_returns.append(num / denom + 1 / lamb)
        return shaped_returns
    batch_size = len(returns)
    assert batch_size == args.n
    assert len(random_seeds) == batch_size
    shaped_returns = fitness_shaping()
    print('Episode num: %d\n'
          'Sigma: %f\n'
          'Total num frames seen: %d\n'
          'Unperturbed reward: %f' %
          (num_eps,
           args.sigma,num_frames,
           unperturbed_results))

    for i in range(args.n):
        np.random.seed(random_seeds[i])
        reward = shaped_returns[i]
        for k,v in synced_model.es_params():
            eps = np.random.normal(0,1,v.size())

            v += torch.from_numpy(args.lr/(args.n*args.sigma)*
                                  (reward*eps)).float()
        
    return synced_model,shaped_returns

def delta_calculator(args,synced_model,base_model,env):
    delta_model = ES(env.observation_space.shape[0],
                     env.action_space)
    for (k_b, v_b), (k_s, v_s), (k_d, v_d) in zip(base_model.es_params(),
                                                  synced_model.es_params(), delta_model.es_params()):

        v_d = v_b - v_s

    return delta_model

def gradient_update_iw(args,synced_model,delta_model,shaped_returns,random_seeds,
                    num_eps, num_frames, unperturbed_results,importance_weights):
    batch_size = len(shaped_returns)
    assert batch_size == args.n
    assert len(random_seeds) == batch_size
    print('Episode num: %d\n'
          'Sigma: %f\n'
          'Total num frames seen: %d\n'
          'Unperturbed reward: %f' %
          (num_eps,
           args.sigma, num_frames,
           unperturbed_results))
    for i in range(args.n):
        np.random.seed(random_seeds[i])
        reward = shaped_returns[i]
        importance_weight = importance_weights[i]
        for (k, v),(k_d,v_d) in zip(synced_model.es_params(), delta_model.es_params()):
            eps = np.random.normal(0, 1, v.size())
            eps = eps * args.sigma + v_d.numpy()
            v += torch.from_numpy(args.lr/(args.sigma*args.sigma)*reward*eps*importance_weight).float()
    return synced_model

def importance_weight(args,delta_model,random_seeds, return_queue,shaped_returns):
    np.random.seed(random_seeds)
    shaped_return = []
    shaped_return.append(shaped_returns)
    importance_weight = []
    iw = 0
    for (k,v) in delta_model.es_params():
        eps = np.random.normal(0, 1, v.size())*args.sigma
        iw -= torch.sum(v*eps).numpy()/(args.sigma*args.sigma)
    iw = math.exp(iw)
    importance_weight.append(iw)
    return_queue.put((random_seeds, importance_weight,shaped_return))

def generate_seeds_and_models(args, synced_model, env):
    """
    Returns a seed and 2 perturbed models
    """
    np.random.seed()
    random_seed = np.random.randint(2 ** 30)
    models = perturb_model(args, synced_model, random_seed, env)
    return random_seed, models


def train_loop_iw_es(args, synced_model, env):
    Unperturbed_reward=[]
    def flatten(raw_results, index):
        notflat_results = [result[index] for result in raw_results]
        return [item for sublist in notflat_results for item in sublist]

    print("Using baseline to train.")
    print("Num params in network %d" % synced_model.count_parameters())
    num_eps = 0
    total_num_frames = 0
    start_time = time.time()
    for gradient_updates in range(args.max_gradient_updates):
        processes = []
        return_queue = mp.Queue()

        if gradient_updates % args.k == 0:
            all_seeds, all_models = [], []
            # Generate a perturbation and its antithesis
            for j in range(args.n):
                random_seed, models = generate_seeds_and_models(args,synced_model,env)
                    # Add twice because we get two models with the same seed
                all_seeds.append(random_seed)
                all_models += models
            assert len(all_seeds) == len(all_models)
            while all_models:
                perturbed_model = all_models.pop()
                seed = all_seeds.pop()
                p = mp.Process(target=do_rollouts, args=(args,
                                                         [perturbed_model],
                                                         [seed],
                                                         return_queue,
                                                         env))
                p.start()
                processes.append(p)
            assert len(all_seeds) == 0
            # Evaluate the unperturbed model as well
            p = mp.Process(target=do_rollouts, args=(args, [synced_model],
                                                     ['dummy_seed'],
                                                     return_queue, env))
            p.start()
            processes.append(p)
            for p in processes:
                p.join()
            raw_results = [return_queue.get() for p in processes]
            seeds, results, num_frames = [flatten(raw_results, index)for index in [0, 1, 2]]
            # Separate the unperturbed results from the perturbed results
            unperturbed_index = seeds.index('dummy_seed')
            seeds.pop(unperturbed_index)
            unperturbed_results = results.pop(unperturbed_index)
            Unperturbed_reward.append(unperturbed_results)
            _ = num_frames.pop(unperturbed_index)
            total_num_frames += sum(num_frames)
            num_eps += len(results)
            base_model = ES(env.observation_space.shape[0],
                           env.action_space)
            base_model.load_state_dict(synced_model.state_dict())

            synced_model,shaped_returns = gradient_update(args, synced_model, results, seeds, num_eps, total_num_frames,
                                           unperturbed_results)

            delta_model = delta_calculator(args,synced_model,base_model,env)
            print('Time: %.1f\n' % (time.time() - start_time))
        else:
            all_seeds = seeds
            while all_seeds:
                seed = all_seeds.pop()
                shaped_return = shaped_returns.pop()
                p = mp.Process(target=importance_weight, args=(args,
                                                         delta_model,
                                                         [seed],
                                                         return_queue,shaped_return
                                                         ))
                p.start()
                processes.append(p)
            assert len(all_seeds) == 0
            p = mp.Process(target=do_rollouts, args=(args, [synced_model],
                                                     ['dummy_seed'],
                                                     return_queue, env))
            p.start()
            processes.append(p)
            for p in processes:
                p.join()
            raw_results = [return_queue.get() for p in processes]
            seeds, importance_weights,shaped_returns = [flatten(raw_results, index)for index in [0, 1, 2]]
            unperturbed_index = seeds.index('dummy_seed')
            seeds.pop(unperturbed_index)
            unperturbed_results = importance_weights.pop(unperturbed_index)
            Unperturbed_reward.append(unperturbed_results)
            _ = shaped_returns.pop(unperturbed_index)


            importance_weights/=np.sum(importance_weights)

            num_eps += len(importance_weights)

            gradient_update_iw(args, synced_model, delta_model, shaped_returns, seeds,
                               num_eps, total_num_frames, unperturbed_results, importance_weights)
            delta_model = delta_calculator(args, synced_model, base_model, env)
            print('Time: %.1f\n' % (time.time() - start_time))
    print(Unperturbed_reward)
    plt.plot(Unperturbed_reward)
    plt.show()