from __future__ import absolute_import, division, print_function
import os
import math
import numpy as np

import torch


import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.autograd import Variable

from model import ES
import time


def do_rollouts(args, models, random_seeds, return_queue, env, are_negative):
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
    return_queue.put((random_seeds, all_returns, all_num_frames, are_negative))


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

def gradient_update(args, synced_model, returns,random_seeds, neg_list,
                    num_eps, num_frames, unperturbed_results,env):

    def fitness_shaping():
        """
        A rank transformation on the rewards, which reduces the chances
        of falling into local optima early in training.
        """
        sorted_returns_backwards = sorted(returns)[::-1]
        lamb = len(returns)
        shaped_returns = []
        denom = sum([math.log(lamb, 2) - math.log(sorted_returns_backwards.index(r) + 1, 2)
                     for r in returns])
        for r in returns:
            num = math.log(lamb, 2) - math.log(sorted_returns_backwards.index(r) + 1, 2)
            shaped_returns.append(num / denom + 1 / lamb)
        return shaped_returns
    batch_size = len(returns)
    assert batch_size == args.n
    assert len(random_seeds) == batch_size
    shaped_returns = fitness_shaping()
    print('Episode num: %d\n'
          'Average reward: %f\n'
          'Max reward: %f\n'
          'Sigma: %f\n'
          'Total num frames seen: %d\n'
          'Unperturbed reward: %f' %
          (num_eps, np.mean(returns),max(returns),
           args.sigma,num_frames,
           unperturbed_results))

    for i in range(args.n):
        np.random.seed(random_seeds[i])
        multiplier = -1 if neg_list[i] else 1
        reward = shaped_returns[i]

        for k,v in synced_model.es_params():
            eps = np.random.normal(0,1,v.size())
            v += torch.from_numpy(args.lr/(args.n* 0.05)*
                                  (reward*multiplier*eps)).float()
    return synced_model



def generate_seeds_and_models(args, synced_model, env):
    """
    Returns a seed and 2 perturbed models
    """
    np.random.seed()
    random_seed = np.random.randint(2 ** 30)
    two_models = perturb_model(args, synced_model, random_seed, env)
    return random_seed, two_models


def train_loop_es(args, synced_model, env):

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
        all_seeds, all_models = [], []
        # Generate a perturbation and its antithesis
        for j in range(int(args.n)):
            random_seed, two_models = generate_seeds_and_models(args,
                                                                synced_model,
                                                                env)
            # Add twice because we get two models with the same seed
            all_seeds.append(random_seed)
            all_models += two_models

        assert len(all_seeds) == len(all_models)

        # Keep track of which perturbations were positive and negative
        # Start with negative true because pop() makes us go backwards
        is_negative = False
        # Add all peturbed models to the queue
        while all_models:
            perturbed_model = all_models.pop()
            seed = all_seeds.pop()
            p = mp.Process(target=do_rollouts, args=(args,
                                                     [perturbed_model],
                                                     [seed],
                                                     return_queue,
                                                     env,
                                                     [is_negative]))
            p.start()
            processes.append(p)

            #is_negative = not is_negative
        assert len(all_seeds) == 0
        # Evaluate the unperturbed model as well
        p = mp.Process(target=do_rollouts, args=(args, [synced_model],
                                                 ['dummy_seed'],
                                                 return_queue, env,
                                                 ['dummy_neg']))
        p.start()
        processes.append(p)
        for p in processes:
            p.join()
        raw_results = [return_queue.get() for p in processes]
        seeds, results, num_frames, neg_list = [flatten(raw_results, index)
                                                                      for index in [0, 1, 2, 3]]
        # Separate the unperturbed results from the perturbed results
        _ = unperturbed_index = seeds.index('dummy_seed')
        seeds.pop(unperturbed_index)
        unperturbed_results = results.pop(unperturbed_index)
        _ = num_frames.pop(unperturbed_index)
        _ = neg_list.pop(unperturbed_index)

        total_num_frames += sum(num_frames)
        num_eps += len(results)
        args.sigma = max(0.005,0.25-(gradient_updates) *0.25/50)
        synced_model = gradient_update(args, synced_model, results, seeds,
                                       neg_list, num_eps, total_num_frames,
                                       unperturbed_results,env)
        print('Time: %.1f\n' % (time.time()-start_time))