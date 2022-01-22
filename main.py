import os.path
import torch.cuda
import argparse

import utils
from train import train
import json
import time
from utils import create_experiment_directory, pretty_time_delta

def parse_arguments(dir_name):
    # Instantiate dicts
    algo_dict = {
        '0': 'DQNAgent',
        '1': 'DoubleDQNAgent',
        '2': 'DuelingDQNAgent',
        '3': 'DuelingDoubleDQNAgent',
        '4': 'RandomAgent'
    }

    # TO THE REPORT: 1 and 3/4
    env_dict = {
        # Simple
        '0': 'CartPole-v1',
        '1': 'Acrobot-v1',
        '2': 'MountainCar-v0',
        # Hard
        '3': 'TennisNoFrameskip-v0',
        '4': 'PongNoFrameskip-v4',
        '5': 'SuperMarioBros-v0'
    }

    parser = argparse.ArgumentParser(description='Neural Information Processing Systems Project')
    parser.add_argument('-n_games', type=int, default=1, help='Number of games.')
    parser.add_argument('-lr', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('-gamma', type=float, default=0.99, help='Discount factor for reward decay.')
    parser.add_argument('-epsilon', type=float, default=1., help='Starting value for epsilon.')
    parser.add_argument('-eps_dec', type=float, default=1e-5, help='Decay for epsilon greedy action selection.')
    parser.add_argument('-eps_min', type=float, default=0.1, help='Min value for epsilon.')
    parser.add_argument('-max_mem', type=int, default=50_000, help='Maximum size for memory replay buffer.')
    parser.add_argument('-stack', type=int, default=4, help='Num of frames to stack for environment.')
    parser.add_argument('-bs', type=int, default=32, help='Batch size for replay memory sampling.')
    parser.add_argument('-replace', type=int, default=1000, help='Interval for replacing target network.')
    parser.add_argument('-env', type=str, default='5', help='\n'.join([k+':'+v for k,v in env_dict.items()]))
    parser.add_argument('-load_checkpoint', type=int, default=0, help='Weather you want to load model (True=testing).')
    parser.add_argument('-algo', type=str, default='0', help='\n'.join([k+':'+v for k,v in algo_dict.items()]))
    parser.add_argument('-print_time', type=int, default=1, help='Do you want to print the time it takes to run the algorithm?')
    #parser.add_argument('-path', type=str, default='tmp/', help='Path for model loading/saving.')

    parser.add_argument('-save_parameters', type=int, default=1, help='Should I save these parameters or load existing ones?') # REQUIRED ARGUMENT!!!
    parser.add_argument('-parameters_file', type=str, default='parameters.txt', help='File to where I should save parameters or from where I should load them.')

    args = parser.parse_args()

    # Convert algorithm IDs to names
    args.algo = algo_dict[args.algo] if args.algo in algo_dict.keys() else args.algo
    if args.algo not in algo_dict.values():
        raise NotImplementedError("The algorithm you are trying to use is not implemented.")

    # Convert environment IDs to names
    args.env = env_dict[args.env] if args.env in env_dict.keys() else args.env
    if args.env not in env_dict.values():
        raise NotImplementedError("The environment you are trying to use is not implemented.")

    # Do you need to use simple network (MLP) or hard network (CNN)
    t = 3
    args.simple = True if any([v==args.env for k, v in env_dict.items() if int(k) < t]) else False

    # If model is being saved, save it to a new experiment directory
    #if not args.load_checkpoint:
    #    args.path = dir_name + args.path

    # Save or Load the parameters from argparse
    if args.save_parameters:
        args.parameters_file = os.path.join(dir_name, args.parameters_file)
        with open(args.parameters_file, 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    else:
        if not os.path.exists(args.parameters_file):
            raise FileNotFoundError('Parameters file does not exist.')

        with open(args.parameters_file, 'r') as f:
            args.__dict__ = json.load(f)

    return args

if __name__ == '__main__':
    dir_name = create_experiment_directory()
    args = parse_arguments(dir_name)

    print("USING GPU: ", torch.cuda.get_device_name())
    start = time.time()
    train(args, dir_name)
    duration = time.time() - start
    print("DURATION:", pretty_time_delta(duration))