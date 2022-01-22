from utils import plot_learning_curve, save_scores
import agent as Agents
import numpy as np
from make_environment import make_environment
import os

def train(args, dir_name):
    env = make_environment(args)
    best_score = -np.inf  # Here we put the worst possible score we can get (in the case of pong, it is -inf)
    load_checkpoint = args.load_checkpoint # Don't load, we are training (this is bool for train/test)
    n_games = args.n_games

    agent_ = getattr(Agents, args.algo)
    agent = agent_(gamma=args.gamma, epsilon=args.epsilon, lr=args.lr,
                     input_dims=(env.observation_space.shape),
                     n_actions=env.action_space.n, mem_size=args.max_mem, eps_min=args.eps_min,
                     batch_size=args.bs, replace=args.replace, eps_dec=args.eps_dec,
                     chkpt_dir=os.path.join(dir_name, 'models'), algo=args.algo,
                     env_name=args.env, simple=args.simple) # If you have less RAM, mem_size should be smaller

    if load_checkpoint:
        agent.load_models()

    n_steps = 0
    scores, eps_history, steps = [], [], []

    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()

        # Episode
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward

            if not load_checkpoint:
                # If you arent training, you should learn
                agent.store_transition(observation, action, reward, observation_, int(done))
                agent.learn()

            observation = observation_
            n_steps += 1

        scores.append(score)
        steps.append(n_steps)

        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score: ', score,
              'avg score %.1f best score %.1f epsilon %.2f epsiode_score %.5f' % (avg_score, best_score, agent.epsilon, score / n_steps))

        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)

    plot_learning_curve(steps, scores, eps_history, dir_name)
    save_scores(scores, eps_history, dir_name)
