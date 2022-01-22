import numpy as np
import os
from matplotlib import pyplot as plt

def plot_figures(scores_info, epsilon_info, env_name, save_fig_dir, avg_constant=80, linewidth=0.8, title='Acrobot v1', reward_range=(-500,0)):
    # Setup multiple figures
    figure_file = os.path.join(save_fig_dir, 'results.png')
    fig, axs = plt.subplots(ncols=1, nrows=2)

    # Set Metadata of the figure
    axs_eps, axs_reward = axs
    axs_eps.set_title(title)
    plt.xlabel('Episodes')
    axs_reward.set_ylabel('Cumulative Reward')
    axs_eps.set_ylabel('Epsilon')

    axs_eps.set_ylim(0, 1)
    axs_reward.set_ylim(reward_range[0], reward_range[1])

    # Data of the figure
    for epsilons, (score, net_name) in zip(epsilon_info, scores_info):
        net_name = net_name.replace('_', ' ').title().replace('Dqn', 'DQN')
        moving_average_score = [np.mean(score[max(0, t-avg_constant): (t+1)]) for t in range(len(score))]
        axs_eps.plot(np.arange(len(epsilons)), epsilons)
        axs_reward.plot(np.arange(len(moving_average_score)), moving_average_score, label=net_name, linewidth=linewidth)

    # Save the figure
    plt.legend(loc=0)
    plt.savefig(figure_file)

# Has to be run once for each environment
if __name__ == '__main__':
    env_results_root_folder = "C:\\Users\\hajdi\\Desktop\\Radboud\\Neural Information Processing Systems\\Project\\Final Experiments\\Acrobot_v1_"
    env_results = [folder for folder in [os.path.join(env_results_root_folder, file_folder) for file_folder in os.listdir(env_results_root_folder)] if os.path.isdir(folder)]
    env_name = env_results_root_folder.split('\\')[-1]

    scores_info = []
    epsilon_info = []
    for dir in env_results:
        # Load scores and the network name
        score_file = os.path.join(dir, 'scores.npy')
        score = np.load(score_file)
        net_name = dir.split('\\')[-1]

        # Save scores and network name to the list
        scores_info.append((score, net_name))

        # Load epsilon files
        epsilon_file = os.path.join(dir, 'epsilons.npy')
        epsilons = np.load(epsilon_file)
        epsilon_info.append(epsilons)

    # Plot the results and save them in a figure
    plot_figures(scores_info, epsilon_info, env_name, env_results_root_folder)


