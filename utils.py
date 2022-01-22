from matplotlib import pyplot as plt
import numpy as np
import os
from datetime import datetime

# Save scores per epipsode and epislons per episode
def save_scores(scores, epsilons, dir_name):
    score_file = os.path.join(dir_name, 'scores.npy')
    with open(score_file, 'wb') as f:
        np.save(f, scores)

    epsilon_file = os.path.join(dir_name, 'epsilons.npy')
    with open(epsilon_file, 'wb') as f:
        np.save(f, epsilons)

    print("Scores have been saved!")

def plot_learning_curve(x, scores, epsilons, dir_name):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis="x", color="C0")
    ax.tick_params(axis="y", color="C0")

    # Calculate running average (every timestep, avg over all prev actions)
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-100): (t+1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors='C1')

    figure_file = os.path.join(dir_name, f'score_{int(scores[-1])}.png')
    plt.savefig(figure_file)

    print("Sanity-check loss function has been saved!")

def create_experiment_directory(root_exp_dir='experiments'):
    # Create experiments directory if one does not exist
    if not os.path.exists(root_exp_dir):
        os.makedirs(root_exp_dir)

    # Create directory for current experiment
    dir_name = datetime.now().strftime('%m_%d__%H_%M_%S')
    dir_name = os.path.join(root_exp_dir, dir_name)

    # Check if this experiment already has directory (if you ran experiment 2 times in the same second), else create it
    if os.path.exists(dir_name):
        raise OSError("Wait 1 second and then restart experiment.")
    else:
        os.makedirs(dir_name)

    # Make needed directories
    #os.makedirs(dir_name + 'tmp/dqn')
    os.makedirs(os.path.join(dir_name, 'models'))
    #os.makedirs(os.path.join(dir_name, 'plots'))

    return dir_name

def pretty_time_delta(seconds):
    sign_string = '-' if seconds < 0 else ''
    seconds = abs(int(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days > 0:
        return '%s%dd%dh%dm%ds' % (sign_string, days, hours, minutes, seconds)
    elif hours > 0:
        return '%s%dh%dm%ds' % (sign_string, hours, minutes, seconds)
    elif minutes > 0:
        return '%s%dm%ds' % (sign_string, minutes, seconds)
    else:
        return '%s%ds' % (sign_string, seconds)