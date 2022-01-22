from replay_buffer import ReplayBuffer
from network import DeepQNetwork, DuelingDeepQNetwork
import numpy as np
import torch as T

class Agent:
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, algo=None, env_name=None, chkpt_dir='models', simple=False):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace # For the memory buffer
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.simple = simple

        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

    def choose_action(self, observation):
        raise NotImplementedError

    # Store the memory
    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    # Sample from the memory
    def sample_memory(self):
        state, action, reward, state_, done = self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(state_).to(self.q_eval.device)

        return states, actions, rewards, states_, dones

    # Replace q_eval net with q_next network when learn_step_counter is divisible by "replace_target_cnt"
    # Every "replace_target_cnt" number, network gets replaced
    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(
                self.q_eval.state_dict())  # Replaces the parameters of q_eval with the parameters of q_next

    # Decrement epsilon
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    # Load model weights of 2 networks
    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    # Save model weights of 2 networks
    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def learn(self):
        raise NotImplementedError

class DQNAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.q_eval = DeepQNetwork(self.lr, self.n_actions,
                                   input_dims=self.input_dims,
                                   name='q_eval',
                                   chkpt_dir=self.chkpt_dir,
                                   simple=self.simple)  # Network to evaluate Q-value
        self.q_next = DeepQNetwork(self.lr, self.n_actions,
                                   input_dims=self.input_dims,
                                   name='q_next',
                                   chkpt_dir=self.chkpt_dir,
                                   simple=self.simple)  # Network to predict the max q_value of the next state

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            # If random is greater than epsilon, we take the greedy action
            state = T.tensor(np.array([observation]), dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item() # Take the action that gives the max reward
        else:
            # If random is less than epsilon, we take a random action
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_).max(dim=1)[0] # [0] is just because of the way .max() is implemented

        dones = dones.type(T.bool)
        q_next[dones] = 0.0
        q_target = rewards + self.gamma * q_next

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

class DoubleDQNAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.q_eval = DeepQNetwork(self.lr, self.n_actions,
                                   input_dims=self.input_dims,
                                   name='q_eval',
                                   chkpt_dir=self.chkpt_dir,
                                   simple=self.simple)  # Network to evaluate Q-value
        self.q_next = DeepQNetwork(self.lr, self.n_actions,
                                   input_dims=self.input_dims,
                                   name='q_next',
                                   chkpt_dir=self.chkpt_dir,
                                   simple=self.simple)  # Network to predict the max q_value of the next state

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            # If random is greater than epsilon, we take the greedy action
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item() # Take the action that gives the max reward
        else:
            # If random is less than epsilon, we take a random action
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        # 2 options for when to start learning: 1. now (without having memory) (bail out if the mem_cntr is smaller than batch_size)
        #                                       2. after you fill up the memory (until then you randomly explore, requires a lot more time: random exploration each episode)

        # ALSO, we learn from the tuples in the memory (from random self.batch_size tuples in memory), not from the current observation (every time learning happens for "self.memory.mem_cntr" instances) if self.memory.mem_cntr < self.batch_size:
            # The agent does not learn for the first "self.batch_size" steps
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()  # Every step, we evaluate the current state (replaces the network only IF appropriate)
        states, actions, rewards, states_, dones = self.sample_memory()  # The agent takes actions state by state', but learning happens for only the "self.batch_size" tuples from the "self.memory.mem_cnt" recent tuples

        # q_pred is the q_value in the current state, q_pred is the max q_value of the next state
        indices = np.arange(self.batch_size)  # Because dims are batch_size x n_action

        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_)
        q_eval = self.q_eval.forward(states_)

        max_actions = T.argmax(q_eval, dim=1)

        dones = dones.type(T.bool)
        q_next[dones] = 0.0  # Everywhere where we remember that we ended, we add 0.0, because that is the value of the finishing state (and the finishing state will have q_target=rewards)

        q_target = rewards + self.gamma * q_next[indices, max_actions]

        # Train the "q_eval" network
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)  # We only train q_eval network, and then every "self.replace_target_cnt" times we copy the parameters
        loss.backward()
        self.q_eval.optimizer.step()

        self.learn_step_counter += 1
        self.decrement_epsilon()

class DuelingDQNAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.q_eval = DuelingDeepQNetwork(self.lr, self.n_actions,
                                          input_dims=self.input_dims,
                                          name='q_eval',
                                          chkpt_dir=self.chkpt_dir,
                                          simple=self.simple)  # Network to evaluate Q-value
        self.q_next = DuelingDeepQNetwork(self.lr, self.n_actions,
                                          input_dims=self.input_dims,
                                          name='q_next',
                                          chkpt_dir=self.chkpt_dir,
                                          simple=self.simple)  # Network to predict the max q_value of the next state

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            # If random is greater than epsilon, we take the greedy action
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            _, advantage = self.q_eval.forward(state)
            action = T.argmax(advantage).item() # Take the action that gives the max reward
        else:
            # If random is less than epsilon, we take a random action
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()

        indices = np.arange(self.batch_size)

        V_s, A_s = self.q_eval.forward(states)
        V_s_, A_s_ = self.q_next.forward(states_)

        q_pred = T.add(V_s, A_s - A_s.mean(dim=1, keepdim=True))[indices, actions]
        q_next = T.add(V_s_, A_s_ - A_s_.mean(dim=1, keepdim=True)).max(dim=1)[0]
        dones = dones.type(T.bool)
        q_next[dones] = 0.0

        q_target = rewards + self.gamma * q_next

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

class DuelingDoubleDQNAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.q_eval = DuelingDeepQNetwork(self.lr, self.n_actions,
                                          input_dims=self.input_dims,
                                          name='q_eval',
                                          chkpt_dir=self.chkpt_dir,
                                          simple=self.simple)  # Network to evaluate Q-value
        self.q_next = DuelingDeepQNetwork(self.lr, self.n_actions,
                                          input_dims=self.input_dims,
                                          name='q_next',
                                          chkpt_dir=self.chkpt_dir,
                                          simple=self.simple)  # Network to predict the max q_value of the next state

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            # If random is greater than epsilon, we take the greedy action
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            _, advantage = self.q_eval.forward(state)
            action = T.argmax(advantage).item()
        else:
            # If random is less than epsilon, we take a random action
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()

        indices = np.arange(self.batch_size)

        V_s, A_s = self.q_eval.forward(states)
        V_s_, A_s_ = self.q_next.forward(states_)
        V_s_eval, A_s_eval = self.q_eval.forward(states_)

        q_pred = T.add(V_s, A_s - A_s.mean(dim=1, keepdim=True))[indices, actions]
        q_next = T.add(V_s_, A_s_ - A_s_.mean(dim=1, keepdim=True))
        q_eval = T.add(V_s_eval, A_s_eval - A_s_eval.mean(dim=1, keepdim=True))

        max_actions = T.argmax(q_eval, dim=1)
        dones = dones.type(T.bool)
        q_next[dones] = 0.0

        q_target = rewards + self.gamma * q_next[indices, max_actions]

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

class RandomAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_list = np.arange(self.n_actions)

    def choose_action(self, _):
        action = np.random.choice(self.action_list)
        return action

    # Load model weights of 2 networks
    def load_models(self):
        pass

    # Save model weights of 2 networks
    def save_models(self):
        pass

    def learn(self):
        pass
