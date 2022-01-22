import gym
import numpy as np
import cv2
import collections

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# gym.Wrapper is for "step" and "reset", and gym.ObservationWrapper is for "observation"
class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, shape, env=None):
        super().__init__(env)
        self.shape = (shape[2], shape[0], shape[1]) # The desired shape for PyTorch (c,h,w)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=self.shape, dtype=np.float32) #

    def observation(self, obs):
        new_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY) # Turn observation to grayscale (color doesn't convey information here)
        resized_screen = cv2.resize(new_frame, self.shape[1:], interpolation=cv2.INTER_AREA) # Resize the image to (h,w) shape
        new_obs = np.array(resized_screen, dtype=np.uint8).reshape(self.shape) # Turn obs to numpy
        new_obs = new_obs / 255.0 # Scale the obs

        return new_obs

class RepeatActionAndMaxFrame(gym.Wrapper):
    def __init__(self, env=None, repeat=4, clip_reward=False, no_ops=0, fire_first=False):
        super().__init__(env)
        self.repeat = repeat # Each action will be repeated "self.repeat" times
        self.shape = env.observation_space.low.shape  # Shape of the enivronment observation
        self.frame_buffer = np.zeros_like(
            (2, self.shape))  # "np.zeros_like" returns the array of the same shape as a passed array
        # Used to keep track of our 2 (hopefully the same) observations
        self.clip_reward = clip_reward
        self.no_ops = no_ops
        self.fire_first = fire_first

    # Step in the environment repeats the action for "repeat" times
    def step(self, action):
        t_reward = 0.0
        done = False
        for i in range(self.repeat):
            obs, reward, done, info = self.env.step(action) # Do action for each of the observations in self.repeat
            if self.clip_reward:
                reward = np.clip(np.array([reward]), -1, 1)[0] # Clips reward between -1 and +1
            t_reward += reward
            idx = i % self.frame_buffer.shape[0] # Even though we repeat action 4 times, we only want to save 2 observations
            self.frame_buffer[idx] = obs  # Always keeps only 2 observations
            if done:
                break

        max_frame = np.maximum(self.frame_buffer[0],
                               self.frame_buffer[1])  # The frame (observation), that is passed back (the max frame)
        return max_frame, t_reward, done, info

    def reset(self):
        obs = self.env.reset()
        no_ops = np.random.randint(self.no_ops) + 1 if self.no_ops > 0 else 0

        # Number of operations after the env is reseted
        for _ in range(no_ops):
            _, _, done, _ = self.env.step(0)
            if done:
                self.env.reset()

        # Fire first
        if self.fire_first:
            assert self.env.unwrapped.get_action_meanings()[1] == 'FIRE'
            obs, _, _, _ = self.env.step(1)

        self.frame_buffer = np.zeros_like((2, self.shape))  # Puts the first observation into the frame buffer
        self.frame_buffer[0] = obs

        return obs

# Stacks 4 ("repeat" parameter) most recent frames
class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, repeat):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            env.observation_space.low.repeat(repeat, axis=0),
            env.observation_space.high.repeat(repeat, axis=0),
            dtype=np.float32
        ) # Observation space
        self.stack = collections.deque(maxlen=repeat) # Define stack (takes the length of stack as arg)

    def reset(self):
        self.stack.clear()
        observation = self.env.reset()
        for _ in range(self.stack.maxlen):
            self.stack.append(observation)

        return np.array(self.stack).reshape(self.observation_space.low.shape)

    def observation(self, observation):
        self.stack.append(observation) # Append the new observation to the stack
        return np.array(self.stack).reshape(self.observation_space.low.shape) # pass back the stack of observations (as a numpy)

# Function that does all necessary changes on the environment
def make_environment(args, shape=(84,84,1), repeat=4, clip_rewards=False, no_ops=0, fire_first=False): # clip_rewards,no_ops,fire_first will only be activated in testing
    if args.simple:
        env = gym.make(args.env)
    else:
        env = JoypadSpace(gym_super_mario_bros.make(args.env), SIMPLE_MOVEMENT) if args.env.startswith('SuperMarioBros') else gym.make(args.env)

        env = RepeatActionAndMaxFrame(env, repeat, clip_rewards, no_ops, fire_first)
        env = PreprocessFrame(shape, env)
        env = StackFrames(env, repeat)

    return env