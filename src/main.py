from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import MOVEMENT
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Uses GPU for computations if you have CUDA set up, otherwise it will use CPU
# to(device) -> Use this to send specific code to the device specified
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NeuralNet(nn.Module):
    """
    This class is the model of the neural network that we can potentially use later on
    if we need to switch from Q-learning to Deep Q-learning.
    """
    def __init__(self, screen_h, screen_w, num_actions):
        super(NeuralNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(screen_w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(screen_h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, num_actions)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(), -1))


def get_action(epsilon, Q_value, current_state):
    """
    Returns a new action to take
    :param epsilon: epsilon
    :return: the action
    """

    rand = np.random.random()

    if rand < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q_value[current_state])


#Q Learning algorithm
def policy_iteration(env, nS, nA, iterations, gamma, alpha):
    """
    I think we can make this pretty much the same as the assignment.
    Might have to convert to Deep Q-Learning later on so it isn't
    just a copy paste of assignment 2.
    """

    Q_value = np.zeros((nS, nA))
    policy = np.ones((nS, nA)) / nA
    epsilon = 1
    current_state = env.reset()

    episodes = 0
    while episodes < iterations:
        done = False
        t = 0
        while not done and epsilon > 0.001:
            action = get_action(epsilon=epsilon, Q_value=Q_value, current_state=current_state)
            next_state, reward, done, _ = env.step(action)
            Q_value[current_state][action] += alpha * (reward + gamma * np.amax(Q_value[next_state]) - Q_value[current_state][action])
            t += 1
            epsilon = 1 / t

            current_state = next_state

        episodes += 1
        epsilon = 1
        current_state = env.reset()
        print("Episode", episodes, "done")

    return policy


def render_env(policy, max_steps):
    episode_reward = 0
    board_state = env.reset()

    for step in range(max_steps):
        env.render()

        a = policy[board_state]
        board_state, reward, done, info = env.step(a)
        episode_reward += reward
        if done:
            break

    env.render()


def sample_state_space(env):
    """
    Determine how many states each episode will have
    Since Tetris can be played indefinitely, it doesn't have a set amount of states unlike Frozen-Lake assignments
    One solution would be to define the number of states ourselves or we could play through a game and use the number of actions there
    """
    iterations = 0
    done = False

    while not done:
        state, reward, done, info = env.step(env.action_space.sample())
        iterations += 1
        # env.render()

    return iterations


if __name__ == "__main__":
    env = gym_tetris.make('TetrisA-v0')
    env = JoypadSpace(env, MOVEMENT)

    nS = sample_state_space(env=env)
    nA = len(MOVEMENT)

    pol = policy_iteration(env=env, nS=nS, nA=nA, iterations=1000, gamma=0.9, alpha=0.1)

    render_env(pol, 5000)
