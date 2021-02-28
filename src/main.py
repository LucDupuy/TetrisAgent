from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import MOVEMENT





def policy_iteration():
    return






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





if __name__ == "__main__":
    env = gym_tetris.make('TetrisA-v0')
    env = JoypadSpace(env, MOVEMENT)

    pol = policy_iteration()


    render_env(pol, 5000)