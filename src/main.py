import sys
import time
import random

import gym_tetris
import numpy as np
from PIL import Image
from gym_tetris.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

env = gym_tetris.make('TetrisA-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Moveset for convenience
move = {
    "CW": 1,
    "CCW": 2,
    "right": 3,
    "left": 4,
    "down": 5,
}
np.set_printoptions(precision=6, suppress=True, floatmode='fixed', linewidth=np.inf)

def get_board(state):
    """
    Returns the current state of the board as a 20x10 array.
    """
    # Extract board from state i.e. crop board from snapshot of game
    # Convert array to Image to resize it
    board = Image.fromarray(state[47:207, 95:175])
    # Resize to workable 10x20 array
    board = board.resize((10, 20), Image.NEAREST)
    board = np.array(board)
    # Add color channels together to 'flatten' to one dimension
    board = np.sum(board, 2)
    # Change so that 1 = block, 0 = nothing   
    board = (board > 0).astype(int)
    return board


def get_heights(board):
    """
    Returns the height of each column in the given state of the board.
    """
    heights = np.zeros(10, dtype=int)
    for i in range(len(board[0])):
        if np.max(board[:, i]) == 0:
            continue
        else:
            nonzeros = np.argwhere(board[:, i])
            heights[i] = (20 - nonzeros[0]).astype(int)
    return heights


def get_holes(board, heights):
    """
    Returns the number of holes in each column in the given state of the board.
    """
    holes = np.zeros(10, dtype=int)
    for i in range(len(board[0])):
        if heights[i] > 0:
            max_h = (20 - heights[i]).astype(int)
            zeros = np.argwhere(board[max_h:, i] == 0)
            holes[i] = len(zeros)
    return holes


def get_cleared_lines(board):
    """
    Returns the number of cleared lines i.e. lines with just 1's on the given state of the board. Also returns a new board with those lines cleared.
    """
    lines = 0
    for i in range(len(board) - 1, 0, -1):
        if np.min(board[i]) == 1:
            lines += 1
    if lines > 0:  # If there a filled lines, clear them and update board
        board = board[~np.all(board == 1, axis=1)]
        board = np.concatenate([np.zeros((lines, 10), dtype=np.int8), board])
    return lines, board


def get_block_matrix(block):
    """
    Returns matrix representation of the given block.

    Example
    -------
    For the T block the matrix would be:\n
    0 0 0\n
    1 1 1\n
    0 1 0\n

    """
    if block.startswith('I'):
        block_m = np.zeros((4, 4))
        block_m[2] = 1
    elif block.startswith('O'):
        block_m = np.zeros((4, 4))
        block_m[1:3, 1:3] = 1
    elif block.startswith('J'):
        block_m = np.zeros((3, 3))
        block_m[1] = 1
        block_m[2, 2] = 1
    elif block.startswith('L'):
        block_m = np.zeros((3, 3))
        block_m[1] = 1
        block_m[2, 0] = 1
    elif block.startswith('S'):
        block_m = np.zeros((3, 3))
        block_m[1, 1:3] = 1
        block_m[2, 0:2] = 1
    elif block.startswith('Z'):
        block_m = np.zeros((3, 3))
        block_m[1, 0:2] = 1
        block_m[2, 1:3] = 1
    elif block.startswith('T'):
        block_m = np.zeros((3, 3))
        block_m[1] = 1
        block_m[2, 1] = 1
    return block_m


def check_collision(board, block_m, pos, dir):
    """
    Checks if the given block at the given pos can move in the given dir on the given state of the board.

    Returns
    -------
    True if there is a collision, false otherwise.
    """
    new_pos = pos + dir
    block_pieces = np.argwhere(block_m)
    for _, pcs_pos in enumerate(block_pieces):
        new_pos_rel = new_pos + pcs_pos
        # Collision if new relative position of piece in block is outside walls
        if new_pos_rel[0] >= 20 or new_pos_rel[1] < 0 or new_pos_rel[1] >= 10:
            return True
        # Collision if new relative position of piece in block already occupied
        elif new_pos_rel[0] >= 0 and board[new_pos_rel[0], new_pos_rel[1]] == 1:
            return True
    return False


def find_next_states(board, block):
    """
    Returns all possible next states i.e. the lockdown positions of the current block given the current state of the board.
    """
    pos = [-1, 4]
    if block.startswith('O'):
        pos = [-1, 3]
    elif block.startswith('I'):
        pos = [-2, 3]

    next_states = []

    # Get matrix representation of block
    block_m = get_block_matrix(block)

    # Remove current block from board so it doesn't conflict when finding next states
    b = np.array(board)
    block_pieces = np.argwhere(block_m)
    for _, pcs_pos in enumerate(block_pieces):
        pcs_pos_rel = pos + pcs_pos
        b[pcs_pos_rel[0], pcs_pos_rel[1]] = 0

    # Trim number of rotations for certain blocks
    rotations = 4
    if block.startswith('O'):
        rotations = 1
    elif block.startswith('I') or block.startswith('S') or block.startswith('Z'):
        rotations = 2

    # Try all possible rotations
    for rot in range(rotations):
        action_set = []  # Construct set of actions dynamically

        action_set.extend([move["CCW"], 0] * rot)  # Add CCW action rot times to action set
        bm = np.rot90(block_m, rot)  # Rotate block matrix 90*rot degrees CCW
        bp = np.argwhere(bm)  # Get the actual positions of each piece of block

        # First move to the farthest possible left
        for i in range(9):
            if check_collision(b, bm, pos, np.array([0, -i - 1])): break
        action_set.extend([move["left"], 0] * i)
        p = np.array(pos) + np.array([0, -i])

        # Drop block in each 'column' to find new states
        while True:
            # Drop block to lowest possible
            for i in range(20):
                if check_collision(b, bm, p, np.array([i + 1, 0])): break
            p_down = np.array(p) + np.array([i, 0])

            # Make a copy of board with block 'recorded' on it
            new_board = np.array(b)
            game_over = False
            for _, pcs_pos in enumerate(bp):
                pcs_pos_rel = p_down + pcs_pos
                if np.all(pcs_pos_rel >= 0):
                    new_board[pcs_pos_rel[0], pcs_pos_rel[1]] = 1
                else:
                    game_over = True

            # Get necessary features for new state and append it to list
            lines, new_board = get_cleared_lines(new_board)
            heights = get_heights(new_board)
            bumpiness = np.absolute(np.diff(heights))
            holes = get_holes(new_board, heights)
            new_state = {
                "score": 40 * lines,
                "holes": np.sum(holes),
                "bumpiness": bumpiness,
                "heights": heights,
                "max_height": np.sum(heights),
                "action_set": action_set.copy(),
                "game_over": game_over
            }
            next_states.append(new_state)

            # Move right from top if possible
            if check_collision(b, bm, p, np.array([0, 1])):
                break
            else:
                p += np.array([0, 1])
                # Pop left action if available or append right action
                if action_set and action_set[-2] == move["left"]:
                    action_set.pop()
                    action_set.pop()
                else:
                    action_set.extend([move["right"], 0])

    return next_states


def get_features(state):
    """
    Returns the features of this state as a 1D array.
    """
    f0_9 = state["heights"]  # Features 0-9 are the column heights
    f10_18 = state["bumpiness"]  # Ft. 10-18 are the abs. differences between columns
    f19 = state["max_height"]  # Ft. 19 is the tallest height
    f20 = state["holes"]  # Ft. 20 is the total number of holes
    # Ft. 21 is just 1.
    return np.concatenate((f0_9, f10_18, f19, f20, 1), axis=None)


def find_best_state(states, weights, gamma=0.9):
    """
    Calculate the approximation value function for each state using the given weights.
    Returns the state with the greatest value within the given collection of states
    as well as its value function.
    """
    # Calculate a value of each state with the given paramters and find the greatest one
    Q_values = np.array([np.dot(weights, get_features(s)) if not s["game_over"] else -9e300 for s in states])

    if random.random() < 0.95:
        best_index = np.argmax(Q_values)
    else:
        best_index = np.random.choice(range(Q_values.size))

    return states[best_index], Q_values[best_index]


def find_best_weights(Q_features, Q_values, weights):
    # Get new weights using least squares on best Q_values and their features
    x = np.linalg.lstsq(np.array(Q_features), np.array(Q_values), rcond=None)
    potential_weights = [w for w in x if not np.isscalar(w) and len(w) == len(weights)]
    return np.array(potential_weights).min(axis=0)


def normalize(value, height=10000, width=10000):
    return np.tanh(value / width) * height


def run(episode, Q_values, Q_features, weights, gamma=0.9):
    start = time.time()  # Measure how long each episode lasts
    state = env.reset()
    action = 0  # The current action to take (see move dictionary)
    action_set = []  # The sequence of actions to take
    stats = []  # The amount of each block dropped onto the playfield
    best_Qvalue = 0  # The value function of the best next state
    running_rewards = 0

    while True:
        state, reward, done, info = env.step(action)
        running_rewards += reward

        if done:
            end = time.time()
            print("Ep: {:3d}".format(episode),
                  "Score: {:6d}".format(info['score']),
                  "Lines: {:3d}".format(info['number_of_lines']),
                  "Time: {:6.2f}s".format(end - start),
                  "Running qv list: {:8d}".format(len(Q_values)),
                  weights,
                  sep=" | ")

            # weights[np.nanargmax(weights)] -= info['score']
            break

        if stats != info['statistics']:
            block = info['current_piece']
            stats = info['statistics']
            board = get_board(state)

            next_states = find_next_states(board, block)
            best_state, best_Qvalue = find_best_state(next_states, weights)

            best_features = get_features(best_state)
            # Add Q value and feature set associated with best state to a collection
            if np.isfinite(best_Qvalue):
                Q_values.append(best_Qvalue)
                Q_features.append(reward + gamma * best_features)

            weights = find_best_weights(Q_features, Q_values, weights)
            action_set = best_state["action_set"]
            action = 0
            running_rewards = 0

        elif action_set:
            action = action_set.pop(0)
        elif not action_set:
            action = move['down']
        # env.render()

    return Q_values, Q_features, weights


def learn(episodes=1000, gamma=0.9):
    """
    Find the best weights, training the agent with the given number of episodes.
    """
   #  weights = np.concatenate((-np.ones(10), -2 * np.ones(9), -40, -1, 10), axis=None)
    weights = -np.ones(22)

    Q_values = []  # List of all best_Qvalues
    Q_features = []  # List of all feature vectors of the best_Qvalues

    for i in range(episodes):
        Q_values, Q_features, weights = run(i, Q_values, Q_features, weights, gamma)

    # Return weights from episode with greatest score
    return weights


if __name__ == "__main__":

    # Find optimal weights using value function approximation
    start = time.time()
    training_episodes = 1000
    print("Finding optimal weights using", training_episodes, "training episodes...", )
    weights = learn(training_episodes, gamma=0.9)
    end = time.time()
    print("Total time elapsed: ", time.strftime("%H:%M:%S", time.gmtime(end - start)))
    print("Best weights:\n", weights)

    # Show 100000 iterations of using the best weights
    state = env.reset()  # Screenshot/rgb array of game at a given time
    action = 5  # The current action to take (see move dictionary)
    action_set = []  # The sequence of actions to take
    block = ''  # The name of the current block that is falling
    stats = []  # The amount of each block dropped onto the playfield
    best_score = 0
    for i in range(100000):
        state, _, done, info = env.step(action)
        if done:
            if info['score'] > best_score:
                best_score = info['score']
            env.reset()
            action_set = []
            action = 0

        # Only determine next set of actions when current block changes i.e. statistics update
        if stats != info['statistics']:
            # Get necessary info
            block = info['current_piece']
            stats = info['statistics']
            board = get_board(state)

            # Get the action sequence of the next best state
            next_states = find_next_states(board, block)
            best_state, _ = find_best_state(next_states, weights)
            action_set = best_state["action_set"]

            action = 0  # Must set action to NOOP (no operation) when current block changes
        # Get the next action in the action set
        elif action_set:
            action = action_set.pop(0)
        # Drop block to lockdown when action set is empty
        elif not action_set:
            action = move['down']
        env.render()

    print("Best score using best weights:", best_score)
    env.close()
