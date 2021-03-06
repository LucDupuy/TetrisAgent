from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import SIMPLE_MOVEMENT
import numpy as np
from PIL import Image
import pandas as pd
import os

env = gym_tetris.make('TetrisA-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# For convenience
move = {
    "CW": 1,
    "CCW": 2,
    "right": 3,
    "left": 4,
    "down": 5,
}

# nS = 20 * 10
# nA = len(SIMPLE_MOVEMENT)
# Qs = np.zeros((nS, nA))
# policy = np.ones((nS, nA)) / nA
# epsilon = 1
# alpha = 0.1
# gamma = 0.9

state = env.reset() # Screenshot/rgb array of game at a given time
action = 5 # The current action to take (see move dictionary)
action_set = [] # The sequence of actions to take
block = '' # The name of the current block that is falling
stats = [] # The amount of each block dropped onto the playfield

ITERATIONS = 100000
last_num_iterations = 0

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
    Returns the number of cleared lines i.e. lines with just 1's on the given state of the board.
    """
    lines = 0
    for i in range(len(board)-1, 0, -1):
        if np.min(board[i]) == 1:
            lines += 1
    return lines

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

def find_next_states(board, block, pos):
    """
    Returns all possible next states i.e. the lockdown positions of the current block given the current state of the board.
    """
    
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
    if block.startswith('O'): rotations = 1
    elif block.startswith('I') or block.startswith('S') or block.startswith('Z'): rotations = 2

    # Try all possible rotations
    for rot in range(rotations):
        action_set = [] # Construct set of actions dynamically

        # Add CCW action rot times to action set
        for times in range(rot):
            action_set.append(move["CCW"])
            action_set.append(0)

        # Rotate block matrix 90 degrees CCW
        bm = np.rot90(block_m, rot)
        bp = np.argwhere(bm)

        # First move to the farthest possible left
        p = np.array(pos)
        dir = np.array([0, 0])
        for i in range(1, 7):
            if check_collision(b, bm, p, np.array([0, -i])):
                break
            else: 
                dir = np.array([0, -i])
                action_set.append(move["left"])
                action_set.append(0) 
        p += dir

        # Drop block in each 'column' to find new states
        while True:
            # Drop block to lowest possible
            dir = np.array([0,0])
            for i in range(1, 20):
                if check_collision(b, bm, p, np.array([i, 0])):
                    break
                else: 
                    dir = np.array([i, 0])
            p_down = np.array(p) + dir

            # Make a copy of board with block 'recorded' on it
            new_board = np.array(b)
            for _, pcs_pos in enumerate(bp):
                pcs_pos_rel = p_down + pcs_pos
                new_board[pcs_pos_rel[0], pcs_pos_rel[1]] = 1

            # Construct necessary info for new state and append it to list
            heights = get_heights(new_board)
            bumpiness = np.sum(np.absolute(np.diff(heights)))
            holes = get_holes(new_board, heights)
            lines = get_cleared_lines(new_board)
            new_state = {
                "board": new_board,
                "cleared_lines": lines,
                "holes": np.sum(holes),
                "bumpiness": bumpiness,
                "total_height": np.sum(heights),
                "action_set": action_set.copy()
            }
            next_states.append(new_state)

            # Move right if possible
            if check_collision(b, bm, p, np.array([0, 1])):
                break # Stop if not
            else:
                p += np.array([0, 1])
                # Pop left action if available or append right action
                if action_set and action_set[-2] == move["left"]:
                    action_set.pop()
                    action_set.pop()
                else:
                    action_set.append(move["right"])
                    action_set.append(0)

    return next_states

def find_best_state(states):
    """
    Using a genetic algorithm borrowed from https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/, find the next best state.
    """
    best_index = 0
    best_val = -99999999

    # Calculate a value of each state with the given paramters and find the greatest one
    for i in range(len(states)):
        tallest_height = states[i]["total_height"]
        num_cleared_lines = states[i]["cleared_lines"]
        num_holes = states[i]["holes"]
        difference_in_heights = states[i]["bumpiness"]

        r = [-0.510066, 0.760666, -0.35663, -0.184483]
        val = np.dot(r, [tallest_height, num_cleared_lines, num_holes, difference_in_heights])
        if val > best_val:
            best_val = val
            best_index = i

    return states[best_index]


def makeResultsSpreadsheet(score, iterations):
    """

    :param score: the score achieved before loss
    :param iterations: iterations taken to acheive this score
    """

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):

        results_file = "./Results.xls"

        new_row = {'Score': [score], 'Iterations': [iterations]}

        if os.path.isfile(results_file):
            df = pd.read_excel(results_file)
            df = df.append(new_row, ignore_index=True)


            df.to_excel('./Results.xls',
                             header=True, index=False)

        else:
            new_df = pd.DataFrame(data=new_row)
            new_df.to_excel('./Results.xls',
                             header=True, index=False)


for i in range(ITERATIONS):
    state, _, done, info = env.step(action)


    # Only determine next set of actions when current block changes i.e. statistics update
    if stats != info['statistics']:
        # Get necessary info
        block = info['current_piece']
        stats = info['statistics']
        board = get_board(state)

        # Relative position of block on board
        pos = [-1, 4]
        if block.startswith('O'):
            pos = [-1, 3]
        elif block.startswith('I'): 
            pos = [-2, 3] 

        # Get the action sequence of the next best state
        next_states = find_next_states(board, block, pos)
        best_state = find_best_state(next_states) # np.random.choice(next_states)
        action_set = best_state["action_set"]

        action = 0 # Must set action to NOOP (no operation) when current block changes
    # Get the next action in the action set
    elif action_set:
        action = action_set.pop(0)
    # Drop block to lockdown when action set is empty
    elif not action_set:
        action = move['down']

    if done:
        round_iterations = abs(last_num_iterations - i)
        last_num_iterations = i
        # Print out the score to see how we improve

        if i is not ITERATIONS:
            makeResultsSpreadsheet(info.get('score'), round_iterations)
        env.reset()
        action_set = []
        action = 0
    env.render()

env.close()