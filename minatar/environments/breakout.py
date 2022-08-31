################################################################################################################
# Authors:                                                                                                      #
# Kenny Young (kjyoung@ualberta.ca)                                                                             #
# Tian Tian (ttian@ualberta.ca)                                                                                 #
# Robert Joseph (rjoseph1@ualberta.ca)                                                                          #        
################################################################################################################
from re import X
import numpy as np

from minatar.utils import choice, try2jit


#####################################################################################################################
# Env
#
# The player controls a paddle on the bottom of the screen and must bounce a ball tobreak 3 rows of bricks along the
# top of the screen. A reward of +1 is given for each brick broken by the ball.  When all bricks are cleared another 3
# rows are added. The ball travels only along diagonals, when it hits the paddle it is bounced either to the left or
# right depending on the side of the paddle hit, when it hits a wall or brick it is reflected. Termination occurs when
# the ball hits the bottom of the screen. The balls direction is indicated by a trail channel.
#
#####################################################################################################################
class Env:
    def __init__(self, ramping = None, random_state = None):
        self.channels ={
            'paddle':0,
            'ball':1,
            'trail':2,
            'brick':3,
        }
        self.action_map = ['n','l','u','r','d','f']
        if random_state is None:
            self.random = np.random.RandomState()
        else:
            self.random = random_state
        self.reset()

    # Update environment according to agent action
    def act(self, a):
        r = 0
        if self.terminal:
            return r, self.terminal

        # Resolve player action
        self.pos = _move_paddle(self.pos, a)

        # Update ball position
        self.last_x = self.ball_x
        self.last_y = self.ball_y

        (
            self.ball_x,
            self.ball_y,
            self.dx,
            self.dy,
            r,
            self.terminal
         ) = _move_ball(self.ball_x, self.ball_y, self.dx, self.dy, self.brick_map, self.pos)

        return r, self.terminal

    # Query the current level of the difficulty ramp, difficulty does not ramp in this game, so return None
    def difficulty_ramp(self):
        return None

    # Process the game-state into the 10x10xn state provided to the agent and return
    def state(self):
        return _build_state(self.ball_x, self.ball_y, self.pos, self.last_x, self.last_y, self.brick_map)

    # Reset to start state for new episode
    def reset(self):
        self.ball_y = 3
        ball_start = choice([(0,1,1),(9,-1,1)], self.random)
        self.ball_x, self.dx, self.dy = ball_start
        self.pos = 4
        self.brick_map = np.zeros((10,10))
        self.brick_map[1:4,:] = 1
        self.strike = False
        self.last_x = self.ball_x
        self.last_y = self.ball_y
        self.terminal = False

    # Dimensionality of the game-state (nx10x10)
    def state_shape(self):
        return [len(self.channels), 10, 10]

    # Subset of actions that actually have a unique impact in this environment
    def minimal_action_set(self):
        minimal_actions = ['n','l','r']
        return [self.action_map.index(x) for x in minimal_actions]

# -----------------------
# -- Utility functions --
# -----------------------
# These are all jit-compiled and cached to disk
# as long as numba has been installed correctly
# otherwise, they are called as pure python functions
# jit-compiling has approximately a 4x speedup

@try2jit
def _move_paddle(x, a):
    if a == 1:
        return max(0, x-1)
    elif a == 3:
        return min(9, x+1)

    return x

@try2jit
def _move_ball(x, y, dx, dy, bricks, px):
    new_x = x + dx
    new_y = y + dy

    r = 0
    terminal = False

    # first handle wall collisions
    if new_x < 0:
        new_x = 1
        dx = -dx

    elif new_x > 9:
        new_x = 8
        dx = -dx

    # then handle top of board collision
    if new_y < 0:
        new_y = 0
        dy = -dy

    # check for brick collision
    elif bricks[new_y, x] == 1:
        r = 1
        bricks[new_y, x] = 0
        new_y = y
        new_x = x
        dy = -dy

    elif bricks[new_y, new_x] == 1:
        r = 1
        bricks[new_y, new_x] = 0
        new_y = y
        new_x = x
        dy = -dy
        dx = -dx

    # check for bottom of board collision
    elif new_y == 9:
        # if there are no bricks
        # then reinitialize the first 4 rows to bricks
        if np.sum(bricks) == 0:
            bricks[1:4] = 1

        # if the ball hit the paddle
        if new_x == px:
            dy = -dy
            new_y = y

        elif x == px:
            dy = -dy
            dx = -dx
            new_y = y

        else:
            terminal = True

    return new_x, new_y, dx, dy, r, terminal

@try2jit
def _build_state(x, y, px, lx, ly, bricks):
    state = np.zeros((4, 10, 10), dtype='bool')

    # set ball
    state[1, y, x] = 1

    # set paddle
    state[0, 9, px] = 1

    # set trail
    state[2, ly, lx] = 1

    # set bricks
    state[3, :, :] = bricks

    return state
