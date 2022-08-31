################################################################################################################
# Authors:                                                                                                      #
# Kenny Young (kjyoung@ualberta.ca)                                                                             #
# Tian Tian (ttian@ualberta.ca)                                                                                 #                             
# Robert Joseph (rjoseph1@ualberta.ca)                                                                          #                           
################################################################################################################
import numpy as np
from minatar.utils import choice, sample, try2jit


#####################################################################################################################
# Constants
#
#####################################################################################################################
ramp_interval = 100
init_spawn_speed = 10
init_move_interval = 5
shot_cool_down = 5


#####################################################################################################################
# Env
#
# The player can move freely along the 4 cardinal directions. Enemies and treasure spawn from the sides. A reward of
# +1 is given for picking up treasure. Termination occurs if the player makes contact with an enemy. Enemy and
# treasure direction are indicated by a trail channel. Difficulty is periodically increased by increasing the speed
# and spawn rate of enemies and treasure.
#
#####################################################################################################################
class Env:
    def __init__(self, ramping = True, random_state = None):
        self.channels ={
            'player':0,
            'enemy':1,
            'trail':2,
            'gold':3
        }
        self.action_map = ['n','l','u','r','d','f']
        self.ramping = ramping
        if random_state is None:
            self.random = np.random.RandomState()
        else:
            self.random = random_state

        # build an empty memory
        self.entities = np.ones((8, 4), dtype=np.int) * -1
        self.reset()

    # Update environment according to agent action
    def act(self, a):
        r = 0
        if(self.terminal):
            return r, self.terminal

        # Spawn enemy if timer is up
        if(self.spawn_timer==0):
            self._spawn_entity()
            self.spawn_timer = self.spawn_speed

        # Resolve player action
        self.player_x, self.player_y = _move_player(self.player_x, self.player_y, a)

        # Update entities
        if(self.move_timer==0):
            self.move_timer = self.move_speed
            _update_entities(self.entities)

        r, self.terminal = _check_collisions(self.entities, self.player_x, self.player_y)

        # Update various timers
        self.spawn_timer -= 1
        self.move_timer -= 1

        # Ramp difficulty if interval has elapsed
        if self.ramping and (self.spawn_speed>1 or self.move_speed>1):
            if(self.ramp_timer>=0):
                self.ramp_timer-=1
            else:
                if(self.move_speed>1 and self.ramp_index%2):
                    self.move_speed-=1
                if(self.spawn_speed>1):
                    self.spawn_speed-=1
                self.ramp_index+=1
                self.ramp_timer=ramp_interval
        return r, self.terminal

    # Spawn a new enemy or treasure at a random location with random direction (if all rows are filled do nothing)
    def _spawn_entity(self):
        # check if the entity memory is already full
        slot_options, = np.where(self.entities[:, 0] < 0)

        # if so, no need to do expensive sampling
        if len(slot_options) == 0:
            return

        # determine direction and color randomly
        lr = choice([1, 0], self.random)
        is_gold = sample([2/3, 1/3], self.random)
        x = 0 if lr else 9

        # stick in a currently empty memory slot
        slot = slot_options[0]
        self.entities[slot] = [x, slot + 1, lr, is_gold]

    # Query the current level of the difficulty ramp, could be used as additional input to agent for example
    def difficulty_ramp(self):
        return self.ramp_index

    # Process the game-state into the 10x10xn state provided to the agent and return
    def state(self):
        return _build_state(self.entities, self.player_x, self.player_y)

    # Reset to start state for new episode
    def reset(self):
        self.player_x = 5
        self.player_y = 5
        self.shot_timer = 0
        self.entities[:, :] = -1
        self.spawn_speed = init_spawn_speed
        self.spawn_timer = self.spawn_speed
        self.move_speed = init_move_interval
        self.move_timer = self.move_speed
        self.ramp_timer = ramp_interval
        self.ramp_index = 0
        self.terminal = False

    # Dimensionality of the game-state (nx10x10)
    def state_shape(self):
        return [len(self.channels),10,10]

    # Subset of actions that actually have a unique impact in this environment
    def minimal_action_set(self):
        minimal_actions = ['n','l','u','r','d']
        return [self.action_map.index(x) for x in minimal_actions]

# -----------------------
# -- Utility functions --
# -----------------------
# These are all jit-compiled and cached to disk
# as long as numba has been installed correctly
# otherwise, they are called as pure python functions
# jit-compiling has approximately a 4x speedup

# takes an entity memory
# and a player (x, y) state and generates
# a (10, 10, 4) representation of game state
@try2jit
def _build_state(entities, x, y):
    state = np.zeros((4, 10, 10), dtype='bool')

    # add player entity
    state[0, y, x] = 1

    # add non-player entities
    for i in range(len(entities)):
        entity = entities[i]
        ex, ey, di, gold = entity

        if ex < 0:
            continue

        # if gold, then put in channel 3
        c = 3 if gold else 1
        state[c, ey, ex] = 1

        back_x = ex - 1 if di else ex + 1
        if back_x >= 0 and back_x <= 9:
            # add trail
            state[2, ey, back_x] = 1

    return state

# takes an entity memory and increments
# their position by a single state
# removing out-of-bounds entities
@try2jit
def _update_entities(entities):
    for i in range(len(entities)):
        entity = entities[i]
        ex, _, di, _ = entity

        # update position
        if ex >= 0:
            ex += 1 if di else -1
            entities[i][0] = ex

        # check bounds
        if ex < 0 or ex > 9:
            entities[i] = -1

# checks if the player entity has collided
# with any entity in memory
@try2jit
def _check_collisions(entities, x, y):
    r = 0
    terminal = False
    for i in range(len(entities)):
        entity = entities[i]
        # entity coords, and color
        ex, ey, _, gold = entity
        if ex >= 0:
            # if we collided
            if ex == x and ey == y:
                if gold:
                    entities[i] = -1
                    r += 1
                else:
                    terminal = True

    return r, terminal

# takes a player position and direction
# and returns a new within-bounds position
@try2jit
def _move_player(x, y, a):
    if a == 1:
        x = max(0, x - 1)
    elif a == 2:
        y = max(1, y - 1)
    elif a == 3:
        x = min(9, x + 1)
    elif a == 4:
        y = min(8, y + 1)

    return x, y
