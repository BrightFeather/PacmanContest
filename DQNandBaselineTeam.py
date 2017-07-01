# PacmanDQNTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# Pacman game
from game import Grid
from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
import capture
from util import nearestPoint, manhattanDistance
from pacman import Directions
import layout

import numpy as np
import random
import util
import time
import sys
import re

# Replay memory
from collections import deque

# Neural nets
import tensorflow as tf
from DQN import *

from baselineTeam import ReflexCaptureAgent, DefensiveReflexAgent #OffenseDefenseAgents

params = {
    # Model backups
    'load_file': 'saves/model-1775078_1174317',
    'save_file': True,
    'save_interval': 5000,

    # Training parameters
    'train_start': 10000, #5000,  # Episodes before training starts
    'batch_size': 32, #32 # Replay memory batch size
    'mem_size': 10000,  # Replay memory size
    'numTraining': 10000, # number of training epochs

    'discount': 0.95,  # Discount rate (gamma value)
    'lr': .0002,  # Learning reate
    'rms_decay': 0.99,  # RMS Prop decay
    'rms_eps': 1e-6,  # RMS Prop epsilon

    # Epsilon value (epsilon-greedy)
    'eps': 1.0,  # Epsilon start value
    'eps_final': 0.1,  # Epsilon end value
    'eps_step': 10000  #10000 Epsilon steps between start and end (linear)
}


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveDQNAgent', second = 'DefensiveReflexAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(firstIndex,isRed), eval(second)(secondIndex)]

    # return [eval(first)(firstIndex,numTraining, offense=offense,model_file=model_file), eval(second)(secondIndex)]

##########
# Agents #
##########
class DQNAgent(CaptureAgent):
    """
    Class for offensive DQN agent
    """

    def __init__(self, index, isRed, logstdout=True, timeForComputing=.1):
        """
            Lists several variables you can query:
            self.index = index for this agent
            self.red = true if you're on the red team, false otherwise
            self.agentsOnTeam = a list of agent objects that make up your team
            self.distancer = distance calculator (contest code provides this)
            self.observationHistory = list of GameState objects that correspond
                to the sequential order of states that have occurred so far this game
            self.timeForComputing = an amount of time to give each turn for computing maze distances
                (part of the provided distance calculator)
            """
        # Agent index for querying state
        self.index = index

        # Whether or not you're on the red team
        self.red = isRed

        # Agent objects controlling you and your teammates
        self.agentsOnTeam = None

        # Maze distance calculator
        self.distancer = None

        # A history of observations
        self.observationHistory = []

        # Time to spend each turn on computing maze distances
        self.timeForComputing = timeForComputing

        # Access to the graphics
        self.display = None
        print("Initialise DQN Agent")

        # Load parameters from user-given arguments
        self.params = params

        self.params['width'] = 32  # self.layout.width #layout.width
        self.params['height'] = 16  # layout.height
        self.params['depth'] = 6

        # Offensive player
        self.offense = True

        # Set this to False if you do not want to see detailed information of agent's each move
        self.logstdout = logstdout

        # Start Tensorflow session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.qnet = DQN(self.params)

        # time started
        self.general_record_time = time.strftime("%a_%d_%b_%Y_%H_%M_%S", time.localtime())
        # Q and cost
        self.Q_global = []
        self.cost_disp = 0

        # Stats
        self.cnt = self.qnet.sess.run(self.qnet.global_step)
        self.local_cnt = 0

        self.training_cnt = 0

        # if model_file is not None:
        #     tmp = re.findall('-(\d+)', model_file)
        #     self.numeps = eval(tmp[0])
        # if self.params['load_file'] is not None:
        #     tmp = re.findall('-(\d+)', self.params['load_file'])
        #     self.numeps = eval(tmp[0])
        # else:
        self.numeps = 0

        self.last_score = 0
        self.s = time.time()
        self.last_reward = 0.

        self.replay_mem = deque()
        self.last_scores = deque()

    def registerInitialState(self, gameState):  # inspects the starting state

        # print str(gameState)

        # Reset reward
        self.last_score = 0
        self.current_score = 0
        self.last_reward = 0.
        self.ep_rew = 0

        # Reset state
        self.last_state = None
        self.current_state = gameState

        # Reset actions
        self.last_action = None

        # Reset vars
        self.terminal = None
        self.won = True
        self.Q_global = []
        self.delay = 0

        # Next
        self.frame = 0
        self.numeps += 1

        # log for what happened after last action
        self.log = None

        # from CaptureAgent
        self.start = gameState.getAgentPosition(self.index)

        # self.red = gameState.isOnRedTeam(self.index)
        # self.distancer = distanceCalculator.Distancer(gameState.data.layout)
        CaptureAgent.registerInitialState(self, gameState)

    def getMove(self, state):
        # Exploit / Explore
        stateMatrix = self.getStateMatrices(self.current_state)
        if np.random.rand() > self.params['eps']:
            # Exploit action
            # if self.offense:
            self.Q_pred = self.qnet.sess.run(
                self.qnet.y,
                feed_dict={self.qnet.x: np.reshape(stateMatrix,
                                                   (1, self.params['width'], self.params['height'],
                                                    self.params['depth'])),
                           self.qnet.q_t: np.zeros(1),
                           self.qnet.actions: np.zeros((1, 4)),
                           self.qnet.terminals: np.zeros(1),
                           self.qnet.rewards: np.zeros(1)})[0]
            self.Q_global.append(max(self.Q_pred))
            a_winner = np.argwhere(self.Q_pred == np.amax(self.Q_pred))
            if len(a_winner) > 1:
                move = self.get_direction(a_winner[np.random.randint(0, len(a_winner))][0])
            else:
                move = self.get_direction(a_winner[0][0])
        else:
            # Random:
            move = self.get_direction(np.random.randint(0, 4))

            # Save last_action
        if not self.red:
            move = Directions.REVERSE[move]

        self.last_action = self.get_value(move)
        print move

        return move

    def get_value(self, direction):
        if direction == Directions.NORTH:
            return 0.
        elif direction == Directions.EAST:
            return 1.
        elif direction == Directions.SOUTH:
            return 2.
        else:
            return 3.

    def get_direction(self, value):
        if value == 0.:
            return Directions.NORTH
        elif value == 1.:
            return Directions.EAST
        elif value == 2.:
            return Directions.SOUTH
        else:
            return Directions.WEST

    def observation_step(self, state):
        raise NotImplementedError

    def observationFunction(self, state):
        # Do observation
        self.terminal = False
        self.observation_step(state)

        return state

    def final(self, state):
        # Next
        self.ep_rew += self.last_reward

        # Do observation
        self.terminal = True
        self.observation_step(state)

        # Print stats
        log_file = open('./logs/' + str(self.general_record_time) + '-l-' + str(self.params['width']) + '-m-' + str(
            self.params['height']) + '-x-' + str(self.params['num_training']) + '.log', 'a')
        log_file.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f " %
                       (self.numeps, self.local_cnt, self.cnt, time.time() - self.s, self.ep_rew,
                        self.params['eps']))
        log_file.write(
            "| Q: %10f | won: %r \n" % (
            (max(self.Q_global) if len(self.Q_global) != 0 else float('nan'), self.won)))
        # python 3
        # log_file.write("| Q: %10f | won: %r \n" % ((max(self.Q_global, default=float('nan')), self.won)))
        sys.stdout.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f " %
                         (self.numeps, self.local_cnt, self.cnt, time.time() - self.s, self.ep_rew,
                          self.params['eps']))
        sys.stdout.write(
            "| Q: %10f | won: %r \n" % (
            (max(self.Q_global) if len(self.Q_global) != 0 else float('nan'), self.won)))
        # python 3
        # sys.stdout.write("| Q: %10f | won: %r \n" % ((max(self.Q_global, default=float('nan')), self.won)))
        sys.stdout.flush()

    def train(self):
        # Train
        # if (self.local_cnt > self.params['train_start']):
        batch = random.sample(self.replay_mem, self.params['batch_size'])  # --> sample randomly?
        batch_s = []  # States (s)
        batch_r = []  # Rewards (r)
        batch_a = []  # Actions (a)
        batch_n = []  # Next states (s')
        batch_t = []  # Terminal state (t)

        for i in batch:
            batch_s.append(i[0])
            batch_r.append(i[1])
            batch_a.append(i[2])
            batch_n.append(i[3])
            batch_t.append(i[4])
        batch_s = np.array(batch_s)
        batch_r = np.array(batch_r)
        batch_a = self.get_onehot(np.array(batch_a))
        batch_n = np.array(batch_n)
        batch_t = np.array(batch_t)

        self.cnt, self.cost_disp = self.qnet.train(batch_s, batch_a, batch_t, batch_n, batch_r)
        print ' | Exploiting. Training Cost : %.4f' % self.cost_disp

    def get_onehot(self, actions):
        """ Create list of vectors with 1 values at index of action in list """
        actions_onehot = np.zeros((self.params['batch_size'], 4))
        for i in range(len(actions)):
            actions_onehot[i][int(actions[i])] = 1
        return actions_onehot

    # def mergeStateMatrices(self, stateMatrices):
    #     """ Merge state matrices to one state tensor """
    #     stateMatrices = np.swapaxes(stateMatrices, 0, 2)
    #     total = np.zeros((7, 7))
    #     for i in range(len(stateMatrices)):
    #         total += (i + 1) * stateMatrices[i] / 6
    #     return total

    def getStateMatrices(self, state):
        """ Return wall, ghosts, food, capsules matrices """

        def getEnemyMatrix(self, state):
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width))
            matrix.dtype = int
            if self.red:
                enemyIndices = state.blueTeam
            else:
                enemyIndices = state.redTeam
            for i in xrange(len(state.data.agentStates)):
                if i in enemyIndices:
                    agentState = state.data.agentStates[i]
                    pos = agentState.configuration.getPosition()
                    cell = 1
                    matrix[-1 - int(pos[1])][int(pos[0])] = cell
            return matrix

        def getAllyMatrix(self, state):
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width))
            matrix.dtype = int
            if self.red:
                allyIndices = state.redTeam
            else:
                allyIndices = state.blueTeam
            for i in xrange(len(state.data.agentStates)):
                if i in allyIndices:
                    agentState = state.data.agentStates[i]
                    pos = agentState.configuration.getPosition()
                    cell = 1
                    matrix[-1 - int(pos[1])][int(pos[0])] = cell
            return matrix

        def getWallMatrix(state):
            """ Return matrix with wall coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            grid = state.data.layout.walls
            matrix = np.zeros((height, width))
            matrix.dtype = int

            for i in range(grid.height):
                for j in range(grid.width):
                    # Put cell vertically reversed in matrix
                    cell = 1 if grid[j][i] else 0
                    matrix[-1 - i][j] = cell
            return matrix

        def getPacmanMatrix(state):
            """ Return matrix with pacman coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width))
            matrix.dtype = int

            for agentState in state.data.agentStates:
                if agentState.isPacman:
                    pos = agentState.configuration.getPosition()
                    cell = 1
                    matrix[-1 - int(pos[1])][int(pos[0])] = cell

            return matrix

        def getSelfMatrix(state):
            '''Return matrix with self coordinates set to 1'''
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width))
            matrix.dtype = int

            pos = state.getAgentPosition(self.index)
            cell = 1
            matrix[-1 - int(pos[1])][int(pos[0])] = cell

            return matrix

        def getSelfGhostMatrix(state):
            '''Return matrix with self coordinates set to 1'''
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width))
            matrix.dtype = int

            if not state.data.agentStates[self.index].scaredTimer > 0:
                pos = state.getAgentPosition(self.index)
                cell = 1
                matrix[-1 - int(pos[1])][int(pos[0])] = cell

            return matrix

        def getSelfScaredGhostMatrix(state):
            '''Return matrix with self coordinates set to 1'''
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width))
            matrix.dtype = int

            if state.data.agentStates[self.index].scaredTimer > 0:
                pos = state.getAgentPosition(self.index)
                cell = 1
                matrix[-1 - int(pos[1])][int(pos[0])] = cell

            return matrix

        def getGhostMatrix(state):
            """ Return matrix with ghost coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width))
            matrix.dtype = int

            for agentState in state.data.agentStates:
                if not agentState.isPacman:
                    if not agentState.scaredTimer > 0:
                        pos = agentState.configuration.getPosition()
                        cell = 1
                        matrix[-1 - int(pos[1])][int(pos[0])] = cell

            return matrix

        def getEnemyPacmanMatrix(self, state):

            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width))
            matrix.dtype = int

            if self.red:
                enemyIndices = state.blueTeam
            else:
                enemyIndices = state.redTeam

            cell = 1
            for i in enemyIndices:
                if state.data.agentStates[i].isPacman:
                    pos = state.data.agentStates[i].configuration.getPosition()
                    matrix[-1 - int(pos[1])][int(pos[0])] = cell
            return matrix

        def getEnemyGhostMatrix(self, state):
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width))
            matrix.dtype = int

            if self.red:
                enemyIndices = state.blueTeam
            else:
                enemyIndices = state.redTeam

            cell = 1
            for i in enemyIndices:
                if not state.data.agentStates[i].isPacman and state.data.agentStates[i].scaredTimer == 0:
                    pos = state.data.agentStates[i].configuration.getPosition()
                    matrix[-1 - int(pos[1])][int(pos[0])] = cell
            return matrix

        def getScaredGhostMatrix(state):
            """ Return matrix with ghost coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width))
            matrix.dtype = int

            for agentState in state.data.agentStates:
                if not agentState.isPacman:
                    if agentState.scaredTimer > 0:
                        pos = agentState.configuration.getPosition()
                        cell = 1
                        matrix[-1 - int(pos[1])][int(pos[0])] = cell

            return matrix

        def getEnemyScaredGhostMatrix(self, state):
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width))
            matrix.dtype = int

            if self.red:
                enemyIndices = state.blueTeam
            else:
                enemyIndices = state.redTeam

            cell = 1
            for i in enemyIndices:
                if not state.data.agentStates[i].isPacman and state.data.agentStates[i].scaredTimer > 0:
                    pos = state.data.agentStates[i].configuration.getPosition()
                    matrix[-1 - int(pos[1])][int(pos[0])] = cell
            return matrix

        def getFoodMatrix(state):
            """ Return matrix with food coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            grid = state.data.food
            matrix = np.zeros((height, width))
            matrix.dtype = int

            for i in range(grid.height):
                for j in range(grid.width):
                    # Put cell vertically reversed in matrix
                    cell = 1 if grid[j][i] else 0
                    matrix[-1 - i][j] = cell

            return matrix

        def getCapsulesMatrix(state):
            """ Return matrix with capsule coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            capsules = self.getCapsules(state)
            matrix = np.zeros((height, width))
            matrix.dtype = int

            for i in capsules:
                # Insert capsule cells vertically reversed into matrix
                matrix[-1 - i[1], i[0]] = 1

            return matrix

        # Create observation matrix as a combination of
        # wall, pacman, ghost, food and capsule matrices
        width, height = state.data.layout.width, state.data.layout.height
        # width, height = self.params['width'], self.params['height']

        observation = np.zeros((6, height, width))

        observation[0] = getWallMatrix(state)
        observation[1] = getSelfMatrix(state)
        observation[2] = getEnemyGhostMatrix(self, state)
        observation[3] = getEnemyScaredGhostMatrix(self, state)
        observation[4] = getFoodMatrix(state)
        observation[5] = getCapsulesMatrix(state)

        if not self.red:
            observation = np.rot90(observation,2)

        observation = np.swapaxes(observation, 0, 2)

        return observation

    def chooseAction(self, state):
        move = self.getMove(state)

        # Stop moving when not legal
        legal = state.getLegalActions(self.index)

        if move not in legal:
            move = Directions.STOP

        return move

class OffensiveDQNAgent(DQNAgent):
    # def registerInitialState(self, gameState):
    #     DQNAgent.registerInitialState(self,gameState)
    #     if self.red:
    #         self.start = (8,16)
    #     else:
    #         self.start = (9,17)

    # import __main__
    # if '_display' in dir(__main__):
    #     self.display = __main__._display

    def observation_step(self, state):
        if self.last_action is not None:
            # Process current experience state
            self.last_state = self.current_state.deepCopy()
            self.current_state = state.deepCopy()

            '''
            Offensive agent:
            1. get food: small positive reward(+2)
            2. get capsult: medium positive reward(+5)
            3. get back to own territory with food: big positive reward(+10 x #food)
            4. get eaten: big negative reward(-500)
            5. travel in enemy territory: small negative reward(-1)
            6. travel in own territory vertically: medium negative reward(-2)
            7. travel in own territory horizontally towards own: medium negative reward(-2)
            8. travel in own territory horizontally towards enemy: small negative reward(-1)
            9. stop: medium negative reward(-5)
            '''
            agentPosition = state.getAgentPosition(self.index)
            # Process current experience reward
            self.current_score = state.getScore()
            reward = self.current_score - self.last_score
            self.last_score = self.current_score

            # if pacman is eaten by ghost, dead
            if self.last_state.data.agentStates[self.index].isPacman and \
                    self.current_state.data.agentStates[self.index].configuration == self.current_state.data.agentStates[self.index].start:
                self.last_reward = -100
                self.log = 'Eaten by Ghost'
                self.terminal = True
                self.won = False
                # state.data._lose = True
                # state.data._win = False

            # if currently in own territory
            elif not self.current_state.data.agentStates[self.index].isPacman:
                # keeps walking in own territory
                if not self.last_state.data.agentStates[self.index].isPacman:
                    # stop
                    if self.last_state.getAgentPosition(self.index) == self.current_state.getAgentPosition(
                            self.index):
                        self.last_reward = -10.
                        self.log = 'Stop'
                    # horizontally towards self
                    elif self.red and self.last_action == 3 or \
                                    not self.red and self.last_action == 1:
                        self.last_reward = -5.
                        self.log = 'Travel'
                    # vertically
                    elif self.last_action in [0, 2]:  # or \
                        self.last_reward = -2.
                        self.log = 'Travel'
                    # horizontally towards enemey
                    elif (self.red and self.last_action == 1) or \
                            (not self.red and self.last_action == 3):
                        self.last_reward = -1.
                        self.log = 'Travel'
                    else:
                        self.last_reward = -1.
                        self.log = 'Travel'
                # gets back with food from enemy territory
                elif not self.last_state.data.agentStates[self.index].isPacman \
                        and self.last_state.data.agentStates[self.index].numCarrying > 0:
                    # reward depends on number of food taken times reward per food and cost of getting back
                    self.last_reward = 10. * self.current_state.data.agentStates[self.index].numCarrying - 1.
                    self.log = 'Get Back!'

            # else if currently in enemies' territory
            else:
                lastPosition = agentPosition  # self.last_state.getAgentPosition(self.index)
                x, y = lastPosition
                # ate food
                if self.last_state.data.food[x][y]:
                    self.last_reward = 5.
                    self.log = 'Eat food'
                # ate capsule
                elif (x, y) in self.last_state.data.capsules:
                    self.last_reward = 5.
                    self.log = 'Eat capsule'
                # stop
                elif self.last_state.getAgentPosition(self.index) == self.current_state.getAgentPosition(
                        self.index):
                    self.last_reward = -10.
                    self.log = 'Stop'
                # travel
                else:
                    self.last_reward = -1.
                    self.log = 'Travel'

            self.ep_rew += self.last_reward

            if self.logstdout:

                print 'Game #' + str(self.numeps) + ', Move #' + str(
                    self.frame) + ': Reward for DQN agent\'s last action ***' + self.log + '*** : ' + str(
                    self.last_reward) + '. ',
                print 'Position: ' + str(agentPosition),

            # Store last experience into memory
            experience = (self.getStateMatrices(self.last_state), float(self.last_reward), self.last_action, \
                          self.getStateMatrices(self.current_state), self.terminal)
            self.replay_mem.append(experience)
            if len(self.replay_mem) > self.params['mem_size']:
                self.replay_mem.popleft()

            # Save model
            if (self.params['save_file']):
                if self.local_cnt > self.params['train_start'] and self.local_cnt % self.params[
                    'save_interval'] == 0:
                    self.qnet.save_ckpt(
                        'saves/O/model-' + str(self.cnt) + '_' + str(self.numeps))
                    print('Model saved')

            # Train
            if (self.training_cnt > self.params['train_start']):
                self.train()
                self.qnet.write_summay(self.local_cnt)
            elif self.logstdout:
                print ' | Exploring...'

        # Next
        self.training_cnt += 1
        self.local_cnt += 1
        self.frame += 1
        self.params['eps'] = max(self.params['eps_final'],
                                 1.00 - float(self.cnt) / float(self.params['eps_step']))

    #


class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):

    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}

# class StandStillAgent(CaptureAgent):
#     """An agent that does not move"""
#     def registerInitialState(self, gameState):
#         self.start = gameState.getAgentPosition(self.index)
#         CaptureAgent.registerInitialState(self, gameState)
#
#     def chooseAction(self,gameState):
#         return Directions.STOP





