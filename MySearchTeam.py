# MySearchTeam.py
# ---------------
# by Zongjian Li
# partner Weijia Chen
# for AI course final project

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint

#################
#     Params    #
#################

default_params = {
    "particle_sum": 3000,   # used in position inference
    "max_depth": 50,  # used in expectimax agents, it can be very large, but will be limited by actionTimeLimit
    "max_position": 1,  # used in expectimax agents. How many inferenced positions for each agent are used to evaluate state/reward.
    "action_time_limit": 1.0,  # higher if you want to search deeper
    "fully_observed": False,  # not ready yet
    "consideration_distance_factor": 1.5,  # agents far than (search_distance * factor) will be considered stay still
    "expand_factor": 1.0,  # factor to balance searial and parallel work load, now 1.0 is okay
    "truncate_remain_time_percent": 0.1,  #
    "eval_total_reward": False,  # otherwise eval state. It controls whether add up values.

    "enable_stay_inference_optimization": True,  # used in position inference
    "enable_stop_action": False,  # used in many agents, whether enable STOP action.
    "enable_stop_transition": False,  # used in position inference, enable this to allow STOP transition
    "enable_print_log": True,  # print or not
    "enable_coarse_partition": True,  # used in parallel agents, coarse partition or fine partition.

    "discount": 0.9,  # used in q-learning agent
    "learning_rate": 0.01,  # used in q-learning agent
    "filename": "./q-learning-weights.txt",  # used in q-learning agent. saving & load weights. it can be "None"
    "save_interval": 200,  # used in q-learning agent

    "exploration": 1.414,  # used in mcts agent, in ucb formula
    "max_sample": 100,  # used in mcts agent
}

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
                first='StateEvaluationOffensiveAgent',
                second='StateEvaluationDefensiveAgent',
                particleSum = None,
                maxDepth = None,
                maxPosition = None,
                actionTimeLimit = None,
                fullyObserved = None,
                considerationDistanceFactor = None,
                expandFactor = None,
                truncateRemainTimePercent = None,
                evalTotalReward = None,
                enableStayInferenceOptimization = None,
                enableStopAction = None,
                enableStopTransition = None,
                enablePrintLog = None,
                enableCoarsePartition = None,

                discount = None,
                learningRate = None,
                filename = None,
                saveInterval = None,

                exploration = None,
                maxSample = None,
               ):
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
    if particleSum is not None: default_params["particle_sum"] = int(particleSum)
    if maxDepth is not None: default_params["max_depth"] = int(maxDepth)
    if maxPosition is not None: default_params["max_position"] = int(maxPosition)
    if actionTimeLimit is not None: default_params["action_time_limit"] = float(actionTimeLimit)
    if fullyObserved is not None: default_params["fully_observed"] = bool(fullyObserved)
    if considerationDistanceFactor is not None: default_params["consideration_distance_factor"] = int(considerationDistanceFactor)
    if expandFactor is not None: default_params["expand_factor"] = float(expandFactor)
    if truncateRemainTimePercent is not None: default_params["truncate_remain_time_percent"] = float(truncateRemainTimePercent)
    if evalTotalReward is not None: default_params["eval_total_reward"] = bool(evalTotalReward)
    if enableStayInferenceOptimization is not None: default_params["enable_stay_inference_optimization"] = enableStayInferenceOptimization.lower() == "true"
    if enableStopAction is not None: default_params["enable_stop_action"] = enableStopAction.lower() == "true"
    if enableStopTransition is not None: default_params["enable_stop_transition"] = enableStopTransition.lower() == "true"
    if enablePrintLog is not None: default_params["enable_print_log"] = enablePrintLog.lower() == "true"
    if enableCoarsePartition is not None: default_params["enable_coarse_partition"] = enableCoarsePartition.lower() == "true"
    if discount is not None: default_params["discount"] = float(discount)
    if learningRate is not None: default_params["learning_rate"] = float(learningRate)
    if filename is not None: default_params["filename"] = filename
    if saveInterval is not None: default_params["save_interval"] = int(saveInterval)
    if exploration is not None: default_params["exploration"] = float(exploration)
    if maxSample is not None: default_params["max_sample"] = int(maxSample)

    return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

###########################
#                         #
# a virtual class         #
# provide basic functions #
#                         #
###########################
class LogClassification:
    """A class(enum) representing log levels."""
    DETAIL = "Detail"
    INFOMATION = "Infomation"
    WARNING = "Warning"
    ERROR = "Error"

class UtilAgent(CaptureAgent):
    """A virtual agent calss. Providing basic functions."""

    ######################
    # overload functions #
    ######################

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.actionTimeLimit = default_params["action_time_limit"]

    def chooseAction(self, gameState):
        self.log("Agent %d:" % (self.index,))
        self.time = {"START": time.time()}
        action = self.takeAction(gameState)
        self.time["END"] = time.time()
        self.printTimes()
        return action

    #####################
    # virtual functions #
    #####################

    def takeAction(self, gameState):
        util.raiseNotDefined()

    ##############
    # interfaces #
    ##############

    def log(self, content, classification=LogClassification.INFOMATION):
        if default_params["enable_print_log"]:
            print(str(content))
        pass

    def timePast(self):
        return time.time() - self.time["START"]

    def timePastPercent(self):
        return self.timePast() / self.actionTimeLimit

    def timeRemain(self):
        return self.actionTimeLimit - self.timePast()

    def timeRemainPercent(self):
        return self.timeRemain() / self.actionTimeLimit

    def printTimes(self):
        timeList = list(self.time.items())
        timeList.sort(key=lambda x: x[1])
        relativeTimeList = []
        startTime = self.time["START"]
        totalTime = timeList[len(timeList) - 1][1] - startTime
        reachActionTimeLimit = totalTime >= self.actionTimeLimit
        for i in range(1, len(timeList)):
            j = i - 1
            k, v = timeList[i]
            _, lastV = timeList[j]
            time = v - lastV
            if time >= 0.0001:
                relativeTimeList.append("%s:%.4f" % (k, time))
        prefix = "O " if not reachActionTimeLimit else "X "
        prefix += "Total %.4f " % (totalTime,)
        self.log(prefix + str(relativeTimeList))

    def getSuccessor(self, gameState, actionAgentIndex, action):
        """Finds the next successor which is a grid position (location tuple)."""
        successor = gameState.generateSuccessor(actionAgentIndex, action)
        pos = successor.getAgentState(actionAgentIndex).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(actionAgentIndex, action)
        else:
            return successor

######################################################
#                                                    #
# a virtual class                                    #
# inference agent positions using particle filtering #
#                                                    #
######################################################

class PositionInferenceAgent(UtilAgent):
    """A virtual agent class. Inherite this class to get position inference ability. It uses particle filtering algorithm."""

    ######################
    # overload functions #
    ######################

    isFullyObserved = None  # not implemented yet

    def registerInitialState(self, gameState):
        UtilAgent.registerInitialState(self, gameState)
        PositionInferenceAgent.isFullyObserved = default_params["fully_observed"]  # TODO:
        self.initPositionInference(gameState)

    def takeAction(self, gameState):
        self.time["BEFORE_POSITION_INFERENCE"] = time.time()
        self.updatePositionInference(gameState)
        self.checkPositionInference(gameState)
        self.updateBeliefDistribution()
        self.displayDistributionsOverPositions(self.bliefDistributions)
        self.getCurrentAgentPostions(self.getTeam(gameState)[0])
        self.time["AFTER_POISITION_INFERENCE"] = time.time()
        # for index in range(gameState.getNumAgents()): self.log("AGENT", index, "STATE", gameState.data.agentStates[index], "CONF", gameState.data.agentStates[index].configuration)
        bestAction = self.selectAction(gameState)
        return bestAction

    def final(self, gameState):
        PositionInferenceAgent.particleSum = None
        UtilAgent.final(self, gameState)

    #####################
    # virtual functions #
    #####################

    def selectAction(self, gameState):
        util.raiseNotDefined()

    ##############
    # interfaces #
    ##############

    def getCurrentAgentPositionsAndPosibilities(self, agentIndex):
        '''get inference positions and posibilities'''
        gameState = self.getCurrentObservation()
        if agentIndex in self.getTeam(gameState):
            result = [(gameState.getAgentPosition(agentIndex), 1.0)]
        else:
            result = self.bliefDistributions[agentIndex].items()
            result.sort(key=lambda x: x[1], reverse=True)
        return result

    def getCurrentAgentPostions(self, agentIndex):
        '''get inference positions'''
        result = self.getCurrentAgentPositionsAndPosibilities(agentIndex)
        result = [i[0] for i in result]
        return result

    def getCurrentMostLikelyPosition(self, agentIndex):
        return self.getCurrentAgentPostions(agentIndex)[0]

    #############
    # inference #
    #############

    width = None
    height = None
    particleSum = None
    particleDicts = None
    walls = None

    def initPositionInference(self, gameState):
        if PositionInferenceAgent.particleSum is None:
            PositionInferenceAgent.width = gameState.data.layout.width
            PositionInferenceAgent.height = gameState.data.layout.height
            PositionInferenceAgent.particleSum = default_params["particle_sum"]
            PositionInferenceAgent.particleDicts = [None for _ in range(gameState.getNumAgents())]
            PositionInferenceAgent.walls = gameState.getWalls()

            for agentIndex in self.getOpponents(gameState):
                self.initParticleDict(agentIndex)

    def updatePositionInference(self, gameState):
        def update(particleDict, sonarDistance, isStay):
            if isStay and default_params["enable_stay_inference_optimization"]:
                transferedParticleDict = particleDict
            else:
                transferedParticleDict = util.Counter()
                for tile, sum in particleDict.items():
                    x, y = tile
                    available = [tile] if default_params["enable_stop_transition"] else []
                    if not PositionInferenceAgent.walls[x][y + 1]: available.append((x, y + 1))
                    if not PositionInferenceAgent.walls[x][y - 1]: available.append((x, y - 1))
                    if not PositionInferenceAgent.walls[x - 1][y]: available.append((x - 1, y))
                    if not PositionInferenceAgent.walls[x + 1][y]: available.append((x + 1, y))
                    # assume equal trans prob
                    for newTile in available:
                        transferedParticleDict[newTile] += sum / len(available)
                    remainSum = sum % len(available)
                    for _ in range(remainSum):
                        newTile = random.choice(available)
                        transferedParticleDict[newTile] += 1
            agentX, agentY = agentPosition
            candidateParticleDict = util.Counter()
            for tile, sum in transferedParticleDict.items():
                x, y = tile
                distance = abs(agentX - x) + abs(agentY - y)
                newProbability = gameState.getDistanceProb(distance, sonarDistance) * sum
                if newProbability > 0:
                    candidateParticleDict[tile] += newProbability
            if len(candidateParticleDict) > 0:
                newPariticleDict = util.Counter()
                for _ in range(PositionInferenceAgent.particleSum):
                    tile = self.weightedRandomChoice(candidateParticleDict)
                    newPariticleDict[tile] += 1
                return newPariticleDict
            else:
                self.log("Lost target", classification=LogClassification.WARNING)
                return self.getFullParticleDict()

        agentPosition = gameState.getAgentPosition(self.index)

        for agentIndex in range(gameState.getNumAgents()):
            if agentIndex in self.getOpponents(gameState):
                particleDict = PositionInferenceAgent.particleDicts[agentIndex]
                sonarDistance = gameState.agentDistances[agentIndex]
                isStay = not (agentIndex == self.index - 1 or agentIndex == self.index + gameState.getNumAgents() - 1)
                # self.log("Opponent Agent %d is %s" % (agentIndex, "STAY" if isStay else "MOVE"))
                PositionInferenceAgent.particleDicts[agentIndex] = update(particleDict, sonarDistance, isStay)

    def checkPositionInference(self, gameState):
        for agentIndex in range(gameState.getNumAgents()):
            if agentIndex in self.getOpponents(gameState):  # postion of teammates are always available
                # when eat pacman (not for sure)
                def eatPacmanJudge():
                    previous = self.getPreviousObservation()
                    if previous is not None:
                        previousOppoPos = previous.getAgentPosition(agentIndex)
                        if previousOppoPos is not None:
                            if previousOppoPos == gameState.getAgentPosition(self.index):
                                return True
                    return False

                if eatPacmanJudge():
                    self.initParticleDict(agentIndex)

                # when observed
                agentPosition = gameState.getAgentPosition(agentIndex)
                if agentPosition is not None:
                    PositionInferenceAgent.particleDicts[agentIndex] = util.Counter()
                    PositionInferenceAgent.particleDicts[agentIndex][agentPosition] = PositionInferenceAgent.particleSum

    def updateBeliefDistribution(self):
        self.bliefDistributions = [dict.copy() if dict is not None else None for dict in PositionInferenceAgent.particleDicts]
        for dict in self.bliefDistributions: dict.normalize() if dict is not None else None

    #########
    # utils #
    #########

    def getFullParticleDict(self):
        result = util.Counter()
        xStart = 1
        xEnd = PositionInferenceAgent.width - 1
        yStart = 1
        yEnd = PositionInferenceAgent.height - 1
        total = 0
        for x in range(xStart, xEnd):
            for y in range(yStart, yEnd):
                if not PositionInferenceAgent.walls[x][y]:
                    total += 1
        for x in range(xStart, xEnd):
            for y in range(yStart, yEnd):
                if not PositionInferenceAgent.walls[x][y]:
                    result[(x, y)] = PositionInferenceAgent.particleSum / total
        return result

    def initParticleDict(self, opponentAgentIndex):
        if self.red:
            xStart = PositionInferenceAgent.width - 2
            xEnd = PositionInferenceAgent.width - 1
            yStart = PositionInferenceAgent.height / 2
            yEnd = PositionInferenceAgent.height - 1
        else:
            xStart = 1
            xEnd = 2
            yStart = 1
            yEnd = PositionInferenceAgent.height / 2
        total = 0
        for x in range(xStart, xEnd):
            for y in range(yStart, yEnd):
                if not PositionInferenceAgent.walls[x][y]:
                    total += 1
        PositionInferenceAgent.particleDicts[opponentAgentIndex] = util.Counter()
        for x in range(xStart, xEnd):
            for y in range(yStart, yEnd):
                if not PositionInferenceAgent.walls[x][y]:
                    PositionInferenceAgent.particleDicts[opponentAgentIndex][(x, y)] = PositionInferenceAgent.particleSum / total

    def weightedRandomChoice(self, weightDict):
        weights = []
        elems = []
        for elem in weightDict:
            weights.append(weightDict[elem])
            elems.append(elem)
        total = sum(weights)
        key = random.uniform(0, total)
        runningTotal = 0.0
        chosenIndex = None
        for i in range(len(weights)):
            weight = weights[i]
            runningTotal += weight
            if runningTotal >= key:
                chosenIndex = i
                return elems[chosenIndex]
        raise Exception('Should not reach here')

##############################################
#                                            #
# a virtual class                            #
# linear combination of features and weights #
#                                            #
##############################################

class ReflexAgent(PositionInferenceAgent):
    """A virtual agent class. Basiclly same with ReflexAgent in baselineTeam.py, but inherited from PositionInferenceAgent."""

    ######################
    # overload functions #
    ######################

    def registerInitialState(self, gameState):
        PositionInferenceAgent.registerInitialState(self, gameState)
        self.start = gameState.getAgentPosition(self.index)

    def selectAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = gameState.getLegalActions(self.index)
        foodLeft = len(self.getFood(gameState).asList())
        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, self.index, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        self.time["BEFORE_REFLEX"] = time.time()
        bestAction = self.pickAction(gameState)
        self.time["AFTER_REFLEX"] = time.time()

        return bestAction

    #####################
    # virtual functions #
    #####################

    def getFeatures(self, gameState, actionAgentIndex, action):
        util.raiseNotDefined()

    def getWeights(self, gameState, actionAgentIndex, action):
        util.raiseNotDefined()

    ######################
    # linear combination #
    ######################

    def pickAction(self, gameState):
        bestValue = float("-inf")
        bestAction = None
        for action in gameState.getLegalActions(self.index):
            value = self.evaluate(gameState, self.index, action)
            if value > bestValue:
                bestValue = value
                bestAction = action
        return bestAction

    #########
    # utils #
    #########

    def evaluate(self, gameState, actionAgentIndex, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, actionAgentIndex, action)
        weights = self.getWeights(gameState, actionAgentIndex, action)
        return features * weights


###############################
#                             #
# a virtual class             #
# search expectimax in serial #
#                             #
###############################

class TimeoutException(Exception):
    """A custom exception for truncating search."""
    pass

class ExpectimaxAgent(ReflexAgent):
    """A virtual agent class. It uses depth first search to find the best action. It can stop before time limit."""

    ######################
    # overload functions #
    ######################

    def registerInitialState(self, gameState):
        ReflexAgent.registerInitialState(self, gameState)
        self.start = gameState.getAgentPosition(self.index)
        self.maxDepth = default_params["max_depth"]
        self.maxInferencePositionCount = default_params["max_position"]

    def pickAction(self, gameState):
        self.time["BEFORE_SEARCH"] = time.time()
        _, bestAction = self.searchTop(gameState)
        self.time["AFTER_SEARCH"] = time.time()
        return bestAction

    ##########
    # search #
    ##########

    def getCurrentReward(self, gameState, agentIndex, action):
        if default_params["eval_total_reward"]:
            return self.evaluate(gameState, agentIndex, action)
        else:
            return 0

    def searchWhenGameOver(self, gameState):
        return self.evaluate(gameState, self.index, Directions.STOP), Directions.STOP

    def searchWhenZeroDepth(self, gameState, agentIndex):
        isTeam = agentIndex in self.getTeam(gameState)
        bestValue = float("-inf") if isTeam else float("inf")
        bestAction = None
        assert agentIndex == self.index
        legalActions = gameState.getLegalActions(agentIndex)
        legalActions.remove(Directions.STOP)  # STOP is not allowed, to avoid the problem of discontinuous evaluation function
        for action in legalActions:
            value = self.evaluate(gameState, agentIndex, action)
            if (isTeam and value > bestValue) or (not isTeam and value < bestValue):
                bestValue = value
                bestAction = action
        return bestValue, bestAction

    def searchWhenNonTerminated(self, gameState, agentIndex, searchAgentIndices, depth, alpha = float("-inf"), beta = float("inf")):
        nextAgentIndex, nextDepth = self.getNextSearchableAgentIndexAndDepth(gameState, searchAgentIndices, agentIndex, depth)
        bestValue = None
        bestAction = None
        # if agentIndex in self.getTeam(gameState):  # team work
        if agentIndex == self.index:  # no team work, is better
            bestValue = float("-inf")
            legalActions = gameState.getLegalActions(agentIndex)
            if not default_params["enable_stop_action"]:
                legalActions.remove(Directions.STOP)  # STOP is not allowed
            for action in legalActions:
                successorState = gameState.generateSuccessor(agentIndex, action)
                newAlpha, _ = self.searchRecursive(successorState, nextAgentIndex, searchAgentIndices, nextDepth, alpha, beta)
                currentReward = self.evaluate(gameState, agentIndex, action) if default_params["eval_total_reward"] else 0
                newAlpha += currentReward
                if newAlpha > bestValue:
                    bestValue = newAlpha
                    bestAction = action
                if newAlpha > alpha: alpha = newAlpha
                if alpha >= beta: break
        else:
            bestValue = float("inf")
            for action in gameState.getLegalActions(agentIndex):
                successorState = gameState.generateSuccessor(agentIndex, action)
                newBeta, _ = self.searchRecursive(successorState, nextAgentIndex, searchAgentIndices, nextDepth, alpha, beta)
                if newBeta < bestValue:
                    bestValue = newBeta
                    bestAction = action
                if newBeta < beta: beta = newBeta
                if alpha >= beta: break
        return bestValue, bestAction

    def searchRecursive(self, gameState, agentIndex, searchAgentIndices, depth, alpha = float("-inf"), beta = float("inf")):
        if gameState.isOver():
            result =  self.searchWhenGameOver(gameState)
        elif depth == 0:
            result = self.searchWhenZeroDepth(gameState, agentIndex)
        else:
            self.ifTimeoutRaiseTimeoutException()
            result = self.searchWhenNonTerminated(gameState, agentIndex, searchAgentIndices, depth, alpha, beta)
        return result

    def searchTop(self, gameState):
        inferenceState = gameState.deepCopy()
        legalActions = gameState.getLegalActions(self.index)
        agentInferencePositionsAndPosibilities = [self.getCurrentAgentPositionsAndPosibilities(agentIndex) for agentIndex in range(gameState.getNumAgents())]
        agentInferencePositions = [self.getCurrentAgentPostions(agentIndex) for agentIndex in range(gameState.getNumAgents())]
        initPointers = [0 for _ in range(gameState.getNumAgents())]
        pointers = None
        upLimits = [min(self.maxInferencePositionCount, len(agentInferencePositionsAndPosibilities[agentIndex])) for agentIndex in range(gameState.getNumAgents())]
        myPosition = inferenceState.getAgentPosition(self.index)
        def changePointer():
            changeAgentIndex = None
            minPointer = 9999
            for agentIndex in range(gameState.getNumAgents()):
                if pointers[agentIndex] + 1 < upLimits[agentIndex] and pointers[agentIndex] < minPointer:
                    minPointer = pointers[agentIndex]
                    changeAgentIndex = agentIndex
            if changeAgentIndex is not None:
                pointers[changeAgentIndex] += 1
                return True
            else:
                return False
        def setConfigurations(origionState, inferenceState):
            totalPosibility = 1.0
            for agentIndex in range(origionState.getNumAgents()):
                if origionState.getAgentState(agentIndex).configuration is None:
                    position, posibility = agentInferencePositionsAndPosibilities[agentIndex][pointers[agentIndex]]
                    inferenceState.data.agentStates[agentIndex].configuration = game.Configuration(position, Directions.STOP)
                else:
                    posibility = 1.0
                totalPosibility *= posibility
            return totalPosibility
        def getSearchAgentIndices(gameState, myPosition, searchMaxDistance):
            searchAgentIndices = []
            for agentIndex in range(gameState.getNumAgents()):
                agentPosition = gameState.getAgentPosition(agentIndex)
                if agentPosition is not None and self.getMazeDistance(agentPosition, myPosition) <= searchMaxDistance:  # the origion is mahattan distance
                    searchAgentIndices.append(agentIndex)
            return searchAgentIndices
        bestAction = None
        bestValue = float("-inf")
        for searchDepth in range(self.maxDepth + 1):
            searchSuccess = False
            localBestValue = None
            localBestAction = None
            try:
                localAverageBestValue = 0.0
                totalPosibility = 1.0
                localResults = []
                considerationDistance = int(searchDepth * default_params["consideration_distance_factor"])
                # self.log("Search depth [%d], consideration distance is [%d]" % (searchDepth, considerationDistance))
                pointers = initPointers
                while True:
                    posibility = setConfigurations(gameState, inferenceState)
                    searchAgentIndices = getSearchAgentIndices(inferenceState, myPosition, considerationDistance)
                    self.log("Take agents %s in to consideration" % searchAgentIndices)
                    value, action = self.searchRecursive(inferenceState, self.index, searchAgentIndices, searchDepth)
                    localResults.append([value, action])
                    totalPosibility *= posibility
                    localAverageBestValue += posibility * value
                    if not changePointer(): break
                localAverageBestValue /= totalPosibility
                minDifference = float("inf")
                for value, action in localResults:
                    difference = abs(value - localAverageBestValue)
                    if difference < minDifference:
                        minDifference = difference
                        localBestAction = action
                        localBestValue = value
                searchSuccess = True
            except TimeoutException:
                pass
            # except multiprocessing.TimeoutError: pass  # Coment this line if you want to use keyboard interrupt
            if searchSuccess:
                bestValue = localBestValue
                bestAction = localBestAction
            else:
                self.log("Failed when search max depth [%d]" % (searchDepth,))
                break
        self.log("Take action [%s] with evaluation [%.6f]" % (bestAction, bestValue))
        return bestValue, bestAction

    #########
    # utils #
    #########

    def ifTimeoutRaiseTimeoutException(self):
        if self.timeRemainPercent() < default_params["truncate_remain_time_percent"]:
            raise TimeoutException()

    def getNextAgentIndex(self, gameState, currentAgentIndex):
        nextAgentIndex = currentAgentIndex + 1
        nextAgentIndex = 0 if nextAgentIndex >= gameState.getNumAgents() else nextAgentIndex
        return nextAgentIndex

    def getNextSearchableAgentIndexAndDepth(self, gameState, searchAgentIndices, currentAgentIndex, currentDepth):
        nextAgentIndex = currentAgentIndex
        nextDepth = currentDepth
        while True:
            nextAgentIndex = self.getNextAgentIndex(gameState, nextAgentIndex)
            nextDepth = nextDepth - 1 if nextAgentIndex == self.index else nextDepth
            if nextAgentIndex in searchAgentIndices: break
        return nextAgentIndex, nextDepth

#################################
#                               #
# a virtual class               #
# search expectimax in parallel #
#                               #
#################################

import multiprocessing
import threading

def searchSerial(agent, gameState, agentIndex, searchAgentIndices, depth, alpha, beta):
    """A function for processes search in serial."""
    if gameState.isOver():
        return agent.searchWhenGameOver(gameState)
    elif depth == 0:
        return agent.searchWhenZeroDepth(gameState, agentIndex)
    else:
        agent.ifTimeoutRaiseTimeoutException()
        nextAgentIndex, nextDepth = agent.getNextSearchableAgentIndexAndDepth(gameState, searchAgentIndices, agentIndex, depth)
        bestValue = None
        bestAction = None
        # if agentIndex in self.getTeam(gameState):  # team work
        if agentIndex == agent.index:  # no team work, is better
            bestValue = float("-inf")
            legalActions = gameState.getLegalActions(agentIndex)
            if not default_params["enable_stop_action"]:
                legalActions.remove(Directions.STOP)  # STOP is not allowed
            for action in legalActions:
                successorState = gameState.generateSuccessor(agentIndex, action)
                newAlpha, _ = searchSerial(agent, successorState, nextAgentIndex, searchAgentIndices, nextDepth, alpha, beta)
                currentReward = agent.evaluate(gameState, agentIndex, action) if default_params["eval_total_reward"] else 0
                newAlpha += currentReward
                if newAlpha > bestValue:
                    bestValue = newAlpha
                    bestAction = action
                if newAlpha > alpha: alpha = newAlpha
                if alpha >= beta: break
        else:
            bestValue = float("inf")
            for action in gameState.getLegalActions(agentIndex):
                successorState = gameState.generateSuccessor(agentIndex, action)
                newBeta, _ = searchSerial(agent, successorState, nextAgentIndex, searchAgentIndices, nextDepth, alpha, beta)
                if newBeta < bestValue:
                    bestValue = newBeta
                    bestAction = action
                if newBeta < beta: beta = newBeta
                if alpha >= beta: break
        return bestValue, bestAction

class SearchThread(threading.Thread):
    """A thread sub-class for expanding search tree and distributing tasks to processes"""

    def __init__(self, agent, action, gameState, agentIndex, searchAgentIndices, depth, alpha, beta):
        threading.Thread.__init__(self)
        self.agent = agent
        self.action = action
        self.gameState = gameState
        self.agentIndex = agentIndex
        self.searchAgentIndices = searchAgentIndices
        self.depth = depth
        self.alpha = alpha
        self.beta = beta
        self.subThreads = []
        self.success = True
        self.exception = None
        self.bestValue = None
        self.bestAction = None

    def expandWhenNonTerminated(self, agent, gameState, agentIndex, searchAgentIndices, depth, alpha, beta):
        agent.ifTimeoutRaiseTimeoutException()
        nextAgentIndex, nextDepth = agent.getNextSearchableAgentIndexAndDepth(gameState, searchAgentIndices, agentIndex, depth)
        bestValue = None
        bestAction = None
        # if agentIndex in self.getTeam(gameState):  # team work
        if agentIndex == agent.index:  # no team work, is better
            bestValue = float("-inf")
            legalActions = gameState.getLegalActions(agentIndex)
            if not default_params["enable_stop_action"]:
                legalActions.remove(Directions.STOP)  # STOP is not allowed
            ExpectimaxParallelAgent.threadCountLock.acquire()
            for action in legalActions:
                successorState = gameState.generateSuccessor(agentIndex, action)
                subThread = SearchThread(agent, action, successorState, nextAgentIndex, searchAgentIndices, nextDepth, alpha, beta)
                self.subThreads.append(subThread)
                agent.threadCount += 1
                subThread.setDaemon(True)
                subThread.start()
            agent.threadCount -= 1
            ExpectimaxParallelAgent.threadCountLock.release()
            for thread in self.subThreads:
                thread.join()
                newAlpha, _ = thread.getResult()
                currentReward = agent.evaluate(gameState, agentIndex, thread.action) if default_params["eval_total_reward"] else 0
                newAlpha += currentReward
                if newAlpha > bestValue:
                    bestValue = newAlpha
                    bestAction = thread.action
                if newAlpha > alpha: alpha = newAlpha
                if alpha >= beta: break
        else:
            bestValue = float("inf")
            legalActions = gameState.getLegalActions(agentIndex)
            ExpectimaxParallelAgent.threadCountLock.acquire()
            for action in legalActions:
                successorState = gameState.generateSuccessor(agentIndex, action)
                subThread = SearchThread(agent, action, successorState, nextAgentIndex, searchAgentIndices, nextDepth, alpha, beta)
                self.subThreads.append(subThread)
                agent.threadCount += 1
                subThread.setDaemon(True)
                subThread.start()
            agent.threadCount -= 1
            ExpectimaxParallelAgent.threadCountLock.release()
            for thread in self.subThreads:
                thread.join()
                newBeta, _ = thread.getResult()
                if newBeta < bestValue:
                    bestValue = newBeta
                    bestAction = thread.action
                if newBeta < beta: beta = newBeta
                if alpha >= beta: break
        return bestValue, bestAction

    def expandRecursive(self, agent, gameState, agentIndex, searchAgentIndices, depth, alpha, beta):
        ExpectimaxParallelAgent.threadCountLock.acquire()
        if agent.threadCount >= ExpectimaxParallelAgent.threadCountThreashold:
            # ExpectimaxParallelAgent.printLock.acquire()
            # agent.log("%s find there are %d threads, it is larger then %d, so start a process search depth %d" % (threading.currentThread().getName(), agent.threadCount, ExpectimaxParallelAgent.threadCountThreashold, depth))
            # ExpectimaxParallelAgent.printLock.release()

            if not default_params["enable_coarse_partition"]:
                agent.threadCount -= 1  # enable this line can divide task in small grant size, but increase data transformation and synchronization
            ExpectimaxParallelAgent.threadCountLock.release()

            ExpectimaxParallelAgent.poolLock.acquire()
            asyncResult = ExpectimaxParallelAgent.pool.apply_async(searchSerial, (agent, gameState, agentIndex, searchAgentIndices, depth, alpha, beta))
            ExpectimaxParallelAgent.poolLock.release()

            # return asyncResult.get(agent.actionTimeLimit * 2)  # add a time out to solve a python KeyboardInterrupt bug
            return asyncResult.get()  # the origion one
        else:
            ExpectimaxParallelAgent.threadCountLock.release()

            if gameState.isOver():
                return agent.searchWhenGameOver(gameState)
            elif depth == 0:
                return agent.searchWhenZeroDepth(gameState, agentIndex)
            else:
                return self.expandWhenNonTerminated(agent, gameState, agentIndex, searchAgentIndices, depth, alpha, beta)

    def run(self):
        try:
            self.bestValue, self.bestAction = self.expandRecursive(self.agent, self.gameState, self.agentIndex, self.searchAgentIndices, self.depth, self.alpha, self.beta)
        except Exception as e:
            self.success = False
            self.exception = e

    def getResult(self):
        if self.success:
            return self.bestValue, self.bestAction
        else:
            raise self.exception

class ExpectimaxParallelAgent(ExpectimaxAgent):
    """A virtual agent class. Search expectimax in parallel. The speed-up ratio will be large if the time limit is loose."""

    ######################
    # overload functions #
    ######################

    pool = None
    processorCount = None
    poolLock = None
    threadCountLock = None
    printLock = None
    threadCountThreashold = None

    def registerInitialState(self, gameState):
        ExpectimaxAgent.registerInitialState(self, gameState)
        if ExpectimaxParallelAgent.processorCount == None:
            ExpectimaxParallelAgent.processorCount = multiprocessing.cpu_count()
            ExpectimaxParallelAgent.pool = multiprocessing.Pool(ExpectimaxParallelAgent.processorCount)
            ExpectimaxParallelAgent.threadCountLock = threading.Lock()
            ExpectimaxParallelAgent.poolLock = threading.Lock()
            ExpectimaxParallelAgent.printLock = threading.Lock()
            ExpectimaxParallelAgent.threadCountThreashold = int(ExpectimaxParallelAgent.processorCount * default_params["expand_factor"])

    def final(self, gameState):
        if ExpectimaxParallelAgent.processorCount is not None:
            ExpectimaxParallelAgent.pool.close()
            ExpectimaxParallelAgent.pool.terminate()
            ExpectimaxParallelAgent.pool.join()
            ExpectimaxParallelAgent.processorCount = None
        ExpectimaxAgent.final(self, gameState)

    ##########
    # search #
    ##########

    def searchRecursive(self, gameState, agentIndex, searchAgentIndices, depth, alpha = float("-inf"), beta = float("inf")):
        self.threadCount = 1
        thread = SearchThread(self, None, gameState, agentIndex, searchAgentIndices, depth, alpha, beta)
        thread.start()
        thread.join()
        return thread.getResult()

######################################
#                                    #
# a virtual class                    #
# using q-learning to learn Weights  #
#                                    #
######################################

class ApproximateQLearningAgent(ExpectimaxAgent):
    """A virtual agent class. Using for learn weights of features automatically. It can save and load results."""

    ######################
    # overload functions #
    ######################

    def registerInitialState(self, gameState):
        default_params["eval_total_reward"] = False
        default_params["max_depth"] = 0
        ExpectimaxAgent.registerInitialState(self, gameState)
        self.weights = self.getInitWeights()
        self.actionHistory = [Directions.STOP]
        self.step = 0
        filename = default_params["filename"]
        if filename is not None:
            import os
            if os.path.exists(filename):
                self.loadWeights(filename)
                self.log("Load weights from " + filename)

    def final(self, gameState):
        if default_params["filename"] is not None:
            self.saveWeights(default_params["filename"])
        ExpectimaxAgent.final(self, gameState)

    def getWeights(self, gameState, actionAgentIndex, action):
        return self.weights

    def searchTop(self, gameState):
        bestValue, bestAction = ExpectimaxAgent.searchTop(self, gameState)
        self.learn(bestValue, bestAction)
        return bestValue, bestAction

    #####################
    # virtual functions #
    #####################

    def getInitWeights(self):
        util.raiseNotDefined

    def getFeatures(self, gameState, actionAgentIndex, action):
        util.raiseNotDefined

    def getReward(self, previousState, previousAction, currentState, currentAction):
        util.raiseNotDefined

    ##############
    # q-learning #
    ##############

    def learn(self, bestValue, bestAction):
        self.time["BEFORE_Q-LEARNING"] = time.time()
        self.step += 1
        self.log("Step %d" % (self.step,))
        currentGameState = self.getCurrentObservation()
        previousGameState = self.getPreviousObservation()
        if previousGameState is None:
            previousGameState = currentGameState
        previousAction = self.actionHistory[-1]
        reward = self.getReward(previousGameState, previousAction, currentGameState, bestAction)
        discount = default_params["discount"]
        difference = (reward + discount * bestValue) - self.evaluate(previousGameState, self.index, previousAction)
        learningRate = default_params["learning_rate"]
        features = self.getFeatures(previousGameState, self.index, previousAction)
        for k, v in features.items():
            if k in self.weights:
                self.weights[k] += learningRate * difference * v

        if self.step % default_params["save_interval"] == 0 and default_params["filename"] is not None:
            self.saveWeights(default_params["filename"])
        self.actionHistory.append(bestAction)
        self.time["AFTER_Q-LEARNING"] = time.time()

    #########
    # utils #
    #########

    def loadWeights(self, filename):
        with open(filename) as f:
            self.step = eval(f.readline())
            self.weights = eval(f.readline())

    def saveWeights(self, filename):
        with open(filename, "w") as f:
            f.write(str(self.step) + "\n")
            f.write(str(self.weights) + "\n")


#################################
#                               #
# monte carlo tree search agent #
#                               #
#################################

class StateTreeNode:
    """A class(struct) for representing a monte carlo tree node."""
    def __init__(self, parent, gameState, agentIndex):
        self.parent = parent
        self.state = gameState
        self.agentIndex = agentIndex
        self.accessTime = 0
        self.reward = 0
        self.children = None

import math

class MonteCarloTreeSearchAgent(PositionInferenceAgent):
    """An agent class. Implemented UCT algorithm. It simulate the game randomlly. You can use it directly. I think MCTS is not good for solving this question, because it is too slow."""

    ######################
    # overload functions #
    ######################

    def registerInitialState(self, gameState):
        PositionInferenceAgent.registerInitialState(self, gameState)
        pass

    def selectAction(self, gameState):
        self.time["BEFORE_SAMPLE"] = time.time()
        inferenceState = self.setMostLikelyPositionConfigurations(gameState)
        root = StateTreeNode(None, inferenceState, self.index)
        self.sample(root)
        self.time["AFTER_SAMPLE"] = time.time()
        bestUCB = float("-inf")
        bestAction = None
        for i in range(len(root.children)):
            ucb = self.getUCB(root, i)
            if ucb > bestUCB:
                bestUCB = ucb
                _, bestAction = root.children[i]
        return bestAction

    ##############
    # interfaces #
    ##############

    def initFastActionEnvironment(self, gameState):
        pass

    def clearFastActionEnvironment(self, gameState):
        pass

    def getFastAction(self, gameState, agentIndex):
        """Random choose an action. You can overload this function to use better simulate strategy."""
        legalActions = gameState.getLegalActions(agentIndex)
        if not default_params["enable_stop_action"]:
            legalActions.remove(Directions.STOP)
        return random.choice(legalActions)

    ###############
    # monte carlo #
    ###############

    def getFinalState(self, gameState, agentIndex):
        """Simulate until the game is finished."""
        if gameState.isOver():
            return gameState
        else:
            state = gameState.deepCopy()
            self.initFastActionEnvironment(state)
            while not state.isOver():
                action = self.getFastAction(state, agentIndex)
                state = self.getSuccessor(state, agentIndex, action)
                agentIndex = (agentIndex + 1) % gameState.getNumAgents()
            self.clearFastActionEnvironment(state)
            return state

    def getFinalStateReward(self, finalState):
        """Get final state reward."""
        score = self.getScore(finalState)
        if score > 0:
            return 1
        elif score < 0:
            return -1
        else:
            return 0

    def getUCB(self, node, index):
        child, _ = node.children[index]
        w = float(child.reward)
        n = float(child.accessTime)
        c = default_params["exploration"]
        t = float(node.accessTime)
        return w / n + c * math.sqrt(math.log(t, math.e) / n)

    def getUCBs(self, node):
        return [self.getUCB(node, i) for i in range(len(node.children))]

    def selectNode(self, node):
        if node.state.isOver():
            return node
        else:
            if node.children is None:
                legalActions = node.state.getLegalActions(node.agentIndex)
                if not default_params["enable_stop_action"]:
                    legalActions.remove(Directions.STOP)
                node.children = [[None, action] for action in legalActions]
            nextAgentIndex = (node.agentIndex + 1) % node.state.getNumAgents()
            nextNode = None
            for pair in node.children:
                child, action = pair
                if child is None:
                    nextState = self.getSuccessor(node.state, node.agentIndex, action)
                    nextNode = StateTreeNode(node, nextState, nextAgentIndex)
                    pair[0] = nextNode
                    break
            if nextNode is not None:
                return nextNode
            else:
                ucbs = self.getUCBs(node)
                # if node.agentIndex in self.getTeam(node.state):  # team work
                if node.agentIndex == self.index:
                    ucb = max(ucbs)
                else:
                    ucb = min(ucbs)
                i = ucbs.index(ucb)
                nextNode, action = node.children[i]
            return self.selectNode(nextNode)

    def sampleOnce(self, root):
        leaf = self.selectNode(root)
        finalState = self.getFinalState(leaf.state, leaf.agentIndex)
        reward = self.getFinalStateReward(finalState)
        node = leaf
        while True:
            node.reward += reward
            node.accessTime += 1
            node = node.parent
            if node is None:
                break

    def sample(self, root):
        number = default_params["max_sample"]
        count = 0
        for _ in range(number):
            self.sampleOnce(root)
            count += 1
            self.log("%d sample, accumulated reward is %.3f" % (count, root.reward), classification=LogClassification.DETAIL)
            if self.timeRemainPercent() < default_params["truncate_remain_time_percent"]:
                break
        self.log("Totally sampled %d times, reward is %.3f" % (count, root.reward))

    #########
    # utils #
    #########

    def setMostLikelyPositionConfigurations(self, origionState):
        inferenceState = origionState.deepCopy()
        for agentIndex in range(origionState.getNumAgents()):
            if origionState.getAgentState(agentIndex).configuration is None:
                position = self.getCurrentMostLikelyPosition(agentIndex)
                inferenceState.data.agentStates[agentIndex].configuration = game.Configuration(position, Directions.STOP)
        return inferenceState

###########################################
#                                         #
# a virtual class                         #
# Optimized monte carlo tree search agent #
#                                         #
###########################################

import imp

class OptimizedMonteCarloTreeSearchAgent(MonteCarloTreeSearchAgent):
    """A virtual agent class. Using custom strategy to replace random strategy in super class."""

    ######################
    # overload functions #
    ######################

    def registerInitialState(self, gameState):
        MonteCarloTreeSearchAgent.registerInitialState(self, gameState)
        moduleFilename = self.getModuleFileName()
        if not moduleFilename.endswith(".py"):
            moduleFilename += ".py"
        module = imp.load_source("SimulateAgentModule", moduleFilename)
        self.setModule(module)
        agentClassNames = [self.getAgentClassName(i) for i in range(gameState.getNumAgents())]
        self.agents = [getattr(module, agentClassNames[i])(i) for i in range(gameState.getNumAgents())]

    def initFastActionEnvironment(self, gameState):
        for agent in self.agents:
            if "registerInitialState" in dir(agent):
                agent.registerInitialState(gameState)

    def clearFastActionEnvironment(self, gameState):
        for agent in self.agents:
            if "final" in dir(agent):
                agent.final(gameState)

    def getFastAction(self, gameState, agentIndex):
        return self.agents[agentIndex].getAction(gameState)

    #####################
    # virtual functions #
    #####################

    def getModuleFileName(self):
        """Return a module file name, which comtaing class(es) implemented CaptureAgent class."""
        util.raiseNotDefined()

    def setModule(self, module):
        """Overload this function to set params if nessesary."""
        pass

    def getAgentClassNames(self, index):
        util.raiseNotDefined()

################################################
#                                              #
# baseline offensive & defensive reflex agents #
#                                              #
################################################

class BasicOffensiveReflexAgent(ReflexAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, actionAgentIndex, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, actionAgentIndex, action)
    foodList = self.getFood(successor).asList()
    features['successorScore'] = -len(foodList)#self.getScore(successor)

    # Compute distance to the nearest food

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, actionAgentIndex, action):
    return {'successorScore': 100, 'distanceToFood': -1}

class BasicDefensiveReflexAgent(ReflexAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, actionAgentIndex, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, actionAgentIndex, action)

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

  def getWeights(self, gameState, actionAgentIndex, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}

#######################################
#                                     #
# DEPRECATED!                         #
# offensive & defensive reflex agents #
#                                     #
#######################################

class BaseAgent(ExpectimaxParallelAgent):
    def registerInitialState(self, gameState):
        ExpectimaxParallelAgent.registerInitialState(self, gameState)
        self.homePos = []
        self.alertDistance = 5
        self.catchDistance = 5
        self.eatCapsuleDistance = 5
        x = gameState.data.layout.width / 2 - 1 if self.red else gameState.data.layout.width / 2
        for y in range(gameState.data.layout.height):
            if not gameState.data.layout.walls[x][y]:
                self.homePos.append((x, y))

class OffensiveReflexAgent(BaseAgent):

    def getFeatures(self, gameState, actionAgentIndex, action):
        """
        features:
        successorScore: minus number of food moving towards the given action
        distanceToFood: maze distance to the closest food
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, actionAgentIndex, action)
        myState = successor.getAgentState(self.index)
        myPos = successor.getAgentState(self.index).getPosition()
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        foodList = self.getFood(successor).asList()

        # Distance to closest ghost. The more the better. Only matters if distance greater than self.alertDistance

        ghostPos = [a.getPosition() for a in enemies if not a.isPacman and a.scaredTimer == 0 and a.getPosition() != None]
        if len(ghostPos) > 0:
          features['ghostDistance'] = min([self.getMazeDistance(myPos, pos) for pos in ghostPos])
        # if distance greater than alertDistance, mute this feature
          if features['ghostDistance'] > self.alertDistance:
            features['ghostDistance'] = 0
        else:
          features['ghostDistance'] = 0

        # Distance to scared ghost. The less the better. Only matters if distance greater than self.catchDistance
        scaredGhostPos = [a.getPosition() for a in enemies if not a.isPacman and a.scaredTimer > 0 and  a.getPosition() != None]
        if len(scaredGhostPos) > 0:
          features['scaredGhostDistance'] = min([self.getMazeDistance(myPos, pos) for pos in scaredGhostPos])
        # if distance greater than alertDistance, mute this feature
          if features['scaredGhostDistance'] > self.catchDistance:
            features['scaredGhostDistance'] = 0
        else:
          features['scaredGhostDistance'] = 0

        # Distance to Capsule. The less the better. Only matters if distance greater than self.eatCapsuleDistance
        capsulePos = self.getCapsules(successor)
        if len(capsulePos) == 0 or capsulePos[0] > self.eatCapsuleDistance:
          features['capsuleDistance'] = 0
        else:
          features['capsuleDistance'] = self.getMazeDistance(myPos, capsulePos[0])

        # Distance to nearest food. The less the better. Always matters
        if len(foodList) > 0:
          features['foodDistance'] = min([self.getMazeDistance(myPos, food) for food in foodList])
          # features['foodDistance'] = -features['foodDistance']
        else:
          features['foodDistance'] = 0

        # Distance to own territory, the less the better. The further the agent is from center, the more dangerous it is.
        features['homeDistance'] = min([self.getMazeDistance(myPos, d) for d in self.homePos])

        # Whether pacman is in a tunnel, bad if in tunnel
        # features['inTunnel'] = 0
        # successorActions = successor.getLegalActions(self.index)
        # if myState.isPacman and len(successorActions) == 2:
        #     features['inTunnel'] = 1

        # Does not prefer stop action and reverse action. Negative weights
        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1



        # Number of food left, the less the better. A successor state with less food is favored!
        foodList = self.getFood(successor).asList()
        features['foodLeft'] = len(foodList)  # self.getScore(successor)
        return features

    def getWeights(self, gameState, actionAgentIndex, action):
        return {'ghostDistance': 100, 'scaredGhostDistance':-50, 'capsuleDistance': -40, 'foodDistance': -30, 'homeDistance': -5,\
                'foodLeft': -100, 'stop':-100, 'reverse':-15}


class DefensiveReflexAgent(BaseAgent):

    def getFeatures(self, gameState, actionAgentIndex, action):

        features = util.Counter()
        successor = self.getSuccessor(gameState, actionAgentIndex, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        # Computes whether the agent is a scared ghost, whether we're currently a scared ghost:
        features['isScared'] = 0
        if not myState.isPacman and myState.scaredTimer > 0:
          features['isScared'] = 1

        # Number of invaders
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)

        # Distance to invaders we can see. The less the better.
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        # Distance to center of own territory. The less the better.
        if self.red:
          center = (gameState.data.layout.width / 2 - 1 ,  gameState.data.layout.height / 2 - 1)
        else:
          center = (gameState.data.layout.width / 2, gameState.data.layout.height / 2)
        features['centerDistance'] = self.getMazeDistance(center,myPos)


        # Does not prefer stop action and reverse action. Negative weights
        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def getWeights(self, gameState, actionAgentIndex, action):
        return {'onDefense': 100, 'isScared':100, 'numInvaders': -1000, 'invaderDistance': -10, 'centerDistance': -1, }#'stop': -100, 'reverse': -2}

###########################
#                         #
# Reward evaluation agent #
#                         #
###########################

from game import Actions
class ActionEvaluationAgent(ExpectimaxAgent):
    """An agent class. It evaluate the reward not the state. You can use it directly."""

    def getFeatures(self, gameState, actionAgentIndex, action):
        foodList = self.getFood(gameState).asList()
        capsuleList = self.getCapsules(gameState)
        walls = gameState.getWalls()
        height = walls.height
        width = walls.width
        area = (self.index - self.index % 2) / 2
        isPacman = self.getSuccessor(gameState, actionAgentIndex, action).getAgentState(self.index).isPacman

        def getAgent(agentIndex):
            return gameState.getAgentState(agentIndex)
        def isNearby(pos1, pos2):
            return pos1 in Actions.getLegalNeighbors(pos2, walls)
        def isScared(agent):
            return agent.scaredTimer > 0

        position = getAgent(self.index).getPosition()
        nextPosition = Actions.getSuccessor(position, action)

        teammates = [getAgent(i) for i in self.getTeam(gameState)]
        opponents = [getAgent(i) for i in self.getOpponents(gameState)]
        chaserList = [a for a in opponents if not a.isPacman and a.getPosition() is not None]
        preyerList = [a for a in opponents if a.isPacman and a.getPosition() is not None]

        features = util.Counter()

        if action == Directions.STOP:
            features["stopped"] = 1.0

        for agent in chaserList:
            if nextPosition == agent.getPosition():
                if isScared(agent):
                    features["eat_ghost"] += 1
                else:
                    features["nearby_harmful_ghost_count"] = 1
            elif isNearby(nextPosition, agent.getPosition()):
                if isScared(agent):
                    features["nearby_harmless_ghost_count"] += 1
                elif isPacman:
                    features["nearby_harmful_ghost_count"] += 1

        for agent in chaserList:
            if (nextPosition == agent.getPosition() and not isScared(agent)) or (isPacman and isNearby(nextPosition, agent.getPosition())):
                features["nearby_harmless_ghost_count"] = 0
                break

        if not isScared(getAgent(self.index)):
            for agent in preyerList:
                if nextPosition == agent.getPosition:
                    features["eat_invader"] = 1
                elif isNearby(nextPosition, agent.getPosition()):
                    features["nearby_harmless_invader_count"] += 1
        else:
            for agent in opponents:
                if agent.getPosition() is not None:
                    if nextPosition == agent.getPosition():
                        features["eat_invader"] = -10
                    elif isNearby(nextPosition, agent.getPosition()):
                        features["nearby_harmless_invader_count"] += -10

        for capsule in capsuleList:
            if nextPosition == capsule:
                features["eat_capsule"] = 1

        if not features["nearby_harmful_ghost_count"]:
            if nextPosition in foodList:
                features["eat_food"] = 1
            if len(foodList) > 0:
                inChargeFoodList = []
                for food in foodList:
                    foodX, foodY = food
                    if (foodY > area * height / 3 and foodY < (area + 1) * height / 3):
                        inChargeFoodList.append(food)
                if len(inChargeFoodList) == 0:
                    inChargeFoodList = foodList
                if len(inChargeFoodList) > 0:
                    nearestFoodDistance = min([self.getMazeDistance(nextPosition, food) for food in inChargeFoodList])
                    features["nearest_food_distance"] = float(nearestFoodDistance) / (height * width)

        features["food_left"] = len(foodList)

        return features

    def getWeights(self, gameState, actionAgentIndex, action):
        return {"stopped": -1.0, "food_left": -2.0, "eat_food": 1.0, "eat_capsule": 10.0, "eat_invader": 5.0, "eat_ghost": 3.0, "nearest_food_distance": -1,
                "nearby_harmless_invader_count": 1.0, "nearby_harmful_ghost_count": -10.0, "nearby_harmless_ghost_count": 0.1}

class ActionEvaluationOffensiveAgent(ActionEvaluationAgent):
    """An agent class. Optimized for offense. You can use it directly."""

    ######################
    # overload functions #
    ######################

    def getWeights(self, gameState, actionAgentIndex, action):
        return {"stopped": -1.0, "food_left": -2.0, "eat_food": 1.0, "eat_capsule": 10.0, "eat_invader": 5.0, "eat_ghost": 3.0, "nearest_food_distance": -1,
                "nearby_harmless_invader_count": 0.0, "nearby_harmful_ghost_count": -10.0, "nearby_harmless_ghost_count": 0.1}

###########################
#                         #
# State evaluation agents #
#                         #
###########################

class StateEvaluationAgent(ExpectimaxAgent):
    """An agent class. Evaluate the state, not the reward. You can use it directly."""

    ######################
    # overload functions #
    ######################

    def getFeatures(self, gameState, actionAgentIndex, action):
        assert actionAgentIndex == self.index
        successor = self.getSuccessor(gameState, actionAgentIndex, action)

        walls = successor.getWalls()
        position = successor.getAgentPosition(self.index)
        teamIndices = self.getTeam(successor)
        opponentIndices = self.getOpponents(successor)
        foodList = self.getFood(successor).asList()
        foodList.sort(key=lambda x: self.getMazeDistance(position, x))
        defendFoodList = self.getFoodYouAreDefending(successor).asList()
        capsulesList = self.getCapsules(successor)
        capsulesList.sort(key=lambda x: self.getMazeDistance(position, x))
        defendCapsulesList = self.getCapsulesYouAreDefending(successor)
        scaredTimer = successor.getAgentState(self.index).scaredTimer
        foodCarrying = successor.getAgentState(self.index).numCarrying
        foodReturned = successor.getAgentState(self.index).numReturned
        stopped = action == Directions.STOP
        reversed = action != Directions.STOP and Actions.reverseDirection(successor.getAgentState(self.index).getDirection()) == gameState.getAgentState(self.index).getDirection()

        def isPacman(state, index):
            return state.getAgentState(index).isPacman
        def isGhost(state, index):
            return not isPacman(state, index)
        def isScared(state, index):
            return state.data.agentStates[index].scaredTimer > 0  # and isGhost(state, index)
        def isInvader(state, index):
            return index in opponentIndices and isPacman(state, index)
        def isHarmfulInvader(state, index):
            return isInvader(state, index) and isScared(state, self.index)
        def isHarmlessInvader(state, index):
            return isInvader(state, index) and not isScared(state, self.index)
        def isHarmfulGhost(state, index):
            return index in opponentIndices and isGhost(state, index) and not isScared(state, index)
        def isHarmlessGhost(state, index):
            return index in opponentIndices and isGhost(state, index) and isScared(state, index)

        def getDistance(pos):
            return self.getMazeDistance(position, pos)
        def getPosition(state, index):
            return state.getAgentPosition(index)
        def getScaredTimer(state, index):
            return state.getAgentState(index).scaredTimer
        def getFoodCarrying(state, index):
            return state.getAgentState(index).numCarrying
        def getFoodReturned(state, index):
            return state.getAgentState(index).numReturned
        def getPositionFactor(distance):
            return (float(distance) / (walls.width * walls.height))

        features = util.Counter()

        features["stopped"] = 1 if stopped else 0

        features["reversed"] = 1 if reversed else 0

        features["scared"] = 1 if isScared(successor, self.index) else 0

        features["food_returned"] = foodReturned

        features["food_carrying"] = foodCarrying

        features["food_defend"] = len(defendFoodList)

        features["nearest_food_distance_factor"] = float(getDistance(foodList[0])) / (walls.height * walls.width) if len(foodList) > 0 else 0

        features["nearest_capsules_distance_factor"] = float(getDistance(capsulesList[0])) / (walls.height * walls.width) if len(capsulesList) > 0 else 0

        returnFoodX = walls.width / 2 - 1 if self.red else walls.width / 2
        nearestFoodReturnDistance = min([getDistance((returnFoodX, y)) for y in range(walls.height) if not walls[returnFoodX][y]])
        features["return_food_factor"] = float(nearestFoodReturnDistance) / (walls.height * walls.width) * foodCarrying

        features["team_distance"] = float(sum([getDistance(getPosition(successor, i)) for i in teamIndices if i != self.index])) / (walls.height * walls.width)

        harmlessInvaders = [i for i in opponentIndices if isHarmlessInvader(successor, i)]
        features["harmless_invader_distance_factor"] = max([getPositionFactor(getDistance(getPosition(successor, i))) * (getFoodCarrying(successor, i) + 5) for i in harmlessInvaders]) if len(harmlessInvaders) > 0 else 0

        harmfullInvaders = [i for i in opponentIndices if isHarmfulInvader(successor, i)]
        features["harmful_invader_distance_factor"] = max([getPositionFactor(getDistance(getPosition(successor, i))) for i in harmfullInvaders]) if len(harmfullInvaders) > 0 else 0

        harmlessGhosts = [i for i in opponentIndices if isHarmlessGhost(successor, i)]
        features["harmless_ghost_distance_factor"] = max([getPositionFactor(getDistance(getPosition(successor, i))) for i in harmlessGhosts]) if len(harmlessGhosts) > 0 else 0

        return features

    def getWeights(self, gameState, actionAgentIndex, action):
        return {
            "stopped": -2.0,
            "reversed": -1.0,
            "scared": -2.0,
            "food_returned": 10.0,
            "food_carrying": 8.0,
            "food_defend": 5.0,
            "nearest_food_distance_factor": -1.0,
            "nearest_capsules_distance_factor": -0.5,
            "return_food_factor": -0.5, # 1.5
            "team_distance": 0.5,
            "harmless_invader_distance_factor": -0.4,
            "harmful_invader_distance_factor": 0.4,
            "harmless_ghost_distance_factor": -0.2,
        }

class StateEvaluationOffensiveAgent(StateEvaluationAgent):
    """An agent class. Optimized for offense. You can use it directly."""

    ######################
    # overload functions #
    ######################

    def getWeights(self, gameState, actionAgentIndex, action):
        return {
            "stopped": -2.0,
            "reversed": -1.0,
            "scared": -2.0,
            "food_returned": 10.0,
            "food_carrying": 8.0,
            "food_defend": 0.0,
            "nearest_food_distance_factor": -1.0,
            "nearest_capsules_distance_factor": -1.0,
            "return_food_factor": -0.5, # 1.5
            # "team_distance": 0.5,
            "harmless_invader_distance_factor": -0.1,
            "harmful_invader_distance_factor": 0.1,
            "harmless_ghost_distance_factor": -0.2,
        }

class StateEvaluationDefensiveAgent(StateEvaluationAgent):
    """An agent class. Optimized for defence. You can use it directly."""

    ######################
    # overload functions #
    ######################

    def getWeights(self, gameState, actionAgentIndex, action):
        return {
            "stopped": -2.0,
            "reversed": -1.0,
            "scared": -2.0,
            "food_returned": 1.0,
            "food_carrying": 0.5,
            "food_defend": 5.0,
            "nearest_food_distance_factor": -1.0,
            "nearest_capsules_distance_factor": -0.5,
            "return_food_factor":  1.5,
            # "team_distance": 0.5,
            "harmless_invader_distance_factor": -1.0,
            "harmful_invader_distance_factor": 2.0,
            "harmless_ghost_distance_factor": -0.1,
        }

#####################################
#                                   #
# Test approximate q-learning agent #
#                                   #
#####################################

class TestApproximateQLearningAgent(ApproximateQLearningAgent):
    """An agent class. For testing whether the ApproximateQLearningAgent works or not. You can use it directly, only for test."""

    ######################
    # overload functions #
    ######################

    def getInitWeights(self):
        return {'successorScore': 100, 'distanceToFood': -1}

    def getFeatures(self, gameState, actionAgentIndex, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, actionAgentIndex, action)
        foodList = self.getFood(successor).asList()
        features['successorScore'] = -len(foodList)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance
        return features

    def getReward(self, previousState, previousAction, currentState, currentAction):
        return self.getScore(currentState) - self.getScore(previousState)

################################################
#                                              #
# Test optimized monte carlo tree search agent #
#                                              #
################################################

class TestOptimizedMonteCarloTreeSearchAgent(OptimizedMonteCarloTreeSearchAgent):
    """An agent class. For testing whether the OptimizedMonteCarloTreeSearchAgent works or not. You can use it directly, only for test."""

    ######################
    # overload functions #
    ######################

    def getModuleFileName(self):
        return "baselineTeam"  # baselineTeam.py have to be in the same directory.

    def getAgentClassName(self, index):
        if index % 2 == 0:
            return "OffensiveReflexAgent"
        else:
            return "DefensiveReflexAgent"
