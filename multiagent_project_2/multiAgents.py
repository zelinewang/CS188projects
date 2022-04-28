# multiAgents.py
# --------------
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


from telnetlib import GA
from typing import List
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # look at HW2 Q12 to see what is a good evaluation function
        # just to write a heuristic function to evaluate where should we go or search which action we should take
        closest = 42069
        newFoodList = newFood.asList() # a list of (x,y) in newFood
        hasFood = 0 # can not use the isWin to determine if there is any FOOD left, use hasFood boolean to figure that out
        heuristicFood = 0
        for (x, y) in newFoodList:
            if newFood[x][y]: #if there is a Food
                MD = manhattanDistance(newPos, (x,y))
                hasFood = 1
                if (MD < closest): #choose the distance to the closest food
                    closest = MD
        if hasFood == 0:
            return successorGameState.getScore()
        else:
            heuristicFood = 1/(closest + 1) #heuristic is the reciprocal (inverse) of the distance to closest food
            return heuristicFood + successorGameState.getScore()
        # always consider the situation where heu is 0 (when is the goal state)
        #return the cost + heuristic (as the evaluation of Astar search)

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def minimaxValue(self, depth, index, state: GameState):
        if (depth == 0 or state.isWin() or state.isLose()): # DO () for all the IFs
            return self.evaluationFunction(state)
        elif index == 0:
            return self.maxValue(depth, index, state)
        elif index > 0:
            return self.minValue(depth, index, state)

    def maxValue(self, depth, index, state: GameState):
        v = float('-inf')
        actions  = state.getLegalActions(index)
        successors = [state.generateSuccessor(index, action) for action in actions]
        # all the successors this agent could get to
        for successor in successors:
            v = max(v, self.minimaxValue(depth, index + 1, successor))
            # take the max of minmax value for next agent
        return v

    def minValue(self, depth, index, state: GameState):
        v = float('inf')
        actions  = state.getLegalActions(index)
        successors = [state.generateSuccessor(index, action) for action in actions]
        if (index == state.getNumAgents() - 1):
            depth = depth - 1
            newIndex = 0
        elif (index < state.getNumAgents()):
            newIndex = index + 1
        for successor in successors:
            v = min(v, self.minimaxValue(depth, newIndex, successor))
        return v

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # set up the game and spread different agents, but always Pacman moving first
        numAgents = gameState.getNumAgents()
        legalActions = gameState.getLegalActions(self.index) 
        # get legal actions for the first Pacman move
        highest = float('-inf')
        hightestAction = legalActions[0]
        if (numAgents == 1): # there is no ghost
            for action in legalActions:
                successor = gameState.generateSuccessor(self.index, action)
                if self.evaluationFunction(successor) >= highest:
                    highest = self.evaluationFunction(successor)
                    hightestAction = action
        else:
            for action in legalActions:
                successor = gameState.generateSuccessor(self.index, action)
                maxvalue = self.minimaxValue(self.depth, self.index + 1, successor) 
                # if you wanna call a method inside of this class, do self.function, not just function
                # to get the minimax value for the next agent move, which is def a ghost
                if (maxvalue >= highest):
                    highest = maxvalue
                    hightestAction = action
        
        return hightestAction
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    # a is the best choice so far for MAX, b is the best choice so far for MIN
    # MAX only changes a, MIN only changes b
    def minimaxValue(self, depth, index, state: GameState, a, b): 
        if (depth == 0 or state.isWin() or state.isLose()):
            return self.evaluationFunction(state)
        elif index == 0:
            return self.maxValue(depth, index, state, a, b)
        elif index > 0:
            return self.minValue(depth, index, state, a, b)

    def maxValue(self, depth, index, state: GameState, a, b):
        v = float('-inf')
        actions  = state.getLegalActions(index)
        # all the successors this agent could get to
        for action in actions:
            successor = state.generateSuccessor(index, action)
            # putting successor inside of the loop
            # cuz we are pruning, so we are cutting down the looping times
            # and the autograder checks for the GENERATESUCCESSOR method, 
            # so we need to put it inside of the loop to cut it down
            v = max(v, self.minimaxValue(depth, index + 1, successor, a, b))
            if v > b:
                return v
            a = max(a, v)
            # take the max of minmax value for next agent
        return v

    def minValue(self, depth, index, state: GameState, a, b):
        v = float('inf')
        actions  = state.getLegalActions(index)
        if (index == state.getNumAgents() - 1):
            depth = depth - 1
            newIndex = 0
        elif (index < state.getNumAgents()):
            newIndex = index + 1
        for action in actions:
            successor = state.generateSuccessor(index, action)
            # putting successor inside of the loop
            # cuz we are pruning, so we are cutting down the looping times
            # and the autograder checks for the GENERATESUCCESSOR method, 
            # so we need to put it inside of the loop to cut it down
            v = min(v, self.minimaxValue(depth, newIndex, successor, a, b))
            if v < a:
                return v
            b = min(b, v)
        return v
    

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()
        legalActions = gameState.getLegalActions(self.index)
        highest = float('-inf')
        hightestAction = legalActions[0]
        if (numAgents == 1):
            for action in legalActions:
                successor = gameState.generateSuccessor(self.index, action)
                if self.evaluationFunction(successor) >= highest:
                    highest = self.evaluationFunction(successor)
                    hightestAction = action
        else:
            a = float('-inf')
            b = float('inf')
            for action in legalActions:
                successor = gameState.generateSuccessor(self.index, action)
                maxvalue = self.minimaxValue(self.depth, self.index + 1, successor, a, b) 
                if (maxvalue >= highest):
                    highest = maxvalue
                    hightestAction = action
                # if highest > b:
                #    highest = highest 
                # (nothing happens, cuz there is only one root node for this MAX, we are not pruning on this)
                a = max(highest, a)
                # but we do need to change the Beta (cuz it's MAX)
        
        return hightestAction
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def minimaxValue(self, depth, index, state: GameState): 
        if (depth == 0 or state.isWin() or state.isLose()):
            return self.evaluationFunction(state)
        elif index == 0:
            return self.maxValue(depth, index, state)
        elif index > 0:
            return self.minValue(depth, index, state)

    def maxValue(self, depth, index, state: GameState):
        v = float('-inf')
        actions  = state.getLegalActions(index)
        for action in actions:
            successor = state.generateSuccessor(index, action)
            v = max(v, self.minimaxValue(depth, index + 1, successor))
        return v

    def minValue(self, depth, index, state: GameState):
        sum = 0
        actions  = state.getLegalActions(index)
        if (index == state.getNumAgents() - 1):
            depth = depth - 1
            newIndex = 0
        elif (index < state.getNumAgents()):
            newIndex = index + 1
        for action in actions:
            successor = state.generateSuccessor(index, action)
            sum += self.minimaxValue(depth, newIndex, successor)
        return sum/len(actions)

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()
        legalActions = gameState.getLegalActions(self.index)
        highest = float('-inf')
        hightestAction = legalActions[0]
        if (numAgents == 1):
            for action in legalActions:
                successor = gameState.generateSuccessor(self.index, action)
                if self.evaluationFunction(successor) >= highest:
                    highest = self.evaluationFunction(successor)
                    hightestAction = action
        else:
            for action in legalActions:
                successor = gameState.generateSuccessor(self.index, action)
                maxvalue = self.minimaxValue(self.depth, self.index + 1, successor) 
                if (maxvalue >= highest):
                    highest = maxvalue
                    hightestAction = action
        return hightestAction
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
     # A new Expectimax Agent (__init__)
    action = ExpectimaxAgent.getAction(currentGameState)
    # using Q1 stat-action evaluation and Q4 expectimax agent

    # cuz the ghost is a random agent
    # so we should use expectimax agent to find an optimal action with high probability,

    # then with this good action with a high chance, evaluate it in Q1 state-action pair evaluation
    # which is def a "better" evaluation because of "better" action (at least with a high probability)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    closest = 42069
    newFoodList = newFood.asList() 
    hasFood = 0 
    heuristicFood = 0
    for (x, y) in newFoodList:
        if newFood[x][y]:
            MD = manhattanDistance(newPos, (x,y))
            hasFood = 1
            if (MD < closest): 
                closest = MD
    if hasFood == 0:
        return successorGameState.getScore()
    else:
        heuristicFood = 1/(closest + 1)
        return heuristicFood + successorGameState.getScore()
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
