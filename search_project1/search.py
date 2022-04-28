# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from operator import ne
import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    from util import Stack
    fringe = Stack()
    expanded = {problem.getStartState()}
    # a set to store the state that already visited, which is the different between graph search and tree search
    for successor in problem.getSuccessors(problem.getStartState()):
        path = [successor[1]] # store the pathes (lists of actions) to get to that successor
        fringe.push((successor, path))
    while not fringe.isEmpty():
        next = fringe.pop()
        thispath = next[1]
        nextSuccessor = next[0]
        nextState = nextSuccessor[0]
        thisaction = nextSuccessor[1]
        expanded.add(nextState)
        if problem.isGoalState(nextState):
            return thispath
        else:
            for subSuccessor in problem.getSuccessors(nextState):
                if subSuccessor[0] not in expanded:
                    newpath = []
                    newpath.extend(thispath)
                    newpath.append(subSuccessor[1])
                    fringe.push((subSuccessor, newpath))
    util.raiseNotDefined()


"""     OR !
    while not fringe.isEmpty():
    next = fringe.pop()
    thispath = next[1]
    nextSuccessor = next[0]
    nextState = nextSuccessor[0]
    thisaction = nextSuccessor[1]
    expanded.add(nextState)
    actions.append(thisaction)
    if problem.isGoalState(nextState):
        return thispath
    else:
        for subSuccessor in problem.getSuccessors(nextState):
            if subSuccessor[0] not in expanded:
                newpath = []
                newpath.extend(thispath)
                newpath.append(subSuccessor[1]) """
    

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue
    fringe = Queue()
    expanded = {problem.getStartState()}
    for successor in problem.getSuccessors(problem.getStartState()):
        path = [successor[1]] # store the pathes (lists of actions) to get to that successor
        fringe.push((successor, path))
    while not fringe.isEmpty():
        next = fringe.pop()
        thispath = next[1]
        nextSuccessor = next[0]
        nextState = nextSuccessor[0]
        thisaction = nextSuccessor[1]
        if nextState not in expanded:
            expanded.add(nextState)
            if problem.isGoalState(nextState):
                return thispath
            else:
                for subSuccessor in problem.getSuccessors(nextState):
                    newpath = []
                    newpath.extend(thispath)
                    newpath.append(subSuccessor[1])
                    fringe.push((subSuccessor, newpath))
    
    util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    fringe = PriorityQueue()
    expanded = {problem.getStartState()}
    for successor in problem.getSuccessors(problem.getStartState()):
        path = [successor[1]] # store the pathes (lists of actions) to get to that successor
        fringe.push((successor, path), successor[2])
    while not fringe.isEmpty():
        next = fringe.pop()
        thispath = next[1]
        nextSuccessor = next[0]
        nextState = nextSuccessor[0]
        thisaction = nextSuccessor[1]
        if nextState not in expanded:
            expanded.add(nextState)
            totalcost = problem.getCostOfActions(thispath)
            if problem.isGoalState(nextState):
                return thispath
            else:
                for subSuccessor in problem.getSuccessors(nextState):
                    newpath = []
                    newpath.extend(thispath)
                    newpath.append(subSuccessor[1])
                    fringe.update((subSuccessor, newpath), totalcost + subSuccessor[2])
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    fringe = PriorityQueue()
    expanded = {problem.getStartState()}
    for successor in problem.getSuccessors(problem.getStartState()):
        path = [successor[1]] # store the pathes (lists of actions) to get to that successor
        fringe.push((successor, path), successor[2] + heuristic(successor[0], problem))
    while not fringe.isEmpty():
        next = fringe.pop()
        thispath = next[1]
        nextSuccessor = next[0]
        nextState = nextSuccessor[0]
        thisaction = nextSuccessor[1]
        if nextState not in expanded:
            expanded.add(nextState)
            totalcost = problem.getCostOfActions(thispath)
            if problem.isGoalState(nextState):
                return thispath
            else:
                for subSuccessor in problem.getSuccessors(nextState):
                    newpath = []
                    newpath.extend(thispath)
                    newpath.append(subSuccessor[1])
                    fringe.update((subSuccessor, newpath), totalcost + subSuccessor[2] + heuristic(subSuccessor[0], problem))
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
