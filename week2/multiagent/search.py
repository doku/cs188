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
    path = [s, s, w, s, w, w, s, w]
    print path
    return path

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    closed = set()
    nodes = util.Stack() 
    nodes.push((problem.getStartState(), None, 0, None))
    while(not nodes.isEmpty()):
        node = nodes.pop()

        if problem.isGoalState(node[0]):
            #nodes.push((node[0], node[1], c3, node))
            #print "Goal Node : ", node
            break

        if node[0] not in closed:
            closed.add(node[0])
            for c1, c2, c3 in problem.getSuccessors(node[0]):
                nodes.push((c1, c2, c3, node))
    
    directions = []
    pnt = node
    while pnt:
        if pnt[1] is not None:
            directions.append(pnt[1])
        pnt = pnt[3]
    return directions[::-1]


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    closed = set()
    nodes = util.Queue()
    nodes.push((problem.getStartState(), None, 0, None))
    while(not nodes.isEmpty()):
        node = nodes.pop()

        if problem.isGoalState(node[0]):
            break

        if node[0] not in closed:
            closed.add(node[0])
            for c1, c2, c3 in problem.getSuccessors(node[0]):
                nodes.push((c1, c2, c3, node))

    directions = [] 
    pnt = node
    while pnt:
        if pnt[1] is not None:
            directions.append(pnt[1])
        pnt = pnt[3]
    return directions[::-1]



def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    closed = set()
    nodes = util.PriorityQueue()
    nodes.push((problem.getStartState(), None, 0,  None), 0)
    while( not nodes.isEmpty()):
        node = nodes.pop()
        if problem.isGoalState(node[0]):
            break
        if node[0] not in closed:
            closed.add(node[0])
            for c1, c2, c3 in problem.getSuccessors(node[0]):
                nodes.push((c1, c2, c3 + node[2], node), c3 + node[2])
    directions = []
    pnt = node
    while pnt:
        if pnt[1] is not None:
            directions.append(pnt[1])
        pnt = pnt[3]
    return directions[::-1]

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    closed = set()
    nodes = util.PriorityQueue()
    start = problem.getStartState()
    nodes.push((start, None, 0, None), heuristic(start, problem))
    while(not nodes.isEmpty()):
        node = nodes.pop()
        if problem.isGoalState(node[0]):
            break
        if node[0] not in closed:
            closed.add(node[0])
            for c1, c2, c3 in problem.getSuccessors(node[0]):
                nodes.push((c1, c2, c3 + node[2], node), c3 + node[2] + heuristic(c1, problem)) 
    directions = []
    pnt = node
    while pnt:
        if pnt[1] is not None:
            directions.append(pnt[1])
        pnt = pnt[3]
    return directions[::-1]


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
