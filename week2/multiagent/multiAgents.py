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


from util import manhattanDistance
from game import Directions
from game import Actions
import random, util, math

from game import Agent

#import search
#import searchAgents as sa

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """
    #goal = (-1,-1)
    #action_state = "search"

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        #print scores,
        #print legalMoves[chosenIndex]
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
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
        newGhostPositions = successorGameState.getGhostPositions()
        newCapsule = successorGameState.getCapsules()
        total_force = successorGameState.getScore()

        food_search_size = 5
        closest_food = sorted([(float(util.manhattanDistance(newPos, food)), food) for food in newFood.asList()], key=lambda k: k[0])[:food_search_size]
        if len(newFood.asList()):
            problem = PositionSearchProblem(successorGameState, start=newPos, goal=closest_food[0][1],  warn=False, visualize=False)
            close_food = breadthFirstSearch(problem) 
            total_force += 10.0/len(close_food)
            #print "food :", 10.0/len(close_food),
            
        if len(newCapsule):
            caps = []
            for cap in newCapsule:
                problem = PositionSearchProblem(successorGameState, start=newPos, goal=cap, warn=False, visualize=False)
                c = breadthFirstSearch(problem)
                caps.append(c)
                total_force += 5.0/len(c)
        
        #tot_dis.append(( breadthFirstSearch(problem) , closest_food[0][1] ))
        #tot_dis_sor = sorted(tot_dis, key=lambda x: len(x[0])) 
        #closest_food = [tot_dis_sor[i][0][0] for i in range(food_search_size)]
        #print closest_food
        #count = {}
        #for i in closest_food:
        #    if i in count:
        #       count[i] += 1
        #    else:
        #        count[i] = 1
        #food_power = sorted(count.items(), key=lambda x: x[1] , reverse=True)[0]
        #print food_power[0]
        #if action == food_power[0]:
        #    total_force += 3
        #tot_dis_sor_sum = sum([len(x[0]) for x in tot_dis_sor])
        ghost_force = sum([15.0 *y / x if y > 0 else -15.0 / x for x,y,z in [(len(breadthFirstSearch(PositionSearchProblem(successorGameState, start=newPos, goal=i.getPosition(), warn=False, visualize=False))), ghostState.scaredTimer, i.getPosition()) for i in newGhostStates] if x < 10 and x > 0])
        #ghost_force_multiplier = 
        #print ghost_force*ghost_force , 100.0/tot_dis_sor_sum, action
        #print dir(currentGameState.getPacmanPosition())
        #total_force += ghost_force**2 + 100.0/tot_dis_sor_sum
        total_force +=  ghost_force
        if action == Directions.WEST or action == Directions.NORTH:
            total_force += random.random()
        
        #print Actions.vectorToDirection((newPos[0] - currentGameState.getPacmanPosition()[0], newPos[1] - currentGameState.getPacmanPosition()[1]))
            
        #print ghost_force , total_force
        if action == Directions.STOP:
            total_force *= .01
        return total_force
        #return ghost_force*ghost_force + 100.0/tot_dis_sor_sum
        #ghost_force = sum([(2**k) for k in [float(util.manhattanDistance(newPos, i)) for i in newGhostPositions] if k <= 6])
        """
        if (newScaredTimes[0] == 0):
            ghost_force = sum([(2**k) for k in [float(util.manhattanDistance(newPos, i)) for i in newGhostPositions] if k <= 6])
            
            food_search_size = 3
            closest_food = sorted([(float(util.manhattanDistance(newPos, food)), food) for food in newFood.asList()], key=lambda k: k[0])[:food_search_size]
            food_force = sum([((food_search_size-y)**2/(x*x)) for y,x in enumerate(zip(*closest_food)[0])])
            #food_force = random.random()*100
            total_force = sum([ghost_force, food_force])
            if action == Directions.STOP:
                total_force *= 0.1
            print ghost_force, food_force, total_force, action

            return total_force
        """
        return successorGameState.getScore()
        #return total_force
        
def vector_to_direction(vector):
    dx, dy = vector
    direction_y = Directions.NORTH
    direction_x = Directions.EAST
    if abs(dy) > abs(dx):
        if dy > 0:
            return Directions.NORTH
        else:
            return Directions.SOUTH
    else:
        if dx > 0:
            return Directions.EAST
        else:
            return Directions.WEST
    

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


    
def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(breadthFirstSearch(prob))

def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    foodList = foodGrid.asList()
    wallList = problem.walls.asList()
    def mean():
        pass
    def euclideDist(pos1, pos2):
        return (( pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1]) **2) **0.5
    def manDist(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    def modManDist(pos1, pos2):
        pass
    return max([manDist(position, food) for food in foodList] or [0])
    #return max([manDist(position, food) for food in foodList] or [0]) - ([manDist(position, wall) for wall in wallList] or [0])
    #return max([mazeDistance(position, food, problem.startingGameState) for food in foodList] or [0]) 

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

    
class PositionSearchProblem(SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print 'Warning: this does not look like a regular search maze'

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost    
    
        
class FoodSearchProblem:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """
    def __init__(self, startingGameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0 # DO NOT CHANGE
        self.heuristicInfo = {} # A dictionary for the heuristic to store information]
        self.ghostState = startingGameState.getGhostStates()

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1 # DO NOT CHANGE
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                successors.append( ( ((nextx, nexty), nextFood), direction, 1) )
        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x,y= self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost 
    
        
def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
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
        
        def helperAction(gameState, ind):
            if gameState.isLose() or gameState.isWin():
                return (self.evaluationFunction(gameState), Directions.STOP)
            if ind / gameState.getNumAgents() == self.depth:
                #print "self eval", self.evaluationFunction(gameState), gameState.getNumAgents(), self.index
                return (self.evaluationFunction(gameState), Directions.WEST)
            current_index = ind % gameState.getNumAgents()
            scores = []
            for act in gameState.getLegalActions(current_index):
                successorState = gameState.generateSuccessor(current_index, act)
                scor, _ = helperAction(successorState, ind + 1)
                scores.append((scor,act))
            if current_index == 0:
                return max(scores, key=lambda x: x[0])
            else:
                return min(scores, key=lambda x: x[0])
        
        return helperAction(gameState, 0)[1]
        


            

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        def alpha_beta(state, idx, alpha, beta):
            if state.isLose() or state.isWin() or (idx / state.getNumAgents()) == self.depth:
                return self.evaluationFunction(state)
            if idx % state.getNumAgents() == 0:
                v = float("-inf")
            else:
                v = float("inf")
            ind = idx % state.getNumAgents()
            for action in state.getLegalActions(ind):
                successor = state.generateSuccessor(ind, action)
                if ind == 0:
                    v = max(v, alpha_beta(successor, idx + 1, alpha, beta))
                    if v > beta:
                        return v
                    alpha = max(alpha, v)
                else:
                    v = min(v, alpha_beta(successor, idx + 1, alpha, beta))
                    if v < alpha:
                        return v
                    beta = min(beta, v)
            #scores = [alpha_beta(x, idx+1, alpha, beta) for x in [state.generateSuccessor(idx, action) for action in state.getLegalActions(idx)]]
            return v
        v = (float("-inf"), "Stop")
        alpha = float("-inf")
        beta = float("inf")
        #v_max_action = "Stop"
        for action in gameState.getLegalActions(0):
            #print v, alpha, beta, action
            v = max(v, (alpha_beta(gameState.generateSuccessor(0, action), 1, alpha, beta), action), key= lambda x: x[0])
            if v[0] > beta:
                return v[1]
            alpha = max(alpha, v[0])
        return v[1]
        #return max([(alpha_beta(gameState.generateSuccessor(0, action), 1, float('-inf'), float('inf')), action) for action in gameState.getLegalActions(0)], key=lambda x:x[0])[1]
        
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        def expectimax(state, idx):
            if state.isLose() or state.isWin() or (idx / state.getNumAgents()) == self.depth:
                return (self.evaluationFunction(state), None)
            ind = idx % state.getNumAgents()
            if ind == 0:
                v = (float('-inf'), None)
            else:
                v = (0, None)
            actions = state.getLegalActions(ind)
            len_childrends = len(actions)
            vv = [(expectimax(state.generateSuccessor(ind, action), idx+1)[0], action) for action in actions]
            if ind == 0:
                v = max(vv, key=lambda x:x[0])
            else:
            
                #print vv, sum(zip(*vv)[0])
                v = (sum(zip(*vv)[0])/float(len_childrends), None)
            return v
        
        return expectimax(gameState, 0)[1]
                
                
def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    #legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    #scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    #bestScore = max(scores)
    #bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    #chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    "Add more of your code here if you want to"
    #print scores,
    #print legalMoves[chosenIndex]
    #return legalMoves[chosenIndex]
    #v = []
    #for action in currentGameState.getLegalActions():
    
    successorGameState = currentGameState #.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    newGhostPositions = successorGameState.getGhostPositions()
    newCapsule = successorGameState.getCapsules()
    total_force = successorGameState.getScore()

    food_search_size = 5
    closest_food = sorted([(float(util.manhattanDistance(newPos, food)), food) for food in newFood.asList()], key=lambda k: k[0])[:food_search_size]
    if len(newFood.asList()):
        problem = PositionSearchProblem(successorGameState, start=newPos, goal=closest_food[0][1],  warn=False, visualize=False)
        close_food = breadthFirstSearch(problem) 
        total_force += 10.0/len(close_food)
        #print "food :", 10.0/len(close_food),
        
    if len(newCapsule):
        caps = []
        for cap in newCapsule:
            problem = PositionSearchProblem(successorGameState, start=newPos, goal=cap, warn=False, visualize=False)
            c = breadthFirstSearch(problem)
            caps.append(c)
            total_force += 5.0/len(c)
    
    ghost_force = sum([10.0 *y / x if y > 0 else -15.0 / x for x,y,z in [(len(breadthFirstSearch(PositionSearchProblem(successorGameState, start=newPos, goal=i.getPosition(), warn=False, visualize=False))), ghostState.scaredTimer, i.getPosition()) for i in newGhostStates] if x < 10 and x > 0])

    total_force +=  ghost_force
    #if action == Directions.WEST or action == Directions.NORTH:
    #    total_force += random.random()
      
    #print ghost_force , total_force
    #if action == Directions.STOP:
    #    total_force *= .01
    return total_force
    #v.append((total_force, action)) 
    #print v
    #if len(v):
    #    return max(v, key=lambda x: x[0])[0]
    #return Directions.STOP
    #print direction[1]
    #legalMoves = currentGameState.getLegalActions()
    #chosenIndex = random.choice(range(0,len(legalMoves)-1)) # Pick randomly among the best
    #return legalMoves[chosenIndex]
    #return direction[0]


# Abbreviation
better = betterEvaluationFunction

