# myAgentP1.py
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
# This file was based on the starter code for student bots, and refined 
# by Mesut (Xiaocheng) Yang


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint

#########
# Agent #
#########


class myAgentP1(CaptureAgent):
  """  
            Chun Hao (Jason) Wang, Chanzy Huang
            Phase Number: 1  
            Description of Bot: 
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """
    
    self.weights = {'successorScore': 100, 'teammateDistance': 100, 'closestFood': 100,
                'numRepeats': 100, 'teammateFoodDistance':100, 'scores':100,
                'turnsTaken':100, 'pacmanDeath':100}
                
    #self.r = -500
    self.alpha = .8
    self.gamma = .5
    
    self.Qs = util.Counter()
                
    # Make sure you do not delete the following line. 
    # If you would like to use Manhattan distances instead 
    # of maze distances in order to save on initialization 
    # time, please take a look at:
    # CaptureAgent.registerInitialState in captureAgents.py.
    CaptureAgent.registerInitialState(self, gameState)
    self.start = gameState.getAgentPosition(self.index)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    prevState = self.getPreviousObservation()
    prevScore = 0
    prevQs = 0
    if prevState:
        prevActions = prevState.getLegalActions(self.index)
        prevQs = max([self.evaluate(prevState, a) for a in prevActions])
        prevScore = prevState.getScore()
    
    
    observedState = self.getCurrentObservation()
    Qval = self.evaluate(observedState, observedState.getAgentState(self.index).getDirection())
    
    reward = prevScore - observedState.getScore()
    diff = (reward + self.gamma * prevQs) - Qval
    theState = (observedState.getAgentPosition(self.index), observedState.getAgentState(self.index).getDirection())
    self.Qs[theState] = self.Qs[theState] + self.alpha * diff
    feats = self.getFeatures(observedState, theState[1])
    for k in self.weights.keys():
        self.weights[k] = self.weights[k] + self.alpha * diff * feats[k]
        
    #Qs[(observedState.getAgentPosition(self.index), observedState.getAgentState(self.index).getDirection())]
    #if (observedState, 
    #observerdState. 
    
    
    #Qval = Qs[(observerdState.getAgentPosition(self.index), observedState.getAgentDirection(self.index) )]
    #diff = ((prevQs.getScore()-observedState.getScore()) + self.gamma*prevQs) - max(Qvalues, keys=lambda x: x[0])[0]
    
    #if (self.getAgentPosition(self.index), ) in self.Qs
    
    print self.weights
    values = [(self.evaluate(gameState, a), a) for a in actions]
    
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    # INSERT YOUR LOGIC HERE
    #print "val " + str(values)
    return max(values, key=lambda x:x[0])[1] 
    #return "North"
    
  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return sum(x*y for x,y in zip(features.values(), weights.values()))
    
  def getFeatures(self, gameState, action):
    features = util.Counter()

    ### Useful information you can extract from a GameState (pacman.py) ###
    successorGameState = gameState.generateSuccessor(self.index, action)
    newPos = successorGameState.getAgentPosition(self.index)
    oldFood = gameState.getFood()
    newFood = successorGameState.getFood()
    ghostIndices = self.getOpponents(successorGameState)
    
    # Determines how many times the agent has already been in the newPosition in the last 20 moves
    numRepeats = sum([1 for x in self.observationHistory[-20:] if x.getAgentPosition(self.index) == newPos])

    foodPositions = oldFood.asList()
    foodDistances = [self.getMazeDistance(newPos, foodPosition) for foodPosition in foodPositions]
    closestFood = min( foodDistances ) + 1.0

    ghostPositions = [successorGameState.getAgentPosition(ghostIndex) for ghostIndex in ghostIndices]
    ghostDistances = [self.getMazeDistance(newPos, ghostPosition) for ghostPosition in ghostPositions]
    ghostDistances.append( 1000 )
    closestGhost = min( ghostDistances ) + 1.0

    teammateIndices = [index for index in self.getTeam(gameState) if index != self.index]
    assert len(teammateIndices) == 1, "Teammate indices: {}".format(self.getTeam(gameState))
    teammateIndex = teammateIndices[0]
    teammatePos = successorGameState.getAgentPosition(teammateIndex)
    teammateDistance = self.getMazeDistance(newPos, teammatePos) + 1.0

    pacmanDeath = successorGameState.data.num_deaths

    features['successorScore'] = self.getScore(successorGameState)
    features['teammateDistance'] = teammateDistance
    features['closestFood'] = closestFood
    features['numRepeats'] = numRepeats
    features['teammateFoodDistance'] = min([self.getMazeDistance(teammatePos, foodPosition) for foodPosition in foodPositions])
    features['scores'] = self.getScore(gameState)
    if self.getNumTurnsTaken():
        features['turnsTaken'] = self.getNumTurnsTaken()
    else:
        features['turnsTaken'] = 0
    features['pacmanDeath'] = pacmanDeath
    
    # CHANGE YOUR FEATURES HERE

    return features

  def getWeights(self, gameState, action):
    # CHANGE YOUR WEIGHTS HERE
    
    
    return self.weights