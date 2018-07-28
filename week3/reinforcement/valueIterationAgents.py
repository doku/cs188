# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        for _ in range(self.iterations):
            states = self.mdp.getStates()
            vPrime = self.values.copy()
            for state in states:  
                a_lis = []
                actions = self.mdp.getPossibleActions(state)
                if len(actions) != 0:
                    for action in actions:
                        t_q = 0
                        for t, p in self.mdp.getTransitionStatesAndProbs(state, action):
                            q = p * (self.mdp.getReward(state, action, t) + self.discount * vPrime[t])
                            t_q += q
                        a_lis.append(t_q)
                    self.values[state] = max(a_lis)


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        actions = self.mdp.getPossibleActions(state)
        if len(actions) != 0:
            if action in actions:
                q = 0
                for t, p in self.mdp.getTransitionStatesAndProbs(state, action):
                    q += p * (self.mdp.getReward(state, action, t) + self.discount * self.values[t])
                return q
            else:
                print "Action not allowed"
        else:
            print "In terminal state"

            
    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        actions = self.mdp.getPossibleActions(state)
        if len(actions) != 0:
            a_list = []
            for action in actions:
                t_q = 0
                for t, p in self.mdp.getTransitionStatesAndProbs(state, action):
                    q = p * (self.mdp.getReward(state, action, t) + self.discount * self.values[t])
                    t_q += q
                a_list.append((t_q, action))
            return max(a_list, key=lambda x: x[0])[1]
        

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()
        statesLen = len(states)
        for i in range(self.iterations):
            state = states[i % statesLen]
            actions = self.mdp.getPossibleActions(state)
            a_list = []
            if len(actions) != 0:
                for action in actions:
                    q = 0
                    for t, p in self.mdp.getTransitionStatesAndProbs(state, action):
                        q += p * (self.mdp.getReward(state, action, t) + self.discount * self.values[t])
                    a_list.append(q)
                self.values[state] = max(a_list)
            

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        pred = util.Counter()
        ss = self.mdp.getStates()
        pQueue = util.PriorityQueue()
        for s in ss:
            pred[s] = set()
        for s in ss:
            if not self.mdp.isTerminal(s):
                for action in self.mdp.getPossibleActions(s):
                    for t, p in self.mdp.getTransitionStatesAndProbs(s, action):
                        if p > 0:
                            #print pred[t]
                            pred[t].add(s)
                            #print pred[t]
        for s in ss:
            if not self.mdp.isTerminal(s):
                diff = abs(self.getValue(s) - max([self.computeQValueFromValues(s, action) for action in self.mdp.getPossibleActions(s)]))
                pQueue.push(s, -1 * diff)
        for i in range(self.iterations):
            if pQueue.isEmpty():
                break
            s = pQueue.pop()
            
            self.values[s] = max([self.computeQValueFromValues(s, action) for action in self.mdp.getPossibleActions(s)])
            """if not self.mdp.isTerminal(s):
                actions = self.mdp.getPossibleActions(s)
                if len(actions) != 0:
                    a_list = []
                    for action in actions:
                        q = 0
                        for t, p in self.mdp.getTransitionStatesAndProbs(s, action):
                            q += p * (self.mdp.getReward(s, action, t) + self.discount * self.values[t])
                        a_list.append(q)
                    self.values[s] = max(a_list)
            """
            
            for p in pred[s]:
                diff = abs(self.getValue(p) - max([self.computeQValueFromValues(p, action) for action in self.mdp.getPossibleActions(p)]))
                if diff > self.theta:
                    pQueue.update(p, -1 * diff)

