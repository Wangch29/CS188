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

    def __init__(self, mdp, discount=0.9, iterations=100):
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
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        states: list = self.mdp.getStates()

        for _ in range(self.iterations):
            new_values: dict = util.Counter()
            for state in states:
                new_values[state] = self.__state_max_qvalue(state)
            self.values = new_values

    def __state_max_qvalue(self, state) -> float:
        """ Calculate a single state's iteration value. """
        actions: list = self.mdp.getPossibleActions(state)
        action_values = util.Counter()

        for action in actions:
            transitions: list = self.mdp.getTransitionStatesAndProbs(state, action)
            action_values[action] = 0
            for transition in transitions:
                next_state = transition[0]
                prob = transition[1]
                action_values[action] += (
                        prob * (self.mdp.getReward(state, action, next_state) + self.discount * self.values[next_state])
                )

        return action_values[action_values.argMax()]

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
        "*** YOUR CODE HERE ***"
        qValue: float = 0
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)

        for transition in transitions:
            nextState = transition[0]
            prob = transition[1]
            qValue += prob * (self.mdp.getReward(state, action, nextState) + self.discount * self.values[nextState])

        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.mdp.getPossibleActions(state)
        action_values = util.Counter()

        if len(actions) == 0:
            return None
        for action in actions:
            action_values[action] = self.computeQValueFromValues(state, action)

        return action_values.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        self.queue = util.PriorityQueue()
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states: list = self.mdp.getStates()

        predecessors = self.__computePredecessors()

        for state in states:
            self.__state_max_qvalue(state, True)

        for _ in range(self.iterations):
            if self.queue.isEmpty():
                return
            state = self.queue.pop()
            self.values[state] = self.__state_max_qvalue(state, False)
            self.__add_predecessor(state, predecessors)

    def __computePredecessors(self):
        """
        Compute predecessors for each state.
        """
        predecessors = {}
        states = self.mdp.getStates()
        for state in states:
            if not self.mdp.isTerminal(state):
                for action in self.mdp.getPossibleActions(state):
                    for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                        if nextState in predecessors:
                            predecessors[nextState].add(state)
                        else:
                            predecessors[nextState] = {state}
        return predecessors

    def __state_max_qvalue(self, state, toQueue: bool) -> float:
        """ Calculate a single state's iteration value. """
        if self.mdp.isTerminal(state):
            return self.values[state]

        actions: list = self.mdp.getPossibleActions(state)
        action_values = util.Counter()

        for action in actions:
            transitions: list = self.mdp.getTransitionStatesAndProbs(state, action)
            action_values[action] = 0
            for transition in transitions:
                next_state = transition[0]
                prob = transition[1]
                action_values[action] += (
                        prob * (self.mdp.getReward(state, action, next_state) + self.discount * self.values[next_state])
                )

        max_qvalue = action_values[action_values.argMax()]

        if toQueue:
            diff = abs(self.values[state] - max_qvalue)
            self.queue.update(state, -diff)

        return max_qvalue

    def __add_predecessor(self, state, predecessors):
        """ Calculate a single state's iteration value. """
        if self.mdp.isTerminal(state):
            return

        # Q-value Calculate.
        for pstate in predecessors[state]:
            max_qvalue = self.__state_max_qvalue(pstate, False)
            diff = abs(self.values[pstate] - max_qvalue)
            if diff > self.theta:
                self.queue.update(pstate, -diff)
