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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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
        score: float = successorGameState.getScore()

        newFood_list: list = newFood.asList()
        distance_to_next_food: float = float("inf")
        distances_to_food = [util.manhattanDistance(newPos, food) for food in newFood_list]
        score += 10 / min(distances_to_food) if len(distances_to_food) != 0 else 0

        distance_to_ghost: float = float("inf")
        for ghost, scared in zip(newGhostStates, newScaredTimes):
            if scared == 0 and util.manhattanDistance(newPos, ghost.getPosition()) < distance_to_ghost:
                distance_to_ghost = util.manhattanDistance(newPos, ghost.getPosition())
        score += float("-inf") if distance_to_ghost == 0 or distance_to_ghost == 1 else 1 / (1 - distance_to_ghost)
        # print(f"score = {score}")
        return score


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

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
        return self.getActionHelper(gameState, 1, 0)

    def getActionHelper(self, gameState: GameState, depth: int, agentIndex: int):
        """ Helper function for self.getAction(). """
        agentNumbers: int = gameState.getNumAgents()

        if depth > self.depth and agentIndex == 0:
            return self.evaluationFunction(gameState)

        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        next_index: int = (agentIndex + 1) % agentNumbers
        next_depth: int = depth if next_index != 0 else (depth + 1)
        action_values = {
            action: self.getActionHelper(gameState.generateSuccessor(agentIndex, action), next_depth, next_index)
            for action in gameState.getLegalActions(agentIndex)}

        if agentIndex == 0:
            if depth == 1:
                return max(action_values, key=action_values.get)
            else:
                return max(action_values.values())
        else:
            return min(action_values.values())


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.getActionHelper(gameState, 1, 0, float("-inf"), float("inf"))

    def getActionHelper(self, gameState: GameState, depth: int, agentIndex: int, alpha: float, beta: float):
        """ Helper function for self.getAction(). """
        agentNumbers: int = gameState.getNumAgents()

        if depth > self.depth and agentIndex == 0:
            return self.evaluationFunction(gameState)

        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        next_index: int = (agentIndex + 1) % agentNumbers
        next_depth: int = depth if next_index != 0 else (depth + 1)

        # Max node.
        if agentIndex == 0:
            nowMax: float = float("-inf")
            bestAction: str = ""
            for action in gameState.getLegalActions(agentIndex):
                newValue = self.getActionHelper(gameState.generateSuccessor(agentIndex, action),
                                                next_depth, next_index, alpha, beta)
                if nowMax < newValue:
                    nowMax = newValue
                    if depth == 1:
                        bestAction = action

                alpha = max(alpha, nowMax)
                if alpha > beta:
                    if depth == 1:
                        return action
                    else:
                        return alpha

            if depth == 1:
                return bestAction
            else:
                return nowMax

        # Min node.
        else:
            nowMin: float = float("inf")
            for action in gameState.getLegalActions(agentIndex):
                nowMin = min(nowMin, self.getActionHelper(gameState.generateSuccessor(agentIndex, action),
                                                          next_depth, next_index, alpha, beta))
                beta = min(beta, nowMin)
                if beta < alpha:
                    return beta
            return nowMin


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
