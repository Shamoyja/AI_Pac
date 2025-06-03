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

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


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
        foodgen = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodgen = successorGameState.getFood().asList()
        minimumfood = float("inf")
        for food in foodgen:
            minimumfood = min(minimumfood, manhattanDistance(newPos, food))

        for ghos in successorGameState.getGhostPositions():
            if (manhattanDistance(newPos, ghos) < 2):
                mini = - float('inf')
                return mini
        
        return successorGameState.getScore() + 1/minimumfood

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
        """
        "*** YOUR CODE HERE ***"       """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.
        """
        def maxValue(state, depth):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            v = float('-inf')
            for action in state.getLegalActions(0):
                successorState = state.generateSuccessor(0, action)
                v = max(v, minValue(successorState, depth, 1))
            return v

        def minValue(state, depth, agentIndex):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            v = float('inf')
            numAgents = state.getNumAgents()
            for action in state.getLegalActions(agentIndex):
                successorState = state.generateSuccessor(agentIndex, action)
                if agentIndex == numAgents - 1:  
                    v = min(v, maxValue(successorState, depth + 1))
                else:
                    v = min(v, minValue(successorState, depth, agentIndex + 1))
            return v

        legalActions = gameState.getLegalActions(0)
        bestAction = None
        bestValue = float('-inf')

        for action in legalActions:
            successorState = gameState.generateSuccessor(0, action)
            value = minValue(successorState, 0, 1)  
            if value > bestValue:
                bestValue = value
                bestAction = action
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """

        def maxValue(state, depth, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            v = float('-inf')
            for action in state.getLegalActions(0):
                successorState = state.generateSuccessor(0, action)
                v = max(v, minValue(successorState, depth, 1, alpha, beta))
                if v > beta:  
                    return v
                alpha = max(alpha, v) 
            
            return v

        def minValue(state, depth, agentIndex, alpha, beta):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            v = float('inf')
            numAgents = state.getNumAgents()
            for action in state.getLegalActions(agentIndex):
                successorState = state.generateSuccessor(agentIndex, action)
                if agentIndex == numAgents - 1: 
                    v = min(v, maxValue(successorState, depth + 1, alpha, beta))
                else:
                    v = min(v, minValue(successorState, depth, agentIndex + 1, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        legalActions = gameState.getLegalActions(0)
        bestAction = None
        bestValue = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        for action in legalActions:
            successorState = gameState.generateSuccessor(0, action)
            value = minValue(successorState, 0, 1, alpha, beta)  
            if value > bestValue:
                bestValue = value
                bestAction = action
            alpha = max(alpha, bestValue)
        return bestAction

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
        "*** YOUR CODE HERE ***"
        def maxValue(state, depth):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            v = float('-inf')
            for action in state.getLegalActions(0):
                successorState = state.generateSuccessor(0, action)
                v = max(v, expValue(successorState, depth, 1)) 
            return v
        def expValue(state, depth, agentIndex):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            v = 0
            legalActions = state.getLegalActions(agentIndex)
            p = 1.0 / len(legalActions)

            numAgents = state.getNumAgents()
            for action in legalActions:
                successorState = state.generateSuccessor(agentIndex, action)
                
                if agentIndex == numAgents - 1: 
                    v += p * maxValue(successorState, depth + 1)
                else: 
                    v += p * expValue(successorState, depth, agentIndex + 1)
            return v
        legalActions = gameState.getLegalActions(0)
        bestAction = None
        bestValue = float('-inf')
        for action in legalActions:
            successorState = gameState.generateSuccessor(0, action)
            value = expValue(successorState, 0, 1) 
            if value > bestValue:
                bestValue = value
                bestAction = action
        return bestAction        

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <First I found, pacman's distance to the closest food, then found pacman's proximity to ghosts 
      (avoiding active ghosts) then I have to make sure pacman's eating scared ghosts for extra points while keeping
      track of the number of pellets left and last calculate the current game score and combine all using a weighted 
      sum to calculate the evaluation score.>
    """
    "*** YOUR CODE HERE ***"
    pacmanPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    capsuleList = currentGameState.getCapsules()
    score = currentGameState.getScore()
    if foodList:
        minFoodDist = min(manhattanDistance(pacmanPos, food) for food in foodList)
        score += 10.0 / minFoodDist 
    for ghostState in ghostStates:
        ghostPos = ghostState.getPosition()
        ghostDist = manhattanDistance(pacmanPos, ghostPos)

        if ghostState.scaredTimer > 0:
            score += 20.0 / ghostDist
        else:
            if ghostDist < 2:
                score -= 500  
            else:
                score -= 5.0 / ghostDist 
    score -= 100 * len(capsuleList)
    return score

# Abbreviation
better = betterEvaluationFunction

