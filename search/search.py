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
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
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
    start = goal = (problem.getStartState(), [])
    if problem.isGoalState(start[0]):
        return []

    stack = util.Stack()
    stack.push(start)
    explored = []

    while not stack.isEmpty() and not problem.isGoalState(goal[0]):
        node = stack.pop()
        explored.append(node[0])
        for child in problem.getSuccessors(node[0]):
            if child[0] not in explored:
                goal = (child[0], child[1], node)
                stack.push(goal)

    if goal is not start:
        actions = []
        while len(goal) == 3:
            actions.append(goal[1])
            goal = goal[2]
        actions.reverse()
        return actions

    util.raiseNotDefined()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    start = (problem.getStartState(), [])
    queue = util.Queue()
    queue.push(start)
    explored = [start[0]]

    while not queue.isEmpty():
        node = queue.pop()
        if problem.isGoalState(node[0]):
            actions = []
            while len(node) == 3:
                actions.append(node[1])
                node = node[2]
            actions.reverse()
            return actions
        for child in problem.getSuccessors(node[0]):
            if child[0] not in explored:
                explored.append(child[0])
                queue.push((child[0], child[1], node))

    util.raiseNotDefined()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    start = (problem.getStartState(), [], 0)
    p_queue = util.PriorityQueue()
    p_queue.push(start, 0)
    explored = []

    while not p_queue.isEmpty():
        node = p_queue.pop()
        if node[0] not in explored:
            explored.append(node[0])
            if problem.isGoalState(node[0]):
                actions = []
                while len(node) == 4:
                    actions.append(node[1])
                    node = node[3]
                actions.reverse()
                return actions
            for c in problem.getSuccessors(node[0]):
                p_queue.push((c[0], c[1], c[2] + node[2], node), c[2] + node[2])

    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    start = (problem.getStartState(), [], 0)
    p_queue = util.PriorityQueue()
    p_queue.push(start, 0)
    explored = []

    while not p_queue.isEmpty():
        node = p_queue.pop()
        if node[0] not in explored:
            explored.append(node[0])
            if problem.isGoalState(node[0]):
                actions = []
                while len(node) == 4:
                    actions.append(node[1])
                    node = node[3]
                actions.reverse()
                return actions
            for c in problem.getSuccessors(node[0]):
                p_queue.push((c[0], c[1], c[2] + node[2], node), c[2] + node[2] + heuristic(c[0], problem))

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
