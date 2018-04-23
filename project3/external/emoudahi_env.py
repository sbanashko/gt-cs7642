import sys
from six import StringIO
from enum import IntEnum
from gym.envs.toy_text import discrete
import numpy as np


MAP = [
    "+----+",
    "|    |",
    "|    |",
    "+----+",
]


class SoccerEnv(discrete.DiscreteEnv):
    """
    Soccer environment defined by Greenwald-Hall

    Soccer is a discrete environment defined by the location of the two players in the grid world, and who currently has
    possession of the ball.

    The grid is sized 2x4 with the first column representing the goal area for player 1, and the final column representing
    the goal for player 2.

    actions: N, S, E, W, stick

    states: location of 1, location of 2,
    """

    num_rows = 2
    num_columns = 4
    max_row = num_rows - 1
    max_column = num_columns - 1

    class Action(IntEnum):
        N = 0
        S = 1
        E = 2
        W = 3
        Stick = 4

    def __init__(self):

        self.desc = np.asarray(MAP, dtype='c')

        # 8 possible positions for A, 8 possible positions for B, 2 possible possessions
        num_states = SoccerEnv.num_rows**2 * SoccerEnv.num_columns**2 * 2

        num_actions = 25 # Five possible actions for each player

        # Transition matrix.
        P = {s: {a: [] for a in range(num_actions)} for s in range(num_states)}

        # Initial starting distribution.
        isd = np.zeros(num_states)

        for player1_row in range(SoccerEnv.num_rows):
            for player1_column in range (SoccerEnv.num_columns):
                for player2_row in range (SoccerEnv.num_rows):
                    for player2_column in range(SoccerEnv.num_columns):

                        # The players may not occupy the same state.
                        if player1_row == player2_row and player1_column == player2_column:
                            continue

                        for player1_possession in range(2):

                            state = SoccerEnv.encode_state(player1_row, player1_column, player2_row, player2_column, player1_possession)

                            # Identify the initial starting states.
                            if (player1_possession and player1_column != 0 and player1_column != SoccerEnv.max_column) \
                                or (not player1_possession and player2_column != 0 and player2_column != SoccerEnv.max_column):
                                isd[state] += 1
                            else:
                                # No need to calculate the transition probabilities for the done states.
                                continue

                            for player1_action in SoccerEnv.Action:
                                for player2_action in SoccerEnv.Action:

                                    action = SoccerEnv.encode_action(player1_action, player2_action)

                                    # Get all transitions out of the current state
                                    transitions = SoccerEnv.transitions(player1_row, player1_column, player2_row,
                                                                        player2_column, player1_possession,
                                                                        player1_action, player2_action)

                                    # All transitions are equally likely.
                                    p = 1.0/len(transitions)

                                    # Add entries in the transition matrix for
                                    for next_state, reward, done in transitions:
                                        P[state][action].append((p, next_state, reward, done))
        isd /= isd.sum()

        discrete.DiscreteEnv.__init__(self, num_states, num_actions, P, isd)
        pass

    @staticmethod
    def resolve_player1_action(s, a, is_first_to_act):
        """
        :param s: The initial state s
        :param a: The action
        :param is_first_to_act: a boolean indicating whether player1 is first to act.
        :return: next_s, collision
        """

        player1_row, player1_column, player2_row, player2_column, player1_possession = s

        # There's a change of possession if the player with the ball moves second.
        is_change_of_possession = not is_first_to_act and player1_possession
        collision_possession = not player1_possession if is_change_of_possession else player1_possession

        if a is SoccerEnv.Action.N:
            collision = player1_row == player2_row + 1 and player1_column == player2_column
            if collision:
                return [player1_row, player1_column, player2_row, player2_column, collision_possession], True
            else:
                return [max(player1_row - 1, 0), player1_column, player2_row, player2_column, player1_possession], False
        elif a is SoccerEnv.Action.E:
            collision = player1_row == player2_row and player1_column + 1 == player2_column
            if collision:
                return [player1_row, player1_column, player2_row, player2_column, collision_possession], True
            else:
                return [player1_row, min(player1_column + 1, SoccerEnv.max_column), player2_row, player2_column,
                        player1_possession], False
        elif a is SoccerEnv.Action.W:
            collision = player1_row == player2_row and player1_column == player2_column + 1
            if collision:
                return [player1_row, player1_column, player2_row, player2_column, collision_possession], True
            else:
                return [player1_row, max(player1_column - 1, 0), player2_row, player2_column, player1_possession], False
            pass
        elif a is SoccerEnv.Action.S:
            collision = player1_row + 1 == player2_row and player1_column == player2_column
            if collision:
                return [player1_row, player1_column, player2_row, player2_column, collision_possession], True
            else:
                return [min(player1_row + 1, SoccerEnv.max_row), player1_column, player2_row, player2_column,
                        player1_possession], False
            pass
        elif a is SoccerEnv.Action.Stick:
            return s, False
        else:
            raise NotImplementedError

    @staticmethod
    def resolve_player2_action(s, a, is_first_to_act):
        player1_row, player1_column, player2_row, player2_column, player1_possession = s

        # There's a change of possession if the player with the ball moves second.
        is_change_of_possession = not is_first_to_act and not player1_possession
        collision_possession = not player1_possession if is_change_of_possession else player1_possession

        if a is SoccerEnv.Action.N:
            collision = player1_row + 1 == player2_row and player1_column == player2_column
            if collision:
                return [player1_row, player1_column, player2_row, player2_column, collision_possession], True
            else:
                return [player1_row, player1_column, max(player2_row - 1, 0), player2_column, player1_possession], False
        elif a is SoccerEnv.Action.E:
            collision = player1_row == player2_row and player1_column == player2_column + 1
            if collision:
                return [player1_row, player1_column, player2_row, player2_column, collision_possession], True
            else:
                return [player1_row, player1_column, player2_row, min(player2_column + 1, SoccerEnv.max_column),
                        player1_possession], False
        elif a is SoccerEnv.Action.S:
            collision = player1_row == player2_row + 1 and player1_column == player2_column
            if collision:
                return [player1_row, player1_column, player2_row, player2_column, collision_possession], True
            else:
                return [player1_row, player1_column, min(player2_row + 1, SoccerEnv.max_row), player2_column,
                        player1_possession], False
        elif a is SoccerEnv.Action.W:
            collision = player1_row == player2_row and player1_column + 1 == player2_column
            if collision:
                return [player1_row, player1_column, player2_row, player2_column, collision_possession], True
            else:
                return [player1_row, player1_column, player2_row, max(player2_column - 1, 0), player1_possession], False
        elif a is SoccerEnv.Action.Stick:
            return s, False
        else:
            raise NotImplementedError

    @staticmethod
    def transitions(player1_row, player1_column, player2_row, player2_column, player1_possession, player1_action, player2_action):
        """ This is a helper method for constructing the transitions probability table.
        :param player1_row:
        :param player1_column:
        :param player2_row:
        :param player2_column:
        :param player1_possession:
        :param player1_action:
        :param player2_action:
        :return: Returns a list of transitions of the form [new state, reward, done]
        """

        s = (player1_row, player1_column, player2_row, player2_column, player1_possession)
        transitions = []

        # Case: Player 1 goes first
        next_s, collision = SoccerEnv.resolve_player1_action(s, player1_action, True)
        if not collision:
            next_s, collision = SoccerEnv.resolve_player2_action(next_s, player2_action, False)
        transitions.append((SoccerEnv.encode_state(next_s[0], next_s[1], next_s[2], next_s[3], next_s[4]),
                            SoccerEnv.reward(next_s), SoccerEnv.done(next_s)))

        # Case: Player 2 goes first:
        next_s, collision = SoccerEnv.resolve_player2_action(s, player2_action, True)
        if not collision:
            next_s, collision = SoccerEnv.resolve_player1_action(next_s, player1_action, False)
        transitions.append((SoccerEnv.encode_state(next_s[0], next_s[1], next_s[2], next_s[3], next_s[4]),
                            SoccerEnv.reward(next_s), SoccerEnv.done(next_s)))

        return transitions

    @staticmethod
    def reward(s):
        player1_row, player1_column, player2_row, player2_column, player1_possession = s
        if player1_possession and player1_column == 0:
            return 100
        elif player1_possession and player1_column == SoccerEnv.max_column:
            return -100
        elif not player1_possession and player2_column == 0:
            return 100
        elif not player1_possession and player2_column == SoccerEnv.max_column:
            return -100
        else:
            return 0

    @staticmethod
    def done(s):
        player1_row, player1_column, player2_row, player2_column, player1_possession = s
        if player1_possession and player1_column == 0:
            return True
        elif player1_possession and player1_column == SoccerEnv.max_column:
            return True
        elif not player1_possession and player2_column == 0:
            return True
        elif not player1_possession and player2_column == SoccerEnv.max_column:
            return True
        else:
            return False

    @staticmethod
    def encode_action(action1, action2):
        return action2 * 5 + action1

    @staticmethod
    def decode_action(x):
        action1 = x % 5
        action2 = x // 5
        return action1, action2

    @staticmethod
    def encode_state(player1_row, player1_column, player2_row, player2_column, player1_possession):
        player1_key = player1_column + player1_row * 4
        player2_key = player2_column + player2_row * 4
        location_key = (player2_key + player1_key * 8)
        return location_key + player1_possession * 64

    @staticmethod
    def decode_state(x):
        location_key = x % 64
        player1_possession = x // 64
        player2_key = location_key % 8
        player1_key = location_key // 8
        player1_column = player1_key % 4
        player1_row = player1_key // 4
        player2_column = player2_key % 4
        player2_row = player2_key // 4
        return player1_row, player1_column, player2_row, player2_column, player1_possession

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]
        player1_row, player1_column, player2_row, player2_column, player1_possession = self.decode_state(self.s)
        print(player1_row, player1_column, player2_row, player2_column, player1_possession)
        out[player1_row + 1][player1_column + 1] = 'A' if player1_possession else 'a'
        out[player2_row + 1][player2_column + 1] = 'b' if player1_possession else 'B'

        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        if self.lastaction is not None:
            action1, action2 = SoccerEnv.decode_action(self.lastaction)
            outfile.write("  ({},{})\n".format(action1, action2))
        else:
            outfile.write("\n")

        # No need to return anything for human
        if mode != 'human':
            return outfile
