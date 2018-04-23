from project3.vars import NORTH, SOUTH, WEST, EAST, FIELD_DIM_X, FIELD_DIM_Y
import numpy as np


class World:
    def __init__(self, player, opponent, debug=False):
        self.player = player
        self.opponent = opponent
        self.debug = debug

    def reset(self):
        self.player.reset()
        self.opponent.reset()
        return self._discretize_state()

    def step(self, player_action, opponent_action):
        if np.random.random() < 0.5:
            actions = [player_action, opponent_action]
            player_first = True
        else:
            actions = [opponent_action, player_action]
            player_first = False

        if self.debug:
            if player_first:
                print('Player action: {}'.format(self._get_direction(player_action)))
                print('Opponent action: {}'.format(self._get_direction(opponent_action)))
            else:
                print('Opponent action: {}'.format(self._get_direction(opponent_action)))
                print('Player action: {}'.format(self._get_direction(player_action)))

        for i, a in enumerate(actions):
            first_move = i == 0
            player_move = (player_first and first_move) or (not player_first and not first_move)
            self._check_collision(player_move, first_move, a)

        new_state = self._discretize_state()
        reward, done = self._check_goal()
        details = None  # TODO

        return new_state, reward, done, details

    def render(self):
        # +-----------+
        # |A |  |  |  |
        # +--+--+--+--+
        # |  |  |B*|  |
        # +-----------+

        p_output = 'A*' if self.player.possession else 'A '
        o_output = 'B*' if self.opponent.possession else 'B '
        empty_output = '  '

        print('+-----------+')
        for y in range(FIELD_DIM_Y):
            row_str = ''
            for x in range(FIELD_DIM_X):
                row_str += '|'
                if self.player.x == x and self.player.y == y:
                    row_str += p_output
                elif self.opponent.x == x and self.opponent.y == y:
                    row_str += o_output
                else:
                    row_str += empty_output

            row_str += '|\n+-----------+'
            print(row_str)
        print(' AG       BG \n')

    def _check_collision(self, player_move, first_move, action):
        collision = False
        same_row = self.player.y == self.opponent.y
        same_col = self.player.x == self.opponent.x

        if player_move:
            if action == NORTH and self.player.y > 0 and self.opponent.y == self.player.y - 1 and same_col:
                collision = True
            elif action == SOUTH and self.player.y < FIELD_DIM_Y - 1 and self.opponent.y == self.player.y + 1 and same_col:
                collision = True
            elif action == WEST and self.player.x > 0 and self.opponent.x == self.player.x - 1 and same_row:
                collision = True
            elif action == EAST and self.player.x < FIELD_DIM_X - 1 and self.opponent.x == self.player.x + 1 and same_row:
                collision = True

            if not collision:
                self.player.move(action)
            elif not first_move and self.player.possession:
                self.player.possession = False
                self.opponent.possession = True

        else:
            if action == NORTH and self.opponent.y > 0 and self.player.y == self.opponent.y - 1 and same_col:
                collision = True
            elif action == SOUTH and self.opponent.y < FIELD_DIM_Y - 1 and self.player.y == self.opponent.y + 1 and same_col:
                collision = True
            elif action == WEST and self.opponent.x > 0 and self.player.x == self.opponent.x - 1 and same_row:
                collision = True
            elif action == EAST and self.opponent.x < FIELD_DIM_X - 1 and self.player.x == self.opponent.x + 1 and same_row:
                collision = True

            if not collision:
                self.opponent.move(action)
            elif not first_move and self.opponent.possession:
                self.opponent.possession = False
                self.player.possession = True

    def _discretize_state(self):
        """
        If player has possession, then state in [0, 64)
            If player in square 0, then state in [0, 8)
                If opponent in square 0, then state == 0
                Elif opponent in square 1, then state == 1
                ...
            If player in square 1, then state in [8, 16)
            ...
        Elif opponent has possession, then state in [64, 128)
            If player in square 0, then state in [64, 72)
                If opponent in square 0, then state == 64
                ...
            ...
        :return:
        """
        player_possession = 1 if self.player.possession else 0
        player_pos = FIELD_DIM_X * self.player.y + self.player.x
        opponent_pos = FIELD_DIM_X * self.opponent.y + self.opponent.x
        return (FIELD_DIM_X * FIELD_DIM_Y) ** 2 * (1 - player_possession) + \
               (FIELD_DIM_X * FIELD_DIM_Y) * player_pos + \
               opponent_pos

    def _check_goal(self):
        reward = 0
        # Player goal
        if self.player.possession and self.player.x == 0:
            reward = 100
        # Opponent own goal
        elif self.opponent.possession and self.opponent.x == 0:
            reward = 100
        # Opponent goal
        elif self.opponent.possession and self.opponent.x == FIELD_DIM_X - 1:
            reward = -100
        # Player own goal
        elif self.player.possession and self.player.x == FIELD_DIM_X - 1:
            reward = -100

        return reward, reward != 0

    def _get_direction(self, action):
        if action == NORTH:
            return '^'
        elif action == SOUTH:
            return 'v'
        elif action == WEST:
            return '<'
        elif action == EAST:
            return '>'
        else:
            return '.'


class Player:
    def __init__(self, x, y, possession, name):
        self.init_x = self.x = x
        self.init_y = self.y = y
        self.init_possession = self.possession = possession
        self.name = name

    def reset(self):
        self.x = self.init_x
        self.y = self.init_y
        self.possession = self.init_possession

    def move(self, action):
        if action == NORTH and self.y > 0:
            self.y -= 1
        elif action == SOUTH and self.y < FIELD_DIM_Y - 1:
            self.y += 1
        elif action == WEST and self.x > 0:
            self.x -= 1
        elif action == EAST and self.x < FIELD_DIM_X - 1:
            self.x += 1

    def set_possession(self, possession):
        self.possession = possession


class Referee:
    def __init__(self):
        self.joint_strategies = []

    def select_actions(self):
        return np.random.choice(self.joint_strategies)
