import numpy as np
import random
import pandas as pd
from IPython.display import display, clear_output
import tensorflow as tf
import keras
from keras.callbacks import History
from keras import layers

def best_of(moves):
    """
    Returns the index of the highest value in an array, corresponding to the predicted best move.

    Arguments:
    moves: an array of shape (27,1) of floats representing the predicted move values for the board state.
    Returns:
    move: the index of the maximum value of the array.
    """
    return np.argmax(moves)


def flip_coin():
    """
    randomly returns a 1 or 0
    """
    return random.randint(0, 1)


mat = None


def init_wincon_matrix():
    def put():
        mat[x + 3 * y + 9 * z, col] = 1

    global mat
    mat = np.zeros((27, 49))
    col = 0

    for x in range(3):
        for y in range(3):
            for z in range(3):
                put()
            col += 1
        for z in range(3):
            for y in range(3):
                put()
            col += 1
        for y in range(3):
            z = y
            put()
        col += 1
        for y in range(3):
            z = 2 - y
            put()
        col += 1

    for z in range(3):
        for y in range(3):
            for x in range(3):
                put()
            col += 1
        for y in range(3):
            x = y
            put()
        col += 1
        for y in range(3):
            x = 2 - y
            put()
        col += 1

    for y in range(3):
        for z in range(3):
            x = z
            put()
        col += 1
        for z in range(3):
            x = 2 - z
            put()
        col += 1

    for x in range(3):
        y = z = x
        put()
    col += 1

    for x in range(3):
        y = z = 2 - x
        put()
    col += 1
    for x in range(3):
        y = x
        z = 2 - x
        put()
    col += 1
    for x in range(3):
        z = x
        y = 2 - x
        put()
    col += 1


class BaselinePlayer():
    def __init__(self):
        self.name = "Baseline"

    def get_move(self, board):
        legal_moves = board.legal_moves()
        return random.choice(legal_moves)

    def finalize(self, board, is_x):
        pass


class HumanPlayer():
    def __init__(self, name="Human"):
        self.name = name

    def get_move(self, board):
        clear_output()
        board.display()
        legal_moves = board.legal_moves()
        move = self.query_human()
        while move not in legal_moves:
            move = self.query_human(insult=True)
        return move

    def query_human(self, insult=False):
        print('Please enter a number the x, y, and z coordinates of '
              'your move in that order, separated by a space')
        if insult:
            print('Try a LEGAL move, dipshit')
        coords = input().split()
        x, y, z = map(int, coords)
        move = coords_to_index(x, y, z)

        return move

    def finalize(self, board, is_x):
        board.display()
        if board.score == 1:
            print("Xs win")
        elif board.score == -1:
            print("Os win")
        elif board.score == 0:
            print("draw")
        elif board.score is None:
            print("Game incomplete")
        else:
            assert False, "score is invalid"


class AIPlayer():

    def __init__(self, model, name="CPU", train = False, quiet = False):
        self.model = model
        self.train = train
        self.lastboard = None
        self.lastmove = None
        self.name = name
        self.losses = []
        self.quiet = False

    def print(self, *args, **kwargs):
        if not self.quiet:
            print(*args, **kwargs)

    def get_move(self, board):
        legal_moves = set(board.legal_moves())
        if board.turn == 'Os':
            current_board = np.concatenate([board.arr[27:],board.arr[:27]], axis=0)
            current_board = current_board.reshape(1,54)
        else:
            current_board = board.arr.reshape(1,54)

        move_values = self.model.predict(current_board)
        current_move = np.argmax(move_values)

        if current_move not in legal_moves:
            self.print(f"idiot tried to play{current_move}")
            current_move = random.choice(list(legal_moves))
            self.print(f"playing {current_move} instead")
        #backprop
        if self.lastboard is not None and self.train:
            Y = np.full((1,27), 2.)
            for move in range(27):
                if move not in legal_moves:
                    Y[0, move] = -1
            Y[0, self.lastmove] = move_values[0,current_move]
            history = self.model.fit(self.lastboard, Y, verbose=0 if self.quiet else 2)
            self.losses.append(history.history['loss'][0])
        self.lastboard = current_board.copy()
        self.lastmove = current_move
        return current_move

    def finalize(self, board, is_x):
        legal_moves = set(board.legal_moves())
        if self.train:
            Y = np.full((1,27), 2.)
            for move in range(27):
                if move not in legal_moves:
                    Y[0, move] = -1
            Y[0, self.lastmove] = board.score * (-1 if not is_x else 1)
            history = self.model.fit(self.lastboard, Y, verbose=0 if self.quiet else 2)
            self.losses.append(history.history['loss'][0])
            self.print(f"XXX average loss is {sum(self.losses)/len(self.losses)}")
            self.lastloss = self.losses

        self.lastboard = None
        self.lastmove = None
        self.losses = []

class Board():

    def __init__(self, player1, player2):
        self.arr = np.zeros((54))
        self.score = None
        self.turn = 'Xs'
        self.player1 = player1
        self.player2 = player2

    def legal_moves(self):
        moves = np.concatenate(np.argwhere(
            (self.arr[:27] == 0) & (self.arr[27:] == 0)))
        return moves

    def display(self):
        def getvals(i):
            if self.arr[i]:
                return "X"
            if self.arr[i + 27]:
                return "O"
            return " "

        vals = list(map(getvals, range(27)))

        outer = []
        for z in range(3):
            inner = []
            for y in range(3):
                inner.append((f"{z} " if y == 1 else "  ") + f"{y}| " +
                             " | ".join(vals[9 * z + 3 * y: 9 * z + 3 * y + 3]))
            xline = "     0   1   2\n"
            edge = "   +-----------+\n"

            outer.append(xline + edge + " |\n   |-----------|\n".join(inner) +
                         " |\n" + edge)

        print(f"{self.player1} as Xs vs {self.player2} as Os\n")
        print("\n\n".join(outer))
        print(f"{self.turn} to play:")

    def game_over(self):
        board = self.arr.reshape((1, 54))
        if mat is None:
            init_wincon_matrix()
        if 3 in np.dot(board[:, :27], mat):
            self.score = 1
            return True
        if 3 in np.dot(board[:, 27:], mat):
            self.score = -1
            return True
        for i in range(27):
            if not board[0, i] or board[0, i + 27]:
                break
        else:
            self.score = 0
            return True
        return False

    def play_move(self, move):
        """
        Applies a supplied move to a supplied board and s

        Arguments:
            move: an integer between 0 and 26 representing an index into `board`

        Returns:
            islegal: a boolean, whether the passed move is legal on the passed board.
            board: the updated board state if the move is legal.  Otherwise, the same board that was passed.
        """
        islegal = True
        if self.arr[move] == 0 and self.arr[move + 27] == 0:
            self.arr[move if self.turn == "Xs" else move + 27] = 1
            if self.turn == "Xs":
                self.turn = "Os"
            else:
                self.turn = "Xs"
        else:
            islegal = False
        return islegal

def mikesfirstmodel():
    model = keras.Sequential([

        layers.Dense(16, activation='relu',
                    #kernel_regularizer=tf.keras.regularizers.l1(0.01),
                    #activity_regularizer=tf.keras.regularizers.l2(0.05)
                    ),

        layers.Dense(27, activation = 'tanh')
        ])

    #Model Hyperparameters
    optimizer = tf.keras.optimizers.Adam(learning_rate=.1)
    metrics = [tictacloss]

    loss = tictacloss
    #loss = 'mean_squared_error'


    model.compile(loss=loss, optimizer=optimizer, metrics=metrics, callback = [History])
    return model

def play_loop(exs, ohs):
    board = Board(exs.name, ohs.name)
    current_player = exs
    next_player = ohs
    while not board.game_over():
        move = current_player.get_move(board)
        board.play_move(move)
        temp = current_player
        current_player = next_player
        next_player = temp
    exs.finalize(board, True)
    ohs.finalize(board, False)


def coords_to_index(x, y, z):
    move = x + 3 * y + 9 * z
    return move


def index_to_coords(i):
    x = i % 3
    y = (i // 3) % 3
    z = i // 9
    return x, y, z


def tictacloss(y_true, y_pred):
    squares = (y_true - y_pred) ** 2
    squares = tf.where(y_true == 2, tf.zeros_like(squares), squares)
    return tf.reduce_sum(squares, axis=0)
