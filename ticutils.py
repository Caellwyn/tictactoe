import numpy as np
import random
from IPython.display import display, clear_output
import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt
import os
import statistics as stat


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


class BaselinePlayer:
    def __init__(self):
        self.name = "Baseline"

    def get_move(self, board):
        legal_moves = board.legal_moves()
        return random.choice(legal_moves)

    def finalize(self, board, is_x):
        pass


class SmartPlayer:
    def __init__(self):
        self.name = "Smart"

    def get_move(self, board):
        legal_moves = board.legal_moves()
        if mat is None:
            init_wincon_matrix()
        if board.turn == 'Os':
            current_board = np.concatenate([board.arr[27:], board.arr[:27]], axis=0)
            current_board = current_board.reshape(1, 54)
        else:
            current_board = board.arr.reshape(1, 54)
        my_progress = np.dot(current_board[:, :27], mat)
        o_progress = np.dot(current_board[:, 27:], mat)

        m_score = np.zeros((27,))
        opponent_blocker = []
        for m in legal_moves:
            for c in range(mat.shape[1]):
                if mat[m, c]:
                    if my_progress[0, c] == 2 and o_progress[0, c] == 0:
                        return m
                    if my_progress[0, c] == 0 and o_progress[0, c] == 2:
                        opponent_blocker.append(m)
                    if my_progress[0, c] + o_progress[0, c] == 1:
                        m_score[m] += 2
                    if my_progress[0, c] + o_progress[0, c] == 0:
                        m_score[m] += 1
        if opponent_blocker:
            return opponent_blocker[0]
        if np.max(m_score) == 0:
            return random.choice(legal_moves)
        return random.choice(np.concatenate(np.argwhere(m_score == np.max(m_score))))

    def finalize(self, board, is_x):
        pass


class HumanPlayer:
    def __init__(self, name="Human"):
        self.name = name

    def get_move(self, board):
        board.display()
        legal_moves = board.legal_moves()
        move = self.query_human()
        while move not in legal_moves:
            move = self.query_human(insult=True)
        clear_output()
        return move

    def query_human(self, insult=False):
        print('Please enter a number the x, y, and z coordinates of '
              'your move in that order, separated by a space')
        if insult:
            print('Try a LEGAL move, dipshit')
        move = None
        while move is None:
            try:
                coords = input().split()
                x, y, z = map(int, coords)
                move = coords_to_index(x, y, z)
            except Exception:
                print("Didn't understand you, try again. \nPlease "
                      "input your answer like '0 2 1', \nbut, you know, "
                      "without the quotes")

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


class SmartishPlayer(SmartPlayer):

    def __init__(self, smarts=.5):
        SmartPlayer.__init__(self)
        self.smarts = smarts

    def get_move(self, board):
        if random.random() < self.smarts:
            return SmartPlayer.get_move(self, board)
        else:
            legal_moves = board.legal_moves()
            return random.choice(legal_moves)


class AIPlayer:

    def __init__(self, model, name="CPU", train=False, quiet=False):
        self.model = model
        self.train = train
        self.lastboard = None
        self.lastmove = None
        self.lastlegal = None
        self.name = name
        self.losses = []
        self.quiet = quiet
        self.current_illegal_move_count = 0
        self.last_illegal_move_count = None
        self.current_move_count = 0
        self.last_move_count = None
        self.lastloss = None

    def print(self, *args, **kwargs):
        if not self.quiet:
            print(*args, **kwargs)

    def get_move(self, board):
        self.current_move_count += 1
        legal_moves = set(board.legal_moves())
        if board.turn == 'Os':
            current_board = np.concatenate([board.arr[27:], board.arr[:27]], axis=0)
            current_board = current_board.reshape(1, 54)
        else:
            current_board = board.arr.reshape(1, 54)

        move_values = self.model.predict(current_board)
        current_move = np.argmax(move_values)

        if current_move not in legal_moves:
            self.current_illegal_move_count += 1
            self.print(f"idiot tried to play{current_move}")
            current_move = random.choice(list(legal_moves))
            self.print(f"playing {current_move} instead")
        # backprop
        if self.lastboard is not None and self.train:
            Y = np.full((1, 27), 2.)
            for move in range(27):
                if move not in self.lastlegal:
                    Y[0, move] = -2
            Y[0, self.lastmove] = move_values[0, current_move]
            history = self.model.fit(self.lastboard, Y, verbose=0 if self.quiet else 2)
            self.losses.append(history.history['loss'][0])
        self.lastboard = current_board.copy()
        self.lastmove = current_move
        self.lastlegal = legal_moves
        return current_move

    def finalize(self, board, is_x):
        if self.train:
            Y = np.full((1, 27), 2.)
            for move in range(27):
                if move not in self.lastlegal:
                    Y[0, move] = -2
            Y[0, self.lastmove] = board.score * (-1 if not is_x else 1) + 4
            history = self.model.fit(self.lastboard, Y, verbose=0 if self.quiet else 2)
            self.losses.append(history.history['loss'][0])
            self.print(f"XXX average loss is {sum(self.losses) / len(self.losses)}")
            self.lastloss = self.losses

        self.lastboard = None
        self.lastmove = None
        self.lastlegal = None
        self.losses = []
        self.last_illegal_move_count = self.current_illegal_move_count
        self.current_illegal_move_count = 0
        self.last_move_count = self.current_move_count
        self.current_move_count = 0


class Board:

    def __init__(self, player1, player2):
        self.arr = np.zeros(54)
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
                     # kernel_regularizer=tf.keras.regularizers.l1(0.01),
                     # activity_regularizer=tf.keras.regularizers.l2(0.05)
                     ),

        layers.Dense(27)
    ])

    # Model Hyperparameters
    optimizer = keras.optimizers.Adam(learning_rate=.1)

    loss = TicTacLoss()
    # loss = 'mean_squared_error'

    model.compile(loss=loss, optimizer=optimizer)
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
    return board.score


def training_loop(ai, opponents=[BaselinePlayer()], epochs=1, alpha=.9,
                  round_robin=False, train=True,
                  save_path=None, display_results=True,
                  progress_frequency=10, save_frequency = 10000):

    cache_quiet = ai.quiet
    cache_train = ai.train
    ai.quiet = True
    ai.train = train
    try:
        isxs = True
        scores = []
        movelosses = []
        finallosses = []
        avgillegal = []
        wingamelen = []
        lossgamelen = []
        checkpointpath = None


        if not isinstance(opponents, list):
            opponents = [opponents]

        for game in range(epochs):

            isxs = not isxs
            if round_robin:
                foe = opponents[game % len(opponents)]
            else:
                foe = random.choice(opponents)
            if isxs:
                score = play_loop(ai, foe)
            else:
                score = -play_loop(foe, ai)
            finallosses.append(ai.lastloss[-1])
            movelosses.append(stat.mean(ai.lastloss[:-1]))
            scores.append(score)
            avgillegal.append(ai.last_illegal_move_count / ai.last_move_count)
            if score == 1:
                wingamelen.append(ai.last_move_count)
            if score == -1:
                lossgamelen.append(ai.last_move_count)

            if game % progress_frequency == 0:
                print('This is game number: ', game)

            if save_path:
                if game % save_frequency == 0 or game == epochs-1:

                    tempcheckpointpath = f"{save_path}Epochs:{game}"
                    ai.model.save(tempcheckpointpath)
                    print(f"saving model to {tempcheckpointpath}")
                    if checkpointpath is not None:
                        os.remove(checkpointpath)
                    checkpointpath = tempcheckpointpath



        avgscores = running_average(scores, alpha)
        avgavgillegal = running_average(avgillegal, alpha)

        # game len is not the total number of moves made, but just the number of moves taken by the ai.

        avgwingamelen = running_average(wingamelen, alpha)
        avglossgamelen = running_average(lossgamelen, alpha)

        if display_results:

            fig, axes = plt.subplots(3, 2)
            ax1, ax2, ax3, ax4, ax5, ax6 = axes.reshape((6,))

            ax1.plot(avgscores)
            ax1.set_title('Final Scores')
            ax2.plot(movelosses)
            ax2.set_title('move losses')
            ax3.plot(finallosses)
            ax3.set_title('Final losses')
            ax4.plot(avgavgillegal)
            ax4.set_title('average illegal moves per move')
            ax5.plot(avgwingamelen)
            ax5.set_title('average length of games won')
            ax6.plot(avglossgamelen)
            ax6.set_title('average length of games lost')
            fig.set_size_inches(12, 12)
            plt.show()

        if wingamelen:
            meanwingamelen = stat.mean(wingamelen)
        else:
            meanwingamelen = 0

        if lossgamelen:
            meanlossgamelen = stat.mean(lossgamelen)
        else:
            meanlossgamelen = 0


        stats = {"score": stat.mean(scores),
            "move loss": stat.mean(movelosses),
            "final loss": stat.mean(finallosses),
            "avg illegal": stat.mean(avgillegal),
            'win game length': meanwingamelen,
            'loss game length': meanlossgamelen}

        print('Training Complete')
        return stats


    finally:
        ai.quiet = cache_quiet
        ai.train = cache_train


def running_average(data, alpha):
    runningaverage = [data[0]]
    for i in range(1, len(data)):
        runningaverage.append((runningaverage[i - 1] * alpha) + (data[i] * (1 - alpha)))
    return runningaverage


def coords_to_index(x, y, z):
    move = x + 3 * y + 9 * z
    return move


def index_to_coords(i):
    x = i % 3
    y = (i // 3) % 3
    z = i // 9
    return x, y, z


class TicTacLoss:
    def __init__(self, decay=1.):
        self.decay = decay
        self.__name__ = f"loss_decay_{decay}".replace('.', '')

    def __call__(self, y_true, y_pred):
        illegal_moves = y_true == -2
        dont_cares = y_true == 2
        y_pred = tf.where(tf.math.abs(y_pred) <= 1., self.decay * y_pred, y_pred)
        y_pred = tf.where(y_pred >= 3, y_pred - 4, y_pred)

        squares = (y_true - y_pred) ** 2
        squares = tf.where(illegal_moves & (y_pred < -2.), tf.zeros_like(squares), squares)
        squares = tf.where(dont_cares & (tf.math.abs(y_pred) > 1), tf.math.abs(y_pred) - 1, squares)
        squares = tf.where(dont_cares & (tf.math.abs(y_pred) <= 1), tf.zeros_like(squares), squares)

        return tf.reduce_sum(squares, axis=0)

    @staticmethod
    def from_config(cfg):
        return TicTacLoss(cfg["decay"])

    def get_config(self):
        # assert False
        return {"decay": self.decay}


def load_model(path):
    with keras.utils.CustomObjectScope({"TicTacLoss": TicTacLoss}):
        model = keras.models.load_model(path)
        return model
