import numpy as np
import random
from IPython.display import display, clear_output
import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt
import os
import statistics as stat
import pandas as pd
from google.colab import drive
DRIVEPATH = '/gdrive/My Drive/Colab Notebooks/tic tac toe/'


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

mats = None

def init_wincon_matrix():
    step = [0]

    def put():

        mats[x + 3 * y + 9 * z, step, col] = 1
        step[0] = (step[0] + 1) % 3

    global mat
    global mats
    mats = np.zeros((27, 3, 49))

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
    mat = np.sum(mats, axis=1)


class BaselinePlayer:
    def __init__(self):
        self.name = "Baseline"

    def get_move(self, board):
        legal_moves = board.legal_moves()
        return random.choice(legal_moves)

    def finalize(self, board, is_x):
        pass


class SmartPlayer():
    def __init__(self,iq=1):
        self.name = "SmartPlayer: iq " + str(iq)
        self.iq = iq

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
        winning_moves = []
        m_score = np.zeros((27,))
        opponent_blocker = []
        for m in legal_moves:
            for c in range(mat.shape[1]):
                if mat[m, c]:
                    if my_progress[0, c] == 2 and o_progress[0, c] == 0:
                        winning_moves.append(m)
                    if my_progress[0, c] == 0 and o_progress[0, c] == 2:
                        opponent_blocker.append(m)
                    if my_progress[0, c] + o_progress[0, c] == 1:
                        m_score[m] += 2
                    if my_progress[0, c] + o_progress[0, c] == 0:
                        m_score[m] += 1
        if winning_moves:
            return random.choice(winning_moves)
        if opponent_blocker:
            return random.choice(opponent_blocker)
        if np.max(m_score) == 0:
            return random.choice(legal_moves)
        for i in range(len(legal_moves-1)):
            if random.random() > self.iq:
                best_move = random.choice(np.concatenate(np.argwhere(m_score == np.max(m_score))))
                m_score[best_move] = 0

            else:
                break
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



class AIPlayer:

    def __init__(self, model, name="CPU", train=False, verbose=1, exploration = 0):
        self.model = model
        self.train = train
        self.lastboard = None
        self.lastmove = None
        self.lastlegal = None
        self.name = name
        self.losses = []
        self.verbose = verbose
        self.current_illegal_move_count = 0
        self.last_illegal_move_count = None
        self.current_move_count = 0
        self.last_move_count = None
        self.lastloss = [0,0]
        self.exploration = exploration
        self.last_move_values = None
        self.model.opponent_history = []


    def print(self, *args, **kwargs):
        if "level" in kwargs:
            level = kwargs["level"]
            del kwargs["level"]
        else:
            level = 1
        if self.verbose >= level:
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
        proposed_move = current_move

        # Returns random move if exploring, or if an illegal move was proposed.
        exploring = random.random() < self.exploration
        if current_move not in legal_moves or exploring:
            current_move = random.choice(list(legal_moves))
            if exploring:
                self.print(f"Exploring:")
            else:
                self.print(f"idiot tried to play {current_move} illegally:")
                self.current_illegal_move_count += 1
            self.print(f"playing {current_move} instead {proposed_move}")

        # backprop
        if self.lastboard is not None:
            Y = np.full((1, 27), 2.)
            for move in range(27):
                if move not in self.lastlegal:
                    Y[0, move] = -2
            Y[0, self.lastmove] = move_values[0, current_move]
            self.print(f"XXX X is {self.lastboard}\n Yhat is {self.last_move_values}, \nY is {Y}", level=2)
            if self.train:
                history = self.model.fit(self.lastboard, Y, verbose=self.verbose)
                self.losses.append(history.history['loss'][0])

            elif self.model.optimizer is not None:
                loss1 = self.model.evaluate(self.lastboard, Y, verbose=self.verbose)
                self.losses.append(loss1)
        self.lastboard = current_board.copy()
        self.lastmove = current_move
        self.lastlegal = legal_moves
        self.last_move_values = move_values

        return current_move


    def finalize(self, board, is_x):
        if self.train:
            Y = np.full((1, 27), 2.)
            for move in range(27):
                if move not in self.lastlegal:
                    Y[0, move] = -2
            Y[0, self.lastmove] = board.score * (-1 if not is_x else 1) + 4
            self.print(f"XXX Final X is {self.lastboard}\n Yhat is {self.last_move_values}, \nY is {Y}", level=2)
            history = self.model.fit(self.lastboard, Y, verbose=self.verbose)
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
        # Set back to initial conditions?
        # self.last_move_values = None


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
    cache_verbose = ai.verbose
    cache_train = ai.train
    ai.train = train
    ai.verbose = 0
    opponentnames = [opponent.name for opponent in opponents]
    ai.model.opponent_history.append((opponentnames,epochs))
    try:
        isxs = True
        scores = ([], [])
        movelosses = ([], [])
        finallosses = ([], [])
        avgillegal = ([], [])
        wingamelen = ([], [])
        lossgamelen = ([], [])
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
            finallosses[int(isxs)].append(ai.lastloss[-1])
            movelosses[int(isxs)].append(stat.mean(ai.lastloss[:-1]))
            scores[int(isxs)].append(score)
            avgillegal[int(isxs)].append(ai.last_illegal_move_count / ai.last_move_count)
            if score == 1:
                wingamelen[int(isxs)].append(ai.last_move_count)
            if score == -1:
                lossgamelen[int(isxs)].append(ai.last_move_count)

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
        # XXX NEED TO FIX PLOTTING AND STATS
        if display_results:

            fig, axes = plt.subplots(6, 2)
            row1, row2, row3, row4, row5, row6 = axes
            for i in range(2):
                turn = ["ohs", "exs"][i]

                row1[i].plot(avgscores[i])
                row1[i].set_title('Final Scores for ' + turn)
                row2[i].plot(movelosses[i])
                row2[i].set_title('move losses for ' + turn)
                row3[i].plot(finallosses[i])
                row3[i].set_title('Final losses for ' + turn)
                row4[i].plot(avgavgillegal[i])
                row4[i].set_title('average illegal moves per move for ' + turn)
                row5[i].plot(avgwingamelen[i])
                row5[i].set_title('average length of games won for ' + turn)
                row6[i].plot(avglossgamelen[i])
                row6[i].set_title('average length of games lost for ' + turn)
            fig.set_size_inches(12, 20)
            plt.show()

        meanwingamelen = [0, 0]
        meanlossgamelen = [0, 0]
        stats = {}
        for i in range(2):
            turn = ["ohs", "exs"][i]
            if wingamelen[i]:
                meanwingamelen[i] = stat.mean(wingamelen[i])

            if lossgamelen[i]:
                meanlossgamelen[i] = stat.mean(lossgamelen[i])



            stats["score for "+turn] = stat.mean(scores[i])
            stats["move loss for "+turn] = stat.mean(movelosses[i])
            stats["final loss for "+turn] = stat.mean(finallosses[i])
            stats["avg illegal for "+turn] = stat.mean(avgillegal[i])
            stats["win game length for "+turn] = meanwingamelen[i]
            stats["loss game length for "+turn] = meanlossgamelen[i]

        print('Training Complete')
        return stats


    finally:
        ai.verbose = cache_verbose
        ai.train = cache_train

def eval_loop(model,verbose=0,save=True):
    if save:
        drive.mount('/gdrive')
        savepath = DRIVEPATH + 'Experiment Dataframe.csv'
        df = pd.read_csv(savepath, index_col='model_name')
        if model.name in df.index:
            raise ValueError('model with this name already exists in the Experimental Database.'
            'Please choose a new name for your model')
    ai = AIPlayer(model, train=False, verbose=verbose)
    stats = training_loop(ai,[SmartPlayer(.9)],train=False, epochs=1000)
    data = pd.DataFrame({'model_name':[model.name],
                    'model_summary':[model.get_config()],
                    'optimizer':[model.optimizer.get_config()],
                    'opponent_history':[model.opponent_history],
                    'eval_stats':[stats]}).set_index('model_name')
    if save:
        df = df.append(data, verify_integrity=True,)
        df.to_csv(savepath)
        print('saving results to: ' + savepath)
    return data

def running_average(data, alpha):
    ret = ([], [])
    for i in range(2):
        if data[i]:
            ret[i].append(data[i][0])
            for j in range(1, len(data[i])):
                ret[i].append((ret[i][j - 1] * alpha) + (data[i][j] * (1 - alpha)))
    return ret


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
        y_true = tf.where(tf.math.abs(y_true) <= 1., self.decay * y_true, y_true)
        y_true = tf.where(y_true >= 3, y_true - 4, y_true)

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


class TicRows(layers.Layer):

    def __init__(self, channels, board_edge=3, connection_mat = None,
                 rows_share_weights = False):
        super(TicRows, self).__init__()

        self.channels = channels
        self.board_edge = board_edge
        if connection_mat is None:
            if mats is None:
                init_wincon_matrix()
            connection_mat = mats
        self.connection_mat = connection_mat
        self.rows_share_weights = rows_share_weights

    def build(self, input_shape):
        board_size = self.board_edge**3
        row_count = 3 * (self.board_edge + 1)**2 + 1
        if input_shape[-1] % board_size != 0:
            raise ValueError(f"input shape should a multiple of "
                f"{board_size}, but instead is {input_shape[-1]}")
        input_channels = input_shape[-1] // board_size

        if self.rows_share_weights:
            num_row_weights = 1
        else:
            num_row_weights = row_count
        self.v = self.add_weight(shape=(self.board_edge, input_channels,
                                        num_row_weights, self.channels),
                                 initializer='random_normal',
                                 trainable=True)

        self.b = tf.repeat(self.add_weight(shape=(self.channels * num_row_weights,),
                                 initializer='random_normal',
                                 trainable=True), row_count//num_row_weights,
                           axis = 0)

    def call(self, inputs):
        # mats is (board, rowlen, rows)
        # v is (rowlen, input_channels, rows,  output_channels, )
        # mats is elemnent wise multiplied by v, with v broacasted across board, and mats broatcasted by input_channels and output channels.
        # input is  (batchsize, input channels* board)
        row_count = 3 * (self.board_edge + 1)**2 + 1

        board_size = self.board_edge**3
        input_channels = inputs.shape[-1] // board_size
        if self.rows_share_weights:
            num_row_weights = 1
        else:
            num_row_weights = row_count
        v_reshaped = tf.reshape(self.v, (1, self.board_edge, input_channels, num_row_weights, self.channels))
        v_broadcasted = tf.broadcast_to(v_reshaped, (board_size, self.board_edge, input_channels, row_count, self.channels))
        mats_reshaped = tf.reshape(self.connection_mat, (board_size, self.board_edge, 1, row_count, 1))
        mats_broadasted = tf.broadcast_to(mats_reshaped, (board_size, self.board_edge, input_channels, row_count, self.channels))
        w_unsummed_unshaped = tf.cast(mats_broadasted, "float32") * v_broadcasted
        w_summed_unshaped = tf.math.reduce_sum(w_unsummed_unshaped, axis=1)
        w = tf.reshape(w_summed_unshaped, (input_channels * board_size, self.channels * row_count))
        # return is (batchsize, rows * output_channels)
        return tf.matmul(inputs, w) + self.b

    def compute_output_shape(self, input_shape):
        row_count = 3 * (self.board_edge + 1)**2 + 1
        return (input_shape[0], row_count * self.channels)

    def get_config(self):
        return {"channels": self.channels, "board_edge":self.board_edge, "connection_mat":None}
