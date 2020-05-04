import numpy as np
import random


def play_move(board, move):
    """
    Applies a supplied move to a supplied board

    Arguments:
        board: a np.array of shape (54,1) represent board state before the move is applied.
        move: an integer between 0 and 26 representing an index into `board`

    Returns:
        islegal: a boolean, whether the passed move is legal on the passed board.
        board: the updated board state if the move is legal.  Otherwise, the same board that was passed.
    """
    islegal = True
    if board[move] == 0 and board[move + 27] == 0:
        board[move] = 1
    else:
        islegal = False
    return board, islegal


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
    print(col)


def game_over(board):
    board = board.reshape((1, 54))
    if mat is None:
        init_wincon_matrix()
    if 3 in np.dot(board[:, :27], mat):
        return True, 1
    if 3 in np.dot(board[:, 27:], mat):
        return True, -1
    for i in range(27):
        if not board[i] or board[i + 27]:
            break
    else:
        return True, 0
    return False, 0
