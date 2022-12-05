import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import math
import glob
import traceback
from mcts import mcts

import keras
from keras import layers
import tensorflow as tf
import numpy as np

import sente
from sente import sgf


def sgf_generator(glob_string):
    """

    create an SGF file generator.

    :param glob_string: string to glob files with
    :return: a generator yielding
    """

    # obtain a generator for the glob
    files = glob.iglob(glob_string)

    for file in files:

        try:
            game = sgf.load(file)

            # if the board is not 19x19, skip
            if game.numpy().shape != (19, 19, 4):
                continue
            else:
                # yield the game
                yield game
        except (sente.exceptions.InvalidSGFException,
                ValueError) as error:
            traceback.print_exception(error, file=sys.stderr)
            continue

def get_8_fold_symmetries_x(game: sente.Game):
    """

    generates a numpy array containing a sente game, expanded to all of its 8-fold symmetries

    :param game: the game to get the symmetries for
    :return: 8x19x19x4 array containing all duplicates
    """

    board = game.numpy()
    transpose = np.transpose(board, axes=(1, 0, 2))
    x = np.empty((8,) + board.shape)

    x[0, :, :, :] = board
    x[1, :, :, :] = np.flip(board, axis=0)
    x[2, :, :, :] = np.flip(board, axis=1)
    x[3, :, :, :] = np.flip(board, axis=(0, 1))
    x[4, :, :, :] = transpose
    x[5, :, :, :] = np.flip(transpose, axis=0)
    x[6, :, :, :] = np.flip(transpose, axis=1)
    x[7, :, :, :] = np.flip(transpose, axis=(0, 1))

    return x

def get_8_fold_symmetries_y(move: np.array):
    """

    generates a numpy array containing a sente game, expanded to all of its 8-fold symmetries

    :param move:
    :param game: the game to get the symmetries for
    :return: 8x19x19x4 array containing all duplicates
    """

    transpose = np.transpose(move)
    x = np.empty((8,) + move.shape)

    x[0, :, :] = move
    x[1, :, :] = np.flip(move, axis=0)
    x[2, :, :] = np.flip(move, axis=1)
    x[3, :, :] = np.flip(move, axis=(0, 1))
    x[4, :, :] = transpose
    x[5, :, :] = np.flip(transpose, axis=0)
    x[6, :, :] = np.flip(transpose, axis=1)
    x[7, :, :] = np.flip(transpose, axis=(0, 1))

    return x

def training_data_generator(glob_string: str):
    """

    creates a training data generator object

    :return:
    """

    file_generator = sgf_generator(glob_string)

    for game in file_generator:

        player = game.get_winner()
        y = 1 if player == sente.stone.BLACK else 0
        for move in game.get_default_sequence():

            # break out of the loop as soon as we hit an illegal move
            if not game.is_legal(move):
                break

            """
            # otherwise, generate a label for the move
            move_array = np.zeros(shape=(19, 19))
            move_array[move.get_x(), move.get_y()] = 1
            """

            # get 8-fold symmetries of the board and the correct move

            x = get_8_fold_symmetries_x(game)
            """
            y = get_8_fold_symmetries_y(move_array)
            """
            # go through all the active games and fill the arrays
            for i in range(8):

                # yield the results
                yield x[i], (y,)

            # play the move on the board now
            game.play(move)

def main():
    dataset = tf.data.Dataset.from_generator(lambda: training_data_generator("sgfs-uploaded/2021/*/*/*"),
                                             output_signature=(
                                                 tf.TensorSpec(shape=(19, 19, 4), dtype=tf.uint8),
                                                 tf.TensorSpec(shape=(1,))
                                             )).batch(32)

    test_data = dataset.take(10000)
    training_data = dataset.take(50000)

    # get the numpy spec from a generic game
    input_numpy = sente.Game().numpy()

    # input layer
    input_layer = layers.Input(shape=input_numpy.shape)
    x = input_layer

    # First layer has a kernel size of 5

    x = layers.Conv2D(filters=192,
                      kernel_size=5,
                      padding="same",
                      activation="relu"
                      )(input_layer)
    x = layers.BatchNormalization()(x)

    # subsequent layers have kernel sizes of 3
    for i in range(11):
        x = layers.Conv2D(filters=192,
                          kernel_size=3,
                          activation="relu",
                          padding="same")(x)
        x = layers.BatchNormalization()(x)

    # output layer adds everything together with bias
    x = layers.Conv2D(filters=1,
                      kernel_size=1,
                      activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.BatchNormalization()(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=input_layer, outputs=output, name="Value-Network")
    # keras.utils.plot_model(model)

    model.summary()

    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss="binary_crossentropy")

    history = model.fit(training_data, epochs=5, validation_data=test_data)

    model.save("Value_Network")
if __name__ == "__main__":
    main()
