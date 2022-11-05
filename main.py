"""

Author: Arthur Wesley & Camden Ross

"""

import sente
# from sente import sgf
from tensorflow import keras
from tensorflow.keras import layers


def main():

    game = sente.Game()

    inputs = keras.Input(shape=(784,))

    dense = layers.Dense(64, activation="relu")
    x = dense(inputs)

    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(10)(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
    model.summary()


if __name__ == "__main__":
    main()
