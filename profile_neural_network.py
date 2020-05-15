from keras import optimizers
from keras.activations import elu, linear
from keras.initializers import Constant
from keras.layers import Conv1D, Dense, Input, Reshape, MaxPool1D, Flatten, SpatialDropout1D, concatenate
from keras.models import Model
from keras import backend as K


def sum_squared_error(y_true, y_pred):
    """
    Sum squared error loss function used for training the model
    :param y_true:
    :param y_pred:
    :return:
    """
    return K.sum(K.square(y_true - y_pred), axis=-1)


def root_mean_squared_error(y_true, y_pred):
    """
    Root mean squared loss function used for evaluating the model.
    :param y_true:
    :param y_pred:
    :return:
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def get_profile_neural_network(n_steps_out, n_steps_in):
    """
    Get A instance of the profile neural network
    :param n_steps_out: The number of prediction, the network should make (horizon)
    :return: A keras model
    """

    conv_input = Input(shape=(n_steps_in,), name="hist_input")
    conv = Reshape((n_steps_in, 1))(conv_input)
    conv = Conv1D(4, [3], activation=elu, padding='same')(conv)
    conv = MaxPool1D(pool_size=2)(conv)
    conv = Conv1D(1, [7], activation=elu, padding='same')(conv)
    conv = MaxPool1D(pool_size=2)(conv)
    conv = Flatten()(conv)

    trend_input = Input(shape=(n_steps_out, 10), name="full_trend")
    trend = Dense(8, activation=elu)(trend_input)
    trend = Dense(4, activation=elu)(trend)
    trend = Conv1D(4, [5], activation=elu, padding='same')(trend)
    trend = Conv1D(1, [5], activation=elu, padding='same')(trend)

    dummy_input = Input(shape=(n_steps_out, 16), name="dummy_input")
    dummy = Conv1D(2, [7], activation=elu, padding='same')(dummy_input)
    dummy = Conv1D(1, [7], activation=elu, padding='same')(dummy)
    dummy = Flatten()(dummy)

    conv = Dense(n_steps_out)(conv)
    conv = Reshape((n_steps_out, 1))(conv)
    dummy = Reshape((n_steps_out, 1))(dummy)
    fc = concatenate([dummy, conv], axis=2)
    fc = Conv1D(16, [7], padding='same', activation=elu)(fc)
    fc = SpatialDropout1D(rate=0.3)(fc)
    fc = Conv1D(8, [7], padding='same', activation=elu)(fc)
    fc = SpatialDropout1D(rate=0.3)(fc)
    fc = Conv1D(1, [7], padding='same')(fc)

    profile_input = Input(shape=(n_steps_out,), name="profile")
    profile = Reshape((n_steps_out, 1))(profile_input)

    out = concatenate([fc, profile, trend])
    out = Conv1D(1, [1], padding='same', use_bias=False, activation=linear, kernel_initializer=Constant(value=1 / 3))(
        out)
    pred = Flatten()(out)

    model = Model(inputs=[conv_input, trend_input, dummy_input, profile_input], outputs=pred)
    model.compile(optimizer=optimizers.Adam(), loss=sum_squared_error,
                  metrics=[root_mean_squared_error])
    return model
