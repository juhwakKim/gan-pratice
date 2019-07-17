from keras.models import Model
from keras.layers import *
from keras import backend as K
from keras import metrics
from keras import optimizers

import tensorflow as tf

import numpy as np
from imageloader import data_process
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from scipy.stats import norm
import pandas as pd
latent_dim = 64

def Conv1DTranspose(input_tensor, filters, kernel_size, strides=1, padding='same', activation='tanh',
                    name: str = "", trainable: bool = True):
    x = Lambda(lambda x: K.expand_dims(x, axis=2), name=name + '_deconv1d_part1')(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1),
                        activation=activation, strides=(strides, 1), padding=padding,
                        name=name + '_deconv1d_part2', trainable=trainable)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2), name=name + '_deconv1d_part3')(x)
    return x
    
input_layer = Input(shape=(50, 2))
x = input_layer
x = Conv1D(16,5,padding='SAME',strides=1)(x)
x = BatchNormalization()(x)
x = LeakyReLU(0.2)(x)
x = Conv1D(32,5,padding='SAME',strides=1)(x)
x = BatchNormalization()(x)
x = LeakyReLU(0.2)(x)
x = Conv1D(64,5,padding='SAME',strides=1)(x)
x = BatchNormalization()(x)
x = LeakyReLU(0.2)(x)
x = Conv1D(128,5,padding='SAME',strides=1)(x)
x = BatchNormalization()(x)
x = LeakyReLU(0.2)(x)
x = Conv1D(256,5,padding='SAME',strides=1)(x)
x = BatchNormalization()(x)
x = LeakyReLU(0.2)(x)

shape_before_flattening = K.int_shape(x)

x = GlobalAveragePooling1D()(x)
x = Dense(32,activation="relu")(x)

z_mean = Dense(latent_dim)(x)
z_log_var = Dense(latent_dim)(x)

def sampling(args):

    # Reparameterization trick for back-propagation
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=1)
    return z_mean + K.exp(z_log_var) * epsilon


z = Lambda(sampling)([z_mean, z_log_var])

decoder_input = Input(K.int_shape(z)[1:])

x = Dense(np.prod(shape_before_flattening[1:]),
                 activation='relu')(decoder_input)
x = Reshape(shape_before_flattening[1:])(x)

x = Conv1D(256,5,padding='SAME',strides=1)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv1D(128,5,padding='SAME',strides=1)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv1D(64,5,padding='SAME',strides=1)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv1D(32,5,padding='SAME',strides=1)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv1D(2,5,padding='SAME',strides=1)(x)
x = Activation('relu')(x)

decoder = Model(decoder_input, x)
z_decoded = decoder(z)
decoder.summary()
class CustomVariationalLayer(Layer):

    def vae_loss(self, x, z_decoded):

        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        # 여기에서 - 가 곱해져 있다.
        cross_entropy_loss = metrics.mean_squared_error(x, z_decoded)

        kl_loss = 0.5 * K.mean(K.square(z_mean) + K.exp(z_log_var) - z_log_var - 1, axis=-1)
        #kl_loss = -5e-4 * K.mean(1 + z_log_var -
        #                         K.square(z_mean) - K.exp(z_log_var), axis=-1)

        return K.mean(cross_entropy_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        return x


y = CustomVariationalLayer()([input_layer, z_decoded])

vae = Model(input_layer, y)
vae.compile(optimizer=optimizers.Adam(), loss=None)
vae.summary()


dp = data_process('./aug/all/train/number',True)
dp.point_data_load()
#dp.image_make()
#dp.image_read()
dp.sequence_50()
#dp.data_shuffle()

x_data = []
scaler = MinMaxScaler()
x_data = dp.point.reshape((-1,1))
x_data = scaler.fit_transform(x_data)
x_train = x_data.reshape((-1,50,2))
#for k in range(len(dp.point)):
#    x_data.append(scaler.fit_transform(dp.point[k]))

#x_train = np.array(x_data)

#x_train = dp.point
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

vae.fit(x=x_train, y=None, shuffle=True, epochs=1000,
        batch_size=100, verbose=1)
decoder.save_weights("model_1.h5")
n = 15

grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
z_sample = np.random.randn(150,64) 
print(z_sample)
x_decoded = decoder.predict(z_sample, batch_size=150)
for i in range(len(x_decoded)):
    print(x_decoded[i].shape)
    dec = scaler.inverse_transform(x_decoded[i])
    df = pd.DataFrame(dec.astype(int))
    df.to_csv("./test/test{}".format(i),index=False,header=False)
