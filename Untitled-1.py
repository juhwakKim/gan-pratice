import numpy as np
from scipy import misc
import glob
import imageio
from keras.models import Model
from keras.layers import *
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import Callback
from imageloader import data_process
import tensorflow as tf

'''
imgs = glob.glob('datasets/celebA/*.jpg')
np.random.shuffle(imgs)

height,width = misc.imread(imgs[0]).shape[:2]
center_height = int((height - width) / 2)
'''
# hyper parameters
total_epoch = 100
batch_size = 64
img_dim = 50
z_dim = 256
'''
def imread(f):
    x = misc.imread(f)
    x = x[center_height:center_height+width, :]
    x = misc.imresize(x, (img_dim, img_dim))
    x = x.astype(np.float32) / 255 * 2 - 1
    return x
'''

def data_generator(bs=64):
    X = []
    while True:
        np.random.shuffle(imgs)
        for f in imgs:
            X.append(imread(f))
            if len(X) == bs:
                X = np.array(X)
                yield X,None
                X = []

def Conv1DTranspose(input_tensor, filters, kernel_size, strides=1, padding='same', activation='tanh',
                    name: str = "", trainable: bool = True):
    x = Lambda(lambda x: K.expand_dims(x, axis=2), name=name + '_deconv1d_part1')(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1),
                        activation=activation, strides=(strides, 1), padding=padding,
                        name=name + '_deconv1d_part2', trainable=trainable)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2), name=name + '_deconv1d_part3')(x)
    return x

# Encoder
x_in = Input(shape=(50, 2))
x = x_in
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
x = GlobalAveragePooling1D()(x)

encoder = Model(x_in, x)
encoder.summary()
map_size = K.int_shape(encoder.layers[-2].output)[1:-1]

# Decoder
z_in = Input(shape=K.int_shape(x)[1:])
z = z_in
z = Dense(np.prod(map_size)*z_dim)(z)
z = Reshape(map_size + (z_dim,))(z)
z = Conv1DTranspose(z, 256, 5, trainable=True, padding='SAME',name="deconv_decoder_1")
z = BatchNormalization()(z)
z = Activation('relu')(z)
z = Conv1DTranspose(z, 128, 5, trainable=True, padding='SAME',name="deconv_decoder_2")
z = BatchNormalization()(z)
z = Activation('relu')(z)
z = Conv1DTranspose(z, 64, 5, trainable=True, padding='SAME',name="deconv_decoder_3")
z = BatchNormalization()(z)
z = Activation('relu')(z)
z = Conv1DTranspose(z, 32, 5, trainable=True, padding='SAME',name="deconv_decoder_4")
z = BatchNormalization()(z)
z = Activation('relu')(z)
z = Conv1DTranspose(z, 2, 5, trainable=True, padding='SAME',name="deconv_decoder_5")
z = Activation('tanh')(z)

decoder = Model(z_in, z)
decoder.summary()

class ScaleShift(Layer):
    def __init__(self, **kwargs):
        super(ScaleShift, self).__init__(**kwargs)
    def call(self, inputs):
        z, shift, log_scale = inputs
        z = K.exp(log_scale) * z + shift
        logdet = -K.sum(K.mean(log_scale, 0))
        self.add_loss(logdet)
        return z

z_shift = Dense(z_dim)(x)
z_log_scale = Dense(z_dim)(x)
u = Lambda(lambda z: K.random_normal(shape=K.shape(z)))(z_shift)
z = ScaleShift()([u, z_shift, z_log_scale])

x_recon = decoder(z)
x_out = Subtract()([x_in, x_recon])

recon_loss = 0.5 * K.sum(K.mean(x_out**2, 0)) + 0.5 * np.log(2*np.pi) * np.prod(K.int_shape(x_out)[1:])
z_loss = 0.5 * K.sum(K.mean(z**2, 0)) - 0.5 * K.sum(K.mean(u**2, 0))
vae_loss = recon_loss + z_loss

vae = Model(x_in, x_out)
vae.add_loss(vae_loss)
vae.compile(optimizer=Adam(1e-4))


def sample(path):
    n = 9
    figure = np.zeros((img_dim*n, 2))
    for i in range(n):
        for j in range(n):
            x_recon = decoder.predict(np.random.randn(1, *K.int_shape(x)[1:]))
            digit = x_recon[0]
            figure[i*img_dim: (i+1)*img_dim,
                   j*img_dim: (j+1)*img_dim] = digit
    figure = (figure + 1) / 2 * 255
    figure = figure.astype(np.uint8)
    imageio.imwrite(path, figure)


class Evaluate(Callback):
    def __init__(self):
        import os
        self.lowest = 1e10
        self.losses = []
        if not os.path.exists('VAE_samples'):
            os.mkdir('VAE_samples')
        with open('VAE_samples/architecture.json', 'w') as f:
            f.write(decoder.to_json())
    def on_epoch_end(self, epoch, logs=None):
        path = 'VAE_samples/test_%s.png' % epoch
        sample(path)
        self.losses.append((epoch, logs['loss']))
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            decoder.save_weights('VAE_samples/best_encoder.weights')


evaluator = Evaluate()

dp = data_process('./aug/all/test/number',True)
dp.point_data_load()
#dp.image_make()
#dp.image_read()
dp.sequence_50()
dp.data_shuffle()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

vae.fit(dp.point,epochs=100,batch_size=batch_size)
'''
vae.fit_generator(data_generator(batch_size),
                  epochs=total_epoch,
                  steps_per_epoch=int(len(imgs)/batch_size),
                  callbacks=[evaluator])
'''