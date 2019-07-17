from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Dropout, CuDNNLSTM,  Bidirectional, LSTM, Conv1D,MaxPooling1D,ZeroPadding1D,GlobalMaxPooling1D,UpSampling1D
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras.backend as K
import keras
import pandas as pd
import os
import matplotlib.pyplot as plt
from imageloader import data_process

import sys

import numpy as np

class WGAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        self.scaler = MinMaxScaler()
        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = RMSprop(lr=0.00005)

        # Build and compile the critic
        self.critic = self.build_critic()
        self.critic.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generated imgs
        z = Input(shape=(1000,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.critic.trainable = False

        # The critic takes generated images as input and determines validity
        valid = self.critic(img)

        # The combined model  (stacked generator and critic)
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
            optimizer=RMSprop(lr=0.0005),
            metrics=['accuracy'])

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):
        '''
        model = Sequential()
        model.add(CuDNNLSTM(256, input_shape=(50,2), return_sequences=True))
        model.add(CuDNNLSTM(256,return_sequences=True))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(2, activation='sigmoid'))
        model.summary()
        
        noise = Input(shape=(50,2))
        seq = model(noise)
        '''
        
        model = Sequential()
        model.add(Dense(25*16, input_dim=(1000)))
        model.add(Reshape((25,16)))
        model.add(UpSampling1D())
        model.add(Conv1D(32,10,padding='same',strides=1))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv1D(16,10,padding='same',strides=1))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv1D(2,10,padding='same',strides=1))
        model.add(Activation("relu"))
        model.summary()
        
        noise = Input(shape=(1000,))
        seq = model(noise)

        return Model(noise, seq)

    def build_critic(self):

        model = Sequential()
        model.add(Conv1D(8,10,padding='same',strides=1,input_shape=(50,2)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv1D(16,10,padding='same',strides=1))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv1D(24,10,padding='same',strides=1))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv1D(32,10,padding='same',strides=1))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        seq = Input(shape=(50,2))
        validity = model(seq)

        return Model(seq, validity)


    def train(self, epochs, batch_size=10, sample_interval=50):

        dp = data_process('./aug/all/test/number',True)
        dp.point_data_load()
        #dp.image_make()
        #dp.image_read()
        dp.sequence_50()
        dp.data_shuffle()




        size = int(np.size(dp.point,0) * 0.7)
        X_train = dp.point
        j = 0
        x_data = []
        y_data = []

        
        
 
        '''
        for k in range(len(dp.point)):
            x_data.append(self.scaler.fit_transform(dp.point[k]))
        
            
        X_train = np.array(x_data)
        Y_DATA = np.array(y_data)

        for i in range(np.size(Y_DATA,0)):
            Y_DATA[i] = np.unique(Y_DATA[i],axis=0)
        Y_DATA = Y_DATA.reshape((np.size(Y_DATA,0),1))
        '''

       # X_train, X_test, Y_train, Y_test = train_test_split(X_DATA, Y_DATA, random_state=42)
        #X_train = keras.preprocessing.sequence.pad_sequences(X_DATA, maxlen=40, padding='post', dtype='float32')


        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        for epoch in range(epochs):

            if(epoch % 50 == 0):
                self.sample_images(1)

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, 16, batch_size)
                imgs = X_train[idx]
                
                # Sample noise as generator input
                noise = np.random.normal(0, 1, (batch_size, 1000))

                # Generate a batch of new images
                gen_imgs = self.generator.predict(noise)

                # Train the critic
                d_loss_real = self.critic.train_on_batch(imgs, valid)
                d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                # Clip critic weights
                for l in self.critic.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)


            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))
            # If at save interval => save generated image samples
            #if epoch % sample_interval == 0:
                #self.sample_images(epoch)

    def sample_images(self, epoch):
        '''
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/mnist_%d.png" % epoch)
        plt.close()
        '''
        for i in range(10):
            noise = np.random.normal(0, 1, (10, 1000))
            predictions = self.generator.predict(noise)
            #print(predictions)
            #predictions = np.reshape(predictions,(10,50,2))
            for j in range(10):
                #predict = self.scaler.inverse_transform(predictions[j])
                predict = predictions[j]
                predict = predict.astype(int)
                dataframe = pd.DataFrame(predict, columns= ['x','y'])
                dataframe.to_csv("./model/test_{}".format(i*10+j), index=False)


if __name__ == '__main__':
    config = tf.ConfigProto()

    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    wgan = WGAN()
    wgan.train(epochs=1000, batch_size=10, sample_interval=50)
