from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Dropout, CuDNNLSTM, Bidirectional, LSTM
from keras.layers import BatchNormalization, Activation, ZeroPadding2D,Flatten,concatenate
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
import cv2
import sys

import numpy as np

class WGAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        self.scaler = MinMaxScaler((0,100))
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
        img,seq = self.generator(z)

        # For the combined model we will only train the generator
        self.critic.trainable = False

        # The critic takes generated images as input and determines validity
        valid = self.critic([img,seq])

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
        model.add(Dense(256, input_dim=(1000)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(120))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Reshape((40,3)))
        model.add(CuDNNLSTM(256,return_sequences=True))
        model.add(Dense(2, activation='sigmoid'))
        model.summary()
        '''
        input_ = Input(shape=(1000,))

        x = Dense(128 * 8 * 8,activation='relu')(input_)
        x_1 = Reshape((8, 8, 128))(x)
        x_1 = UpSampling2D()(x_1)
        x_1 = Conv2D(128, kernel_size=4, padding="same")(x_1)
        x_1 = BatchNormalization()(x_1)
        x_1 = Activation("relu")(x_1)
        x_1 = UpSampling2D()(x_1)
        x_1 = Conv2D(64, kernel_size=4, padding="same")(x_1)
        x_1 = BatchNormalization()(x_1)
        x_1 = Activation("relu")(x_1)
        x_1 = Conv2D(1, kernel_size=4, padding="same",activation='tanh')(x_1)

        x_2 = Reshape((32,-1))(x)
        x_2 = CuDNNLSTM(256,return_sequences=True)(x_2)
        x_2 = CuDNNLSTM(128,return_sequences=True)(x_2)
        x_2 = Dense(2, activation='sigmoid')(x_2)
        
        model_1 = Model(inputs=input_,outputs=[x_1, x_2])
        model_1.summary()
        noise = Input(shape=(1000,))
        seq_1,seq_2 = model_1(noise)

        return Model(noise, [seq_1,seq_2])

    def build_critic(self):
        
        model = Sequential()
        model.add(CuDNNLSTM(512, input_shape=(40,2), return_sequences=True))
        model.add(Bidirectional(CuDNNLSTM(512)))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        
        input_1 = Input(shape=(32,32,1))

        x_1 = Conv2D(16, kernel_size=3, strides=2, padding="same")(input_1)
        x_1 = LeakyReLU(alpha=0.2)(x_1)
        x_1 = Dropout(0.25)(x_1)
        x_1 = Conv2D(32, kernel_size=3, strides=2, padding="same")(x_1)
        x_1 = ZeroPadding2D(padding=((0,1),(0,1)))(x_1)
        x_1 = BatchNormalization(momentum=0.8)(x_1)
        x_1 = LeakyReLU(alpha=0.2)(x_1)
        x_1 = Dropout(0.25)(x_1)
        x_1 = Conv2D(64, kernel_size=3, strides=2, padding="same")(x_1)
        x_1 = BatchNormalization(momentum=0.8)(x_1)
        x_1 = LeakyReLU(alpha=0.2)(x_1)
        x_1 = Dropout(0.25)(x_1)
        x_1 = Conv2D(128, kernel_size=3, strides=1, padding="same")(x_1)
        x_1 = BatchNormalization(momentum=0.8)(x_1)
        x_1 = LeakyReLU(alpha=0.2)(x_1)
        x_1 = Dropout(0.25)(x_1)
        x_1 = Flatten()(x_1)

        input_2 = Input(shape=(32,2))
        
        x_2 = CuDNNLSTM(128, return_sequences=True)(input_2)
        x_2 = CuDNNLSTM(32)(x_2)
        x_2 = Dense(256)(x_2)

        merged = concatenate([x_1,x_2])
        m = Dense(1)(merged)

        model = Model(inputs=[input_1,input_2], outputs = m)

        seq_1 = Input(shape=(32,32,1))
        seq_2 = Input(shape=(32,2))
        
        validity = model([seq_1,seq_2])

        return Model([seq_1,seq_2], validity)


    def train(self, epochs, batch_size=10, sample_interval=50):

        j = 0
        x_data = []
        y_data = []
        x_img = []
        xt_img = []
        
        for i in range(10):
            while(os.path.exists("./DATA/{}/{}train_{}".format(i,i,j))):
                j += 1
            for k in range(j):
                df = pd.read_csv("./DATA/{}/{}train_{}".format(i,i,k))
                df_img = df[['x','y']].to_numpy()
                img = np.zeros((500, 500, 1), np.uint8)
                for k in range(len(df_img)):
                    if(k != len(df_img)-1):
                        cv2.line(img, (df_img[k][0],df_img[k][1]), (df_img[k+1][0],df_img[k+1][1]), (255,255,255), 30)
                img = cv2.flip(img, 1)
                img = cv2.resize(img,(32,32),interpolation=cv2.INTER_AREA)
                img = img - np.mean(img, axis=0)

                x_data.append(img)
                y_data.append(self.scaler.fit_transform(df[['x','y']].to_numpy()))

            
        X_DATA = np.array(x_data)
        X_DATA = np.reshape(X_DATA,(-1,32,32,1))
        X_DATA = X_DATA/255
        Y_DATA = np.array(y_data)
        Y_DATA = keras.preprocessing.sequence.pad_sequences(Y_DATA, maxlen=32, padding='post', dtype='float32')


        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        for epoch in range(epochs):

            if(epoch % 50 == 0):
                self.sample_images(epoch)

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------
                # Select a random batch of images
                idx = np.random.randint(0, 200, batch_size)
                imgs = X_DATA[idx]
                seqs = Y_DATA[idx]
                # Sample noise as generator input
                noise = np.random.normal(0, 1, (batch_size, 1000))

                # Generate a batch of new images
                gen_imgs,gen_seq = self.generator.predict(noise)

                # Train the critic
                d_loss_real = self.critic.train_on_batch([imgs,seqs], valid)
                d_loss_fake = self.critic.train_on_batch([gen_imgs,gen_seq], fake)
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
        r, c = 10, 10

        fig, axs = plt.subplots(r, c)
        for i in range(10):
            noise = np.random.normal(0, 1, (10, 1000))
            predict_img,predict_seq = self.generator.predict(noise)
            #predictions = np.reshape(predictions,(10,50,2))
            for j in range(10):
                axs[j,i].imshow(predict_img[j,:,:,0], cmap='gray')
                axs[j,i].axis('off')
                predict = self.scaler.inverse_transform(predict_seq[j])
                predict = predict.astype(int)
                dataframe = pd.DataFrame(predict, columns= ['x','y'])
                dataframe.to_csv("./model/test_{}".format(i*10+j), index=False)
        fig.savefig("image/%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    config = tf.ConfigProto()

    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    wgan = WGAN()
    wgan.train(epochs=1000, batch_size=10, sample_interval=50)
