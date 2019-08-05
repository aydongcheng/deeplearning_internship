# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 17:20:31 2019

@author: 3Ptgoals
"""

# coding=utf-8

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape,Flatten
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from keras.layers.convolutional import UpSampling2D, Conv2D,MaxPooling2D,Conv2DTranspose
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

img_height = 28
img_width = 28
batch_size = 64
mnist = input_data.read_data_sets('../data/MNIST_data', one_hot=True)


def img_to_code(img_input):
    if trained_encoder is not None:
        return trained_encoder.predict(img_input)
    else:
        return img_input
    
def noise(code):
    idx = np.around(np.random.uniform(0.,code.shape[1]-1,size=int(code.shape[1]/5))).astype(np.int)
    noise_code = np.copy(code)
    noise_code[:,idx] = 0
    return noise_code

def train_one_by_one():
    epochs = 3000
    delta = 0
    trained_encoder = None
    layers = [28 * 28, 700, 400, 200, 50,10,2]
    decoders = []
    for i in range(len(layers) - 1):
        
        encoder = Sequential()
        if i == len(layers) -2:
            encoder.add(Dense(units=layers[i + 1]))
        else:
            encoder.add(Dense(units=layers[i + 1],activation='sigmoid'))
        decoder = Sequential()
        decoder.add(Dense(units=layers[i], activation='sigmoid'))
        encoder_input = Input(shape=(layers[i],))
        code = encoder(encoder_input)
        reconstruct_code = decoder(code)
        combined = Model(encoder_input, reconstruct_code)
        optimizer = Adam(0.001)
        combined.compile(loss='mse', optimizer=optimizer)

        for j in range(epochs):
            imgs, _ = mnist.train.next_batch(batch_size)
            # 由之前训练好的encoder 将image转成code  如果没有训练过的encoder 那code就是image
            code = imgs if (trained_encoder is None) else trained_encoder.predict(imgs)
            noise_code = noise(code)
            loss = combined.train_on_batch(noise_code, code)
            if j % 50 == 0:
                print("%d layer,loss:%f"%(i,loss))
                
        # 经过上面的for循环 已经又训练好了一层encoder  将encoder与之前的合并
        if trained_encoder is None:
            trained_encoder = encoder
        else:
            img_input = Input(shape=[img_width * img_height])
            input_code = trained_encoder(img_input)
            code = encoder(input_code)
            trained_encoder = Model(img_input,code)
        
        # 保存当前encoder 对应的decoder
        decoders.append(decoder)
        epochs += delta  #每层增加1000次迭代次数
        
    img_input = Input(shape=[img_width * img_height])
    decoders.reverse()
    last_decode = trained_encoder(img_input)
    for decoder in decoders:
        last_decode = decoder(last_decode)
    combined = Model(img_input, last_decode)
    combined.compile(loss='mse', optimizer=optimizer)
    return trained_encoder,combined

def train(combined,epochs):
    losses = []
    for i in range(epochs):
        imgs, labels = mnist.train.next_batch(batch_size)
        noise_imgs = noise(imgs)
        loss = combined.train_on_batch(noise_imgs, imgs)
        if i >= 20 and i % 5 == 0:
            print("epoch:%d,loss:%f" % (i, loss))
            losses.append(loss)
    plt.plot(np.arange(20, epochs, 5), losses)
    plt.show()

picture_num = 4
def test(combined):
    figure_num = 1
    for _ in range(picture_num):
        imgs, labels = mnist.test.next_batch(3)
        noise_imgs = noise(imgs)
        score = combined.evaluate(noise_imgs, imgs, verbose=0)
        print("Test loss:", score)
        output_imgs = combined.predict(noise_imgs)
        for i in range(3):
            plt.figure(figure_num)
            plt.subplot(2, 3, i + 1)  # 两行一列的第一个子图
            plt.imshow(noise_imgs[i].reshape((28, 28)), cmap='gray')
            plt.subplot(2, 3, i + 1 + 3)  # 两行一列的第二个子图
            plt.imshow(output_imgs[i].reshape((28, 28)), cmap='gray')
        figure_num += 1

def plot_low_dimension():
    imgs, labels = mnist.test.next_batch(3000)
    labels = np.transpose(np.nonzero(labels))[:,1]
    codes = encoder.predict(imgs)
    plt.figure(10)
    i = 0
    for color in ['red','blue','green','black','chocolate','silver','lawngreen','darkcyan','lawngreen','c']:
        y = (labels.T == i)
        x = codes[y]
        plt.scatter(x[:,0],x[:,1],c=color,label=str(i),alpha=0.6,edgecolors='white')
        i+=1
    plt.title('Scatter')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='best')
    plt.show()
    
def plot_3d():
    imgs, labels = mnist.test.next_batch(3000)
    labels = np.transpose(np.nonzero(labels))[:,1]
    codes = encoder.predict(imgs)
    plt.figure(4)
    ax = Axes3D(fig)
    i = 0
    for color in ['red','blue','green','black','chocolate','silver','lawngreen','darkcyan','lawngreen','c']:
        y = (labels.T == i)
        x = codes[y]
        plt.scatter(x[:,0],x[:,1],x[:,2],c=color,label=str(i),alpha=0.6,edgecolors='white')
        i+=1

    # 绘制图例
    ax.legend(loc='best')


    # 添加坐标轴(顺序是Z, Y, X)
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})

    # 展示
    plt.show()

if __name__ == '__main__':
    encoder, combined = train_one_by_one()
    train(combined, 6000)
    test(combined)
    
#     plot_low_dimension()

