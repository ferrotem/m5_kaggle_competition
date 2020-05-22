#%%
import os
GPU = "0"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Concatenate


matrix_root = "./matrixes"


#%%
train_values = np.load(matrix_root+"/train_values.npy")
price_values = np.load(matrix_root+"/price_values.npy")
date_values = np.load(matrix_root+"/date_values.npy")
categorical_values = np.load(matrix_root+"/categorical_values.npy")

#%%
X_train = []
Y_train = []
for i in range(1713):
    X = []
    X.append(train_values[:,i:i+100])
    X.append(price_values[:,i:i+100])
    X_train.append(X)
    Y_train.append(train_values[:,i+100])

X_val = []
Y_val = []
for i in range(1713,1813):
    X = []
    X.append(train_values[:,i:i+100])
    X.append(price_values[:,i:i+100])
    X_val.append(X)
    Y_val.append(train_values[:,i+100])
#%%
# X_train, Y_train = np.array(X_train).reshape((30490,100,2)), np.array(Y_train)
# X_val, Y_val = np.array(X_val).reshape((30490,100,2)), np.array(Y_val)

#%%
def conv_net():
    x_in = Input(shape=(30490,100,2))
    x = Conv2D(32,(7,7),strides=(3,1), padding='same', activation='relu')(x_in)
    x = MaxPooling2D((3,3),padding='same')(x)
    x = Conv2D(64,(7,7),strides=(3,1), padding='same', activation='relu')(x)
    x = MaxPooling2D((3,3),padding='same')(x)
    x = Conv2D(128,(7,7),strides=(3,1), padding='same', activation='relu')(x)
    x = MaxPooling2D((3,3),padding='same')(x)
    x = Conv2D(64,(7,7),strides=(3,1), padding='same', activation='relu')(x)
    x = MaxPooling2D((3,3),padding='same')(x)
    x = Flatten()(x)
    #x = Dense(30490*2, activation='relu')(x)
    #x_out = Dense(30490)(x)
    return tf.keras.Model([x_in],[x_out])
#%%
cnn = conv_net()
cnn.summary()
#%%
def emb_net():
    x_cat = Input(shape=(30490,20))
    x = Conv2D(32,(7,7),strides=(3,3), padding='same', activation='relu')(x_in)
    x = MaxPooling2D((2,2),padding='same')(x)
    x = Conv2D(64,(7,7),strides=(3,3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2,2),padding='same')(x)
    x = Flatten()(x)
    x_out = Dense(320)(x)
    return tf.keras.Model([x_cat],[x_out])
def date_net():
    x_date = Input(shape=(100,65))
    x = Conv2D(32,(7,7),strides=(3,3), padding='same', activation='relu')(x_in)
    x = MaxPooling2D((2,2),padding='same')(x)
    x = Conv2D(64,(7,7),strides=(3,3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2,2),padding='same')(x)
    x = Flatten()(x)
    x_out = Dense(320)(x)
    return tf.keras.Model([x_date],[x_out])

def full_model():
    x_in = Input(shape=(30490,100,2))
    x_cat = Input(shape=(30490,20))
    x_date = Input(shape=(100,65))

    cnn, emb_nn, date_nn = conv_net(), emb_net(), date_net()
    cnn_out = cnn(x_in)
    emb_out = emb_nn(x_cat)
    date_out = date_nn(x_date)
    
    feat = Concatenate()([emb_out,date_out])
    x =Concatenate()([cnn_out,feat])
    x_out = Dense(39490)(x)
    return tf.keras.Model([x_in,x_cat,x_date],[x_out])



#%%

log_dir="logs/"
CLASSIFIER = 'cnn'
os.makedirs(log_dir, exist_ok=True)
summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + ' Classifier={}__'.format(CLASSIFIER) + datetime.datetime.now().strftime("%Y-%m-%d")+"/train/")
val_summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + ' Classifier={}__'.format(CLASSIFIER) + datetime.datetime.now().strftime("%Y-%m-%d")+"/validation/")


loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


checkpoint_dir = 'checkpoints_{}'.format(CLASSIFIER)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
print("checkpoint_prefix", checkpoint_prefix)
os.makedirs(checkpoint_prefix, exist_ok=True)

@tf.function
def train_step(x_input, y_input, trainable = True):
    with tf.GradientTape(persistent=False) as tape:
        prediction = cnn(x_input, training =trainable)
        loss = loss_object(y_input, prediction)
    gradients = tape.gradient(loss, cnn.trainable_weights)
    optimizer.apply_gradients(zip(gradients, cnn.trainable_weights))
    return loss



def train(epochs=50):
    pbar = tqdm(total =epochs, desc="total_epochs")
    for epoch in range(epochs):

        step = 0
        pbar_steps = tqdm(total=len(Y_train), desc="total_steps")
        for (x, y) in zip(X_train, Y_train):
            x, y = tf.convert_to_tensor(x), tf.convert_to_tensor(y)
            x = tf.reshape(x, (1, 30490,100,2))
            y = tf.expand_dims(y, axis = 0)
            train_loss = train_step(x,y)
            if step%100==0:
                total_step = 1713*epoch+step
                with summary_writer.as_default():
                    tf.summary.scalar('loss', np.sum(train_loss.numpy()), step=total_step)
            step += 1
            pbar_steps.update(1)
        pbar_steps.close()


        val_loss = []
        pbar_val = tqdm(total = len(Y_val), desc="val_steps")
        val_step = 0

        for (x,y) in zip(X_val, Y_val):
            x, y = tf.convert_to_tensor(x), tf.convert_to_tensor(y)
            x = tf.reshape(x, (1, 30490,100,2))
            y = tf.expand_dims(y, axis = 0)
            loss = train_step(x,y, False)
            val_loss.append(loss.numpy())
            val_step+=1
            pbar_val.update(1)
        pbar_val.close()

        with val_summary_writer.as_default():
            tf.summary.scalar('val_loss', np.mean(val_loss), step=epoch) 
        
        pbar.update(1)


train()

    