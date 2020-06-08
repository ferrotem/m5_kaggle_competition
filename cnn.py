
import os
GPU ='1'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=GPU
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Concatenate, Conv1D, MaxPooling1D, LSTM,Lambda,BatchNormalization, Add, Activation, LeakyReLU

CLASSIFIER = "Grouped_cnn_leaky"+GPU#'Scaled+CNN+LSTMx1+Dense' +"_GPU_"+GPU  
matrix_root = "./matrixes"
ACTIVATION = 'relu'

train_values = np.load(matrix_root+"/train_values.npy")
price_values = np.load(matrix_root+"/scaled_price_values.npy")
date_values = np.load(matrix_root+"/date_values.npy")
categorical_values = np.load(matrix_root+"/categorical_values.npy")
total_mean =  np.load(matrix_root+"/or_total_mean.npy", allow_pickle=True)
total_diff = np.load(matrix_root+"/or_total_diff.npy", allow_pickle=True)
max_train_values = np.load(matrix_root + "/max_train_values.npy")

def scale_calc(i):
    Z = []
    for j in range(10):
        k_start = 3049*j
        k_finish = 3049*(j+1)
        x = train_values[k_start:k_finish,i:i+100]
        x = np.sum(x, axis = 1)
        Z.append(x)
    Z = np.array(Z)
    s_sum =np.sum(Z, axis = 0)
    scaled  = np.divide(Z,s_sum)
    scaled = np.nan_to_num(scaled)
    return scaled

def dataset_generator(start, end):
    X_train, Y_train, scaled = [],[],[]
    for i in range(start, end):
        X = []
        X.append(train_values[:,i:i+100])
        X.append(price_values[:,i:i+100])
        for el in total_mean:
            X.append(el.T[:,i:i+100])
        for el2 in total_diff:
            X.append(el2.T[:,i:i+100])
        X_train.append(X)
        Y_train.append(train_values[:,i+100]) #    Y_train.append(total_diff[0].T[:,i+100])
        #Pres_train.append( np.sum(train_values[:,i:i+100], axis=1)/np.sum(train_values[:,i:i+100])) #*max_train_values.T
        scaled.append(scale_calc(i))
    return X_train, Y_train, scaled
    

X_train, Y_train, Scaled_train = dataset_generator(0,1713)
X_val, Y_val, Scaled_val= dataset_generator(1713,1813)

# x = np.reshape(X_train[1448],(1, 30490,100,12))
# print("1448: ",x.shape)
# print("1449: ", X_train[1449].shape)

def conv_net():
    x_in = Input(shape=(30490,100,12))
    x = Conv2D(32,(10,10),strides=(3,1), padding='same')(x_in)
    x = LeakyReLU()(x)
    #x = MaxPooling2D((3,3),padding='same')(x)
    x = Conv2D(64,(10,10),strides=(3,1), padding='same')(x)
    x = LeakyReLU()(x)
    #x = MaxPooling2D((3,3),padding='same')(x)
    x = Conv2D(128,(10,10),strides=(3,1), padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(128,(10,10),strides=(3,1), padding='same')(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((3,3),padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(256,(10,10),strides=(3,1), padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(256,(3,3),strides=(3,1), padding='same')(x)
    x = LeakyReLU()(x)
    # x = MaxPooling2D((3,3),padding='same')(x)
    x = Conv2D(64,(3,3),strides=(3,1), padding='same')(x)
    x = LeakyReLU()(x)
    # x = MaxPooling2D((3,3),padding='same')(x)
    x_out = Flatten()(x) #x should be changed for cnn
    #    x_out = Dense(30490, activation='relu')(x)
    return tf.keras.Model([x_in],[x_out])

# def conv_net():
#     x_in = Input(shape=(30490,100,12))
#     x = Conv2D(32,(7,7),strides=(3,1), padding='same', activation=ACTIVATION)(x_in)
#     x = MaxPooling2D((3,3),padding='same')(x)
#     x = Conv2D(64,(7,7),strides=(3,1), padding='same', activation=ACTIVATION)(x)
#     x = MaxPooling2D((3,3),padding='same')(x)
#     x = Conv2D(128,(7,7),strides=(3,1), padding='same', activation=ACTIVATION)(x)
#     x = MaxPooling2D((3,3),padding='same')(x)
#     x = Conv2D(64,(7,7),strides=(3,1), padding='same', activation=ACTIVATION)(x)
#     x = MaxPooling2D((3,3),padding='same')(x)
#     x_out = Flatten()(x) #x should be changed for cnn
#     #    x_out = Dense(30490, activation='relu')(x)
#     return tf.keras.Model([x_in],[x_out])

cnn = conv_net()
cnn.summary()

def emb_net():
    x_cat = Input(shape=(30490,15))
    x = Conv1D(32, 100, strides=7, padding='same', activation=ACTIVATION)(x_cat)
    x = MaxPooling1D(2,padding='same')(x)
    x = Flatten()(x)
    #x = Dense(365, activation='relu' )(x)
    x_out = Dense(320)(x)
    return tf.keras.Model([x_cat],[x_out])
def date_net():
    x_date = Input(shape=(100,61))
    x = Conv1D(32,7,strides=1, padding='same', activation=ACTIVATION)(x_date)
    x = Flatten()(x)
    #x = Dense(365, activation='relu' )(x)
    x_out = Dense(320)(x)
    return tf.keras.Model([x_date],[x_out])

def full_model():
    x_in = Input(shape=(30490,100,12))
    x_cat = Input(shape=(30490,15))
    x_date = Input(shape=(100,61))

    cnn, emb_nn, date_nn = conv_net(), emb_net(), date_net()
    emb_nn.summary()
    date_nn.summary()
    cnn_out = cnn(x_in)
    emb_out = emb_nn(x_cat)
    date_out = date_nn(x_date)
    
    feat = Concatenate()([emb_out,date_out])
    x = Concatenate()([cnn_out,feat])
    # x = Lambda(lambda x: tf.reshape(x, (1,4,320)))(x)#185 pca2, 192, pca1, pca3 382
    # x = LSTM(365, activation='relu', return_sequences=True)(x)
    # x = LSTM(365, activation='relu')(x)
    x = Dense(1000)(x)
    x_out = Dense(3049, activation=tf.nn.leaky_relu)(x)
    return tf.keras.Model([x_in,x_cat,x_date],[x_out])

final_model = full_model()
final_model.summary()


log_dir="logs/"

os.makedirs(log_dir, exist_ok=True)
summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + ' Classifier={}__'.format(CLASSIFIER) + datetime.datetime.now().strftime("%Y-%m-%d")+"/train/")
val_summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + ' Classifier={}__'.format(CLASSIFIER) + datetime.datetime.now().strftime("%Y-%m-%d")+"/validation/")


loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()


checkpoint_dir = 'checkpoints_{}'.format(CLASSIFIER)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
print("checkpoint_prefix", checkpoint_prefix)
os.makedirs(checkpoint_prefix, exist_ok=True)
#checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=final_model)

latest = tf.train.latest_checkpoint(checkpoint_dir)
print("latest: ", latest)
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=final_model)
checkpoint.restore(latest)

@tf.function
def train_step(x_input,x_cat, x_date, y_input,  pres, trainable = True):
    with tf.GradientTape(persistent=False) as tape:
        prediction = final_model([x_input, x_cat, x_date], training =trainable)
        scaled_pred = pres * prediction
        #tf.print("Before: ", scaled_pred.shape)
        scaled_pred = tf.reshape(scaled_pred, [-1])
        #tf.print("After: ", scaled_pred.shape)

        loss = loss_object(y_input, scaled_pred)
    gradients = tape.gradient(loss, final_model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, final_model.trainable_weights))
    return loss, prediction



def train(epochs=50):
    pbar = tqdm(total =epochs, desc="total_epochs")
    x_cat, x_date =tf.convert_to_tensor(categorical_values), tf.convert_to_tensor(date_values)
    x_cat, x_date = tf.reshape(x_cat, (1, 30490,15)), tf.reshape(x_date, (1, 1969,61))
    for epoch in range(epochs):

        step = 0
        pbar_steps = tqdm(total=len(Y_train), desc="total_steps")
        for (x, y, pres) in zip(X_train, Y_train, Scaled_train):
            x = np.reshape(x,(1, 30490,100,12))

            x, y, pres= tf.convert_to_tensor(x), tf.convert_to_tensor(y), tf.convert_to_tensor(pres, tf.float32)
            #x = tf.reshape(x, (1, 30490,100,12))
            y = tf.expand_dims(y, axis = 0)
            x_date100 =x_date[:,step:step+100,:]
            train_loss, prediction = train_step(x, x_cat,x_date100, y,pres )
            #print("loss: ", train_loss.numpy())
            if step%100==0:
                total_step = 1713*epoch+step
                with summary_writer.as_default():
                    tf.summary.scalar('loss', np.sum(train_loss.numpy()), step=total_step)
                    tf.summary.histogram('predicted', prediction.numpy(), step=total_step)
            step += 1
            pbar_steps.update(1)
        pbar_steps.close()
        if epoch%10==0:
            checkpoint.save(file_prefix = checkpoint_prefix)    


        val_loss = []
        pbar_val = tqdm(total = len(Y_val), desc="val_steps")
        val_step = 0

        for (x,y,pres) in zip(X_val, Y_val, Scaled_val):
            x = np.reshape(x,(1, 30490,100,12))
            x, y, pres= tf.convert_to_tensor(x), tf.convert_to_tensor(y), tf.convert_to_tensor(pres, tf.float32)
            #x = tf.reshape(x, (1, 30490,100,12))
            #y = tf.expand_dims(y, axis = 0)
            x_date100 =x_date[:,val_step+1713:val_step+1813,:]
            loss, _ = train_step(x, x_cat,x_date100, y,pres, False)
            val_loss.append(loss.numpy())
            val_step+=1
            pbar_val.update(1)
        pbar_val.close()

        with val_summary_writer.as_default():
            tf.summary.scalar('val_loss', np.mean(val_loss), step=epoch) 
        
        pbar.update(1)


train(150)

