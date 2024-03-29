import os
GPU ='3'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=GPU
import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Concatenate, Conv1D, MaxPooling1D, LSTM,Lambda

CLASSIFIER = "Grouped_cnn1"#'Scaled+CNN+CD' +"_GPU_"+GPU #CNN+cat+date
matrix_root = "./matrixes"

ACTIVATION = 'relu'
scaled_train = np.load(matrix_root+"/train_values.npy")
scaled_price = np.load(matrix_root+"/scaled_price_values.npy")
date_values = np.load(matrix_root+"/date_values.npy")
categorical_values = np.load(matrix_root+"/categorical_values.npy")
max_train_values = np.load(matrix_root + "/max_train_values.npy")
# total_mean =  np.load(matrix_root+"/total_mean.npy", allow_pickle=True)
# total_diff = np.load(matrix_root+"/shiffted.npy", allow_pickle=True)
# max_train_values = np.load(matrix_root + "/max_train_values.npy")



def conv_net():
    x_in = Input(shape=(30490,100,12))
    x = Conv2D(32,(7,7),strides=(3,1), padding='same', activation=ACTIVATION)(x_in)
    #x = MaxPooling2D((3,3),padding='same')(x)
    x = Conv2D(64,(7,7),strides=(3,1), padding='same', activation=ACTIVATION)(x)
    #x = MaxPooling2D((3,3),padding='same')(x)
    x = Conv2D(128,(7,7),strides=(3,1), padding='same', activation=ACTIVATION)(x)
    x = Conv2D(128,(7,7),strides=(3,1), padding='same', activation=ACTIVATION)(x)
    x = MaxPooling2D((3,3),padding='same')(x)
    x = Conv2D(256,(7,7),strides=(3,1), padding='same', activation=ACTIVATION)(x)
    x = Conv2D(256,(7,7),strides=(3,1), padding='same', activation=ACTIVATION)(x)
    # x = MaxPooling2D((3,3),padding='same')(x)
    x = Conv2D(64,(7,7),strides=(3,1), padding='same', activation=ACTIVATION)(x)
    # x = MaxPooling2D((3,3),padding='same')(x)
    x_out = Flatten()(x) #x should be changed for cnn
    #    x_out = Dense(30490, activation='relu')(x)
    return tf.keras.Model([x_in],[x_out])
# cnn = conv_net()
# cnn.summary()

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
    #x = Dense(1000)(x)
    x_out = Dense(3049)(x)
    return tf.keras.Model([x_in,x_cat,x_date],[x_out])
final_model = full_model()
final_model.summary()
optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = 'checkpoints_{}'.format(CLASSIFIER)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
print("checkpoint_prefix", checkpoint_prefix)
os.makedirs(checkpoint_prefix, exist_ok=True)
latest = tf.train.latest_checkpoint(checkpoint_dir)
print("latest: ", latest)
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=final_model)
checkpoint.restore(latest)


@tf.function
def test_step(x_input,x_cat, x_date,  pres, trainable = False):
    with tf.GradientTape(persistent=False) as tape:
        prediction = final_model([x_input, x_cat, x_date], training =trainable)
        scaled_pred = pres * prediction
        scaled_pred = tf.reshape(scaled_pred, [-1])
    return scaled_pred

def scale_calc(i):
    Z = []
    for j in range(10):
        k_start = 3049*j
        k_finish = 3049*(j+1)
        x = scaled_train[k_start:k_finish,i:i+100]
        x = np.sum(x, axis = 1)
        Z.append(x)
    Z = np.array(Z)
    s_sum =np.sum(Z, axis = 0)
    scaled  = np.divide(Z,s_sum)
    scaled = np.nan_to_num(scaled)
    return scaled

def difference( interval, start, end):
    return [scaled_train[:,i] - scaled_train[:,i - interval] for i in range(start, end)]

def mean( interval, start, end ):
    return [np.mean(scaled_train[:,i-interval:i], axis = 1) for i in range(start, end)]
list_inter = [1,7,14,28,365]
def pca_input(step):
    one_input = []
    X_scaled = scaled_train[:,step:step+100]
    one_input.append(X_scaled)
    X_price = scaled_price[:,step:step+100]
    one_input.append(X_price)
    for i in list_inter:
        one_input.append(np.array(difference(i, step, step+100)).T)
    for i in list_inter:
        one_input.append(np.array(mean(i, step, step+100)).T)

    #pres_train =  np.sum(scaled_train[:,i:i+100], axis=1)/np.sum(scaled_train[:,i:i+100])
    pres_train = scale_calc(i)
    return one_input, pres_train


def test(epochs=56):
    pbar = tqdm(total =epochs, desc="total_epochs")
    x_cat, x_date =tf.convert_to_tensor(categorical_values), tf.convert_to_tensor(date_values)
    x_cat, x_date = tf.reshape(x_cat, (1, 30490,15)), tf.reshape(x_date, (1, 1969,61))
    output = []
    global scaled_train
    for epoch in range(epochs):
        step = 1813+epoch
        X, pres = pca_input(step)
        x = np.reshape(X,(1, 30490,100,12))
        x, pres= tf.convert_to_tensor(x),  tf.convert_to_tensor(pres, tf.float32)  #X_train, Y_train, Pres_train = dataset_generator(1813+epoch,1814+epoch)   
        x_date100 =x_date[:,1813+epoch:1913+epoch,:]

        prediction =test_step(x, x_cat,x_date100, pres) #test_step(x, x_cat,x_date100)

        scaled_output = prediction.numpy() #scaled_train[:,-1]+prediction.numpy()#np.mean(scaled_train[:,-100], axis = 0)+prediction.numpy()
        scaled_output = np.reshape(scaled_output, (30490,1))
        print("scaled shape: ", scaled_output.shape)
        scaled_train = np.concatenate([scaled_train, scaled_output], axis=1)
        print("prediction mean:", np.mean(prediction.numpy()) )

        x = tf.convert_to_tensor(x)
        output.append(scaled_output)
        pbar.update(1)
    final_np = np.array(output)
    np.save("./final_pred_{}.npy".format(CLASSIFIER), final_np)

test()

#%%
output = np.load("./final_pred_{}.npy".format(CLASSIFIER))
output.shape
output = output.reshape((56,30490))
sample = pd.read_csv("sample_submission.csv")
n = output[:28,:]#*max_train_values.T
x =output[28:,:]#* max_train_values.T
y =np.concatenate((n,x), axis=1)
y.shape
y = y.T

ss = pd.DataFrame(y)
ss[ss < 0.1] = 0
ss["id"]=sample["id"]
l = ss.columns.to_list()
l = l[-1:]+l[:-1]
ss = ss[l]
l = sample.columns.to_list()
ss.columns =l
ss.to_csv("boran_{}.csv".format(CLASSIFIER), index =False)







#%%
import numpy as np
import pandas as pd
CLASSIFIER = "Grouped_cnn1"
df = pd.read_csv("boran_{}.csv".format(CLASSIFIER))


#%%
