import os
GPU ='2'
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

CLASSIFIER = "cnn"#'Scaled+CNN+CD' +"_GPU_"+GPU #CNN+cat+date
matrix_root = "./matrixes"


train_values = np.load(matrix_root+"/train_values.npy")
price_values = np.load(matrix_root+"/price_values.npy")
date_values = np.load(matrix_root+"/date_values.npy")
categorical_values = np.load(matrix_root+"/categorical_values.npy")




start = 1813
X_test = np.array([train_values[:,start:start+100], price_values[:,start:start+100]])

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
    x_out = Dense(30490, activation='relu')(x)
    return tf.keras.Model([x_in],[x_out])

final_model = conv_net()
#final_model.summary()

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
def test_step(x_input, trainable = False):#,x_cat, x_date
    prediction = final_model([x_input], training =trainable)#([x_input, x_cat, x_date], training =trainable)
    return prediction


def test(epochs=56):
    pbar = tqdm(total =epochs, desc="total_epochs")
    x_cat, x_date =tf.convert_to_tensor(categorical_values), tf.convert_to_tensor(date_values)
    x_cat, x_date = tf.reshape(x_cat, (1, 30490,15)), tf.reshape(x_date, (1, 1969,61))
    x = tf.convert_to_tensor(X_test)
    x = tf.reshape(x, (1, 30490,100,2)) 
    output = []
    for epoch in range(epochs):   
        x_date100 =x_date[:,1813+epoch:1913+epoch,:]
        prediction =test_step(x) #test_step(x, x_cat,x_date100)
        x = x.numpy()
        x[:,:,0:99,:] = x[:,:,1:100,:]
        x[:,:,99, 0] =prediction.numpy()
        print("shape: ", prediction.shape)
        x[:,:,99, 1] =price_values[:,1913+epoch]
        x = tf.convert_to_tensor(x)
        output.append(prediction.numpy())
        pbar.update(1)
    final_np = np.array(output)
    np.save("./final_pred_{}.npy".format(CLASSIFIER), final_np)

test()


output = np.load("./final_pred_{}.npy".format(CLASSIFIER))
output.shape
output = output.reshape((56,30490))
sample = pd.read_csv("sample_submission.csv")
n = output[:28,:]
x =output[28:,:]
y =np.concatenate((n,x), axis=1)
y.shape
y = y.T

ss = pd.DataFrame(y)
ss[ss < 0] = 0
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

df = pd.read_csv("boran_cnn.csv")


#%%
