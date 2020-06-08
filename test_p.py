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
from sklearn.decomposition import PCA
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Concatenate, Conv1D, MaxPooling1D, LSTM,Lambda

CLASSIFIER = "PCAx1_GPU_2"#'Scaled+CNN+CD' +"_GPU_"+GPU #CNN+cat+date
SAMPLE_SIZE = 100

matrix_root = "./matrixes"


#train_values = np.load(matrix_root+"/train_values.npy")
#price_values = np.load(matrix_root+"/price_values.npy")
scaled_train = np.load(matrix_root+"/scaled_train_values.npy")
scaled_price = np.load(matrix_root+"/scaled_price_values.npy")
date_values = np.load(matrix_root+"/date_values.npy")
categorical_values = np.load(matrix_root+"/categorical_values.npy")
max_train_values = np.load(matrix_root + "/max_train_values.npy")




start = 1813



def conv_net():
    x_in = Input(shape=(SAMPLE_SIZE,100,2))
    x = Conv2D(32,(7,7),strides=(3,1), padding='same', activation='relu')(x_in)
    x = MaxPooling2D((3,3),padding='same')(x)
    x = Conv2D(64,(7,7),strides=(3,1), padding='same', activation='relu')(x)
    x = MaxPooling2D((3,3),padding='same')(x)
    x = Conv2D(128,(7,7),strides=(3,1), padding='same', activation='relu')(x)
    x = MaxPooling2D((3,3),padding='same')(x)
    x = Conv2D(64,(7,7),strides=(3,1), padding='same', activation='relu')(x)
    x = MaxPooling2D((3,3),padding='same')(x)

    x_out = Flatten()(x) #x should be changed for cnn
    # x = Lambda(lambda x: tf.reshape(x, (1,1,128)))(x)
    # x = LSTM(100, activation='relu', return_sequences=True)(x)
    # x_out = LSTM(100, activation='relu')(x)
#    x_out = Dense(30490, activation='relu')(x)
    return tf.keras.Model([x_in],[x_out])

# cnn = conv_net()
# cnn.summary()

def emb_net():
    x_cat = Input(shape=(30490,15))
    x = Conv1D(32, 100, strides=7, padding='same', activation='relu')(x_cat)
    x = MaxPooling1D(2,padding='same')(x)
    x = Conv1D(64, 100, strides=7, padding='same', activation='relu')(x)
    x = MaxPooling1D(2,padding='same')(x)
    x = Conv1D(128, 100, strides=7, padding='same', activation='relu')(x)
    x = MaxPooling1D(2,padding='same')(x)
    x = Flatten()(x) 
    x_out = Dense(320)(x)
    return tf.keras.Model([x_cat],[x_out])
def date_net():
    x_date = Input(shape=(100,61))
    x = Conv1D(32,7,strides=1, padding='same', activation='relu')(x_date)
    x = MaxPooling1D(2,padding='same')(x)
    x = Conv1D(64,7,strides=1, padding='same', activation='relu')(x)
    x = MaxPooling1D(2,padding='same')(x)
    x = Flatten()(x)
    x_out = Dense(320)(x)
    return tf.keras.Model([x_date],[x_out])

def full_model():
    x_in = Input(shape=(SAMPLE_SIZE,100,2))
    x_cat = Input(shape=(30490,15))
    x_date = Input(shape=(100,61))

    cnn, emb_nn, date_nn = conv_net(), emb_net(), date_net()
    cnn.summary()
    emb_nn.summary()
    date_nn.summary()
    cnn_out = cnn(x_in)
    emb_out = emb_nn(x_cat)
    date_out = date_nn(x_date)
    
    feat = Concatenate()([emb_out,date_out])
    x = Concatenate()([cnn_out,feat])
    x = Lambda(lambda x: tf.reshape(x, (1,4,192)))(x)#185 pca2, 192, pca1
    x = LSTM(365, activation='relu', return_sequences=True)(x)
    x = LSTM(365, activation='relu')(x)
    x = Dense(1000)(x)
    x_out = Dense(30490, activation='relu')(x)
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
def test_step(x_input,x_cat, x_date, trainable = False):#,x_cat, x_date
    prediction = final_model([x_input, x_cat, x_date], training =trainable)#([x_input, x_cat, x_date], training =trainable)
    return prediction

def pca_calculation(x,i):
    pca = PCA(n_components=100, svd_solver='full')
    pca.fit(x[:,i:i+100].T)
    S_train = pca.transform(x[:,i:i+100].T)
    return S_train.T

def difference(data, interval, start, end):
    return [data[i] - data[i - interval] for i in range( start, end)]

def mean(data, interval, ):
    return [np.mean(data[i-interval:i,:], axis = 0) for i in range(start, end)]
list_inter = [1,7,14,28,365]

def test(epochs=56):
    pbar = tqdm(total =epochs, desc="total_epochs")
    x_cat, x_date =tf.convert_to_tensor(categorical_values), tf.convert_to_tensor(date_values)
    x_cat, x_date = tf.reshape(x_cat, (1, 30490,15)), tf.reshape(x_date, (1, 1969,61))
    X_test_t = scaled_train[:,start:start+100]
    X_test_p = scaled_price[:,start:start+100]
    x1 = pca_cal(X_test_t)
    x2 = pca_cal(X_test_p)
    x = tf.convert_to_tensor(np.array([x1,x2]))
    x = tf.reshape(x, (1, 100,100,2)) 
    output = []


    for epoch in range(epochs):   


        x_date100 =x_date[:,1813+epoch:1913+epoch,:]
        prediction =test_step(x, x_cat,x_date100)
        X_test_t[:,:99]= X_test_t[:,1:100]
        X_test_t[:,99] =prediction.numpy()+X_test_t[:,99]
        X_test_p = scaled_price[:,start+epoch:start+epoch+100]
        
        for inter in list_inter():
            data = difference(Xs)

        print("shape: ", prediction.shape)

        x1, x2 = pca_cal(X_test_t), pca_cal(X_test_p)
        x = tf.convert_to_tensor(np.array([x1,x2]))
        x = tf.reshape(x, (1, 100,100,2))
        
        output.append(prediction.numpy())
        pbar.update(1)
    final_np = np.array(output)
    np.save("./final_pred_{}.npy".format(CLASSIFIER), final_np)

test()

#%%%
CLASSIFIER = "PCAx1_GPU_2"
import numpy as np
import pandas as pd
matrix_root = "./matrixes"
max_train_values = np.load(matrix_root + "/max_train_values.npy")
output = np.load("./final_pred_{}.npy".format(CLASSIFIER))
output.shape
output = output.reshape((56,30490))
sample = pd.read_csv("sample_submission.csv")
n = output[:28,:]*max_train_values.T
x =output[28:,:]* max_train_values.T
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
CLASSIFIER = "PCAx1_GPU_2"#'Scaled+CNN+CD' +"_GPU_"+GPU #CNN+cat+date

matrix_root = "./matrixes"
df = pd.read_csv("boran_{}.csv".format(CLASSIFIER))


#%%
