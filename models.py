# M0 - CNN
class M0():
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

    cnn = conv_net()
    cnn.summary()

    # log_dir="logs/"
    # CLASSIFIER = 'cnn'
    # os.makedirs(log_dir, exist_ok=True)
    # summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + ' Classifier={}__'.format(CLASSIFIER) + datetime.datetime.now().strftime("%Y-%m-%d")+"/train/")
    # val_summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + ' Classifier={}__'.format(CLASSIFIER) + datetime.datetime.now().strftime("%Y-%m-%d")+"/validation/")


    # loss_object = tf.keras.losses.MeanSquaredError()
    # optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


    # checkpoint_dir = 'checkpoints_{}'.format(CLASSIFIER)
    # checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    # print("checkpoint_prefix", checkpoint_prefix)
    # os.makedirs(checkpoint_prefix, exist_ok=True)
    # checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=cnn)

    # @tf.function
    # def train_step(x_input, y_input, trainable = True):
    #     with tf.GradientTape(persistent=False) as tape:
    #         prediction = cnn(x_input, training =trainable)
    #         loss = loss_object(y_input, prediction)
    #     gradients = tape.gradient(loss, cnn.trainable_weights)
    #     optimizer.apply_gradients(zip(gradients, cnn.trainable_weights))
    #     return loss, prediction



    # def train(epochs=50):
    #     pbar = tqdm(total =epochs, desc="total_epochs")
    #     for epoch in range(epochs):

    #         step = 0
    #         pbar_steps = tqdm(total=len(Y_train), desc="total_steps")
    #         for (x, y) in zip(X_train, Y_train):
    #             x, y = tf.convert_to_tensor(x), tf.convert_to_tensor(y)
    #             x = tf.reshape(x, (1, 30490,100,2))
    #             y = tf.expand_dims(y, axis = 0)
    #             train_loss, prediction = train_step(x,y)
    #             if step%100==0:
    #                 total_step = 1713*epoch+step
    #                 with summary_writer.as_default():
    #                     tf.summary.scalar('loss', np.sum(train_loss.numpy()), step=total_step)
    #                     tf.summary.histogram('predicted', prediction.numpy(), step=total_step)
    #             step += 1
    #             pbar_steps.update(1)
    #         pbar_steps.close()
    #         checkpoint.save(file_prefix = checkpoint_prefix)    


    #         val_loss = []
    #         pbar_val = tqdm(total = len(Y_val), desc="val_steps")
    #         val_step = 0

    #         for (x,y) in zip(X_val, Y_val):
    #             x, y = tf.convert_to_tensor(x), tf.convert_to_tensor(y)
    #             x = tf.reshape(x, (1, 30490,100,2))
    #             y = tf.expand_dims(y, axis = 0)
    #             loss, _ = train_step(x,y, False)
    #             val_loss.append(loss.numpy())
    #             val_step+=1
    #             pbar_val.update(1)
    #         pbar_val.close()

    #         with val_summary_writer.as_default():
    #             tf.summary.scalar('val_loss', np.mean(val_loss), step=epoch) 
            
    #         pbar.update(1)


    # train(100)

#M1 - CNN+Cat+Date
class M1():
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
    x_out = Flatten()(x) #x should be changed for cnn
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
        x_in = Input(shape=(30490,100,2))
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
        x = Dense(1000)(x)
        x_out = Dense(30490)(x)
        return tf.keras.Model([x_in,x_cat,x_date],[x_out])

    final_model = full_model()
    final_model.summary()




    log_dir="logs/"
    CLASSIFIER = 'CNN+cat+date'
    os.makedirs(log_dir, exist_ok=True)
    summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + ' Classifier={}__'.format(CLASSIFIER) + datetime.datetime.now().strftime("%Y-%m-%d")+"/train/")
    val_summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + ' Classifier={}__'.format(CLASSIFIER) + datetime.datetime.now().strftime("%Y-%m-%d")+"/validation/")


    loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


    checkpoint_dir = 'checkpoints_{}'.format(CLASSIFIER)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    print("checkpoint_prefix", checkpoint_prefix)
    os.makedirs(checkpoint_prefix, exist_ok=True)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=final_model)

    @tf.function
    def train_step(x_input,x_cat, x_date, y_input,  trainable = True):
        with tf.GradientTape(persistent=False) as tape:
            prediction = final_model([x_input, x_cat, x_date], training =trainable)
            loss = loss_object(y_input, prediction)
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
            for (x, y) in zip(X_train, Y_train):

                x, y = tf.convert_to_tensor(x), tf.convert_to_tensor(y)
                x = tf.reshape(x, (1, 30490,100,2))
                y = tf.expand_dims(y, axis = 0)
                x_date100 =x_date[:,step:step+100,:]
                train_loss, prediction = train_step(x, x_cat,x_date100, y )
                if step%100==0:
                    total_step = 1713*epoch+step
                    with summary_writer.as_default():
                        tf.summary.scalar('loss', np.sum(train_loss.numpy()), step=total_step)
                        tf.summary.histogram('predicted', prediction.numpy(), step=total_step)
                step += 1
                pbar_steps.update(1)
            pbar_steps.close()
            checkpoint.save(file_prefix = checkpoint_prefix)    


            val_loss = []
            pbar_val = tqdm(total = len(Y_val), desc="val_steps")
            val_step = 0

            for (x,y) in zip(X_val, Y_val):
                x, y = tf.convert_to_tensor(x), tf.convert_to_tensor(y)
                x = tf.reshape(x, (1, 30490,100,2))
                y = tf.expand_dims(y, axis = 0)
                x_date100 =x_date[:,val_step+1713:val_step+1813,:]
                loss, _ = train_step(x, x_cat,x_date100, y, False)
                val_loss.append(loss.numpy())
                val_step+=1
                pbar_val.update(1)
            pbar_val.close()

            with val_summary_writer.as_default():
                tf.summary.scalar('val_loss', np.mean(val_loss), step=epoch) 
            
            pbar.update(1)


    train(100)

class pca_x1():
    X_train = np.load(pca_root+"/X_list_train_values.npy")
    Y_train = np.load(pca_root+"/Y_list_train_values.npy")
    X_val = np.load(pca_root+"/X_list_validation_values.npy")
    Y_val = np.load(pca_root+"/Y_list_validation_values.npy") 


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


class Scaled_Dense():

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

        x_out = Flatten()(x) #x should be changed for cnn
        # x = Lambda(lambda x: tf.reshape(x, (1,5,128)))(x)
        # x = LSTM(100, activation='relu', return_sequences=True)(x)
        # x_out = LSTM(100, activation='relu')(x)
    #    x_out = Dense(30490, activation='relu')(x)
        return tf.keras.Model([x_in],[x_out])

    # cnn = conv_net()
    # cnn.summary()

    def emb_net():
        x_cat = Input(shape=(30490,15))
        x = Conv1D(32, 100, strides=100, padding='same', activation='relu')(x_cat)
        #x = MaxPooling1D(2,padding='same')(x)
        x = Flatten()(x)
        # x = Conv1D(64, 100, strides=7, padding='same', activation='relu')(x)
        # x = MaxPooling1D(2,padding='same')(x)
        # x = Conv1D(128, 100, strides=7, padding='same', activation='relu')(x)
        # x = MaxPooling1D(2,padding='same')(x)
        x = Dense(365, activation='relu' )(x)     
        x_out = Dense(320, activation='relu')(x)
        return tf.keras.Model([x_cat],[x_out])
    def date_net():
        x_date = Input(shape=(100,61))
        x = Conv1D(32,7,strides=1, padding='same', activation='relu')(x_date)
        #x = MaxPooling1D(2,padding='same')(x)
        x = Flatten()(x)
        
        # x = Conv1D(32,7,strides=1, padding='same', activation='relu')(x_date)
        # x = MaxPooling1D(2,padding='same')(x)
        # x = Conv1D(64,7,strides=1, padding='same', activation='relu')(x)
        # x = MaxPooling1D(2,padding='same')(x)
        #x = Flatten()(x)
        x = Dense(365, activation='relu' )(x)
        x_out = Dense(320, activation='relu')(x)
        return tf.keras.Model([x_date],[x_out])

    def full_model():
        x_in = Input(shape=(30490,100,2))
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
        x = Lambda(lambda x: tf.reshape(x, (1,4,320)))(x)
        x = LSTM(365, activation='relu', return_sequences=True)(x)
        x = LSTM(365, activation='relu')(x)
        x = Dense(1000)(x)
        x_out = Dense(30490, activation='relu')(x)
        return tf.keras.Model([x_in,x_cat,x_date],[x_out])

    final_model = full_model()
    final_model.summary()

class PCA_Dense():
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
        x = Conv1D(32, 100, strides=100, padding='same', activation='relu')(x_cat)
        x = Flatten()(x)
        x = Dense(365, activation='relu' )(x)     
        x_out = Dense(1000, activation='relu')(x)
        return tf.keras.Model([x_cat],[x_out])
    def date_net():
        x_date = Input(shape=(100,61))
        x = Conv1D(32,7,strides=1, padding='same', activation='relu')(x_date)
        x = Flatten()(x)
        x = Dense(365, activation='relu' )(x)
        x_out = Dense(1000, activation='relu')(x)
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
        x = Lambda(lambda x: tf.reshape(x, (1,4,532)))(x)#185 pca2, 192, pca1, pca3 382
        x = LSTM(3650, activation='relu', return_sequences=True)(x)
        x = LSTM(365, activation='relu')(x)
        x = Dense(1000)(x)
        x_out = Dense(30490)(x)
        return tf.keras.Model([x_in,x_cat,x_date],[x_out])


class PCA_Dense_11_01_tardf():
    def conv_net():
    x_in = Input(shape=(SAMPLE_SIZE,100,11))
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
        x = Conv1D(32, 100, strides=100, padding='same', activation='relu')(x_cat)
        x = Flatten()(x)
        #x = Dense(365, activation='relu' )(x)     
        x_out = Dense(1000)(x)
        return tf.keras.Model([x_cat],[x_out])
    def date_net():
        x_date = Input(shape=(100,61))
        x = Conv1D(32,7,strides=1, padding='same', activation='relu')(x_date)
        x = Flatten()(x)
        #x = Dense(365, activation='relu' )(x)
        x_out = Dense(1000,)(x)
        return tf.keras.Model([x_date],[x_out])

    def full_model():
        x_in = Input(shape=(SAMPLE_SIZE,100,11))
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
        #x = Lambda(lambda x: tf.reshape(x, (1,4,532)))(x)#185 pca2, 192, pca1, pca3 382
        # x = LSTM(3650, activation='relu')(x)
        # x = LSTM(365, activation='relu')(x)
        x = Dense(1000, activation='relu')(x)
        x_out = Dense(30490, activation='relu')(x)
        return tf.keras.Model([x_in,x_cat,x_date],[x_out])


    final_model = full_model()
    final_model.summary()


class working_PCa():

    def conv_net():
        x_in = Input(shape=(SAMPLE_SIZE,100,11))
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
        x = Conv1D(32, 100, strides=100, padding='same', activation='relu')(x_cat)
        x = Flatten()(x)
        #x = Dense(365, activation='relu' )(x)     
        x_out = Dense(1000)(x)
        return tf.keras.Model([x_cat],[x_out])
    def date_net():
        x_date = Input(shape=(100,61))
        x = Conv1D(32,7,strides=1, padding='same', activation='relu')(x_date)
        x = Flatten()(x)
        #x = Dense(365, activation='relu' )(x)
        x_out = Dense(1000)(x)
        return tf.keras.Model([x_date],[x_out])

    def full_model():
        x_in = Input(shape=(SAMPLE_SIZE,100,11))
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
        # x = Lambda(lambda x: tf.reshape(x, (1,4,532)))(x)#185 pca2, 192, pca1, pca3 382
        # x = LSTM(3650, return_sequences=True)(x)
        # x = LSTM(365)(x)
        x = Dense(1000)(x)
        x_out = Dense(30490)(x)
        return tf.keras.Model([x_in,x_cat,x_date],[x_out])


    final_model = full_model()
    final_model.summary()

    class CNN12():
        def conv_net():
        x_in = Input(shape=(30490,100,12))
        x = Conv2D(32,(7,7),strides=(3,1), padding='same', activation=ACTIVATION)(x_in)
        x = MaxPooling2D((3,3),padding='same')(x)
        x = Conv2D(64,(7,7),strides=(3,1), padding='same', activation=ACTIVATION)(x)
        x = MaxPooling2D((3,3),padding='same')(x)
        x = Conv2D(128,(7,7),strides=(3,1), padding='same', activation=ACTIVATION)(x)
        x = MaxPooling2D((3,3),padding='same')(x)
        x = Conv2D(64,(7,7),strides=(3,1), padding='same', activation=ACTIVATION)(x)
        x = MaxPooling2D((3,3),padding='same')(x)
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
        x_out = Dense(1)(x)
        return tf.keras.Model([x_in,x_cat,x_date],[x_out])

    final_model = full_model()
    final_model.summary()

