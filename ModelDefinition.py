'''
**CNN Architechture Definition**

VGG19 is used as a base model for transfer learning, and additional layers are added,
non-linear activation functions "selu" and "relu" are used for these layers,
and the final layer uses "softmax" activation. 
Since the task involves binary classification,
categorical crossentropy is used as the loss function. 
Hyperparameters of stochastic gradient descent, 
the optimizer selected for this architecture, include learning rate, 
learning rate decay, momentum, and dropout. 
A list of callbacks is defined to monitor and store logs of training, 
as well as saving the best overall model every epoch.
Class weights are calculated to ensure that the dataset will not skew training
through class imbalancement. Finally, the model is fitted.
'''
class ModelDefinition():

    def create_model(optimizer=keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True), dropout=0.1, init='uniform'):
      #Parameters
      loss_list = "categorical_crossentropy"
      test_metrics = 'accuracy'
      model_input = keras.Input(shape=(150, 150, 3))

      #Base model
      base_model = VGG19(weights='imagenet', include_top=False)
      for layer in base_model.layers[:]:
          layer.trainable = False
      x = base_model(model_input)
      x = keras.layers.GlobalAveragePooling2D()(x)
      x = keras.layers.Dense(150, activation="selu", kernel_initializer=init)(x)
      x = keras.layers.Dropout(dropout)(x)
      x = keras.layers.Dense(150, activation="selu")(x)
      x = keras.layers.Dropout(dropout)(x)

      #Output net
      y1 = keras.layers.Dense(128, activation='relu')(x)
      y1 = keras.layers.Dropout(dropout)(y1)
      y1 = keras.layers.Dense(64, activation='relu')(y1)
      y1 = keras.layers.Dropout(dropout)(y1)
      y1 = keras.layers.Dense(16, activation='relu')(y1)
      y1 = keras.layers.Dropout(dropout)(y1)

      #Net connections to output layer
      y1 = keras.layers.Dense(2, activation='softmax')(y1)

      #Model compilation
      model = keras.models.Model(inputs=model_input, outputs=y1)
      model.compile(loss=loss_list,
                    optimizer=optimizer,
                    metrics=['accuracy'])
      return model

    #Model checkpoint for most accurate model selection
    def define_callbacks(model_path, logdir):
        checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
        callbacks_list = [checkpoint, tensorboard_callback]
        return callbacks_list

    #Compute class weights
    def class_weights(y_train, y_valid):
        class_weight_list = compute_class_weight('balanced', np.unique(y_train), y_train)
        classWeight = dict(zip(np.unique(y_train), class_weight_list))
        y_train=keras.utils.to_categorical(y_train, 2)
        y_valid=keras.utils.to_categorical(y_valid, 2) 
        return classWeight, y_train, y_valid

    #Model fit training
    def fit_train(X_train, X_valid, y_train, y_valid):
        history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_valid, y_valid), 
                    callbacks=callbacks_list, class_weight=classWeight)
        return history
