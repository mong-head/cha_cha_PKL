'''
**Model Training**

Block that combines multiple classes and methods to generate the dataset, 
define the model, and subsequently train it.
'''
class ModelTraining():
    def do_stuff(image_width, image_height, channels, file_path):
        X_busy, y_busy = DataAcquisition.busy_acquisition(image_width, image_height, channels)
        X_free, y_free = DataAcquisition.free_acquisition(image_width, image_height, channels)
        X_raw, y_raw = DataAcquisition.concatenate_dataset(X_busy, X_free, y_busy, y_free)
        DataAcquisition.save_dataset(X_raw, y_raw, file_path, X_busy, X_free, y_busy, y_free)
        
        X_raw, y_raw = DataAcquisition.load_dataset(file_path)
        X_train, X_valid, y_train, y_valid = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)
        model = ModelDefinition.create_model()
        callbacks_list = ModelDefinition.define_callbacks(model_path, logdir)
        classWeight, y_train, y_valid = ModelDefinition.class_weights(y_train, y_valid)
        history = ModelDefinition.fit_train(X_train, X_valid, y_train, y_valid)