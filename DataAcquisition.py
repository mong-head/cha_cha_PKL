'''
 **Data Acquisition**
 
Acquisition of empty and ocupied data instances,as well as the datset's ground truth. 
The images are attached to a numpy multidimensional array. 
Both classes are concatenated into a single array for inputs, and a single array for targets,
and subsequently stored in a pickle file to be used later if necessary.
'''

class DataAcquisition():
    #Acquire busy class
    def busy_acquisition(image_width, image_height, channels):
        dir_path = file_path+'Dataset/Training/Training Set/Occupied/'
        X_busy = np.ndarray(shape=(len(os.listdir(dir_path)), image_height, image_width, channels), dtype=np.float32)
        for filename in os.listdir(dir_path):
            image_path = dir_path+filename
            print(str(len(os.listdir(dir_path))-(os.listdir(dir_path).index(filename)+1))+' files left in busy class')
            img = cv2.imread(image_path)
            img = cv2.resize(img, (150, 150))
            X_busy[os.listdir(dir_path).index(filename)] = img
        print(X_busy.shape)
        y_busy = np.ones((len(X_busy),1))
        return X_busy, y_busy

    #Acquire free class
    def free_acquisition(image_width, image_height, channels):
        dir_path = file_path+'Dataset/Training/Training Set/Empty/'
        X_free = np.ndarray(shape=(len(os.listdir(dir_path)), image_height, image_width, channels), dtype=np.float32)
        print('X_free shape:'+str(X_free.shape))
        for filename in os.listdir(dir_path):
            image_path = dir_path+filename
            print(str(len(os.listdir(dir_path))-(os.listdir(dir_path).index(filename)+1))+' files left in free class')
            img = cv2.imread(image_path)
            img = cv2.resize(img, (150, 150))
            X_free[os.listdir(dir_path).index(filename)] = img
        print(os.listdir(dir_path).index)
        y_free = np.zeros((len(X_free),1))
        return X_free, y_free

    #Concatenate the dataset
    def concatenate_dataset(X_busy, X_free, y_busy, y_free):
        length = len(X_busy) + len(X_free)
        print(length)
        X_raw = np.ndarray(shape=(length, image_height, image_width, channels), dtype=np.float32)
        for i in range(len(X_busy)):
            X_raw[i] = X_busy[i]
        for x in range(len(X_free)):
            X_raw[i+x+1] = X_free[x]
        y_raw = np.append(y_busy, y_free)
        return X_raw, y_raw

    #Save dataset to file
    def save_dataset(X_raw, y_raw, file_path, X_busy, X_free, y_busy, y_free):
        np.savez((file_path+'Final-Project/Car Dataset v2.npz'), inputs_busy=X_busy, targets_busy=y_busy, inputs_free=X_free, targets_free=y_free)
        print('Saved!')

    #Load dataset from file
    def load_dataset(file_path):
        dataset = np.load((file_path+'Final-Project/Car Dataset v2.npz'))
        X_raw = dataset['inputs']
        y_raw = dataset['targets']
        print('Loaded!')
        return X_raw, y_raw