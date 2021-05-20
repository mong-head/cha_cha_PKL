'''
 ** Data Preprocessing **
At the acquisition part, I already did the gray scaling. So, in this part, I just have to do adaptive threshold. If you want to add more processing, then you can add more function in the Data_Preprocessing class.

Adaptive_Threshold part : At the acquisition part, I saved the all data to have float datatype. But, this function need a uint8 datatype, so I convert each of them.
'''

class Data_Preprocessing():
  def Adaptive_Threshold(X,maxval=255,thresh=0,k=5,C=5):
    X_th = np.ndarray(shape=(len(X), image_height, image_width), dtype=np.uint8)
    i=0
    for img in tqdm(X):
      img = img.astype('uint8')
      th = cv2.adaptiveThreshold(img, maxval, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, k, C)
      X_th[i] = th
      i += 1

    return X_th

   def resize(X,height,width,dtype=np.uint8):
    X_resized = np.ndarray(shape=(len(X), height, width), dtype=dtype)
    i=0
    for img in tqdm(X):
      if dtype==np.uint8:
        img = img.astype('uint8')
      resized_img = cv2.resize(img, (height,width))
      X_resized[i] = resized_img
      i += 1
    return X_resized

# use function

def data_AcquireSave(train_dir=file_path+'Dataset/Training/Training Set'):
  X_def, y_def = Data_Acquisition.def_acquisition(train_dir)
  X_ok, y_ok = Data_Acquisition.ok_acquisition(train_dir)
  X_raw, y_raw = Data_Acquisition.concatenate_dataset(X_ok, X_def, y_ok, y_def)
  Data_Acquisition.save_dataset(X_raw, y_raw, file_path, X_ok, X_def, y_ok, y_def)

def data_AcquireSave_rgb(train_dir=file_path+'Dataset/Training/Training Set'):
  X_def, y_def = Data_Acquisition_rgb.def_acquisition(train_dir)
  X_ok, y_ok = Data_Acquisition_rgb.ok_acquisition(train_dir)
  X_raw, y_raw = Data_Acquisition_rgb.concatenate_dataset(X_ok, X_def, y_ok, y_def)
  Data_Acquisition_rgb.save_dataset(X_raw, y_raw, file_path, X_ok, X_def, y_ok, y_def)

def data_LoadPreprocess(file_path=file_path+'Dataset/Training/Training Set',width=150,height=150):
  X_raw, y_raw, X_ok, y_ok, X_def, y_def = Data_Acquisition.load_dataset(file_path)
  X_raw = Data_Preprocessing.Adaptive_Threshold(X_raw)
  X_raw = Data_Preprocessing.resize(X_raw,height=150,width=150,dtype=np.uint8) # if rgb : dtype=np.float32

  return X_raw, y_raw, X_ok, y_ok, X_def, y_def