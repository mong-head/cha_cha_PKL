'''
**Test Image Segmentation**

Reads parking lot space coordinates from a csv file, 
and stores them in variables in order to generate patches in the image for cropping. 
The cropped images correspond to the segmentation
'''
class ParkingLotSegmentation():
    def crop_image(img, x1, y1, x2, y2, x3, y3, x4, y4):
        top_left_x = min([x1,x2,x3,x4])
        top_left_y = min([y1,y2,y3,y4])
        bot_right_x = max([x1,x2,x3,x4])
        bot_right_y = max([y1,y2,y3,y4])
        roi = img[top_left_y:bot_right_y+1, top_left_x:bot_right_x+1]
        return roi

    def define_patches(model, img, ax, ID, x, y, lot_width, lot_height, center_x, center_y, lot_angle, vacancy):
        cont = 0
        for i in range(len(ID)):
            patch_shape = np.ndarray((4,2))
            patch_shape[0,0] = float(x[i+cont])
            patch_shape[0,1] = float(y[i+cont])
            patch_shape[1,0] = float(x[i+cont+1])
            patch_shape[1,1] = float(y[i+cont+1])
            patch_shape[2,0] = float(x[i+cont+2])
            patch_shape[2,1] = float(y[i+cont+2])
            patch_shape[3,0] = float(x[i+cont+3])
            patch_shape[3,1] = float(y[i+cont+3])
            cont = cont + 3
            roi = ParkingLotSegmentation.crop_image(img, int(patch_shape[0,0]), int(patch_shape[0,1]), int(patch_shape[1,0]), int(patch_shape[1,1]), int(patch_shape[2,0]), int(patch_shape[2,1]), int(patch_shape[3,0]), int(patch_shape[3,1]))
            roi = cv2.resize(roi, (150, 150))
            roi_tr = np.ndarray(shape=(1,150,150,3), dtype=np.float32)
            roi_tr = np.expand_dims(roi, axis=0).astype(np.float32)
            prediction = model.predict(roi_tr)
            print('Slot '+str(ID[i])+': '+str(np.argmax(prediction)))
            if np.argmax(prediction) == 0.0:
                rect = patches.Polygon(patch_shape, linewidth=1,edgecolor='g',facecolor='none')
            else:
                rect = patches.Polygon(patch_shape, linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)

    def csv_input(img, ax, file_path, model,camara_csv):
        with open(file_path+camara_csv, newline='') as csvfile:
            lot_csv = reader = csv.DictReader(csvfile)
            #img = cv2.resize(img,(2592,1944))
            free = 0  #add : free space num
    
            for row in lot_csv:
                #crop_img = img[(int(row['Y'])-int(row['H'])/2):(int(row['Y'])-int(row['H'])/2)+int(row['H']), (int(row['X'])-int(row['W'])/2):(int(row['X'])-int(row['W'])/2)+int(row['W'])]      
                #crop_img = img[int((int(row['Y'])-int(row['H'])/2)):int((int(row['Y'])-int(row['H'])/2)+int(row['H'])), int((int(row['X'])-int(row['W'])/2)):int((int(row['X'])-int(row['W'])/2)+int(row['W']))]
                crop_img = img[int(row['Y']):int(row['Y'])+int(row['H']) , int(row['X']):int(row['X'])+int(row['W'])]
                #roi = cv2.resize(roi, (150, 150))
                
                roi = cv2.resize(crop_img, (150, 150))
                roi_tr = np.ndarray(shape=(1,150,150,3), dtype=np.float32)
                roi_tr = np.expand_dims(roi, axis=0).astype(np.float32)
                prediction = model.predict(roi_tr)
                #print(str(np.argmax(prediction)))
                if np.argmax(prediction) == 0.0:
                    ec='g'
                    free = free + 1
                else:
                    ec='r'
                    
                #rect  = patches.Rectangle(((int(row['X'])-int(row['W'])/2), (int(row['Y'])-int(row['H'])/2)), int(row['W']), int(row['H']), edgecolor=ec, facecolor='none') 
                #rect  = patches.Rectangle(int((int(row['X'])/2.592), int((int(row['Y']))/2.592), int(int(row['W'])/2.592), int(int(row['H'])/2.592), edgecolor=ec, facecolor='none'))
                rect  = patches.Rectangle((int(row['X']), int(row['Y'])), int(row['W']), int(row['H']), edgecolor=ec, facecolor='none')
                ax.add_patch(rect)
        return free

'''
 **Testing**

Takes a parking lot image as an input,
segments it using the coordinates from the provided csv file,
and subsequently runs each segment through the classifier's predictor. 
The results are then graphed into the original image, 
representing the classes by green patches, if empty, and red patches, if occupied.
'''

class ModelTesting():
    def predict_vacancy(file_path):
        #Load trained model
        model = load_model(model_path)
        print('done!')

        #Segment parking lot
        parkinglot_img_list = ['CNR-EXT_FULL_IMAGE_1000x750/FULL_IMAGE_1000x750/OVERCAST/2015-11-16/camera8/2015-11-16_1708.jpg',
                               'CNR-EXT_FULL_IMAGE_1000x750/FULL_IMAGE_1000x750/OVERCAST/2015-11-16/camera8/2015-11-16_0722.jpg',
                               'CNR-EXT_FULL_IMAGE_1000x750/FULL_IMAGE_1000x750/RAINY/2016-02-12/camera9/2016-02-12_1742.jpg',
                               'CNR-EXT_FULL_IMAGE_1000x750/FULL_IMAGE_1000x750/SUNNY/2015-11-27/camera1/2015-11-27_1640.jpg',
                               'school_1_2592.jpg','school_2_2592.jpg','school_3_2592.jpg']
        camera_csv_list = ['CNR-EXT_FULL_IMAGE_1000x750/camera8.csv',
                           'CNR-EXT_FULL_IMAGE_1000x750/camera8.csv',
                           'CNR-EXT_FULL_IMAGE_1000x750/camera9.csv',
                           'CNR-EXT_FULL_IMAGE_1000x750/camera1.csv',
                           'school.csv','school.csv','school2.csv']
        for i in range(len(parkinglot_img_list)):
           #Read parking lot image
           img = cv2.imread(file_path+parkinglot_img_list[i])
           fig,ax = plt.subplots(figsize=(10, 10))
           img = cv2.resize(img,(2592,1944)) #plus

           #Generate patches
           #ParkingLotSegmentation.define_patches(model, img, ax, ID, x, y, lot_width, lot_height, center_x, center_y, lot_angle, vacancy)
           free_num = ParkingLotSegmentation.csv_input(img, ax, file_path, model,camera_csv_list[i])
           ax.imshow(img)
           plt.show()
           print('free space : ',free_num)