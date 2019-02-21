from keras.models import load_model, Model
from utils import *
import os
import itertools
from keras import backend as K


MODEL_PATH = 'model/xception_model_0.05405.h5'
test_folder = 'data_test'
model = load_model(MODEL_PATH,
                   custom_objects={'squeeze_layer': squeeze_layer,
                                   'ctc_loss': ctc_loss,
                                   'fake_loss': fake_loss})
input_layer = model.inputs[0]
output_layer = model.layers[-5].output

model_new = Model(input_layer, output_layer)
# print (model_new.summary())

test_image_list = os.listdir(test_folder)
le = load_label_encoder()

for t in test_image_list:
    image = cv2.imread(os.path.join(test_folder, t), 0)
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image / 255., axis=-1)
    result = model_new.predict(image)
    result = np.squeeze(result)[2:, :]
    result = np.argmax(result, 1)
    result = [k for k, _ in itertools.groupby(result) if k != NO_CLASSES-1]
    result = le.inverse_transform(result)
    result = ''.join(result)
    print ("True label:", t.split('_')[0])
    print ("Predicted :", result)
    print ('_'*150)
