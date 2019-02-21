from keras.models import Model
from keras.layers import *
from keras import backend as K
from tqdm import tqdm
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.xception import Xception
import csv
import re
import os
import pickle
import cv2
from sklearn.preprocessing import LabelEncoder
import unicodedata
from sklearn.utils import shuffle


IMAGE_HEIGHT = 64
IMAGE_WIDTH = None
NO_CHANNEL = 1
TRAIN_VAL_SPLIT = 0.8
BATCH_SIZE = 8
STRIDE = 12
FILTER_SIZE = 32
NO_CLASSES = 66 + 1   # blank token
DATA_FOLDER = 'data_quora'
LABEL_ENCODER_PATH = 'label_encoder.pkl'


class DataGenerator():
    def __init__(self, train_image_list, val_image_list, batch_size=BATCH_SIZE):
        self.train_image_list = train_image_list
        self.val_image_list = val_image_list
        self.batch_size = batch_size
        self.current_train_index = 0
        self.current_val_index = 0
        self.load_label_encoder()

    def load_image(self, image_path):
        image = cv2.imread(image_path, 0)
        image = image / 255.
        image = np.expand_dims(image, axis=-1)
        return image

    def load_label_encoder(self):
        self.le = load_label_encoder()

    def get_batch(self, partition='train'):
        if partition == 'train':
            temp_image_list = self.train_image_list[self.current_train_index:self.current_train_index+self.batch_size]
            temp_image_list = [os.path.join(DATA_FOLDER, t) for t in temp_image_list]
        else:
            temp_image_list = self.val_image_list[self.current_val_index:self.current_val_index+self.batch_size]
            temp_image_list = [os.path.join(DATA_FOLDER, t) for t in temp_image_list]
        image_array = []
        label_array = []
        for ind in range(self.batch_size):
            image_array.append(self.load_image(temp_image_list[ind]))
            label_array.append(temp_image_list[ind].split('/')[-1].split('_')[0])
        max_image_width = max([m.shape[1] for m in image_array])
        max_label_length = max(len(m) for m in label_array)
        input_image = np.ones((self.batch_size, IMAGE_HEIGHT, max_image_width, 1))
        input_true_label = np.ones((self.batch_size, max_label_length)) * NO_CLASSES
        input_time_step = np.zeros((self.batch_size, 1))
        input_label_length = np.zeros((self.batch_size, 1))
        for ind in range(self.batch_size):
            real_width = image_array[ind].shape[1]
            tmp = [self.le.transform([t])[0] for t in label_array[ind]]
            real_label_len = len(tmp)
            input_image[ind, :, :real_width, :] = image_array[ind]
            input_true_label[ind, :real_label_len] = tmp
            input_time_step[ind] = compute_time_step(real_width) - 2
            input_label_length[ind] = real_label_len
        inputs = {
            'input_image': input_image,
            'input_true_label': input_true_label,
            'input_time_step': input_time_step,
            'input_label_length': input_label_length}
        outputs = {'ctc': np.zeros((self.batch_size))}
        return (inputs, outputs)


    def next_train(self):
        while True:
            tmp = self.get_batch('train')
            self.current_train_index += self.batch_size
            if self.current_train_index >= len(self.train_image_list) - self.batch_size:
                self.train_image_list = shuffle(self.train_image_list)
                self.current_train_index = 0
            yield tmp


    def next_val(self):
        while True:
            tmp = self.get_batch('val')
            self.current_val_index += self.batch_size
            if self.current_val_index >= len(self.val_image_list) - self.batch_size:
                self.val_image_list = shuffle(self.val_image_list)
                self.current_val_index = 0
            yield tmp

def compute_time_step(image_width, stride=STRIDE//2):
    for i in range(2):
        tmp = (image_width - 1) // 2 + 1
    tmp = (tmp + stride - 1) // stride
    return tmp


def get_image_list(folder=DATA_FOLDER):
    image_list = os.listdir(folder)
    return image_list
      

def get_all_character():
    image_list = get_image_list()
    all_character_list = []
    for image in image_list:
        tmp = image.split('_')[0]
        all_character_list += tmp
    return all_character_list


def create_label_encoder(all_character_list):
    all_character_list = list(set(all_character_list))
    le = LabelEncoder()
    le.fit(all_character_list)
    with open(LABEL_ENCODER_PATH, 'wb') as f:
        pickle.dump(le, f)


def load_label_encoder():
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        le = pickle.load(f)
    return le


def ctc_loss(args):
    y_pred, y_true, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)


def fake_loss(y_true, y_pred):
    return y_pred


def squeeze_layer(arr, axis=1):
    return K.squeeze(arr, axis)


def create_model(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, NO_CHANNEL)):
    input_image = Input(shape=input_shape, name='input_image')
    input_true_label = Input(shape=(None,), name='input_true_label')
    input_time_step = Input(shape=(1,), name='input_time_step')
    input_label_length = Input(shape=(1,), name='input_label_length')
    x = Conv2D(32, (3, 3), strides=(2, 2), padding='same',
               use_bias=False)(input_image)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    residual = Conv2D(128, (1, 1), strides=(
        2, 1), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 1), padding='same')(x)
    x = add([x, residual])

    residual = Conv2D(256, (1, 1), strides=(
        2, 1), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = Activation('relu')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 1), padding='same')(x)
    x = add([x, residual])
    residual = Conv2D(728, (1, 1), strides=(
        2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = Activation('relu')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = add([x, residual])
    for i in range(2):
        residual = x
        x = Activation('relu')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = add([x, residual])
    residual = Conv2D(1024, (1, 1), strides=(
        2, 1), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = Activation('relu')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 1), padding='same')(x)
    x = add([x, residual])

    x = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(2048, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    sld = Conv2D(
        filters=256,
        kernel_size=(2, FILTER_SIZE//4),
        strides=(2, STRIDE//4),
        padding='same')(x)
    sld = BatchNormalization()(sld)
    sld = ReLU()(sld)
    sld = Lambda(squeeze_layer)(sld)

    gru = CuDNNGRU(units=256, return_sequences=True)(sld)
    gru = BatchNormalization()(gru)
    gru = ReLU()(gru)

    dense = Dense(units=NO_CLASSES)(gru)
    dense = Activation('softmax')(dense)
    loss_out = Lambda(ctc_loss, output_shape=(1,), name='ctc')(
        [dense, input_true_label, input_time_step, input_label_length])
    model = Model([input_image, input_true_label,
                   input_time_step, input_label_length], loss_out)
    print(model.summary())
    return model

# le = load_label_encoder()
# print (le.classes_)