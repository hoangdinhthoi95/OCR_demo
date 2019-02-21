from utils import *
from keras.optimizers import Adam
from keras.callbacks import *
from sklearn.utils import shuffle


all_character_list = get_all_character()
create_label_encoder(all_character_list)

model = create_model()
MODEL_PATH = 'model/xception_model_{val_loss:.5f}.h5'

if not os.path.exists('model'):
    os.mkdir('model')

all_image_list = get_image_list(DATA_FOLDER)
all_image_list = shuffle(all_image_list, random_state=123)
no_train_images = int(TRAIN_VAL_SPLIT * len(all_image_list))
train_image_list = all_image_list[:no_train_images]
val_image_list = all_image_list[no_train_images:]

print("Train on  ", no_train_images, 'images')
print ("Validate on", len(all_image_list) - no_train_images, 'images')

optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-9,
                 decay=1e-9, amsgrad=True, clipnorm=5., clipvalue=0.5)
model.compile(loss=fake_loss, optimizer=optimizer)
data_gen = DataGenerator(train_image_list, val_image_list)
step_val = len(data_gen.val_image_list) // BATCH_SIZE
step_train = len(data_gen.train_image_list) // BATCH_SIZE // 2

checkpointer = ModelCheckpoint(
    filepath=MODEL_PATH, save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(patience=6, verbose=1, facttor=0.75)
model.fit_generator(generator=data_gen.next_train(), steps_per_epoch=step_train, epochs=200, verbose=1,
    callbacks=[checkpointer, reduce_lr], validation_data=data_gen.next_val(), validation_steps=step_val)
