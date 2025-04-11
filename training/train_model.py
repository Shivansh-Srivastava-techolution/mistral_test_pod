import json
import os
# from keras.preprocessing.image import ImageDataGenerator
# from keras.models import Sequential
# from keras.layers import Flatten, Dense
# from keras.applications import VGG16
# from keras.optimizers import Adam
# from keras.callbacks import ModelCheckpoint


def main(dataset_path, models_path, model_details):
    trained_model_path = os.path.join(models_path, "best_model.h5")
    os.system(f"touch {trained_model_path}")
    # Define constants
#     IMG_WIDTH, IMG_HEIGHT = model_details["hyperParameter"]["input_shape"]
#     DATASET_DIR = dataset_path
#     BATCH_SIZE = 64
#     EPOCHS = model_details["hyperParameter"]["epochs"]

#     # Augment the data and split into training and validation sets
#     datagen = ImageDataGenerator(rescale=1. / 255,
#                                  shear_range=0.2,
#                                  zoom_range=0.2,
#                                  horizontal_flip=True,
#                                  validation_split=0.2)  # set validation split

#     # This is a generator that will read pictures found in subfolders of 'data/train', and indefinitely generate
#     # batches of augmented image data
#     train_generator = datagen.flow_from_directory(DATASET_DIR,  # this is the target directory
#                                                   target_size=(IMG_WIDTH, IMG_HEIGHT),
#                                                   # all images will be resized to 150x150
#                                                   batch_size=BATCH_SIZE,
#                                                   class_mode='binary',
#                                                   # since we use binary_crossentropy loss, we need binary labels
#                                                   subset='training')  # set as training data

#     # This is a similar generator, for validation data
#     validation_generator = datagen.flow_from_directory(DATASET_DIR,
#                                                        target_size=(IMG_WIDTH, IMG_HEIGHT),
#                                                        batch_size=BATCH_SIZE,
#                                                        class_mode='binary',
#                                                        subset='validation')  # set as validation data

#     if model_details["hyperParameter"]["model"] == "VGG16":
#         # Load the VGG16 network, ensuring the head FC layer sets are left off
#         base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))

#         # Construct the head of the model that will be placed on top of the base model
#         model = Sequential()
#         model.add(base_model)
#         model.add(Flatten())
#         model.add(Dense(256, activation='relu'))
#         model.add(Dense(1, activation='sigmoid'))

#         for layer in base_model.layers:
#             layer.trainable = False

#         # Configure and compile the model
#         model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
#     else:
#         assert False, "Model not supported"

#     # Save the model every 5 epochs
#     checkpoint = ModelCheckpoint(os.path.join(models_path, "best_model.h5"), monitor='val_accuracy',
#                                  verbose=1, save_best_only=True, mode='max')

#     model.fit_generator(train_generator,
#                         steps_per_epoch=train_generator.samples // BATCH_SIZE,
#                         epochs=EPOCHS,
#                         validation_data=validation_generator,
#                         validation_steps=validation_generator.samples // BATCH_SIZE,
#                         callbacks=[checkpoint])

