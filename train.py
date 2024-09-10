import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Check TensorFlow version
print("TensorFlow version:", tf.__version__)

# MobileNet is designed to work with images of dim 224x224
img_rows, img_cols = 224, 224

# Load the MobileNet model pre-trained on ImageNet
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, 3))

# Set all layers to be trainable
for layer in base_model.layers:
    layer.trainable = True

# Print layers to verify which are trainable
for (i, layer) in enumerate(base_model.layers):
    print(str(i), layer.__class__.__name__, layer.trainable)

# Adding the top/head model for classification
def add_top_model(bottom_model, num_classes):
    """Creates the top or head of the model that will be
    placed on top of the bottom layers"""
    x = bottom_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)
    return x

# Number of classes in your classification task
num_classes = 7  # 7 classes: angry, disgust, fear, happy, sad, surprise, neutral

# Add the head/top model to the MobileNet base
top_model = add_top_model(base_model, num_classes)

# Creating the final model
model = Model(inputs=base_model.input, outputs=top_model)

# Print model summary
print(model.summary())

# Directories for training and validation data
train_data_dir = '/Users/divagarvakeesan/PycharmProjects/MobileNet-Based Real-Time Emotion Recognition from Facial Expressions/train'
validation_data_dir = '/Users/divagarvakeesan/PycharmProjects/MobileNet-Based Real-Time Emotion Recognition from Facial Expressions/validation'

# ImageDataGenerator for data augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1. / 255)

batch_size = 32

# Train data generator
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical'
)

# Validation data generator
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical'
)

# Callbacks for model training
checkpoint = ModelCheckpoint(
    '/Users/divagarvakeesan/PycharmProjects/MobileNet-Based Real-Time Emotion Recognition from Facial Expressions/emotion_face_mobilNet.keras',  # Saving the best model in .keras format
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1
)

earlystop = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=10,
    verbose=1,
    restore_best_weights=True
)

learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_accuracy',
    patience=5,
    verbose=1,
    factor=0.2,
    min_lr=0.0001
)

callbacks = [earlystop, checkpoint, learning_rate_reduction]

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),  # Updated to `learning_rate` argument
    metrics=['accuracy']
)

# Parameters for training
nb_train_samples = 28821  # Number of images in training set (update according to your dataset)
nb_validation_samples = 7066  # Number of images in validation set (update according to your dataset)
epochs = 25  # Increased the number of epochs for better training

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size
)

# Save the final model in the new .keras format
model.save('/Users/divagarvakeesan/PycharmProjects/MobileNet-Based Real-Time Emotion Recognition from Facial Expressions/final_emotion_detection_model.keras')
