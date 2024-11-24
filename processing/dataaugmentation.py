from keras.src.legacy.preprocessing.image import ImageDataGenerator

#Try higher rotation range
#Specifying image data processing
trainDataGenerator = ImageDataGenerator(
    rescale=1./255,
    rotation_range = 15,
    brightness_range=[0.9, 1.1],
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2,
)

"""trainDataGenerator = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    rotation_range=5,
    brightness_range=[0.9, 1.1],
    width_shift_range=0.2,
    zoom_range=0.3,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2,
)"""

#Specifying image data processing
validDataGenerator = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
)

#Get train image data generation
def getTrainImageDataGeneration():
    return trainDataGenerator

#Get validation image data generation
def getValidImageDataGeneration():
    return validDataGenerator