from keras.src.legacy.preprocessing.image import ImageDataGenerator


#Specifying image data processing
trainDataGenerator = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.25,
    rotation_range=15,
    brightness_range=[0.9, 1.1],
    width_shift_range=0.12,
    zoom_range=0.2,
    height_shift_range=0.05,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.15,
)


#Specifying image data processing
validDataGenerator = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.15,
)

#Get train image data generation
def getTrainImageDataGeneration():
    return trainDataGenerator

#Get validation image data generation
def getValidImageDataGeneration():
    return validDataGenerator