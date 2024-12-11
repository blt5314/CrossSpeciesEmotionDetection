#Specifying datasets
datasetName = 'combined'
loadedModelName = "combined_vgg16_multiclass_model.keras"
imageWidth, imageHeight = 224, 224
numberOfClasses = 4
dataDirectory = '../data/' + datasetName
batchSize = 32
classDirectory = './data/' + datasetName
loadModelDirectory = "./models/savedmodels/"