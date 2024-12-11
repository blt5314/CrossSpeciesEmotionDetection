import keras
from config import loadModelDirectory, loadedModelName
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from processing.getdata import validationDataset

#Get confusion matrix for specified model
def getConfusionMatrix(modelFileName):
    # Load model
    modelPath = "." + loadModelDirectory + modelFileName
    loadedModel = keras.saving.load_model(
        modelPath
    )

    #Turn off shuffling
    validationDataset.shuffle = False

    #Get target labels
    targets = validationDataset.classes

    #Get prediction labels
    predictions = np.argmax(loadedModel.predict(validationDataset), axis=1)

    #Create and plot confusion matrix
    cm = confusion_matrix(targets, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=validationDataset.class_indices,yticklabels=validationDataset.class_indices)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    #Reset image generator
    validationDataset.reset()

getConfusionMatrix(loadedModelName)