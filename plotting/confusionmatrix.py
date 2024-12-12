import keras
from config import loadModelDirectory, loadedModelName
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from processing.getdata import validationDataset
from sklearn.metrics import f1_score, precision_score, recall_score

# Define custom metrics: F1 score, Precision, and Recall
def f1_metric(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')

def precision_metric(y_true, y_pred):
    return precision_score(y_true, y_pred, average='weighted')

def recall_metric(y_true, y_pred):
    return recall_score(y_true, y_pred, average='weighted')

#Get confusion matrix for specified model
def getConfusionMatrix(modelFileName):
    # Load model
    modelPath = "." + loadModelDirectory + modelFileName
    loadedModel = load_model(modelPath, custom_objects={'f1_score': f1_metric, 'precision': precision_metric, 'recall': recall_metric})

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