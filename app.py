import keras
import numpy as np
import sys
from pathlib import Path
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.utils.image_utils import load_img, img_to_array
from config import imageWidth
from config import imageHeight
from config import classDirectory

#Get labels from directory
classGenerator = ImageDataGenerator().flow_from_directory(
    classDirectory,
    class_mode = 'categorical',
)
labels = classGenerator.class_indices
labels = {v: k for k, v in labels.items()}

#Load model from foldere
modelFileName = "./models/savedmodels/dog_vgg16_multiclass_model.keras"
loadedModel = keras.saving.load_model(modelFileName)

#Function to load selected image as an array
def load(path):
    image = load_img(path, target_size=(imageWidth, imageHeight))
    image = img_to_array(image)
    image = np.array(image).astype('float32') / 255
    image = np.expand_dims(image, axis=0)
    return image


#Create the main window
class MainWindow(QWidget):
    #Initialize window
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle('Cross-Species Facial Emotion Detection')
        self.setGeometry(100, 100, 500, 500)

        fileBrowse = QPushButton('Select Image')
        fileBrowse.clicked.connect(self.openFileDialog)

        self.imageLabel = QLabel()
        self.imageNameLabel = QLabel()
        self.imageClassLabel = QLabel()

        layout = QVBoxLayout()
        layout.addWidget(fileBrowse)
        layout.addWidget(self.imageNameLabel)
        layout.addWidget(self.imageLabel)
        layout.addWidget(self.imageClassLabel)

        self.setLayout(layout)
        self.show()

    #Open file dialog and update gui with selection
    def openFileDialog(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select a File",
            "",
            "Images (*.png *.jpg *.jpeg)"
        )

        if filename:
            path = str(Path(filename))
            self.imageNameLabel.setText(filename)
            self.imageLabel.setPixmap(QPixmap(path))
            imageFile = load(path)
            predictions = loadedModel.predict(imageFile)
            predictions = np.array(predictions)
            highestProbIndex = np.argmax(predictions)
            self.imageClassLabel.setText(labels.get(highestProbIndex))

#Start application
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())