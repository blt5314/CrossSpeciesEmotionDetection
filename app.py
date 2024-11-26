import keras
import numpy as np
import sys
from pathlib import Path
from PIL.Image import Image
from skimage import transform
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog
from config import imageWidth
from config import imageHeight

modelFileName = "./models/savedmodels/dog_vgg16_multiclass_model.keras"

loadedModel = keras.saving.load_model(modelFileName)

def load(filename):
   npImage = Image.load(filename)
   npImage = np.array(npImage).astype('float32')/255
   npImage = transform.resize(npImage, (imageWidth, imageHeight, 3))
   npImage = np.expand_dims(npImage, axis=0)
   return npImage


#Create the main window
class MainWindow(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle('Cross-Species Facial Emotion Detection')

        fileBrowse = QPushButton('Select Image')
        fileBrowse.clicked.connect(self.openFileDialog)

        self.imageLabel = QLabel()
        self.imageNameLabel = QLabel("test")
        self.imageClassLabel = QLabel("test2")

        layout = QVBoxLayout()
        layout.addWidget(fileBrowse)
        layout.addWidget(self.imageLabel)
        layout.addWidget(self.imageNameLabel)
        layout.addWidget(self.imageClassLabel)

        self.setLayout(layout)
        self.show()

    def openFileDialog(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select a File",
            "",
            "Images (*.png *.jpg *.jpeg)"
        )

        if filename:
            path = Path(filename)
            print(path)
            self.imageNameLabel.setText(filename)
            self.imageLabel.setPixmap(QPixmap(path))
            #image = load(path)
            #self.imageClassLabel.setText(loadedModel.predict(image))

#Start application
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())