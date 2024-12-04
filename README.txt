README for CrossSpeciesEmotionDetection

It is highly recommended that you use the pycharm IDE.

Go to file and then settings in the top left of the IDE.

Go to python interpreter.

Ensure you are on a python 3.12 interpreter.

Click the plus button on this screen.

Add the version 2.18 of tensorflow.

Add the version 6.7.1 of pyqt6.

Add the version 3.6.0 of keras.

Add the version 3.9.2 of matplotlib

Add the version 2.0.2 of numpy

Add the version 0.13.2 of seaborn

Add the version 1.5.2 of scikit-learn

-----------------------------------------------------------------

First you must create the models to use /processing/100test.py and /app.py

The initial dataset by default will be the dog dataset, simply run the /models/createmodel.py script to create
the first dog model

To create the human model, you must change the word dog on line 41 in createmodel.py to human

you must also change line 2 in config.py to human

run createmodel.py again to create the human model

To create the combined model, you must change the word dog on line 41 in createmodel.py to combined

you must also change line 2 in config.py to combined

run createmodel.py again to create the combined model

-----------------------------------------------------------------

If you wish to use the /processing/100test.py script to test the models you generated,
simply go to the script and run it. It does not require any other setup besides having the
models prepared and trained.

If you wish to see app.py, you can choose which model it will use by editing line 17 of the file and
replacing the modelfilename with the modelfilename (located in /models/savedmodels) you wish to use with the GUI.
The default is dog, just replace dog with human or combined to choose which model you want to use.

You may then upload an image of a human or dog to app.py once it is launched to have it analyze the image.