# CrossSpeciesEmotionDetection

Datasets provided by:
https://github.com/iamtomyum/DFEdataset (Dog facial emotion data)
https://www.kaggle.com/datasets/sujaykapadnis/emotion-recognition-dataset (Humann facial emotion data)


### It is highly recommended that you use the pycharm IDE.

**Steps:** 
1. Go to settings in IDE.
2. Under Project, go to Python Interpreter.
3. Ensure you are on a python 3.12 interpreter.
4. Click the plus button on this screen.
*Install these dependencies:*
    tensorflow 2.18
    pyqt6 6.7.1
    keras 3.6.0
    matplotlib 3.9.2
    numpy 2.0.2
    seaborn 0.13.2
    scikit-learn 1.5.2

-----------------------------------------------------------------
**IMPORTANT**
*If you have already created the models, you are not required to do this part*
                |
                |
                V
First you must create the models to use /processing/100test.py and /app.py

1. The initial dataset by default will be the dog dataset, simply run the /models/createmodel.py script to create
the first dog model

2. To create the human model, you must change the word dog on line 41 in createmodel.py to human

3. you must also change line 2 in config.py to human

4. run createmodel.py again to create the human model

5. To create the combined model, you must change the word dog on line 41 in createmodel.py to combined

6. you must also change line 2 in config.py to combined

7. run createmodel.py again to create the combined model

-----------------------------------------------------------------

If you wish to use the /processing/100test.py script to test the models you generated,
simply go to the script and run it. It does not require any other setup besides having the
models prepared and trained.

If you wish to see app.py and confusionmatrix.py, you can choose which model it will use by editing line 17 of the config file and
replacing the modelfilename with the modelfilename (located in /models/savedmodels) you wish to use with the GUI.
The default is combined, just replace dog with human or dog to choose which model you want to use.

You may then upload an image of a human or dog to app.py once it is launched to have it analyze the image.