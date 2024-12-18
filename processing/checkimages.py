import os

#StackOverflow code: https://stackoverflow.com/questions/62544528/tensorflow-decodejpeg-expected-image-jpeg-png-or-gif-got-unknown-format-st
def is_image(filename, verbose=False):

    data = open(filename,'rb').read(10)

    #Check if file is JPG or JPEG
    if data[:3] == b'\xff\xd8\xff':
        if verbose == True:
             print(filename+" is: JPG/JPEG.")
        return True

    #Check if file is PNG
    if data[:8] == b'\x89\x50\x4e\x47\x0d\x0a\x1a\x0a':
        if verbose == True:
             print(filename+" is: PNG.")
        return True

    # check if file is GIF
    if data[:6] in [b'\x47\x49\x46\x38\x37\x61', b'\x47\x49\x46\x38\x39\x61']:
        if verbose == True:
             print(filename+" is: GIF.")
        return True

    return False

dir = ('../data/angry')

#Go through all files in desired folder
for filename in os.listdir(dir):
     #Check if file is actually an image file
     if is_image(os.path.join(dir, filename), verbose=False) == False:
          #If the file is not valid, print it
          print(filename)