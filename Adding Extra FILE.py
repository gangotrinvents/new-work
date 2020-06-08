from pathlib import Path
import glob
import os
import sys

imageDirectory = input("Enter the path of your folder: ")
assert os.path.exists(imageDirectory), "Did not find the file at, "+str(imageDirectory)
for file in glob.glob(imageDirectory+"\*.bmp"):
    p = Path(file)
    p = str(p.stem)
    with open(os.path.join(imageDirectory, p + ".txt"), 'w') as fp:
        pass
