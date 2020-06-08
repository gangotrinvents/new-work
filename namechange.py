import os
os.chdir(r'D:\DataSegregation\OutputImages\Less_than_20%_Tilt_X-axis')
i=1
for file in os.listdir():
    src=file
    dst="part4"+"Less_than_20%_Tilt_X-axis"+str(i)+".bmp"
    os.rename(src,dst)
    i+=1
