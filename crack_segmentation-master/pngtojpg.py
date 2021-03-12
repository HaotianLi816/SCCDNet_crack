import os
from PIL import Image

dirname_read = "C:\\Users\\Administrator\\Desktop\\DeepCrack\\test\\masks\\"
dirname_write = "C:\\Users\\Administrator\\Desktop\\DeepCrack\\test\\masks1\\"
names = os.listdir(dirname_read)
count = 0
for name in names:
    img=Image.open(dirname_read+name)
    name=name.split(".")
    if name[-1] == "png":
        name[-1] = "jpg"
        name = str.join(".", name)
        # r = img.split()
        # img = Image.merge("RGB",(r,0,0))
        to_save_path = dirname_write + name
        img.save(to_save_path)
        count+=1
        print(to_save_path, "------conut：",count)
    else:
        continue
