import os
import random
from PIL import Image, ImageOps
import torch
#对图片进行预处理，旋转，填充白色，对图片进行随机的拉伸和压缩
# 猫的路径
for path in os.listdir(r"D:\tools\Python\pythonProject\woc\train\Cat"):
    image=Image.open(os.path.join(r"D:\tools\Python\pythonProject\woc\train\Cat",path))
    image1=image.rotate(135, expand=True, fillcolor=(0, 0, 0))
    #随机进行+0.3，-0.3的压缩和拉伸
    a=random.randint(70,130)/100
    b=random.randint(70,130)/100
    image1=image1.resize((int(image1.width*a),int(image1.height*b)))
    image1.save(os.path.join(r"D:\tools\Python\pythonProject\woc\ready_picture\cat",path))
    image1.close()
    image.close()
    print(path)
# #狗的路径
for path in os.listdir(r"D:\tools\Python\pythonProject\woc\train\Dog"):
    image=Image.open(os.path.join(r"D:\tools\Python\pythonProject\woc\train\Dog",path))
    image1=image.rotate(135, expand=True, fillcolor=(0, 0, 0))
    #随机进行+0.3，-0.3的压缩和拉伸
    a=random.randint(70,130)/100
    b=random.randint(70,130)/100
    image1=image1.resize((int(image1.width*a),int(image1.height*b)))
    image1.save(os.path.join(r"D:\tools\Python\pythonProject\woc\ready_picture\dog",path))
    image1.close()
    image.close()
    print(path)
# 将图片进行颜色反转

for path in os.listdir(r"D:\tools\Python\pythonProject\woc\ready_picture\dog"):
    image=Image.open(os.path.join(r"D:\tools\Python\pythonProject\woc\ready_picture\dog",path))
    inverted_image = ImageOps.invert(image)
    inverted_image.save(os.path.join(r"D:\tools\Python\pythonProject\woc\ready_picture\dog",path))
#随机抽取
