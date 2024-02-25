import os

import pandas

import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
# 确保模型在GPU上运行，如果有的话
device = torch.device("cuda")
# 加载预训练的ResNet50模型
model = models.resnet50()
model.eval()  # 设置为评估模式
model.fc = nn.Linear(2048, 1)
model.to(device)
transform=transforms.Compose([
    transforms.Resize([224,224]),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
])
submission=pandas.DataFrame()
target=[]
ID=[]
model.load_state_dict(torch.load(r"D:\tools\Python\pythonProject\woc\paremeter\model17"))
with torch.no_grad():
    for path in os.listdir(r"D:\tools\Python\pythonProject\woc\test\T"):
        print(path)
        Pic=Image.open(os.path.join(r"D:\tools\Python\pythonProject\woc\test\T",path))
        Pic=transform(Pic)
        Pic=torch.unsqueeze(Pic,0).to(device)
        prediction = torch.sigmoid(model(Pic))
        output=prediction.item()
        print(output)
        if output>=0.5:
            output=1
        else:
            output=0
        target.append(float(f"{output:.1f}"))
        ID.append(path[:-4])
submission["ID"]=ID
submission["TARGET"]=target
submission.to_csv(r"D:\tools\Python\pythonProject\woc\Submission.csv")
