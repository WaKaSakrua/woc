import torch
from torchvision import models, transforms
from PIL import Image
from torchvision import datasets
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
# 确保模型在GPU上运行，如果有的话
device = torch.device("cuda")

# 加载预训练的ResNet50模型
model = models.resnet50(weights=None)
model.fc = nn.Linear(2048, 1)
model.to(device='cuda:0')
#读取数据，进行数据增强。
transforms=transforms.Compose([
    transforms.Resize([224,224]),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5]),
])

dataset_train=datasets.ImageFolder(r"D:\tools\Python\pythonProject\woc\ready_picture",transform=transforms)
dataset_valiation=datasets.ImageFolder(r"D:\tools\Python\pythonProject\woc\valiation",transform=transforms)
train_loader=torch.utils.data.DataLoader(dataset_train,batch_size=10,shuffle=True,drop_last=True)
valiation_loder=torch.utils.data.DataLoader(dataset_valiation,batch_size=4,shuffle=True,drop_last=True)
optimizer=optim.Adam(model.parameters(),lr=0.002)
# model.load_state_dict(torch.load(r"D:\tools\Python\pythonProject\woc\paremeter\model"))
for epoch in range(50):
    correct=0
    print("This is {} epoch".format(epoch))
    for step,(x,y) in enumerate(train_loader):
        x=x.to(device)
        y=y.unsqueeze(-1)
        y=y.to(device).float()
        optimizer.zero_grad()
        prediction = torch.sigmoid(model(x))
        l = F.binary_cross_entropy(prediction, y)
        l.backward()
        optimizer.step()
        prediction=torch.tensor([[1] if num[0] >= 0.5 else [0] for num in prediction]).to(device)
        correct += prediction.eq(y.long()).sum().item()
        if (step+1)%20==0:
            l.item()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} '.format(
                epoch, (step + 1) * len(x), len(train_loader.dataset),
                       100. * (step + 1) / len(train_loader), l.item()))
    torch.save(model.state_dict(),r"D:\tools\Python\pythonProject\woc\paremeter\model"+str(epoch))
    print("保存成功")
    # test_loss = 0
    # correct = 0
    # with torch.no_grad():
    #     for data, target in valiation_loder:
    #         data, target = data.to(device), target.to(device).float().unsqueeze(-1)
    #         output = model(data)
    #         # print(output)
    #         test_loss += F.binary_cross_entropy(torch.sigmoid(output), target, reduction='sum').item()  # 将一批的损失相加
    #         pred = torch.tensor([[1] if num[0] >= 0.5 else [0] for num in output]).to(device)
    #         correct += pred.eq(target.long()).sum().item()
    #     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #         test_loss, correct, len(valiation_loder),
    #         100. * correct / len(valiation_loder)))



