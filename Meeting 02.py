import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

#导入数据
train_ds = torchvision.datasets.CIFAR10('data',
                                        train = True,
                                        download = True,
                                        transform=torchvision.transforms.ToTensor) #将数据类型转化为Tensor
test_ds = torchvision.datasets.CIFAR10('data',
                                       train = False,
                                       download = True,
                                       transform=torchvision.transforms.ToTensor)
batch_size = 32
train_d1 = torch.utils.DataLoader(train_ds, batch_size = batch_size, shuffle = True)
test_d1 = torch.utils.DataLoader(test_ds, batch_size = batch_size)

#Visualization
plt.figure(figsize = (20, 5))
for i, imgs in enumerate(imgs[:20]):
    npimg = imgs.numpy().transpose((1,2,0));
    plt.subplot(2,10,i+1);
    plt.imshow(npimg, cmap=plt.cm.binary);
    plt.axis('off');

#构建简单的CNN网络
import torch.nn.functional as F

num_classes = 10 #图片的类别数
class Model(nn.Module):
    def __init__(self):
        super().__init__() #提取神经网络
        self.conv1 = nn.Conv2d(3,64,kernel_size=3) #第一层卷积，3x3
        self.pool1 = nn.MaxPool2d(kernel_size = 2) #池化层，2x2
        self.conv2 = nn.Conv2d(64, 64, kernel_size = 3) #第二层卷积，3x3
        self.pool2 = nn.MaxPool2d(kernel_size = 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)  # 第三层卷积，3x3
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        #分类网络
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)
    #向前传播
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv1(x)))
        x = self.pool3(F.relu(self.conv1(x)))

        x = torch.flatten(x, start_dim_1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

from torchinfo import summary
model = Model().to(device)

summary(model)

#设置超参数
loss_fn = nn.CrossEntropyLoss()
learn_rate = 1e-2
opt = torch.optim.SGD(model.parameters(), Ir=learn_rate)
#训练循环
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset) #训练集大小
    num_batches = len(dataloader)

    train_loss, train_acc = 0, 0

    for X,y in dataloader:
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred,y) #计算误差

        optimizer.zero_grad()
        loss.backward() #反向传播
        optimizer.step()

        train_acc += (pred.argmax(dim=1)==y.type(torch.float).sum().item())
        train_loss += loss.item()

    train_acc = train_acc / size
    train_loss = train_loss / num_batches

    return train_acc, train_loss
#测试函数
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, test_acc = 0, 0

    with torch.no_grad():
        for imgs, target in dataloader:
            imgs, target = imgs.to(device), target.to(device)

            target_pred = model(imgs)
            loss = loss_fn(target_pred, target)

            test_loss += loss.item
            test_acc += (target.argmax(dim=1)==target).type(torch.float).sum().item()

    test_acc = test_acc/size
    test_loss = test_loss/num_batches

    return test_acc, test_loss

epochs = 10
train_loss = []
train_acc = []
test_loss = []
test_acc = []

for epoch in range(epochs):
    model.train()
    epoch_train_acc, epoch_train_loss = train(train_d1, model, loss_fn, opt)

    model.eval()
    epoch_test_acc, epoch_test_loss = test(test_d1, model, loss_fn)

    train_acc.append(epoch_train_acc)
    train_loss.append(epoch_train_loss)
    test_acc.append(epoch_test_acc)
    test_loss.append(epoch_test_loss)

    template = ('Epoch:{:2d}, Train_acc:{:.1f}%, Train_loss:{:3f}, Test_acc:{:.1f}%, Test_loss:{:.3f}')
    print(template.format(epoch+1, epoch_train_acc*100, epoch_train_loss, epoch_test_acc*100, epoch_test_loss))
print('Done')

#结果可视化
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans_serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

epochs_range = range(epochs)

plt.figure(figsize = (12,3))
plt.subplot(1,2,1)

plt.plot(epochs_range, train_acc, label = 'Train Accuracy')
plt.plot(epochs_range, test_acc, label = 'Test Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range, train_loss, label = 'Train Loss')
plt.plot(epochs_range, test_loss, label = 'Test Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()