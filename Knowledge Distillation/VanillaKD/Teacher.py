import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchvision import datasets, transforms


# 定义教师网络
class TeacherNet(nn.Module):
    def __init__(self):
        super(TeacherNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.3)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        return output


# 训练过程
def train_teacher(model, device, train_loader, optimizer, epoch):
    # 启用 BatchNormalization 和 Dropout
    model.train()
    trained_samples = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # 搬到指定gpu或者cpu设备上运算
        data, target = data.to(device), target.to(device)
        # 梯度清零
        optimizer.zero_grad()
        # 前向传播
        output = model(data)
        # 计算误差
        loss = F.cross_entropy(output, target)
        # 误差反向传播
        loss.backward()
        # 梯度更新一步
        optimizer.step()

        # 统计已经训练的数据量
        trained_samples += len(data)
        progress = math.ceil(batch_idx / len(train_loader) * 50)

        print('\rTrain epoch: {} {}/{} [{}]{}%'.format(epoch, trained_samples, len(train_loader.dataset), '-'*progress+'>', progress*2), end='')


# 测试过程
def test_teacher(model, device, test_loader):
    # 不启用 BatchNormalization 和 Dropout
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            # 输出预测类别
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print('\nTest: average loss: {:.4f}, accuracy:{}/{},({:.0f}%)'.format(
        test_loss, correct,len(test_loader.dataset), 100 * correct / len(test_loader.dataset)

    ))

    return test_loss, correct / len(test_loader.dataset)


def teacher_main():
    epochs = 10
    batch_size = 64
    torch.manual_seed(0)
    mnist_path = 'data/'

    # 动态设置硬件设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(mnist_path, train=True, download=True,
                       transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(mnist_path, train=False, download=True,
                       transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,),(0.3081,))
                       ])),
        batch_size=1000, shuffle=True
    )

    # 实例化模型
    model = TeacherNet().to(device)
    # 选取优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    teacher_history = []

    for epoch in range(1, epochs+1):
        print("teacher:"+str(epoch))
        train_teacher(model, device, train_loader, optimizer, epoch)
        loss, acc = test_teacher(model, device, test_loader)

        teacher_history.append((loss, acc))

    # 保存模型,state_dict:Returns a dictionary containing a whole state of the module.
    torch.save(model.state_dict(), 'data/model/teacher.pt')

    return model, teacher_history

