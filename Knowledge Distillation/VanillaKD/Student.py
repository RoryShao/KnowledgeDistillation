import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchvision import datasets, transforms


# 构建学生网络
class Student(nn.Module):
    def __init__(self):
        super(Student, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = F.relu(self.fc3(x))
        return output


# 蒸馏部分：定义kd的loss
def distillation(y, labels, teacher_scores, temp, alpha):
    """

    :param y: 学生预测的概率分布
    :param labels: 实际标签
    :param teacher_scores: 老师预测的概率分布
    :param temp: 温度系数
    :param alpha: 损失调整因子
    :return:
    """
    kl_stu_tea = nn.KLDivLoss()(F.log_softmax(y / temp, dim=1), F.softmax(teacher_scores / temp, dim=1)) * temp * temp * 2.0 * alpha
    stu_loss = F.cross_entropy(y, labels) * (1-alpha)

    return kl_stu_tea+stu_loss


# 训练学生网络
def train_student_kd(model, teacher_model, device, train_loader, optimizer, epoch):
    model.train()
    trained_samples = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # 搬到指定gpu或者cpu设备上运算
        data, target = data.to(device), target.to(device)
        # 梯度清零
        optimizer.zero_grad()
        # 前向传播
        output = model(data)
        # 老师输出
        teacher_output = teacher_model(data)
        # 计算误差
        loss = distillation(output, target, teacher_output, temp=10., alpha=.7)
        # 误差反向
        loss.backward()
        # 梯度更新一步
        optimizer.step()

        # 统计已经训练的数据量
        trained_samples += len(data)
        progress = math.ceil(batch_idx / len(train_loader) * 50)

        print('\rTrain epoch: {} {}/{} [{}]{}%'.format(epoch, trained_samples, len(train_loader.dataset), '-'*progress+'>', progress*2), end='')


# 测试学生网络
def test_student_kd(model, device, test_loader):
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
        test_loss, correct, len(test_loader.dataset), 100 * correct / len(test_loader.dataset)

    ))

    return test_loss, correct / len(test_loader.dataset)


def student_kd_main(teacher_m):
    epochs = 10
    batch_size = 64
    torch.manual_seed(0)
    mnist_path = 'data/'

    # 动态设置硬件设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(mnist_path, train=True, download=False,
                       transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,),(0.3081,))
                       ])),
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(mnist_path, train=False, download=False,
                       transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,),(0.3081,))
                       ])),
        batch_size=1000, shuffle=True
    )

    # 实例化模型
    model = Student().to(device)
    # 选取优化器 stduent with kd
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    #  stduent without kd
    # optimizer = torch.optim.Adadelta(model.parameters())

    student_history = []

    for epoch in range(1, epochs+1):
        print("student:"+str(epoch))
        train_student_kd(model, teacher_m, device, train_loader, optimizer, epoch)
        loss, acc = test_student_kd(model, device, test_loader)

        student_history.append((loss, acc))

    # 保存模型,state_dict:Returns a dictionary containing a whole state of the module.
    torch.save(model.state_dict(), 'data/model/student.pt')

    return model, student_history


# 让学生自己学，不使用KD
def train_student(model, device, train_loader, optimizer, epoch):
    model.train()
    trained_samples = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        trained_samples += len(data)
        progress = math.ceil(batch_idx / len(train_loader) * 50)

        print('\rTrain epoch: {} {}/{} [{}]{}%'.format(epoch, trained_samples, len(train_loader.dataset), '-'*progress+'>', progress*2), end='')


def test_student(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest: average loss: {:.4f}, accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, correct / len(test_loader.dataset)


def student_main():
    epochs = 10
    batch_size = 64
    torch.manual_seed(0)
    mnist_path = 'data/'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(mnist_path, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(mnist_path, train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1000, shuffle=True)

    model = Student().to(device)
    optimizer = torch.optim.Adadelta(model.parameters(),  lr=1e-3)
    # optimizer = torch.optim.Adadelta(model.parameters())

    student_history = []

    for epoch in range(1, epochs + 1):
        train_student(model, device, train_loader, optimizer, epoch)
        loss, acc = test_student(model, device, test_loader)
        student_history.append((loss, acc))

    torch.save(model.state_dict(), "data/student.pt")
    return model, student_history

