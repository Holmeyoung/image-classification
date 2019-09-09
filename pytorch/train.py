import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lenet import LeNet

import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type = int, required = False, help = 'batch size', default = 64)
    parser.add_argument('-w', '--worker', type = int, required = False, help = 'load data worker', default = 4)
    parser.add_argument('-lr', '--lr', type = float, required = False, help = 'learning rate', default = 0.001)
    parser.add_argument('-train', '--train', type = str, required = True, help = 'train data dir')
    parser.add_argument('-val', '--val', type = str, required = True, help = 'val data dir')
    args = parser.parse_args()
    return args
args = get_args()

def get_dataloader():
    data_transforms = transforms.Compose([
            transforms.Resize((32, 32), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(args.train, transform = data_transforms),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.worker
        )

    val_loader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(args.val, transform = data_transforms),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.worker
        )
    
    return train_loader, val_loader
train_loader, val_loader = get_dataloader()

classes = 10
net = LeNet(classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model to gpu
net.to(device)

print("Start Training...")
os.makedirs('./expr', exist_ok=True)
for epoch in range(1000):
    net.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if i % 100 == 99:
            print('[%d, %5d] loss: %.6f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
    
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_loader :
            inputs, labels = data
            inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the test images: %.2f' % (100 * correct / total))
    torch.save(net.state_dict(), './expr/%s_%s_%.2f.pth' % (str(epoch), args.net, 100 * correct / total))

print("Done Training!")

