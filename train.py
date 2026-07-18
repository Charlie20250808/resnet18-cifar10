import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class BasicBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1):
        super().__init__()
        self.main=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,stride,1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,3,1,1,bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut=nn.Identity()if stride==1 and in_channels==out_channels else nn.Sequential(
            nn.Conv2d(in_channels,out_channels,1,stride,bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu=nn.ReLU(inplace=True)

    def forward(self,x):
        return self.relu(self.main(x)+self.shortcut(x))

class ResNet18(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()
        self.in_channels=64
        self.stem=nn.Sequential(
            nn.Conv2d(3,64,3,1,1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.features=nn.Sequential(
            self._make_stage(64,2,1),
            self._make_stage(128,2,2),
            self._make_stage(256,2,2),
            self._make_stage(512,2,2),
        )
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Linear(512,num_classes)

    def _make_stage(self,out_channels,blocks,stride):
        layers=[BasicBlock(self.in_channels,out_channels,stride)]
        self.in_channels=out_channels
        layers+=[BasicBlock(out_channels,out_channels)for _ in range (blocks-1)]
        return nn.Sequential(*layers)

    def forward(self,x):
        x=self.features(self.stem(x))
        return self.fc(torch.flatten(self.pool(x),1))

def make_loaders(batch_size=128):
    mean,std=(0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2616)
    train_tf=transforms.Compose([
        transforms.RandomCrop(32,4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean,std),
    ])
    test_tf=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean,std),
    ])
    train_set=datasets.CIFAR10("data",train=True,download=True,transform=train_tf)
    test_set=datasets.CIFAR10("data",train=False,download=True,transform=test_tf)
    use_cuda=torch.cuda.is_available()
    kwargs={
        "batch_size":batch_size,
        "num_workers":4,
        "pin_memory":use_cuda,
        "persistent_workers":True,
    }
    return (
        DataLoader(train_set,shuffle=True,**kwargs),
        DataLoader(test_set,shuffle=False,**kwargs),
    )

def run_epoch(model,loader,loss_fn,device,optimizer=None,scaler=None):
    training=optimizer is not None
    model.train(training)
    total=total_loss=correct=0
    use_amp=scaler is not None and device.type=="cuda"

    with torch.set_grad_enabled(training):
        for images,labels in loader:
            images=images.to(device,non_blocking=True)
            labels=labels.to(device,non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.float16,enabled=use_amp):
                logits=model(images)
                loss=loss_fn(logits,labels)
            if training:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total_loss+=loss.item()*labels.size(0)
            correct+=(logits.argmax(1)==labels).sum().item()
            total+=labels.size(0)
    return total_loss/total,100*correct/total

def main():
    set_seed(42)
    epochs=100
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader,test_loader=make_loaders()
    model=ResNet18().to(device)
    loss_fn=nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer=torch.optim.SGD(
        model.parameters(),lr=0.1,momentum=0.9,weight_decay=0.0005
    )
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)
    scaler=torch.amp.GradScaler("cuda") if device.type=="cuda" else None
    best_acc=0.00

    for epoch in range(1,epochs+1):
        train_loss,train_acc=run_epoch(
            model,train_loader,loss_fn,device,optimizer,scaler,
        )
        test_loss,test_acc=run_epoch(model,test_loader,loss_fn,device)
        scheduler.step()

        if test_acc>best_acc:
            best_acc=test_acc
            torch.save(model.state_dict(),"resnet18_cifar10_best.pth")
        print(
            f"{epoch:03d}/{epochs} | "
            f"train {train_loss:.3f}, {train_acc:.2f}% | "
            f"test {test_loss:.3f}, {test_acc:.2f}% | "
            f"best {best_acc:.2f}%"
        )
if __name__=="__main__":
    main()
