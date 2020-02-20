---
layout: post
title:  Pelee Tutorial [2] PeleeNet PyTorch Code Implementation
date:   2019-02-13
description: Mobile Device에서 Classification을 수행하기 위한 최신 연구인 PeleeNet을 PyTorch로 구현하였습니다. 
comments: true
---


안녕하세요, 오늘은 이전 포스팅에 이어서 Pelee 논문의 Classification을 타겟으로 한 **PeleeNet** 을 PyTorch로 구현할 예정입니다.
<a href="https://hoya012.github.io/blog/DenseNet-Tutorial-2/" target="_blank"> DenseNet PyTorch </a>
포스팅과 마찬가지로 효율적인 실습을 위해 **ipynb** 구현체를 통해 진행할 예정이며 google colab을 이용할 예정입니다. 코드의 전반적인 틀이 DenseNet과 유사하므로 DenseNet의 구현체를 이해하셨다면 쉽게 이해하실 수 있을 것이라 생각됩니다.

PeleeNet의 **ipynb** 구현체는 
<a href="https://github.com/hoya012/pytorch-peleenet" target="_blank"> 해당 github repository </a>
에 업로드 해두었으니 다운 받으셔서 사용하시면 됩니다.

<blockquote> Requirements </blockquote>
지난 DenseNet 코드 실습과 마찬가지로 
<a href=" https://colab.research.google.com/" target="_blank"> google colab </a>
을 이용하여 실습을 진행할 예정이며 **ipynb**를 google drive에 업로드한 뒤 colab으로 실행하시면 아무런 셋팅 없이 바로 코드를 실행해볼 수 있습니다.

<blockquote> Code Implementation </blockquote>
코드 실습에는 논문에서 사용한 ImageNet 데이터셋 대신 CIFAR-10 데이터셋을 사용하였습니다. Colab을 이용한 실습이다 보니 대용량의 ImageNet 데이터셋을 사용하면 우선 데이터를 다운로드 받는데도 굉장히 오랜 시간이 소요되고, 또한 학습 자체도 오랜 시간이 소요되기 때문에 이미지 장수가 적은 CIFAR-10 데이터셋으로 실험을 진행하였습니다.
CIFAR-10 데이터셋은 32x32의 크기를 가지고 있기 때문에, ImageNet 데이터셋과 크기를 맞춰주기 위해 224x224로 resize해주는 방식을 사용하였습니다. 

```python
transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

transform_validation = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])


transform_test = transforms.Compose([
        transforms.Resize(224),     
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)

validset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_validation)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)


num_train = len(trainset)
indices = list(range(num_train))
split = int(np.floor(validation_ratio * num_train))

np.random.seed(random_seed)
np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, sampler=train_sampler, num_workers=0
)

valid_loader = torch.utils.data.DataLoader(
    validset, batch_size=batch_size, sampler=valid_sampler, num_workers=0
)

test_loader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=0
)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```
이 부분은 training, validation, test set으로 split하여 각각의 data loader를 생성해주는 부분을 의미하고, 학습 시에는 random crop, random horizontal flip augmentation을 사용하였습니다. 또한 입력 이미지를 CIFAR-10 데이터셋의 평균, 분산으로 normalize를 해주는 전처리 또한 포함이 되어있습니다. 해당 셀까지 실행을 하면 CIFAR-10 데이터셋을 불러와서 torch data loader class를 생성하게 됩니다.

### Module Class 생성
PeleeNet을 구성하는 Module들을 나타내는 class를 각각 생성하고 이를 조립하여 전체 architecture를 구성할 예정입니다. DenseNet과는 다른 Composite function을 사용하기 때문에 PeleeNet에서 사용하는 Composite function을 구현하면 다음과 같습니다.

```python
class conv_bn_relu(nn.Module):
    def __init__(self, nin, nout, kernel_size, stride, padding, bias=False):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(nin, nout, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm2d(nout)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = self.relu(out)

        return out
```
말 그대로 Convolution - Batch Normalization – ReLU 연산을 차례대로 수행하는 역할을 하며, PeleeNet에서 말하는 Convolution 연산은 다 composite function을 의미하고, 굉장히 자주 사용이 됩니다.

```python
class Transition_layer(nn.Sequential):
  def __init__(self, nin, theta=1):    
      super(Transition_layer, self).__init__()
      
      self.add_module('conv_1x1', conv_bn_relu(nin=nin, nout=int(nin*theta), kernel_size=1, stride=1, padding=0, bias=False))
      self.add_module('avg_pool_2x2', nn.AvgPool2d(kernel_size=2, stride=2, padding=0)) 
```
다음 설명드릴 Transition Layer도 간단하게 구현이 가능합니다. 1x1 convolution 이후 2x2 average pooling을 사용하며, Compression hyper parameter인 theta는 논문의 구현과 같이 1을 사용하였습니다. 이 때 1은 아무런 Compression을 수행하지 않는 것을 의미합니다.

```python
class StemBlock(nn.Module):
    def __init__(self):
        super(StemBlock, self).__init__()
        
        self.conv_3x3_first = conv_bn_relu(nin=3, nout=32, kernel_size=3, stride=2, padding=1, bias=False)
        
        self.conv_1x1_left = conv_bn_relu(nin=32, nout=16, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_3x3_left = conv_bn_relu(nin=16, nout=32, kernel_size=3, stride=2, padding=1, bias=False)
        
        self.max_pool_right = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv_1x1_last = conv_bn_relu(nin=64, nout=32, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        out_first = self.conv_3x3_first(x)
        
        out_left = self.conv_1x1_left(out_first)
        out_left = self.conv_3x3_left(out_left)
        
        out_right = self.max_pool_right(out_first)
        
        out_middle = torch.cat((out_left, out_right), 1)
        
        out_last = self.conv_1x1_last(out_middle)
                
        return out_last 
```
위의 Stem Block은 PeleeNet의 가장 앞 부분에서 수행되는 block이며 입력 image의 크기를 줄여주는 역할을 수행합니다. 3x3 conv를 거친 뒤 2갈래로 나뉘어져서 연산을 수행하고 concat을 거친 뒤 마지막으로 1x1 conv를 거치는 방식으로 쉽게 구현이 가능합니다.

```python
class dense_layer(nn.Module):
  def __init__(self, nin, growth_rate, drop_rate=0.2):    
      super(dense_layer, self).__init__()
      
      self.dense_left_way = nn.Sequential()
      
      self.dense_left_way.add_module('conv_1x1', conv_bn_relu(nin=nin, nout=growth_rate*2, kernel_size=1, stride=1, padding=0, bias=False))
      self.dense_left_way.add_module('conv_3x3', conv_bn_relu(nin=growth_rate*2, nout=growth_rate//2, kernel_size=3, stride=1, padding=1, bias=False))
            
      self.dense_right_way = nn.Sequential()
      
      self.dense_right_way.add_module('conv_1x1', conv_bn_relu(nin=nin, nout=growth_rate*2, kernel_size=1, stride=1, padding=0, bias=False))
      self.dense_right_way.add_module('conv_3x3_1', conv_bn_relu(nin=growth_rate*2, nout=growth_rate//2, kernel_size=3, stride=1, padding=1, bias=False))
      self.dense_right_way.add_module('conv_3x3 2', conv_bn_relu(nin=growth_rate//2, nout=growth_rate//2, kernel_size=3, stride=1, padding=1, bias=False))
      
      self.drop_rate = drop_rate
      
  def forward(self, x):
      left_output = self.dense_left_way(x)
      right_output = self.dense_right_way(x)

      if self.drop_rate > 0:
          left_output = F.dropout(left_output, p=self.drop_rate, training=self.training)
          right_output = F.dropout(right_output, p=self.drop_rate, training=self.training)
          
      dense_layer_output = torch.cat((x, left_output, right_output), 1)
            
      return dense_layer_output
```
다음으로 설명드릴 dense layer는 DenseNet의 bottleneck layer에서 약간의 변형을 시킨 layer입니다. 2-way 로 구성이 되고 있으며 제일 처음 생성한 conv_bn_relu 연산들로 구성이 가능합니다. DenseNet의 핵심 아이디어인 **torch.cat** 을 통해 concatenate 해주는 과정도 기존에는 2개의 feature map을 concat했다면, 2-way로 구성되기 때문에 3개의 feature map을 concat해주는 것이 특징입니다. 

```python
class DenseBlock(nn.Sequential):
  def __init__(self, nin, num_dense_layers, growth_rate, drop_rate=0.0):
      super(DenseBlock, self).__init__()
                        
      for i in range(num_dense_layers):
          nin_dense_layer = nin + growth_rate * i
          self.add_module('dense_layer_%d' % i, dense_layer(nin=nin_dense_layer, growth_rate=growth_rate, drop_rate=drop_rate))
```
마지막으로 설명드릴 DenseBlock은 위에서 생성한 dense layer를 각 DenseBlock의 dense layer 개수에 맞게 이어주는 방식으로 구현이 가능합니다.
이 때 각 Dense Block에서 사용되는 dense layer의 개수는 **num_dense_layers** parameter 이며, 이 parameter와 for 문을 이용하면 DenseBlock을 간단히 구현할 수 있습니다. 

### PeleeNet 구성
다음 설명드릴 부분은 위에서 생성한 module을 기반으로 PeleeNet을 구성하는 부분입니다. 
```python
class PeleeNet(nn.Module):
    def __init__(self, growth_rate=32, num_dense_layers=[3,4,8,6], theta=1, drop_rate=0.0, num_classes=10):
        super(PeleeNet, self).__init__()
        
        assert len(num_dense_layers) == 4
        
        self.features = nn.Sequential()
        self.features.add_module('StemBlock', StemBlock())
        
        nin_transition_layer = 32
        
        for i in range(len(num_dense_layers)):
            self.features.add_module('DenseBlock_%d' % (i+1), DenseBlock(nin=nin_transition_layer, num_dense_layers=num_dense_layers[i], growth_rate=growth_rate, drop_rate=0.0))
            nin_transition_layer +=  num_dense_layers[i] * growth_rate
            
            if i == len(num_dense_layers) - 1:
                self.features.add_module('Transition_layer_%d' % (i+1), conv_bn_relu(nin=nin_transition_layer, nout=int(nin_transition_layer*theta), kernel_size=1, stride=1, padding=0, bias=False))
            else:
                self.features.add_module('Transition_layer_%d' % (i+1), Transition_layer(nin=nin_transition_layer, theta=1))
        
        self.linear = nn.Linear(nin_transition_layer, num_classes)
        
    def forward(self, x):
        stage_output = self.features(x)
        
        global_avg_pool_output = F.adaptive_avg_pool2d(stage_output, (1, 1))  
        global_avg_pool_output_flat = global_avg_pool_output.view(global_avg_pool_output.size(0), -1)
                
        output = self.linear(global_avg_pool_output_flat)
        
        return output
```
Stem Block을 거친 뒤 4개의 Stage에서 각각 Dense Block과 Transition Layer를 통과하는 구조이며, 논문의 구현과 같이 4번째 Stage의 Transition layer만 conv_bn_relu 연산을 거치도록 구현을 하였습니다. 마지막으로 global average pooling을 거친 뒤 fully-connected layer를 거치면 PeleeNet이 완성됩니다.

```python
net = PeleeNet()
net.to(device)
```
이제는 생성한 architecture를 맨 처음 생성한 torch.device에 넣어주면 GPU에서 학습을 할 수 있게 됩니다. 
### Architecture Summary
앞서 생성한 PeleeNet architecture를 torchsummary를 통해 확인하실 수 있습니다. 아래 코드 한 줄만 추가해주면 각 연산마다 output shape이 어떻게 변하는지 확인하실 수 있습니다. 
```python
torchsummary.summary(net, (3, 224, 224))
``` 

### Training
```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=0.9)
learning_rate_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch)

for epoch in range(num_epoch):  
    learning_rate_scheduler.step()
    running_loss = 0.0
    
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()
                
        show_period = 100
        if i % show_period ==  show_period-1:    # print every "show_period" mini-batches
            print('[%d, %5d/50000] loss: %.7f, lr: %.7f' %
                  (epoch + 1, (i + 1)*batch_size, running_loss / show_period, learning_rate_scheduler.get_lr()[0]))
            running_loss = 0.0
            
    # validation part
    correct = 0
    total = 0
    for i, data in enumerate(valid_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    print('[%d epoch] Accuracy of the network on the validation images: %d %%' % 
          (epoch + 1, 100 * correct / total)
         )

print('Finished Training')
```
다음 설명드릴 부분은 학습을 시키는 부분이며 마찬가지로 간단히 구현이 가능합니다. 한 가지 차이점이라면, DenseNet은 전체 epoch의 50%, 75% 지점에서 learning rate를 1/10 해주는 방식이었다면, PeleeNet은 cosine annealing 방식을 사용하였고, restart는 사용하지 않았습니다. 이러한 learning rate scheduling은 **torch.optim.lr_scheduler.CosineAnnealingLR** 을 통해 쉽게 구현하실 수 있습니다. 

### Test
Test Set에 대해 정확도를 측정하는 코드는 다음과 같습니다. Test set에 대해 test를 한 뒤 10가지 클래스마다 정확도를 각각 구하고, 또한 전체 정확도를 구하는 과정이 위에 코드로 구현이 되어있습니다. 

```python
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
                
        for i in range(labels.shape[0]):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))            
            
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i])) 
```

<blockquote> 결론</blockquote>
이번 포스팅에서는 google colab을 통해 PeleeNet을 PyTorch로 구현하고 학습을 해보았습니다. 우선 각 module을 구현하고 이를 조립하는 방식으로 구현을 하였으며 이 외에도 구현한 architecture를 summary 하는 과정까지 설명을 드렸습니다. Shake-Shake, DenseNet, PeleeNet 등 여러 Classification 타겟 논문들을 구현해보았는데요, 다음에도 Image를 타겟으로 하는 여러 논문들을 구현하는 실습 포스팅으로 찾아 뵙겠습니다. 감사합니다!
