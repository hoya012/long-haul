---
layout: post
title:  Image Classification with Stochastic Weight Averaging PyTorch Tutorial
date:   2020-10-22
description: “Averaging Weights Leads to Wider Optima and Better Generalization” 논문에서 제안한 Stochastic Weight Averaging(SWA)을 이용한 Image Classification PyTorch 튜토리얼입니다.
comments: true
---

안녕하세요, 오늘은 지난 <a href="https://hoya012.github.io/blog/SWA/" target="_blank"><b> “Averaging Weights Leads to Wider Optima and Better Generalization 리뷰” </b></a> 글에 이어서, 이 논문에서 제안한 Stochastic Weight Averaging(SWA)을 이용한 Image Classification 튜토리얼을 준비했습니다. 지난 글에서도 말씀드렸지만, PyTorch 1.6에서 SWA를 공식적으로 지원하기 시작해서 이제 손 쉽게 가져다 쓸 수 있는데요, 그래서 오늘 글에서는 짧게 코드를 구현해서 설명 드리고 실험 결과를 소개드릴 예정입니다.

실험에 사용한 코드는 <a href="https://github.com/hoya012/swa-tutorials-pytorch" target="_blank"><b> 제 GitHub Repository</b></a> 에 올려 두었습니다.

<blockquote> PyTorch 1.6 – Stochastic Weight Averaging </blockquote>  
지난 2020년 7월 말, PyTorch의 새로운 버전인 1.6이 릴리즈 되었습니다. <a href="https://hoya012.github.io/blog/SWA/" target="_blank"><b> 릴리즈 노트 </b></a>엔 다양한 기능들이 추가가 되었다고 설명 되어있지만 Stochastic Weight Averaging에 대한 내용이 빠져 있었는데요, 별도의 <a href="https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/" target="_blank"><b> 공식 블로그 글 </b></a>을 통해 SWA의 native 지원을 설명하고 있습니다. 이 글을 보시는 분들은 우선 공식 블로그 글을 보고 오시면 더 이해가 수월하실 것이라 생각됩니다. 공식 블로그 글에서 정리해둔 내용을 한글로 번역하면 다음과 같습니다. 

<figure>
	<img src="{{ '/assets/img/swa/11.PNG | prepend: site.baseurl }}" alt=""> 
</figure>

-	SWA는 standard training (SGD)와 비교했을 때 다양한 computer vision task에서 성능 향상이 가능하다.
-	SWA는 Semi-Supervised Learning, Domain Adaptation의 주요 benchmark에서 SOTA 성능을 달성하게 도와준다.
-	SWA는 language modeling과 강화학습(policy gradient method)에서도 성능 향상이 가능하다.
-	SWA의 후속 연구인 SWAG은 Bayesian model averaging을 approximate할 수 있고, uncertainty calibration에서 SOTA 성능을 달성할 수 있다. 또한 MultiSWAG과 Subspace Inference 등 후속 연구들도 좋은 성능을 보인다.
-	SWA의 Low Precision Training 기법을 적용한 SWALP도 Full-precision SGD training에 준하는 성능을 얻을 수 있다.
-	SWA의 parallel 버전인 SWAP는 큰 batch size와 함께 Neural Network를 학습시켜 학습 속도를 빠르게 할 수 있으며, 27초만에 CIFAR-10에서 94% 정확도를 얻을 수 있다.

이처럼 SWA는 비단 Computer Vision 뿐만 아니라 다양한 task에서도 응용이 될 수 있고, 여러 다양한 학습 기법과도 같이 사용이 될 수 있습니다. 공식 블로그를 참고하시면 위의 내용들을 확인하실 수 있습니다.

<blockquote> Image Classification with SWA Tutorials </blockquote>  
자 이제 Image 분류 문제에 대해 SWA를 적용해보는 시간입니다. PyTorch 1.6 버전으로 코드를 구현하였습니다. 우선 공식 블로그 글에서 예제로 올려 둔 코드는 다음과 같습니다.

```python
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR

loader, optimizer, model, loss_fn = ...
swa_model = AveragedModel(model)
scheduler = CosineAnnealingLR(optimizer, T_max=100)
swa_start = 5
swa_scheduler = SWALR(optimizer, swa_lr=0.05)

for epoch in range(100):
      for input, target in loader:
          optimizer.zero_grad()
          loss_fn(model(input), target).backward()
          optimizer.step()
      if epoch > swa_start:
          swa_model.update_parameters(model)
          swa_scheduler.step()
      else:
          scheduler.step()

# Update bn statistics for the swa_model at the end
torch.optim.swa_utils.update_bn(loader, swa_model)
# Use swa_model to make predictions on test data 
preds = swa_model(test_input)
```

코드가 굉장히 단순하죠? 우선 특별한 부분은 model을 AveragedModel로 묶어주는 부분과, SWA scheduler를 새롭게 선언해주는 부분이 있습니다. 그리고 epoch로 묶여 있는 for loop 안에, epoch가 swa_start보다 크면 swa_model의 parameter를 update해주고, swa_scheduler를 한 step 돌려주는 부분이 있습니다. 마지막으로 학습이 끝난 다음 swa_model의 batch normalization layer를 update해주는 부분이 뒤따릅니다. Batch Normalization의 statistics를 update해주는 부분에 대한 설명은 논문에서 잘 설명이 되어있습니다.

**If the DNN uses batch normal- ization [Ioffe and Szegedy, 2015], we run one additional pass over the data, as in Garipov et al. [2018], to compute the running mean and standard deviation of the activa- tions for each layer of the network with w_SWA weights after the training is finished, since these statistics are not collected during training.**

마지막으로 torch.optim.swa_utils.py 파일을 살펴보시면 각 Class들의 동작 원리를 더 자세히 확인하실 수 있으니 링크를 첨부합니다.
-	Torch.optim.swa_utils.py: https://github.com/pytorch/pytorch/blob/master/torch/optim/swa_utils.py

### 0. Experimental Setup
우선 서론에도 말씀드렸지만 제 GitHub Repository에 있는 <a href="https://github.com/hoya012/swa-tutorials-pytorch" target="_blank"><b> 코드</b></a>를 다운받으신 뒤, 실험에 필요한 라이브러리들을 설치해줍니다.

```python
pip install -r requirements.txt
```

그리고 실험에 사용할 데이터셋도 지난 AMP Tutorial에서 사용했던 **Intel Image Classification** 데이터셋을 또 사용할 예정입니다. 

<figure>
	<img src="{{ '/assets/img/mixed_precision/10.PNG' | prepend: site.baseurl }}" alt=""> 
</figure>

이 데이터셋은 빌딩, 숲, 빙하, 산, 바다, 거리 총 6가지의 class로 구성되어 있고, 150x150 크기의 image 25000장이 제공됩니다. 

### 1. Baseline Training
우선 SWA을 사용하기 전에 기존 방식으로 학습을 시켜서 Baseline 성능을 측정할 예정입니다. Baseline 실험 셋팅은 다음과 같습니다.

- ImageNet Pretrained ResNet-18 from torchvision.models
- Batch Size 256 / Epochs 120 / Initial Learning Rate 0.0001
- Training Augmentation: Resize((256, 256)), RandomHorizontalFlip()
- Adam + Cosine Learning rate scheduling with warmup

최대한 단순하면서 자주 사용되는 기법들을 채택하였습니다. Baseline Training을 돌리기 위한 커맨드 라인 명령어는 다음과 같습니다.

```python
Python main.py --checkpoint_name baseline
```

### 2. Stochastic Weight Averaging Training
다음은 PyTorch 1.6의 SWA 기능을 기존에 사용하던 Image Classification 코드 베이스에 적용하는 과정을 설명 드리겠습니다. 

```python
from torch.optim.swa_utils import SWALR
""" define model and learning rate scheduler for stochastic weight averaging """
swa_model = torch.optim.swa_utils.AveragedModel(model)
swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr)
```

우선 **main.py** 에서 위의 PyTorch 공식 블로그 글의 예제와 같이 model을 AveragedModel로 묶어주고, SWALR scheduler를 선언해줍니다. 

```python
for batch_idx, (inputs, labels) in enumerate(data_loader):
  if not args.decay_type == 'swa':
        self.scheduler.step()
  else:
      if epoch <= args.swa_start:
          self.scheduler.step()

if epoch > args.swa_start and args.decay_type == 'swa':
  self.swa_model.update_parameters(self.model)
  self.swa_scheduler.step()
```

그 뒤, **learning/trainer.py** 에 있는 Data Loader를 enumerate하는 for loop안에 SWA를 추가해줍니다. 우선 현재의 epoch이 SWA를 언제부터 시작할 지 알려주는 **args.swa_start** 보다 커지고, decay type이 swa면 swa_model의 parameter와 swa_scheduler를 update해주면 됩니다. 거의 똑같이 구현을 할 수 있습니다. 

```python
swa_model = swa_model.cpu()
torch.optim.swa_utils.update_bn(train_loader, swa_model)
swa_model = swa_model.cuda() 
```

마지막으로 다시 **main.py**로 돌아와서 swa_model의 batch normalization parameter들을 update해주면 됩니다. 제 구현체에서는 data loader가 CPU로 받아온 뒤, training 혹은 evaluation loop에서 cuda tensor로 변환을 해주기 때문에 위와 같이 cpu와 gpu를 왔다갔다하는 번거로움이 있습니다. 놀랍게도 이게 끝입니다. 거의 이번에도 10줄 정도만 추가했는데 끝이 났죠? 정말 간단하게 기존 코드에 SWA를 추가할 수 있습니다. 

주요 hyper-parameter는 언제부터 SWA를 시작할 지 결정하는 **args.swa_start** 와, SWA의 Learning Rate인 **args.swa_lr**  이며, 이 2가지 값을 적당히 조절해가며 실험을 해볼 예정입니다.

SWA를 적용하여 실험하기 위한 커맨드 라인 명령어는 다음과 같습니다.

```python
python main.py --checkpoint_name swa --decay_type swa --swa_start 90 --swa_lr 5e-5;
```

### 3. 실험 결과
이제 Baseline Training과 SWA Training의 실험 결과를 설명 드리겠습니다.

|   Algorithm  | Test Accuracy | 
|:------------:|:-------------:| 
|  Baseline |      94.10    |
|  SWA_90_0.05|      80.53    |
| SWA_90_5e-4|      93.87    |
|  SWA_90_1e-4|      94.20    |
| SWA_90_5e-5|    **94.57**  |
| SWA_90_1e-5|      94.23    |
| SWA_75_5e-5|      94.27    |
| SWA_60_5e-5|      94.33    |

실험 결과, 원래 논문에서는 SGD에 큰 값의 initial learning rate을 사용했었는데 저는 이번에 Adam을 사용하면서 1e-4의 작은 initial learning rate 값을 사용했습니다. 이 때, 너무 큰 swa_lr을 사용하면 학습이 제대로 되지 않는다는, 어떻게 보면 당연하지만 해봐야 아는 실험을 진행해봤습니다.

그 뒤로는 swa_lr 값을 5e-4 부터 절반씩 줄여가며 1e-5까지 실험을 해봤고, 그 중에 5e-5 일 때가 가장 test accuracy가 높았습니다. 그리고, 1e-4보다 작은 swa_lr을 사용하면 다 Baseline 보다 성능이 좋았습니다. 

마지막으론, SWA을 시작하는 epoch을 전체 120 epoch 중에 기존에는 90 epoch부터 시작을 했는데, 시작 시점을 75, 60 epoch으로 바꿨더니 Baseline보다 성능이 좋긴 했지만, 90 epoch을 사용하였을 때보단 좋지 않은 결과를 보였습니다.

<blockquote> 결론 </blockquote>  
오늘은 지난 글에 이어서, PyTorch 1.6에 공식적으로 지원되기 시작한 Stochastic Weight Averaging(SWA) 기능을 Image Classification Codebase에 구현하여 실험을 진행하고, 실험 결과를 공유 드렸습니다. 이번에도 PyTorch에서 잘 구현을 해준 덕분에 10줄 내외의 코드만 추가하면 쉽게 사용할 수 있었으며, SGD Training 뿐만 아니라 Adam Optimizer를 사용하여도 SWA를 잘 사용하면 Test Accuracy를 높일 수 있다는 실험 결과도 보여드렸습니다. 저는 이 기능도 이제 자주 사용하게 될 것 같네요! 긴 글 읽어 주셔서 감사드리고, 궁금한 점 있으면 언제든 댓글 남겨주세요! 

