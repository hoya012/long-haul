---
layout: post
title:  Image Classification with Automatic Mixed-Precision Training PyTorch Tutorial
date:   2020-08-25
description: PyTorch 1.6에서 공식 지원하기 시작한 Automatic Mixed Precision을 실험해볼 수 있는 Image Classification Tutorial을 제작하였습니다.
comments: true
---

안녕하세요, 지난 <a href="https://hoya012.github.io/blog/Mixed-Precision-Training/" target="_blank"><b> “Mixed-Precision Training of Deep Neural Networks” </b></a> 글에 이어서 오늘은 PyTorch 1.6에서 공식 지원하기 시작한 Automatic Mixed Precision Training 기능을 직접 실험해볼 수 있는 Tutorial 코드와 설명을 글로 작성하였습니다. 

실험에 사용한 코드는 <a href="https://github.com/hoya012/automatic-mixed-precision-tutorials-pytorch" target="_blank"><b> 제 GitHub Repository</b></a> 에 올려 두었습니다. 

<blockquote> PyTorch 1.6 – Automatic Mixed Precision </blockquote>  
지난 2020년 7월 말, PyTorch의 새로운 버전인 1.6이 릴리즈 되었습니다. <a href="https://pytorch.org/blog/pytorch-1.6-released/" target="_blank"><b> 릴리즈 노트</b></a> 에는 전반적인 성능 향상과 Memory Profiling 기능, Distributed Training & RPC(Remote Procedure Call), Frontend APU 업데이트, Torchvision, Torchaudio의 신 버전 공개 등 다양한 내용이 담겨있습니다.

그 중 오늘 소개드릴 Automatic Mixed Precision(이하, AMP) 기능이 저는 가장 인상깊었고 결과적으로 이렇게 블로그 글을 작성하게 되었습니다. Mixed Precision Training에 대한 이론적인 내용은 지난 포스팅에서 설명을 드렸으니 자세한 설명은 생략드리고 어떻게 사용할 수 있는지에 초점을 맞춰서 설명을 드리겠습니다. 

기존에도 NVIDIA에서 2018년 개발한 <a href="https://developer.nvidia.com/blog/apex-pytorch-easy-mixed-precision-training/" target="_blank"><b> Apex</b></a> 를 이용하면 PyTorch에서 Mixed Precision Training을 할 수 있었습니다. 별도의 라이브러리 형태로 존재하였는데 이번 PyTorch 1.6에서 NVIDIA와 Facebook 개발자들이 힘을 합해서 공식적으로 지원하게 되었다고 합니다. (갓비디아, 갓페북 선생님들 감사합니다!!)

하나의 하늘 아래 두 개의 태양이 떠있을 순 없겠죠? PyTorch에 AMP 기능이 합쳐지면서 기존 Apex의 AMP 기능은 지원이 중단될 예정이라고 합니다. (With AMP being added to PyTorch core, we have started the process of deprecating apex.amp.) 하지만 Apex의 AMP 기능을 사용하시던 분들도 코드 몇 줄만 수정하시면 바로 사용하실 수 있습니다. 

Torch.cuda.amp 형태로 접근해서 사용할 수 있으며, AMP 기능의 사용 방법에 대한 공식 문서는 다음과 같습니다. 
-	<a href="https://pytorch.org/docs/stable/amp.html" target="_blank"><b> https://pytorch.org/docs/stable/amp.html </b></a>
-	<a href="https://pytorch.org/docs/stable/notes/amp_examples.html" target="_blank"><b> https://pytorch.org/docs/stable/notes/amp_examples.html </b></a>

<blockquote> Image Classification with AMP Tutorials </blockquote>  
공식 문서에서 예제로 올려둔 코드는 다음과 같습니다. 


```python
import torch 
# Creates once at the beginning of training 
scaler = torch.cuda.amp.GradScaler() 
 
for data, label in data_iter: 
   optimizer.zero_grad() 
   # Casts operations to mixed precision 
   with torch.cuda.amp.autocast(): 
      loss = model(data) 
 
   # Scales the loss, and calls backward() 
   # to create scaled gradients 
   scaler.scale(loss).backward() 
 
   # Unscales gradients and calls 
   # or skips optimizer.step() 
   scaler.step(optimizer) 
 
   # Updates the scale for next iteration 
   scaler.update()
```

기존에 사용하던 코드에서 GradScaler 를 선언해주고, data를 model에 넣어주는 부분을 수정하고, loss와 optimizer를 step 시키는 부분만 수정해주면 바로 AMP를 적용할 수 있습니다. 굉장히 간단하죠?

하지만 직접 처음부터 끝까지 돌려볼 수 있는 Tutorial Code가 없어서 직접 Image Classification 데이터셋으로 실험을 해볼 수 있는 Codebase를 제작하였습니다. 

- 코드 주소: <a href="https://github.com/hoya012/automatic-mixed-precision-tutorials-pytorch" target="_blank"><b> https://github.com/hoya012/automatic-mixed-precision-tutorials-pytorch </b></a>

도움이 되셨다면! 구독과 좋아요, 알림 설정까지!.. 는 아니고 스타 하나씩 눌러 주시면 감사드리겠습니다 ㅎㅎ


### 0. Experimental Setup
우선 코드를 다운받으신 뒤 실험에 필요한 라이브러리들을 설치해줍니다. 

```python
pip install -r requirements.txt
``` 

그 뒤 실험에 사용할 데이터셋을 다운받아야 하는데요, 저는 맨날 쓰는 ImageNet, CIFAR 등의 데이터셋 말고 새로운 데이터셋을 사용해보고 싶어서 이것 저것 찾아보다가 작년 Kaggle에서 진행되었던 **Intel Image Classification** 데이터셋이 마음에 들어서 자주 사용하고 있습니다.

<figure>
	<img src="{{ '/assets/img/mixed_precision/10.PNG | prepend: site.baseurl }}" alt=""> 
</figure>

이 데이터셋은 빌딩, 숲, 빙하, 산, 바다, 거리 총 6가지의 class로 구성되어 있고, 150x150 크기의 image 25000장이 제공됩니다. 비교적 구분이 잘 되는 class긴 한데 직접 데이터를 까보면 class가 애매하거나 잘못 labeling 된 image도 존재해서 나름 재미있습니다. 오늘 실험에서는 이 Intel Classification 데이터셋을 사용할 예정입니다.

### 1. Baseline Training
오늘 실험에서는 기존 방식(FP32)과 AMP를 적용하였을 때를 비교할 예정이며, 기본적인 실험 셋팅은 다음과 같이 사용하였습니다.

-	ImageNet Pretrained ResNet-18 from torchvision.models
-	Batch Size 256 / Epochs 120 / Initial Learning Rate 0.0001
-	Training Augmentation: Resize((256, 256)), RandomHorizontalFlip()
- Adam + Cosine Learning rate scheduling with warmup

굉장히 기본적인 기법들만 적용을 하였으며 실험에 사용한 하드웨어(GPU)는 Tensor Core가 없는 Pascal 세대의 GTX 1080 Ti 1개와, Tensor Core가 있는 Turing 세대의 RTX 2080 Ti 1개를 사용하였습니다. 이 두 개의 GPU가 아무래도 많이 사용이 되기도 하고, 저 같은 서민들이 사용할 수 있는 하이엔드 GPU기도 합니다. (GPU 많은 서버 갖고 싶네요..)

<figure>
	<img src="{{ '/assets/img/mixed_precision/11.PNG | prepend: site.baseurl }}" alt=""> 
</figure>

제가 업로드한 코드를 다운 받으시고 데이터셋을 **data** 폴더에 넣어 주시면 준비는 끝입니다. 
학습을 돌리기 위해선 다음과 같은 Command Line 명령어를 입력해주시면 됩니다. 

```python
Python main.py --checkpoint_name baseline
```

### 2. Automatic Mixed Precision Training
다음은 PyTorch 1.6의 AMP 기능을 추가하여 실험을 돌리는 방법을 설명 드리겠습니다. 

제 코드의 **learning/trainer.py** 에서 training loop가 돌아가는데 이 부분에서 torch.cuda.amp 를 붙여서 AMP 기능을 사용하였습니다. 

```python
""" define loss scaler for automatic mixed precision """
scaler = torch.cuda.amp.GradScaler()

for batch_idx, (inputs, labels) in enumerate(data_loader):
  self.optimizer.zero_grad()

  with torch.cuda.amp.autocast():
    outputs = self.model(inputs)
    loss = self.criterion(outputs, labels)

  # Scales the loss, and calls backward() 
  # to create scaled gradients 
  self.scaler.scale(loss).backward()

  # Unscales gradients and calls 
  # or skips optimizer.step() 
  self.scaler.step(self.optimizer)

  # Updates the scale for next iteration 
  self.scaler.update()
```

실제 코드에서는 args.amp 를 통해 amp를 사용할지 말지를 결정하도록 구현이 되어있습니다. 

마찬가지로 AMP 기능을 사용하여 학습을 돌리기 위한 Command Line 명령어는 다음과 같습니다. 간단하죠?

```python
python main.py --checkpoint_name baseline_amp --amp;
```

### 3. Performance Table
다음은 제 GPU (1080 Ti, 2080 Ti)에서 Baseline(FP32)과 AMP를 적용하였을 때의 성능을 살펴볼 예정입니다. Baseline과 AMP 모두 실험 셋팅(모델, 학습 파라미터 등)을 동일하게 하여 진행하였습니다.
성능에는 Test Accuracy, GPU Memory, 전체 학습 시간을 Metric으로 사용하였고, GPU Memory 사용량은 **nvidia-smi**와 **gpustat**을 통해 측정하였습니다.

|   Algorithm  | Test Accuracy |   GPU Memory   | Total Training Time |
|:------------:|:-------------:|:--------------:|:-------------------:|
|  B - 1080 Ti |      94.13    |     10737MB    |         64.9m       |    
|  B - 2080 Ti |      94.17    |     10855MB    |         54.3m       |    
| AMP - 1080 Ti|      94.07    |     6615MB     |         64.7m       |  
| AMP - 2080 Ti|      94.23    |     7799MB     |         37.3m       |

우선 B는 Baseline을 의미하고 AMP는 Automatic Mixed Precision을 의미합니다. 같은 셋팅이지만 GPU가 달라지면 점유하는 GPU Memory도 달라지는 점이 조금 특이한 결과였습니다. 

모든 셋팅에서 거의 비슷한 Test Accuracy를 보여주었고 AMP를 사용하면 GPU Memory를 FP32보다 적게 점유하는 것을 확인할 수 있습니다. 

또한 Tensor Core가 없는 1080 Ti에서는 학습 시간이 거의 줄어들지 않은 반면, Tensor Core가 있는 2080 Ti에서는 학습 시간이 무려 17분이나 줄어들었습니다. 학습 속도가 약 1.46배 빨라진 셈이죠. GPU Memory도 적게 잡아먹으면서요! 

AMP는 손해보는 것이 거의 없으면서 GPU Memory를 적게 잡아먹어서 Batch Size를 키우거나 더 큰 Model을 학습시킬 수 있는게 가장 큰 장점이며, 최신 GPU에서는 학습 시간도 줄어드는 효과를 얻을 수 있어서 거의 필수로 사용해야 하는 기능이라고 생각합니다.

NVIDIA에서 공식적으로 제공하는 <a href="https://github.com/NVIDIA/DeepLearningExamples" target="_blank"><b> DeepLearningExamples </b></a> 에서 다룬 Image Classification, Object Detection, Segmentation, Natural Language Processing, Recommender Systems, Speech to Text, Text to Speech 에서는 대부분 성능이 좋아집니다. 물론 모든 경우에 성능이 좋아지진 않습니다. 공식 문서에서 다루고 있지 않은 Video Classification에 AMP를 적용해봤는데 저는 오히려 GPU Memory가 늘어나서 사용하지 못했습니다. 원인은 아직 밝혀내지 못한 상황입니다..

저는 1개의 GPU로 1개의 모델에 1개의 Loss, 1개의 Optimizer를 사용하는 가장 단순한 과정만 보여드렸는데요, Gradient를 다양하게 다루는 경우에 대한 예시도 공식 문서에서 확인하실 수 있습니다. 
-	<a href="https://pytorch.org/docs/stable/notes/amp_examples.html" target="_blank"><b> PyTorch Automatic Mixed Precision Examples </b></a>

<blockquote> 결론 </blockquote>  
오늘은 지난 글에 이어서, PyTorch 1.6에 공식적으로 지원되기 시작한 Automatic Mixed Precision(AMP) 기능을 직접 사용해보기 위해 Image Classification Codebase를 만들고 실험을 하여 결과를 공유 드렸습니다. 제 실험 환경에서는 1080 Ti에서는 GPU Memory만 줄어들고 학습 시간은 줄어들지 않은 반면, 2080 Ti에서는 GPU Memory, 학습 시간이 모두 줄어드는 결과를 보였는데요, 더 다양한 모델에 대해 실험을 해보면 경향을 자세히 알 수 있을 것 같긴 합니다. 단 5~6줄 정도의 코드만 추가하면 바로 사용할 수 있는데, 정확도 손실도 거의 없으면서 얻을 수 있는 점이 많은 만큼 현재 진행 중이신 연구에 한 번 적용해보시는 것을 권장 드립니다. 저는 앞으로 자주 사용할 것 같네요. 읽어 주셔서 감사합니다!



