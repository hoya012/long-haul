---
layout: post
title:  “Learning Transferable Architectures for Scalable Image Recognition”
date:   2018-09-19
description: This is a review of 'Learning Transferable Architectures for Scalable Image Recognition' paper.
comments: true
---

안녕하세요, 오늘은 최근 주목받고 있는 AutoML 관련 논문을 리뷰하려고 합니다. 
논문 제목은 “Learning Transferable Architectures for Scalable Image Recognition” 이며 올해, 2018년 CVPR에 발표된 논문입니다. 
이 논문을 소개하기에 앞서 AutoML에 대해 간략하게 설명을 드리고 그 중 오늘 소개드릴 논문을 집중적으로 설명 드리고, 그 후속 연구들에 대해 간단하게 설명 드리겠습니다.  


이전 포스팅과는 다르게 구현체는 따로 올리지 않았습니다. 
구현체를 올리지 않은 가장 큰 이유는 NASNet 학습을 시키려면 고성능의 GPU 500장으로 4일 동안 돌려야 CIFAR-10에 대해 architecture search가 가능하기 때문입니다. 
이 글을 보시는 대부분의 분들은 GPU 500장을 보유하고 계시지 않을 가능성이 높기 때문에 전체 코드 구현체는 따로 구현하지 않았습니다. (만약 계시다면 심심한 사과의 말씀을 드립니다…) 
또한 이 논문에서 CIFAR-10에 대해 찾은 architecture 구조는 쉽게 구현이 가능하며 코드 또한 많이 공개가 되어있어서 따로 구현체를 올리지 않았습니다. 
NASNet architecture 구현체가 필요하신 분들은 
<a href="https://github.com/titu1994/Keras-NASNet" target="_blank"><b>해당 링크</b></a>
에서 확인이 가능합니다. 

<blockquote> AutoML 이란? </blockquote>

AutoML 이란 간단하게 설명을 드리면 Machine Learning 으로 설계하는 Machine Learning을 의미하며, 크게 3가지 방향으로 연구가 진행이 되고 있습니다. 
1.	Automated Feature Learning
2.	Architecture Search
3.	Hyperparameter Optimization  

Machine Learning에서 중요한 부분을 차지하는 **Feature Extraction**, 혹은 **Feature Engineering** 은 학습 모델에 입력을 그대로 사용하지 않고, 유의미한 feature(특징점)를 추출해서 입력으로 사용하는 방법을 의미합니다. 기존에는 사람이 직접 feature 추출 방법을 정해서 하는 방식이라 각 모델마다 최적의 feature 추출 방식을 찾는 과정에 많은 시간이 소요되었습니다.  Automated Feature Learning은 사람이 직접 실험적으로 feature 추출 방법을 정하는 대신 최적의 feature 추출 방법을 학습을 통해서 찾는 방법을 의미합니다. Deep Learning에서는 feature 추출이 뒤에 소개드릴 2가지 요소에 비해 중요성이 비교적 낮은 편이어서 본 포스팅에서는 자세히 다루지는 않을 예정입니다.

**Architecture Search**란 저희가 익히 알고 있는 AlexNet, VGG Net, ResNet, DenseNet 등 CNN과 LSTM, GRU 등 RNN을 구성하는 network 구조, 즉 architecture를 사람이 직접 하나하나 설계하는 대신 학습을 통해 최적의 architecture를 설계하는 방법을 의미합니다. 주로 강화학습(Reinforcement Learning)이나 유전 알고리즘 등을 이용한 연구들이 최근에 많이 발표되고 있으며, 올해에는 gradient 기반으로 한 DARTS 등 활발하게 연구가 진행되고 있습니다. 본 포스팅에서는 Architecture Search를 주로 다룰 예정이며, 강화학습 기반 방법론을 리뷰할 예정입니다.

**Hyperparameter Optimization**이란 학습을 시키기 위해 필요한 hyperparameter들을 학습을 통해 추정하는 것을 의미합니다. 예를 들어 학습률(learning rate), 배치 크기(mini-batch size) 등 학습에 큰 영향을 주는 hyperparameter들을 기존에는 사람이 하나하나 값을 바꿔서 모델을 학습시켜서 가장 성능이 좋았던 hyperparameter를 사용하는 방식이 주로 사용이 되었다면, AutoML에서는 학습을 통해 최적의 hyperparameter를 추정하는 방법을 제안합니다. 이 부분도 Deep Learning에서 중요한 부분을 차지하고 있지만, 본 포스팅에서는 Architecture Search에 대해서만 다룰 예정입니다. 

<blockquote> 강화학습 기반 Architecture Search 방법론 </blockquote>

### 기존 방법론(NAS)

강화학습 기반으로 최적의 architecture를 찾는 연구는 Barret Zoph, Quoc V. Le의 
<a href="https://arxiv.org/pdf/1611.01578.pdf" target="_blank"><b>“Neural Architecture Search with reinforcement learning”(2017) 논문</b></a>
이 가장 잘 알려져 있습니다. 
줄여서 NAS라고 불리며, network의 architecture를 결정하는 요소들, 예를 들면 각 convolutional layer의 filter size, stride 등의 값을 예측하는 RNN Controller와, 이 RNN Controller가 출력한 값들로 구성한 모델을 학습시켜 얻은 validation accuracy를 reward로 하여 RNN controller를 학습시키는 강화학습 모델로 구성이 되어있습니다.  

RNN controller가 출력한 값을 토대로 생성한 architecture를 타겟 데이터셋으로 처음부터 끝까지 학습을 시킨 뒤 성능을 측정하는 이 모든 과정이, 강화학습 모델에게는 학습을 진행하기 위한 하나의 episode에 해당합니다. 
일반적으로 전체 데이터셋을 이용하여 학습을 시킨 뒤, 성능을 측정하는 과정엔 경우에 따라 다르겠지만 적지 않은 시간이 소모됩니다. 
이러한 긴 과정이 강화학습 입장에서는 단 하나의 episode에 해당하니 강화학습을 통해 성능을 높이기 위해선 굉장히 많은 학습을 반복해야 함을 의미합니다.  

실제로 익히 알려진 데이터셋인 CIFAR-10에 대해 최적의 모델을 찾기까지 **800대의 최상급 GPU** 를 사용하여 **거의 한달** 이 걸렸다고 합니다. 
이렇게 해서 찾은 모델은 ResNet보다는 좋은 성능을 보이고, DenseNet과는 거의 유사한 성능을 보이는 것을 확인할 수 있었습니다. 
하지만 CIFAR-10 데이터셋은 앞선 포스팅에서도 다뤘듯이, 이미지의 크기가 32x32로 작은 편이며 전체 학습 이미지의 개수도 5만장밖에 되지 않습니다. 
만약 ImageNet과 같이 이미지의 크기도 크고, 학습 이미지의 개수도 훨씬 많은 경우에는 최적의 모델을 찾기까지 굉장히 많은 시간이 소모될 것입니다. 
이러한 치명적인 한계가 존재하지만, 강화학습을 기반으로 사람이 design한 모델에 버금가는 모델을 찾을 수 있음을 보인 것 자체로 큰 의미를 가질 수 있다고 생각합니다.  

## NASNet

이제 본격적으로 설명드릴 내용은 위의 연구에서 발전된 연구이며, 위의 단락의 말미에서 언급 드린 한계를 해결하는 방법을 제안하였습니다.
논문에 제목에서도 알 수 있듯이 Transferable한 Architecture Search 방법론을 제안하게 됩니다. 
선행 연구인 NAS와 다르게 **image classification** 을 위한 CNN 구조 탐색으로 범위를 한정 지어서 논문을 작성하였습니다. 
결론을 먼저 말씀드리면 본 논문은 CIFAR-10에서 찾은 최적의 모델의 정보를 활용하여 ImageNet 데이터에 대해 적용하였을 때 사람이 design한 기존 State-of-the art 모델에 버금가는 성능을 보일 수 있음을 보여주고 있습니다. 
또한 선행 연구인 NAS보다 학습에 소요되는 시간이 단축되었습니다. 물론 단축된 시간도 굉장히 긴 편입니다. 

-	NAS 
    - 800 GPU, 28 days (NVIDIA K40 GPU)
-	NASNet
    - 500 GPU, 4days (NVIDIA P100s GPU)

우선 두 방식의 가장 큰 차이점은 Search space의 변화입니다. 
Search space, 즉 탐색 공간의 차이로 인해 많은 것을 얻을 수 있었습니다. 
여기서 말하는 탐색 공간이란 Network 구조를 구성하는 요소를 어떻게 정의하여 탐색하는지를 의미합니다.  

기존 방법론(NAS)의 경우 network를 구성하는 각 layer 하나 하나를 RNN controller를 통해 탐색합니다. 
이 경우 좀 더 network를 구체적으로 정의할 수 있지만 그만큼 탐색 공간이 커지는 장단점이 있습니다. 
실제로 CIFAR-10에 대해 NAS를 적용하여 얻은 network를 보면 규칙성을 찾기 힘들 정도로 거의 매 layer마다 다른 모양의 convolution filter를 사용하는 것을 알 수 있습니다. 
반면 소개드릴 방법론은 Search space를 좁혀서 network 구조를 탐색하는 방법을 제안합니다.  

논문에서는 Convolution Cell이라는 단위를 사용하였는데, 전체 network 대신 이 Cell들 탐색한 뒤, 이 Cell들을 조합하여 전체 network를 설계합니다. 
여러분의 이해를 돕기 위해 쉬운 예시를 들어보겠습니다. 
여러분이 듣기 좋은 음악을 작곡한다고 가정해봅시다. 
실제론 그렇지 않겠지만 비유를 위해 가정을 하나 더 하자면, 만든 음악이 가령 기계(oracle)에 의해 좋은 정도에 따라 0 ~ 100점 척도로 점수가 매겨진다고 가정해봅시다. 
이러한 상황에서 기존 NAS의 접근 방법은 매번 새로운 음악을 만든 뒤 점수 평가를 받은 뒤, 그 음악은 버리고 그 느낌을 기억한 채로 다시 새로운 음악을 만드는 과정을 반복하는 것이라고 표현할 수 있습니다.   

본 논문의 방법은 일정 길이를 갖는 멜로디, 예를 들면 후렴구와 같은 멜로디를 여러 개 만든 뒤에 정해진 순서에 맞게 배치하여 곡을 만들고 점수 평가를 받는 과정으로 비유할 수 있습니다. 
이렇게 되면 점수 평가를 받기까지 걸리는 시간이 처음부터 하나의 노래 전체를 작곡하는 것 보다 짧게 되는 장점이 있고 무엇보다 더 긴 노래를 만들어야 하는 상황이 생겼을 때 만든 후렴구들을 이어 붙이기만 하면 긴 노래를 쉽게 만들 수 있다는 장점이 있습니다. 
물론 같은 구간이 반복되어 작곡의 자유도가 떨어지는 단점이 있습니다. 
하지만 긴 노래를 만들어야 하는 상황에서는 처음부터 끝까지 다 작곡하는 방법(NAS)보다는 훨씬 빠르게 작곡이 가능할 것입니다. 
또한 이 논문의 결론에 의하면 이렇게 여러 멜로디를 이어 붙여도 꽤 그럴싸한 노래를 만들 수 있음을 보여주고 있습니다. 
작곡을 architecture search로 치환하면 NAS와 NASNet의 관계가 되는데, 이 비유는 이번 단락을 다 이해하시면 쉽게 와 닿으실 수 있을 것이라 생각합니다.  

