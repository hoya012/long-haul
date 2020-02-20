---
layout: post
title:  Tutorials of Object Detection using Deep Learning [1] What is object detection?
date:   2018-10-18
description: Deep Learning을 이용한 Object detection Tutorial - [1] What is object detection?
comments: true
---

안녕하세요, 이번 포스팅에서는 딥러닝을 이용한 Object detection 방법론에 대해 작성을 할 예정이며, 여러 편으로 구성이 될 예정입니다. 
이전 포스팅과는 다르게 약간은 간단하면서도 짧게 내용을 정리하는 식으로 글을 작성할 예정입니다. 
이번 포스팅은 What is object detection? 이라는 소제목에서 알 수 있듯이 object detection이 무엇인지에 대해 설명을 드리겠습니다. 

<blockquote> What is object detection? </blockquote>

흔히들 딥러닝을 공부할 때 크게 Computer Vision 혹은 자연어 처리(NLP), 강화학습(RL) 3가지로 나뉘어서 공부를 합니다. 
오늘은 Computer Vision과 관련 있는 대표적인 task인 Object Detection에 대해 간단하게 설명을 하려고 합니다.  

<figure>
	<img src="{{ '/assets/img/object_detection_first/fig1_cv_task.PNG' | prepend: site.baseurl }}" alt=""> 
	<figcaption> [그림 1. 대표적인 Computer Vision task] </figcaption>
</figure> 

우선 Object detection을 알기 전에 Image Classification을 알고 넘어가야 합니다. 
Image Classification은 워낙 공부할 자료도 많고 아마 한번쯤은 튜토리얼을 통해서 구현을 해보셨을 것입니다. 
DNN에 입력으로 이미지를 넣으면 그 이미지에 해당하는 Class를 분류해내는 문제를 Image Classification 이라 부르며,
아래 그림과 같이 타겟으로 하는 전체 class에 대한 확률 값들을 출력하게 됩니다.  

<figure>
	<img src="{{ '/assets/img/object_detection_first/fig2_classification_example.PNG' | prepend: site.baseurl }}" alt=""> 
	<figcaption> [그림 2. Image Classification 예시] </figcaption>
</figure> 

Object Detection은 Image Classification task에 사물의 위치를 Bounding Box로 예측하는 Regression task가 추가된 문제라고 생각하시면 됩니다.  

<figure>
	<img src="{{ '/assets/img/object_detection_first/fig3_detection_example.PNG' | prepend: site.baseurl }}" alt=""> 
	<figcaption> [그림 3. Object Detection 예시] </figcaption>
</figure> 

저희가 일반적으로 Object Detection 이라 부르는 문제는 한 이미지에 여러 class의 객체가 동시에 존재할 수 있는 상황을 가정합니다. 
즉, **multi-labeled classification** (한 이미지에 여러 class 존재)과 **bounding box regression** (box의 좌표 값을 예측) 두 문제가 합쳐져 있다고 생각하시면 됩니다. 
그림 3의 예시는 하나의 이미지에 하나의 object가 존재하는 경우를 보여주지만 그림 1과 같이 하나의 이미지에 여러 객체가 존재하여도 검출이 가능하여야 합니다.  

<blockquote> Object Detection = Multi-labeled Classification + Bounding Box Regression </blockquote>

앞서 제가 게시한 포스팅에서도 나와있듯이 Image Classification 분야는 최근에는 AutoML로 얻은 architecture가 사람이 고안한 architecture의 정확도를 넘어서기도 하였으며, 굉장히 많은 연구가 진행이 되어있는 분야입니다. 

<figure>
	<img src="{{ '/assets/img/object_detection_first/fig4_paper_trend_2019.PNG' | prepend: site.baseurl }}" alt=""> 
	<figcaption> [그림 4. Object Detection 논문 흐름도] </figcaption>
</figure> 

반면 Object Detection 분야는 마치 Image Classification이 굉장히 경쟁적으로 논문이 쏟아지던 2010년 중반처럼 현재도 굉장히 많은 논문이 쏟아져 나오고 있으며, 아직까지 굉장히 높은 성능을 달성하지는 않았습니다. 
그림 4의 해당하는 각 논문들에 대한 정리는
<a href="https://github.com/hoya012/deep_learning_object_detection" target="_blank"><b> 제 github repository</b></a>
에서 확인이 가능합니다.

<blockquote> Object detection, before deep learning </blockquote>

요즘은 object detection은 대부분 deep learning 기반으로 연구가 진행이 되고 있습니다. 
하지만 deep learning이 유행을 끌기 훨씬 전부터 object detection에 대한 연구는 진행되고 있었습니다.  

<figure>
	<img src="{{ '/assets/img/object_detection_first/fig5_detection_milestones.PNG' | prepend: site.baseurl }}" alt=""> 
	<figcaption> [그림 5. Object Detection의 milestones] </figcaption>
</figure> 

위의 그림은 올해 나온 Survey 논문의 그림을 인용한 자료입니다. 
대표적인 Object detection 방법론들이 시간 순서로 나열되어 있으며, 아마 대부분 한번쯤은 들어 보셨을 방법들입니다. 
대표적인 SIFT를 시작으로 얼굴 인식으로 굉장히 유명한 Haar Cascades, BoW(Bag of Words), HOG, SURF, DPM 등 굉장히 주옥 같은 연구들이 Deep learning 이전에 진행되고 있었습니다.  

이러한 방법론들 하나하나 설명 드리면 좋지만 그러면 본 글이 굉장히 길어질 것이 우려가 되기도 하고, 실제로 이러한 방법론들은 설명이 잘 되어있는 자료가 많아서 자세하게 서술은 하지 않을 예정입니다.  

이 방법론들 중 개인적으로 중요하다고 생각하면서 deep learning에서도 사용이 되는 대표적인 2가지 방법을 설명 드리겠습니다.

### Sliding Window

Sliding Window 기법은 딥러닝 이전에 가장 자주 사용되던 방법으로, 다양한 scale의 window를 이미지의 왼쪽 위부터 오른쪽 아래까지 sliding하며 score를 계산하는 방법을 의미합니다. 

<figure>
	<img src="{{ '/assets/img/object_detection_first/fig6_sliding_window.PNG' | prepend: site.baseurl }}" alt=""> 
	<figcaption> [그림 6. Sliding Window 예시] </figcaption>
</figure> 

당연히 하나의 이미지에서 여러 번 score를 계산하여야 하므로 속도 측면에서 비효율적이게 됩니다. 
이러한 문제를 해결하기 위해 Deformable Part Model(DPM) 등이 제안이 되기도 합니다. 
이 Sliding window 기법은 deep learning을 이용한 object detection 방법론 들에서도 심심치 않게 보이므로 알아 두시는 것을 추천합니다.  

### Selective Search (SS)

Selective Search는 비교적 최근(2011년)에 나온 방법론이며 마찬가지로 deep learning을 이용한 object detection 방법론에서 사용이 되었습니다. 
아래 그림이 Selective Search를 잘 보여주고 있습니다.  

<figure>
	<img src="{{ '/assets/img/object_detection_first/fig7_selective_search.PNG' | prepend: site.baseurl }}" alt=""> 
	<figcaption> [그림 7. Selective Search 예시] </figcaption>
</figure> 

영상의 계층적 구조를 활용하여 영역을 탐색하고 그룹화하는 과정을 반복하며 객체의 위치를 proposal 해주는 기능을 수행합니다. 
마찬가지로 관련 자료들을 참고하셔서 잘 알아 두시면 좋을 것 같습니다. 

<blockquote> 결론 </blockquote>  

이번 포스팅에서는 object detection에 대해 설명을 드렸습니다. 
Deep learning을 적용하기 이전에 어떤 방법론들이 주를 이루었는지 간단하게 설명 드렸는데요, 
다음 포스팅에서는 Deep learning을 적용한 최초의 object detection 방법론과 성장 과정들에 대해 간단하게 다루고, object detection의 대표적인  대해 설명 드리겠습니다. 
혹시 글을 읽으시다가 잘 이해가 되지 않는 부분은 편하게 댓글에 질문을 주시면 답변 드리겠습니다. 


<blockquote> 참고 문헌 </blockquote>  
- <a href="http://cs231n.stanford.edu/slides/2016/winter1516_lecture8.pdf" target="_blank"> Stanford 강의자료 </a>
- <a href="https://arxiv.org/pdf/1809.02165.pdf" target="_blank"> Object detection Survey 논문 </a>
- <a href="https://github.com/hoya012/deep_learning_object_detection" target="_blank"> hoya012 github </a>

