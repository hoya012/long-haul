---
layout: post
title: “Tutorials of Object Detection using Deep Learning [9] Gaussian YOLOv3. An Accurate and Fast Object Detector Using Localization Uncertainty for Autonomous Driving Review”
date: 2019-10-11
description: ICCV 2019에 accept된 “Gaussian YOLOv3. An Accurate and Fast Object Detector Using Localization Uncertainty for Autonomous Driving” 논문에 대한 리뷰를 수행하였습니다.
comments: true
---

안녕하세요, 이번 포스팅에서는 2019년 ICCV에 accept된 논문인 
 <a href="https://arxiv.org/pdf/1904.04620.pdf" target="_blank"><b> “Gaussian YOLOv3: An Accurate and Fast Object Detector Using Localization Uncertainty for Autonomous Driving” </b></a> 
논문에 대한 리뷰를 수행하려 합니다. 

제목에서 알 수 있듯이 
<a href="https://pjreddie.com/media/files/papers/YOLOv3.pdf" target="_blank"><b> YOLOv3 </b></a>
를 기반으로 하여 기존에 Object Detection 모델들이 가지고 있는 문제를 개선하기 위한 연구를 수행한 논문입니다. 또한 자율 주행 환경을 목표로 하였으며 제가 작성했던 
<a href="https://hoya012.github.io/blog/Tutorials-of-Object-Detection-Using-Deep-Learning-the-application-of-object-detection/" target="_blank"><b> “Tutorials of Object Detection using Deep Learning [3] The application of Object Detection” </b></a> 
에서 설명 드렸듯이 자율 주행에서 사용하기 위해선 Object Detection 모델은 실시간 동작이 가능해야하면서 동시에 정확도도 높아야 하는 어려움이 있습니다. 이러한 점에 주목하여 어떻게 연구를 수행했는지 설명을 드리도록 하겠습니다.

<blockquote> Introduction </blockquote>
딥러닝을 이용한 Object Detection 논문들을 정리한 
<a href="https://github.com/hoya012/deep_learning_object_detection" target="_blank"><b> 제 github repository </b></a> 
는 생각보다 큰 관심을 받았는데요, 들어가 보시면 아시겠지만 2010년대 중반에 비해 최근 굉장히 많은 논문들이 쏟아져 나오고 있습니다. 이번 ICCV 2019 에서도 40편이 넘는 Object Detection 논문이 accept이 되었으니 이 분야에 대한 인기를 알 수 있습니다. 


여러 논문들이 나오면서 정확도 지표는 과거 R-CNN, OverFeat 들에 비해 굉장히 높아진 것을 알 수 있습니다. 하지만 Object Detection을 실생활에 사용하는 대표적인 사례인 자율주행 task에서는 실시간 동작이 필수로 뒷받침이 되어야합니다. 일반적으로 실시간 동작이 가능하기 위해선 적어도 30FPS(Frame Per Second), 즉 1초에 30장의 이미지는 처리를 해야 합니다. 그래서 이 논문은 실시간 동작이 가능한 One-Stage Detector인 YOLOv3를 기반으로 연구를 수행하였습니다. 


또한 자율주행에서는 mislocalization, 즉 False Positive가 굉장히 위험한 결과를 초래할 수 있습니다. False Positive로 인해 차량이 갑자기 급정거를 할 수도 있고, 사고로 이어질 수 있기 때문입니다. 하지만 대부분의 논문들은 오로지 mAP 수치만 높이려고 하지, False Positive 자체를 줄이는 것을 목표로 하지는 않습니다. 이 논문에서는 정확도를 높이면서 동시에 자율주행에 맞게 False Positive를 줄이는 방법을 제안하고 있습니다.


이 논문에서 주목한 점은 Object Detection 알고리즘들의 output은 **bounding box coordinate, class probability** 인데, class에 대한 정보는 확률 값으로 나오지만 bounding box coordinate는 deterministic한 값을 출력하기 때문에 bounding box 예측 결과에 대한 불확실성을 알 수 없습니다. 그래서 저자는 이러한 점에 주목하여 **bounding box coordinate에 Gaussian modeling을 적용하고 loss function을 재설계하여 모델의 정확도를 높이고 localization uncertainty를 예측하는 방법** 을 제안하였습니다. 자세한 내용은 아래에서 다루도록 하겠습니다. 

<blockquote> Gaussian YOLOv3 </blockquote>

## Gaussian Modeling
Gaussian YOLOv3를 설명하기에 앞서 기존 YOLOv3에 대해 정리한 그림이 다음과 같습니다. 
<figure>
	<img src="{{ '/assets/img/object_detection_ninth/1.PNG' | prepend: site.baseurl }}" alt=""> 
	<figcaption> [YOLOv3] </figcaption>
</figure> 

Network 구조는 YOLOv3 논문 혹은 다른 양질의 자료들을 확인하시면 되고, YOLOv3의 예측 결과는 각 Grid 마다 4개의 bounding box coordinate, objectness score, class score 가 한 묶음이 되어 하나의 예측 box를 나타내게 됩니다. 여기에서 objectness score는 이 bounding box 안에 object가 있는지 없는지를 나타내는 확률 값이고, class score는 타겟으로 한 class 마다 object가 해당 class일 확률을 나타내는 값입니다. bounding box coordinate를 구성하는 t parameter 들은 예측 box의 중심 좌표, size를 나타내는 값이며 하나의 정해진 값을 나타내게 됩니다. 즉 objectness score, class score는 확률 값을 나타내어 thresholding 등을 통해 낮은 확률을 갖는 값들을 걸러낼 수 있지만, bounding box coordinate는 확률 값이 아니기 때문에, 예측한 box 좌표들이 얼마나 정확한지, 아닌지를 알 수 없습니다. 이를 해결하기 위해 bounding box coordinate를 구성하는 4개의 t parameter들에 Gaussian Modeling을 적용한 모델이 **Gaussian YOLOv3** 입니다. 

<figure>
	<img src="{{ '/assets/img/object_detection_ninth/2.PNG' | prepend: site.baseurl }}" alt=""> 
	<figcaption> [Gaussian Modeling for YOLOv3] </figcaption>
</figure> 

주어진 test input x에 대한 output y의 Single Gaussian Model은 위의 그림의 (1)과 같이 나타낼 수 있으며, YOLO v3 구조에 맞게 적용한 방식은 그림의 (2) ~ (4)에서 확인하실 수 있습니다. 각 coordinate의 mean 값은 예측된 bounding box 좌표를 의미하고 variance 값은 uncertainty를 의미합니다. 즉 variance가 작으면 확실한 bounding box라고 예측을 하는 것이고, variance가 크다면 예측한 bounding box가 불확실한 것을 의미하는 것입니다. Gaussian Modeling은 기존 YOLOv3의 detection layer만 수정을 하면 되고, 연산량 자체도 미미하게 증가하기 때문에 거의 처리 속도가 유지되며 정확도를 높일 수 있습니다. 


(512x512 input 기준, YOLOv3: 99x10^9 FLOPS / Gaussian YOLOv3: 99.04x10^9 FLOPS)


## Reconstruction of loss function
기존 YOLOv3은 bounding box regression에 sum of the squared error loss function을 사용하고 있습니다. 하지만 위에서 설명 드렸듯이 bounding box coordinate에 Gaussian Modeling을 적용하여 Gaussian parameter로 변환하였기 때문에 loss function 또한 재 설계를 하여야 합니다. 

<figure>
	<img src="{{ '/assets/img/object_detection_ninth/3.PNG' | prepend: site.baseurl }}" alt=""> 
	<figcaption> [Loss function of Gaussian YOLOv3] </figcaption>
</figure> 

본 논문에서는 negative log likelihood(NLL) loss를 이용하였으며 이는 위의 그림의 (5)에서 확인하실 수 있습니다. 또한 GT의 bounding box는 위의 그림의 (6)~(9) 를 통해 계산할 수 있으며 각 식의 notation은 논문에 잘 나와있습니다. 

기존 YOLO v3의 sum of the squared error loss는 학습 과정에서 noisy한 bounding box에 대해 아무런 penalty를 줄 수 없지만 재 구성한 loss function을 사용하면 학습 과정에서 noisy 한 bounding box에 대한 uncertainty를 반영하게 되면서 penalty를 부여할 수 있게 됩니다. 즉 데이터셋이 noisy하더라도 uncertainty를 이용하기 때문에 믿을 수 있는 데이터에 집중하는 효과를 얻을 수 있습니다. 또한 전반적인 정확도의 향상도 얻을 수 있게 됩니다. 

## Utilization of localization uncertainty
마지막으로 제안하는 방법은 Detection Criterion에 localization uncertainty를 적용하는 방법입니다. 

<figure>
	<img src="{{ '/assets/img/object_detection_ninth/6.PNG' | prepend: site.baseurl }}" alt=""> 
	<figcaption> [Detection Criterion using localization uncertainty] </figcaption>
</figure> 

예측한 bounding box 안에 어떤 물체가 있는지 계산할 때 objectness score와 class score를 곱하여 사용하는데 여기에 **(1 – uncertainty)** 를 곱해주며, 이를 통해 False Positive를 줄이고 전반적인 정확도를 향상시킬 수 있습니다. Uncertainty는 각 coordinate의 uncertainty를 평균내서 사용을 하였습니다. 
 
<blockquote> 실험 결과 </blockquote>

## 실험 데이터셋 & 환경
본 논문에서는 자율주행 환경에 적합한 KITTI 데이터셋과 BDD 데이터셋을 이용하였습니다.

<figure>
	<img src="{{ '/assets/img/object_detection_ninth/7.PNG' | prepend: site.baseurl }}" alt=""> 
	<figcaption> [KITTI 데이터셋(좌측), BDD 데이터셋(우측) 예시] </figcaption>
</figure> 


 KITTI 데이터셋은 car, cyclist, pedestrian 3가지 class로 구성이 되어있고 7,481장의 학습 데이터셋, 7,518장의 테스트 데이터셋으로 구성이 되어있습니다. 다만 테스트 데이터셋은 GT가 존재하지 않아서 학습 데이터셋의 절반을 검증용으로 사용하였다고 합니다.  

BDD 데이터셋은 bike, bus, car, motor, person, rider, traffic light, traffic sing, train, truck 총 10가지 class로 구성이 되어있고 학습, 검증, 테스트 데이터셋이 각각 7:1:2 비율로 나뉘어져 있습니다. 

또한 각 데이터셋마다 주로 사용되는 IOU threshold 값은 선행 연구들과 동일한 값들을 사용하였고, 학습에 사용한 hyper parameter, 실험 환경 등은 논문에 잘 나와있습니다.

## 실험 결과
<figure>
	<img src="{{ '/assets/img/object_detection_ninth/4.PNG' | prepend: site.baseurl }}" alt=""> 
	<figcaption> [IOU versus localization uncertainty on KITTI and BDD validation sets] </figcaption>
</figure> 

위의 그림은 KITTI, BDD validation dataset 중 car class에 대해 예측된 bounding box들의 IOU에 따른 localization uncertainty를 그래프로 나타낸 결과입니다. IOU가 높을수록 localization uncertainty는 낮은 것을 보이며 두 값이 반비례하는 경향을 보여주고 있습니다. 이를 통해 제안하고 있는 localization uncertainty가 실제로 예측된 bounding box의 confidence를 잘 나타내고 있음을 확인할 수 있습니다.

<figure>
	<img src="{{ '/assets/img/object_detection_ninth/5.PNG' | prepend: site.baseurl }}" alt=""> 
	<figcaption> [Gaussian YOLOv3의 성능 표] </figcaption>
</figure> 

또한 기존 연구들과 성능을 비교한 표는 다음과 같습니다. 기존 방법들 대비 높은 정확도 수치(mAP)를 보이며 특히 baseline으로 삼은 YOLOv3에 비해 약간의 처리 속도 감소(1FPS 내외)만으로 정확도를 크게 높일 수 있는 것이 주목할 만한 결과입니다. 또한 위의 그림의 Table 4를 보면 YOLOv3 대비 True Positive도 늘었고 False Positive가 약 40% 이상 감소된 것을 확인할 수 있습니다. 즉 Bounding box coordinate에 uncertainty를 부여함으로써 False Positive를 줄이겠다는 논문의 목적이 잘 달성된 것을 확인할 수 있습니다. 

다만 개인적으로 궁금했던 부분은 bounding box coordinate에 uncertainty를 부여하면서 얻을 수 있는 효과 중에 하나인 **noisy data에 robust 해진다는 점** 을 실험적으로도 그런 경향을 확인할 수 있는지 였습니다. 다만 이를 입증하기 위해선 noisy label을 담고 있는 자율주행용, 혹은 Object Detection 용 Public 데이터셋이 필요하기 때문에 본 논문에서는 실험을 할 수 없었겠지만, 추후에 이러한 데이터셋이 나온다면 적용해서 실험 결과를 살펴보는 것도 재미있을 것 같습니다.

<blockquote> 결론 </blockquote>
이번 포스팅에서는 ICCV 2019에 accpet된 논문인 ““Gaussian YOLOv3: An Accurate and Fast Object Detector Using Localization Uncertainty for Autonomous Driving” 논문을 리뷰해보았습니다. 자율주행을 타겟으로 실시간 동작을 하면서 동시에 자율주행에 치명적인 결과를 초래할 수 있는 False Positive를 효과적으로 줄이는 방법을 제안하였으며, 대표적인 자율주행 데이터셋에 대해 높은 성능을 보여주고 있음을 확인할 수 있었습니다. 
다음 번에도 재미있는 논문 리뷰 글로 찾아 뵙도록 하겠습니다. 감사합니다!

