---
layout: post
title:  “Tutorials of Object Detection using Deep Learning [4] How to measure performance of object detection”
date:   2018-11-03
description: Deep Learning을 이용한 Object detection Tutorial - [4] How to measure performance of object detection
comments: true
---

안녕하세요, Tutorials of Object Detection using Deep Learning 네번째 포스팅입니다. 
이번 포스팅에서는 Object Detection의 성능에 대해 설명을 드릴 예정입니다. 
정확도를 어떻게 측정하는 여러 metric에 대해 설명을 드리고, 최근 Object Detection 논문들의 정확도가 어느 정도인지 등을 다룰 예정입니다.

이전 포스팅들은 다음과 같습니다.  

<a href="https://hoya012.github.io/blog/Tutorials-of-Object-Detection-Using-Deep-Learning-what-is-object-detection/" target="_blank"><b> Tutorials of Object Detection using Deep Learning: [1] What is object detection </b></a>  
<a href="https://hoya012.github.io/blog/Tutorials-of-Object-Detection-Using-Deep-Learning-first-object-detection-using-deep-learning/" target="_blank"><b> Tutorials of Object Detection using Deep Learning: [2] First Object Detection using Deep Learning </b></a>
<a href="https://hoya012.github.io/blog/Tutorials-of-Object-Detection-Using-Deep-Learning-the-application-of-object-detection/" target="_blank"><b> Tutorials of Object Detection using Deep Learning [3] The application of Object Detection </b></a>

<blockquote> Object Detection의 성능이란? </blockquote>

Object detection 관련 논문을 읽다 보면 초기의 논문들은 대부분 성능에 “정확도” 지표를 사용하고 있는 것을 확인할 수 있습니다. 
Object Detection 뿐만 아니라 다양한 Task의 논문들을 살펴보면 대부분 연구 초기에는 주로 **“정확도”** 라는 지표를 올리기 위한 연구를 수행합니다. 
Object Detection에서는 이 **“정확도”** 라는 지표를 어떻게 나타낼 수 있을까요?

정확도의 계산은 주로 정답(Ground Truth, 이하 GT)와 모델이 예측한 결과(Prediction) 간의 비교를 통해 이루어집니다. 
Image Classification의 경우에는 GT가 이미지의 class인 반면, Object Detection은 이미지의 각 object의 해당하는 Bounding Box와 Box 안의 class를 의미합니다. 
즉 정확도가 높다는 것은 모델이 GT와 유사한 Bounding Box를 예측(Regression)하면서 동시에 Box 안의 object의 class를 잘 예측(Classification)하는 것을 의미합니다. 
즉 class도 정확하게 예측하면서, 동시에 object의 영역까지 잘 예측을 해야 합니다. 

보통 Object Detection 논문에서 사용하는 정확도의 경우 Class를 예측하지 못하면 실패로 간주됩니다. 
Class를 올바르게 예측하였을 때의 Bounding Box의 정확도를 기준으로 정확도를 측정하게 됩니다. 
이제 이 정확도를 어떻게 측정하는지에 대해 설명을 드리겠습니다.

### IoU (Intersection Over Union)

Object Detection에서 Bounding Box를 얼마나 잘 예측하였는지는 IoU라는 지표를 통해 측정하게 됩니다. 
IoU(Intersection Over Union)는 Object Detection, Segmentation 등에서 자주 사용되며, 영어 뜻 자체로 이해를 하면 “교집합/합집합” 이라는 뜻을 가지고 있습니다. 
실제로 계산도 그러한 방식으로 이루어집니다. 
Object Detection의 경우 모델이 예측한 결과와 GT, 두 Box 간의 교집합과 합집합을 통해 IoU를 측정합니다.  

<figure>
	<img src="{{ '/assets/img/object_detection_fourth/fig1.PNG' | prepend: site.baseurl }}" alt=""> 
	<figcaption> [그림 1. IoU 계산 예시] </figcaption>
</figure> 

처음 보신 분들은 다소 생소할 수 있습니다.
IoU는 교집합이 없는 경우에는 0의 값을 가지며, GT와 모델이 예측한 결과가 100% 일치하는 경우에는 1의 값을 가집니다. 
일반적으로 IoU가 0.5(threshold)를 넘으면 정답이라고 생각하며, 대표적인 데이터셋들마다 약간씩 다른 threshold를 사용하기도 합니다. 

-	PASCAL VOC: 0.5
-	ImageNet: min(0.5, wh/(w+10)(h+10))
-	MS COCO: 0.5, 0.55, 0.6, .., 0.95

<figure>
	<img src="{{ '/assets/img/object_detection_fourth/fig2.PNG' | prepend: site.baseurl }}" alt=""> 
	<figcaption> [그림 2. Object Detection 예측 결과에 따른 IoU 값 예시] </figcaption>
</figure> 

### Precision
Precision은 주로 Recall과 사용되며, Image Classification을 공부하시면서 다들 한번쯤은 보셨을 지표입니다.  

<figure>
	<img src="{{ '/assets/img/object_detection_fourth/fig3.PNG' | prepend: site.baseurl }}" alt=""> 
	<figcaption> [그림 3. Precision 계산 방법] </figcaption>
</figure> 

주로 예측된 결과가 얼마나 정확한지를 나타내는데 사용이 되며 계산 식은 그림에 나와있는 것과 같이 True Positive(실제 Positive를 Positive로 잘 예측한 경우, 이하 TP)를 TP와 False Positive(실제 Negative를 Positive로 잘못 예측한 경우, 이하 FP)의 합으로 나눠줘서 계산을 하게 됩니다. 
즉 Precision을 높이기 위해선 모델이 예측 Box를 신중하게 쳐서 FP를 줄여야 합니다.

### Recall

<figure>
	<img src="{{ '/assets/img/object_detection_fourth/fig4.PNG' | prepend: site.baseurl }}" alt=""> 
	<figcaption> [그림 4. Recall 계산 방법] </figcaption>
</figure> 

Recall은 Precision과는 다르게 입력으로 Positive를 주었을 때 얼마나 잘 Positive로 예측하는지를 나타내는데 사용이 되며, 계산식은 그림과 같이 TP를 TP와 False Negative(실제 Positive를 Negative로 잘못 예측한 경우, 이하 FN)의 합으로 나눠줘서 계산을 하게 됩니다. 
즉 Recall을 높이기 위해선 모델 입장에서는 되도록 Box를 많이 쳐서 정답을 맞혀서 FN을 줄여야 합니다. 
그러므로 잘 아시다시피 Precision과 Recall은 반비례 관계를 갖게 되며, 두 값이 모두 높은 모델이 좋은 모델이라 할 수 있습니다. 

### AP (Average Precision), mAP (mean Average Precision)
앞서 말씀드린 것처럼 Precision과 Recall은 반비례 관계를 갖기 때문에 Object Detection에서는 Average Precision, 이하 **AP** 라는 지표를 주로 사용합니다.  

<figure>
	<img src="{{ '/assets/img/object_detection_fourth/fig5.PNG' | prepend: site.baseurl }}" alt=""> 
	<figcaption> [그림 5. Average Precision 계산 방법] </figcaption>
</figure> 

Average Precision의 계산은 Recall을 0부터 0.1 단위로 증가시켜서 1까지(0, 0.1, 0.2, …, 1) 증가시킬 때 필연적으로 Precision이 감소하는데, 각 단위마다 Precision 값을 계산하여 평균을 내어 계산을 합니다. 
즉 11가지의 Recall 값에 따른 Precision 값들의 평균이 AP를 의미하며, 하나의 Class마다 하나의 AP 값을 계산할 수 있습니다.  

이렇게 전체 Class에 대해 AP를 계산하여 평균을 낸 값이 바로 저희가 논문에서 자주 보는 mean Average Precision, 이하 **mAP** 입니다. 

<figure>
	<img src="{{ '/assets/img/object_detection_fourth/fig6.PNG' | prepend: site.baseurl }}" alt=""> 
	<figcaption> [그림 6. mAP와 AP 예시] </figcaption>
</figure> 

위의 그림은 Faster R-CNN 논문에서 결과로 제시한 표를 가져온 것입니다. 
VOC 데이터셋의 20가지 Class마다 구한 AP 값이 표의 오른쪽 column을 의미하며, 그 값들을 평균 낸 값이 바로 표의 mAP 값입니다. 
즉 논문을 쓸 때 mAP 뿐만 아니라 각 Class 마다 AP 값을 제시하기도 하며 Class가 많은 데이터셋(ex, COCO)은 거의 mAP 지표만 제시하는 경우가 많습니다. 

### FPS (Frame Per Second)
보통 논문에서는 주로 정확도에 초점을 둬서 성능을 올리는데, 실제로 사용하는 입장에서는 정확도뿐만 아니라 속도도 굉장히 중요한 issue입니다. 
속도를 나타낼 때는 보통 초당 몇 장의 이미지가 처리 가능한지를 나타내는 FPS를 사용합니다. 
정확도는 각 방법론마다 비교하기가 수월하지만 속도는 어떤 하드웨어를 사용하였는지에 따라, 혹은 어떤 size의 이미지를 사용하였는지에 따라 수치가 달라지기 때문에 상대적인 비교만 가능하다는 단점이 있습니다. 
하지만 정확도만큼이나 중요한 성능 지표이며 최근 논문들에서는 정확도 못지않게 속도에도 집중을 하는 경향이 있습니다.

<blockquote> 최근 논문들의 성능 수준 </blockquote>  
마지막으로 앞서 설명 드린 성능 지표들 중 정확도에 초점을 두고, 최근 논문들이 어느 정도의 수준인지 간단하게 정리를 하려 합니다.  

보통 논문에서 사용되는 benchmark는 VOC07, VOC12, COCO 등이 있으며, 대부분 이 benchmark들 중 하나 이상을 사용하여 성능을 제시합니다. 
대표적인 방법론들과, 그 방법론들의 benchmark 성능은 다음 표에 제시가 되어있으며, 일부 논문들은 학습 방법에 초점을 두거나, 약간 다른 식으로 benchmark를 사용하는 등 공정한 비교가 어려운 점이 있어서 표에서 제외를 하였습니다. 

## performance table

|   Detector   | VOC07 (mAP@IoU=0.5) | VOC12 (mAP@IoU=0.5) | COCO (mAP) | Published In |
|:------------:|:-------------------:|:-------------------:|:----------:|:------------:| 
|     R-CNN    |         58.5        |          -          |      -     |    CVPR'14   |
|   OverFeat   |           -         |          -          |      -     |    ICLR'14   |
|   MultiBox   |         29.0        |          -          |      -     |    CVPR'14   |
|    SPP-Net   |         59.2        |          -          |      -     |    ECCV'14   |
|    MR-CNN    |     78.2 (07+12)    |     73.9 (07+12)    |      -     |    ICCV'15   |
| AttentionNet |           -         |          -          |      -     |    ICCV'15   |
|  Fast R-CNN  |     70.0 (07+12)    |     68.4 (07++12)   |      -     |    ICCV'15   |
| Faster R-CNN |     73.2 (07+12)    |     70.4 (07++12)   |      -     |    NIPS'15   |
|    YOLO v1   |     66.4 (07+12)    |     57.9 (07++12)   |      -     |    CVPR'16   |
|     G-CNN    |         66.8        |     66.4 (07+12)    |      -     |    CVPR'16   |
|     AZNet    |         70.4        |          -          |    22.3    |    CVPR'16   |
|      ION     |         80.1        |         77.9        |    33.1    |    CVPR'16   |
|   HyperNet   |     76.3 (07+12)    |    71.4 (07++12)    |      -     |    CVPR'16   |
|     OHEM     |     78.9 (07+12)    |    76.3 (07++12)    |    22.4    |    CVPR'16   |
|      MPN     |           -         |          -          |    33.2    |    BMVC'16   |
|      SSD     |     76.8 (07+12)    |    74.9 (07++12)    |      -     |    ECCV'16   |
|    GBDNet    |     77.2 (07+12)    |          -          |    27.0    |    ECCV'16   |
|      CPF     |     76.4 (07+12)    |    72.6 (07++12)    |      -     |    ECCV'16   |
|    MS-CNN    |           -         |          -          |      -     |    ECCV'16   |
|     R-FCN    |     79.5 (07+12)    |    77.6 (07++12)    |    29.9    |    NIPS'16   |
|    PVANET    |          -          |          -          |      -     |   NIPSW'16   |
|  DeepID-Net  |         69.0        |          -          |      -     |    PAMI'16   |
|      NoC     |     71.6 (07+12)    |    68.8 (07+12)     |    27.2    |   TPAMI'16   |
|     DSSD     |     81.5 (07+12)    |    80.0 (07++12)    |      -     |   Arxiv'17   |
|      TDM     |          -          |          -          |    37.3    |    CVPR'17   |
|      FPN     |          -          |          -          |    36.2    |    CVPR'17   |
|    YOLO v2   |     78.6 (07+12)    |    73.4 (07++12)    |      -     |    CVPR'17   |
|      RON     |     77.6 (07+12)    |    75.4 (07++12)    |      -     |    CVPR'17   |
|      DCN     |          -          |          -          |      -     |    ICCV'17   |
|     DeNet    |     77.1 (07+12)    |    73.9 (07++12)    |    33.8    |    ICCV'17   |
|   CoupleNet  |     82.7 (07+12)    |    80.4 (07++12)    |    34.4    |    ICCV'17   |
|   RetinaNet  |          -          |          -          |    39.1    |    ICCV'17   |
|  Mask R-CNN  |          -          |          -          |      -     |    ICCV'17   |
|     DSOD     |     77.7 (07+12)    |    76.3 (07++12)    |      -     |    ICCV'17   |
|      SMN     |         70.0        |          -          |      -     |    ICCV'17   |
|    YOLO v3   |          -          |          -          |    33.0    |   Arxiv'18   |
|      SIN     |     76.0 (07+12)    |    73.1 (07++12)    |    23.2    |    CVPR'18   |
|     STDN     |     80.9 (07+12)    |          -          |      -     |    CVPR'18   |
|   RefineDet  |   **83.8 (07+12)**  |  **83.5 (07++12)**  |    **41.8**    |    CVPR'18   |
|    MegDet    |          -          |          -          |      -     |    CVPR'18   |
|    RFBNet    |     82.2 (07+12)    |          -          |      -     |    ECCV'18   |

최근 나온 방법 중 benchmark에서 가장 좋은 성능을 보이고 있는(2018년 11월 기준) Detector는 **RefineDet** 라는 방법이며 VOC07, VOC12, COCO에서 모두 좋은 성능을 보이고 있습니다. 
추후 성능을 올리기 위한 방법들을 소개드릴 예정인데, 그 때 자세히 살펴보도록 하겠습니다.

<blockquote> 결론 </blockquote>  

이번 포스팅에서는 Object Detection에서 어떤 식으로 성능을 평가하는지에 대해 정리를 하였습니다. 
IoU부터 Recall, Precision과 AP, mAP, 속도를 나타내는 FPS 등 각 성능 지표마다 설명을 드렸습니다. 
다음 글은 Object Detection의 성능 중에 정확도를 올리기 위한 연구들에 대해 설명을 드릴 예정이며 비교적 최근에 나온 논문들을 한 편 한 편 리뷰할 예정입니다.
혹시 글을 읽으시다가 잘 이해가 되지 않는 부분은 편하게 댓글에 질문을 주시면 답변 드리겠습니다.

<blockquote> 참고 문헌 </blockquote>  
- <a href="https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/" target="_blank"> IoU 관련 그림 자료 </a>
- <a href="http://darkpgmr.tistory.com/162" target="_blank"> Average Precision 관련 그림 자료 </a>
- <a href="https://arxiv.org/abs/1506.01497" target="_blank"> mean Average Precision 관련 그림 자료 </a>
