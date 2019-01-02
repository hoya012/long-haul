---
layout: post
title:  “Tutorials of Object Detection using Deep Learning [6] Object Detection Multi Scale Testing Method Review”
date:   2019-01-03
description: Deep Learning을 이용한 Object detection Tutorial - [6] Object Detection Multi Scale Testing Method Review
comments: true
---

안녕하세요, Tutorials of Object Detection using Deep Learning 여섯 번째 포스팅입니다. 
이번 포스팅에서는 Object Detection의 성능 중 정확도를 개선하는 방법 중 Multi-Scale Testing기법에 대해 설명을 드릴 예정입니다. 
간단한 아이디어이지만 mAP 성능을 크게 높일 수 있는 방법이며, 2018년 CVPR에서 발표된
<a href="https://arxiv.org/pdf/1711.06897.pdf" target="_blank"><b> “Single-Shot Refinement Neural Network for Object Detection”, RefineDet </b></a>
방법론을 다룬 논문 등에서 이 방식을 통해 State-of-the-art 성능을 달성하였습니다. (물론 2019년에 State-of-the-art는 다른 논문에게 자리를 넘겨주게 됩니다..)
다만 아직까지 이 방법을 다룬 포스팅을 본 적이 없어서 제가 직접 자료를 만들어서 글을 작성하게 되었습니다. 
본 포스팅에서는 Multi-Scale Testing이 무엇인지 설명드릴 예정이며 실제로 어떻게 코드를 구현하면 되는지에 대해 다루도록 하겠습니다.

이전 포스팅들은 다음과 같습니다.  

<a href="https://hoya012.github.io/blog/Tutorials-of-Object-Detection-Using-Deep-Learning-what-is-object-detection/" target="_blank"><b> Tutorials of Object Detection using Deep Learning: [1] What is object detection </b></a>  
<a href="https://hoya012.github.io/blog/Tutorials-of-Object-Detection-Using-Deep-Learning-first-object-detection-using-deep-learning/" target="_blank"><b> Tutorials of Object Detection using Deep Learning: [2] First Object Detection using Deep Learning </b></a>  
<a href="https://hoya012.github.io/blog/Tutorials-of-Object-Detection-Using-Deep-Learning-the-application-of-object-detection/" target="_blank"><b> Tutorials of Object Detection using Deep Learning: [3] The application of Object Detection </b></a>  
<a href="https://hoya012.github.io/blog/Tutorials-of-Object-Detection-Using-Deep-Learning-how-to-measure-performance-of-object-detection/" target="_blank"><b> Tutorials of Object Detection using Deep Learning: [4] How to measure performance of object detection </b></a>  
<a href="https://hoya012.github.io/blog/Tutorials-of-Object-Detection-Using-Deep-Learning-performance-one/" target="_blank"><b> Tutorials of Object Detection using Deep Learning: [5] Training Deep Networks with Synthetic Data Bridging the Reality Gap by Domain Randomization Review </b></a>

<blockquote> Introduction  </blockquote>

오늘 포스팅에서 다룰 Multi-Scale Testing 방식은 이름에서도 유추가 가능하듯이 하나의 이미지에 대해서 여러 scale에서 test를 하는 방법을 의미합니다.
그동안 하나의 이미지를 여러 scale에서 학습을 하는 논문들은 많았었습니다.
대표적으로 SSD, YOLO이 있으며 SSD에서는 여러 scale의 feature map에 대해서 적용을 하였고, YOLO는 학습 데이터의 해상도를 320x320 부터 608x608까지 다양한 scale로 resize를 하여 학습을 시켰습니다. 
이러한 방식들은 학습 단계에 feature map 혹은 input image 자체에 multi scale을 적용하고 있습니다. 

비슷한 방식으로 test 시에도 multi scale을 적용하는 방식이 바로 Multi-Scale Testing 방식입니다. 
하나의 이미지에 대해 여러 번 test를 해야 하기 때문에 전체적인 takt time은 늘어나지만, 정확도를 많이 높일 수 있습니다. 
이제 이 방식이 소개된 논문의 결과를 보여드리고, 구체적인 알고리즘에 대해 설명을 드리겠습니다.

<blockquote> Multi-Scale Testing 효과 예시  </blockquote>

이번 장에서는 Multi-Scale Testing을 적용하였을 때의 효과를 설명을 드리겠습니다. 
우선 Multi-Scale Testing 방식이 소개되었던 “Single-Shot Refinement Neural Network for Object Detection” 논문에서 제안한 RefineDet 방법론에 대해 간단히 설명을 드리겠습니다.
RefineDet은 SSD 계열의 1-stage detector이며 1-stage detector들의 성격에 맞게 정확하면서도 빠른 detection 방식을 제안하였습니다. 
크게 Anchor Refinement Module(ARM)과 Object Detection Module로 구성이 되어있으며 ARM은 negative anchor들을 걸러내면서, 동시에 anchor들의 위치를 조절하는 역할을 수행합니다. 
이를 그림으로 나타내면 다음과 같습니다. 

<figure>
	<img src="{{ '/assets/img/object_detection_sixth/1.PNG' | prepend: site.baseurl }}" alt=""> 
	<figcaption> [그림 1. RefineDet 구조] </figcaption>
</figure> 

RefineDet 자체로도 꽤 좋은 성능을 보이는 것을 논문에서 확인할 수 있습니다. 
1-Stage Detector 답게 빠른 처리 속도를 보이고 있으며, 2-Stage Detector CoupleNet보다 mAP는 낮지만, 1-stage detector 중에서는 가장 높은 mAP를 보이고 있습니다.  
이는 아래의 그림 2에서 확인이 가능하며 PASCAL VOC 데이터셋에 대한 결과를 보여주고 있습니다.

<figure>
	<img src="{{ '/assets/img/object_detection_sixth/2.PNG' | prepend: site.baseurl }}" alt=""> 
	<figcaption> [그림 2. RefineDet PASCAL VOC 성능 비교표] </figcaption>
</figure> 

이 표를 자세히 보면, RefineDet320+, RefineDet512+ 이 두가지 Method에서는 Input size와 #Boxes, FPS는 나와있지 않지만 다른 방법들에 비해 굉장히 높은 mAP를 달성한 것을 확인할 수 있습니다. 
여기 붙은 **+** 표시가 바로 오늘의 주제인 Multi-Scale Testing을 적용하였을 때의 RefineDet 결과를 의미합니다. 
PASCAL VOC 데이터셋 뿐만 아니라 COCO 데이터셋에서도 Multi-Scale Testing을 적용하였을 때 정확도가 큰 폭으로 오르는 것을 확인할 수 있습니다. 

<figure>
	<img src="{{ '/assets/img/object_detection_sixth/3.PNG' | prepend: site.baseurl }}" alt=""> 
	<figcaption> [그림 3. RefineDet COCO 성능 비교표] </figcaption>
</figure> 

위의 그림 3은 COCO 데이터셋에 대한 여러 방법들의 성능을 비교한 표이며, Multi-Scale Testing을 적용하면 2-stage, 1-stage의 제시된 방법들 중에 가장 높은 AP 성능을 보이고 있는 것을 확인할 수 있습니다. 
특히 **“RefineDet320 with VGG-16”** 과 **“RefineDet320+ with VGG-16”** 를 비교하면 Small Object에서는 거의 2배 높은 AP 성능을 보이기도 합니다. 

RefineDet 논문을 통해 Multi-Scale Testing을 이용하면 정확도를 크게 끌어올릴 수 있는 것을 확인할 수 있었습니다. 
또한 최근 공개된 현 시점에서 가장 최신 논문인 M2Det 논문에서도 Multi-Scale Testing을 이용하였을 때의 정확도 향상 자료를 표로 제공하고 있습니다.

<figure>
	<img src="{{ '/assets/img/object_detection_sixth/4.PNG' | prepend: site.baseurl }}" alt=""> 
	<figcaption> [그림 4. M2Det COCO 성능 비교표] </figcaption>
</figure> 

위의 그림 4를 보시면 제가 파란 색으로 하이라이팅 해 놓은 부분이 바로 Multi-Scale Testing을 적용하기 전 후의 결과를 잘 보여주고 있으며 전반적으로 정확도가 큰 폭으로 향상하는 것을 확인할 수 있습니다. 

이제 이 Multi-Scale Testing을 어떻게 구현하여 사용하는 지에 대해 설명을 드리겠습니다.

<blockquote> Multi-Scale Testing Algorithm </blockquote>  
이번 장에서는 Multi-Scale Testing 알고리즘을 간략하게 설명을 드릴 예정이며, RefineDet 저자의 Official Code를 기반으로 정리를 하였습니다. 
적용하는 방식은 정해져 있지 않고 용도에 맞게 조절할 수 있으며, 설명의 편의를 위해 Official Code의 방식 그대로 설명을 하도록 하겠습니다.
저자의 코드 중 Multi-Scale Testing 관련 코드는 다음과 같습니다.

```python
for i in xrange(num_images):
        im = cv2.imread(imdb.image_path_at(i))

        # ori and flip
        det0 = im_detect(net, im, targe_size)
        det0_f = flip_im_detect(net, im, targe_size)
        det0 = np.row_stack((det0, det0_f))

        det_r = im_detect_ratio(net, im, targe_size, int(0.6*targe_size))
        det_r_f = flip_im_detect_ratio(net, im, targe_size, int(0.6*targe_size))
        det_r = np.row_stack((det_r, det_r_f))

        # shrink: only detect big object
        det1 = im_detect(net, im, int(0.6*targe_size))
        det1_f = flip_im_detect(net, im, int(0.6*targe_size))
        det1 = np.row_stack((det1, det1_f))
        index = np.where(np.maximum(det1[:, 2] - det1[:, 0] + 1, det1[:, 3] - det1[:, 1] + 1) > 32)[0]
        det1 = det1[index, :]

        #enlarge: only detect small object
        det2 = im_detect(net, im, int(1.2*targe_size))
        det2_f = flip_im_detect(net, im, int(1.2*targe_size))
        det2 = np.row_stack((det2, det2_f))
        index = np.where(np.minimum(det2[:, 2] - det2[:, 0] + 1, det2[:, 3] - det2[:, 1] + 1) < 160)[0]
        det2 = det2[index, :]

        det3 = im_detect(net, im, int(1.4*targe_size))
        det3_f = flip_im_detect(net, im, int(1.4*targe_size))
        det3 = np.row_stack((det3, det3_f))
        index = np.where(np.minimum(det3[:, 2] - det3[:, 0] + 1, det3[:, 3] - det3[:, 1] + 1) < 128)[0]
        det3 = det3[index, :]

        det4 = im_detect(net, im, int(1.6*targe_size))
        det4_f = flip_im_detect(net, im, int(1.6*targe_size))
        det4 = np.row_stack((det4, det4_f))
        index = np.where(np.minimum(det4[:, 2] - det4[:, 0] + 1, det4[:, 3] - det4[:, 1] + 1) < 96)[0]
        det4 = det4[index, :]

        det5 = im_detect(net, im, int(1.8*targe_size))
        det5_f = flip_im_detect(net, im, int(1.8*targe_size))
        det5 = np.row_stack((det5, det5_f))
        index = np.where(np.minimum(det5[:, 2] - det5[:, 0] + 1, det5[:, 3] - det5[:, 1] + 1) < 64)[0]
        det5 = det5[index, :]

        det7 = im_detect(net, im, int(2.2*targe_size))
        det7_f = flip_im_detect(net, im, int(2.2*targe_size))
        det7 = np.row_stack((det7, det7_f))
        index = np.where(np.minimum(det7[:, 2] - det7[:, 0] + 1, det7[:, 3] - det7[:, 1] + 1) < 32)[0]
        det7 = det7[index, :]

        # More scales make coco get better performance
        if 'coco' in imdb.name:
            det6 = im_detect(net, im, int(2.0*targe_size))
            det6_f = flip_im_detect(net, im, int(2.0*targe_size))
            det6 = np.row_stack((det6, det6_f))
            index = np.where(np.minimum(det6[:, 2] - det6[:, 0] + 1, det6[:, 3] - det6[:, 1] + 1) < 48)[0]
            det6 = det6[index, :]
            det = np.row_stack((det0, det_r, det1, det2, det3, det4, det5, det7, det6))
        else:
            det = np.row_stack((det0, det_r, det1, det2, det3, det4, det5, det7))

        for j in xrange(1, imdb.num_classes):
            inds = np.where(det[:, -1] == j)[0]
            if inds.shape[0] > 0:
                cls_dets = det[inds, :-1].astype(np.float32)
                if 'coco' in imdb.name:
                    cls_dets = soft_bbox_vote(cls_dets)
                else:
                    cls_dets = bbox_vote(cls_dets)
                all_boxes[j][i] = cls_dets
                if vis:
                    vis_detections(im, imdb.classes[j], cls_dets)
```

Scaling을 주는 방식으로 대표적으로 이미지를 Flip(반전) 시키거나 Up Scaling, Down Scaling 하며 이 코드에서는 Flip은 좌우 반전만 사용하였으며, 원본 이미지가 작을수록 Up Scaling을 많이 사용하는 것을 확인할 수 있습니다. 
Small object의 검출 성능을 높이기 위해 Upscaling을 사용하며 이 경우 false positive를 방지하기 위해 일정 크기 이하의 검출 결과만 사용합니다. 

이렇게 여러 scale에서 검출한 예측 box들을 다 모은 뒤에 NMS 등의 bounding box voting을 거쳐서 최종 결과를 출력하게 됩니다. 
이해를 돕기 위해 그림 자료를 준비했습니다. 

<figure>
	<img src="{{ '/assets/img/object_detection_sixth/5.PNG' | prepend: site.baseurl }}" alt=""> 
	<figcaption> [그림 5. Multi-Scale Testing 그림 예시] </figcaption>
</figure> 

그림 5에서는 이미지에서 무당벌레를 detection하는 과정에서 간단하게 4가지 scale의 testing을 거쳐서 검출 성능을 높이는 것을 보여주고 있습니다. 
원본만 이용하였을 때에는 1마리의 무당벌레를 놓쳤지만 Multi-Scale Testing을 이용하여 검출하는 것을 보여주고 있으며 실제로도 전반적인 Recall이 증가합니다. 

Multi-Scale Testing 에는 정해진 공식은 없으며 여러 hyper-parameter들이 존재하므로, 문제 상황에 맞게 적용하면 좋을 것 같습니다. 
예를 들어 takt time issue보다 정확도가 매우 중요한 경우(ex, Challenge)에는 굉장히 다양한 scale에 대해서 적용을 할 수 있고, takt time이 중요한 경우에는 그림 5와 같이 적은 개수의 scale에 대해서 적용을 할 수 있습니다. 
굉장히 단순하면서도 큰 정확도 향상을 얻을 수 있기 때문에 꼭 사용해보시는 것을 추천 드립니다.  

<blockquote> 결론 </blockquote>  

이번 포스팅에서는 단순하면서도 매우 강력한 방법인 Multi-Scale Testing 방법에 대해 소개를 드렸습니다. 
최신 논문 들에서도 효과가 입증된 방식이니 정확도가 중요한 task를 다루고 계신 분들이라면 꼭 적용해보시는 것을 추천 드립니다! 
다음 포스팅에서는 Object Detection을 잘, 효율적으로 학습하기 위해 제안된 여러 논문들을 간단하게 리뷰할 예정입니다. 
다음에도 알찬 내용으로 찾아 뵙겠습니다! 감사합니다. 

혹시 글을 읽으시다가 잘 이해가 되지 않는 부분은 편하게 댓글에 질문을 주시면 답변 드리겠습니다.

<blockquote> 참고 문헌 </blockquote>  
- <a href="http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Single-Shot_Refinement_Neural_CVPR_2018_paper.pdf" target="_blank"> RefineDet 논문 </a>
- <a href="https://arxiv.org/pdf/1811.04533.pdf" target="_blank"> M2Det 논문 </a>
- <a href="https://github.com/sfzhang15/RefineDet/blob/master/test/lib/fast_rcnn/test.py" target="_blank"> RefineDet Official Code – Multi-Scale Testing implementation  </a>
