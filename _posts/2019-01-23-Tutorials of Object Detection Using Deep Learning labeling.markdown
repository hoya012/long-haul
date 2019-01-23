---
layout: post
title:  “Tutorials of Object Detection using Deep Learning [8] Object Detection Labeling Guide”
date:   2019-01-23
description: Deep Learning을 이용한 Object detection Tutorial - [8] Object Detection Labeling에 대한 설명과 Tool 사용법 등을 소개드립니다.
comments: true
---

안녕하세요, Tutorials of Object Detection using Deep Learning 여덟 번째 포스팅입니다. 
앞선 포스팅 들에서는 성능을 올리기 위한 방법들을 주로 다뤘다면, 이번 포스팅에서는 Object Detection을 실제로 사용하는 단계에서 필요한 labeling과 labeling을 도와주는 tool을 소개드릴 예정입니다.  


<blockquote> Object Detection의 Labeling이란? </blockquote>

일반적으로 Object Detection 논문에서 사용하는 데이터셋인 PASCAL VOC, COCO, ImageNet, Open Images 등의 데이터셋은 이미지마다 bounding box annotation이 존재하여 저희가 만약 이러한 데이터셋을 사용하는 경우에는 별도의 labeling을 할 필요가 없습니다. 

<figure>
	<img src="{{ '/assets/img/object_detection_eighth/1.PNG' | prepend: site.baseurl }}" alt=""> 
	<figcaption> [대표적인 Object Detection 데이터셋의 annotation 예시] </figcaption>
</figure> 

하지만 현업에서는 이미 존재하는 데이터셋을 사용할 수도 있지만 이미지를 직접 취득하는 경우가 대부분입니다.
이 경우 필연적으로 각 이미지 마다 bounding box annotation을 제작해야 합니다. 

다행히 Segmentation 등 더 복잡한 task에 비해선 Object Detection은  labeling cost가 적어서 사용자가 직접 labeling을 하기 어렵지 않다는 점이 장점이라면 장점이라 할 수 있습니다. (물론 쉽진 않습니다..)

Object Detection을 위해 필요한 bounding box 정보는 딱 5가지 정보이며 이 외의 정보는 대부분 5가지 정보로 나타낼 수 있습니다.

-	Class index = Category index = Class name
-	Bounding Box 좌표
    - (x_min, x_max, y_min, y_max) or (x_center, y_center, width, height)

이미지의 bounding box 마다 이 5가지 정보를 생성하는 과정을 **“annotation을 제작한다”** 혹은 **“labeling한다”** 고 표현을 하며 오늘은 이 과정을 도와주는 몇몇 프로그램들에 대해 소개를 드릴 예정입니다. 
그리고 가장 단순하면서도 Object Detection에 최적화된 **BBox-Label-Tool** 이라는 프로그램은 사용법도 같이 설명드릴 예정입니다.

<blockquote> BBox-Label-Tool </blockquote>  

이 프로그램은 오로지 Object Detection의 bounding box 제작을 위해 개발된 프로그램이며 가장 배우기 쉽고 사용하기 쉽다고 생각하여 가장 먼저 소개를 드리게 되었습니다.

BBox-Label-Tool (Single Class): https://github.com/puzzledqs/BBox-Label-Tool  
BBox-Label-Tool (Multi Class): https://github.com/jxgu1016/BBox-Label-Tool-Multi-Class

<figure>
	<img src="{{ '/assets/img/object_detection_eighth/2.PNG' | prepend: site.baseurl }}" alt=""> 
	<figcaption> [BBox-Label-Tool UI 예시] </figcaption>
</figure> 

프로그램은 위의 github repository를 통해 clone하여 사용하실 수 있고 Python 2.7 버전과 Python PIL(Pillow)만 설치 되어있으면 바로 사용이 가능합니다. 
처음에는 single class만 지원이 되었는데 다른 개발자에 의해 multi class도 지원이 가능하게 되었습니다. 
그래서 저는 오늘 **multi class를 기준으로** 사용법을 설명드릴 예정입니다.

### 사용법
1. Github repository에서 clone 받은 뒤 프로젝트 폴더의 **class.txt** 를 열고 labeling 하고자 하는 class 정보를 차례로 입력한다.

<figure>
	<img src="{{ '/assets/img/object_detection_eighth/3.PNG' | prepend: site.baseurl }}" alt=""> 
</figure> 

저는 얼굴(face), 글러브(glove) 2가지 class를 labeling 한다고 가정하여 **class.txt** 파일에 face, glove 2가지를 입력하였습니다.

2. 프로그램을 실행한 뒤 Image directory를 입력하고 **Load** 버튼을 클릭한다.
 
<figure>
	<img src="{{ '/assets/img/object_detection_eighth/4.PNG' | prepend: site.baseurl }}" alt=""> 
</figure> 
 
3. 우측 상단에서 원하는 Class를 고르고 **Confirm Class** 를 클릭한 뒤 해당 class에 해당하는 object들을 **마우스로 드래그하여** labeling 한다.
 
<figure>
	<img src="{{ '/assets/img/object_detection_eighth/5.PNG' | prepend: site.baseurl }}" alt=""> 
</figure>  

4. 하나의 이미지에 모든 object를 labeling 한 경우 **Next** 버튼을 눌러준다. 이 경우 해당 이미지의 labeling 결과가 txt 파일로 저장된다.
 
<figure>
	<img src="{{ '/assets/img/object_detection_eighth/6.PNG' | prepend: site.baseurl }}" alt=""> 
</figure> 

설치와 실행 방법이 간단하다는 장점이 있으며 필요한 기능은 다 갖추고 있으므로 Object Detection Task만 하시는 경우에는 이 프로그램을 추천 드립니다!

<blockquote> LabelMe </blockquote>  
Labelme: https://github.com/wkentaro/labelme

이 프로그램도 open source이며 github에서 쉽게 clone하여 다운로드를 받으실 수 있습니다. 

<figure>
	<img src="{{ '/assets/img/object_detection_eighth/7.PNG' | prepend: site.baseurl }}" alt=""> 
	<figcaption> [LabelMe UI 및 기능 예시] </figcaption>
</figure> 

위에서 설명 드린 BBox-Label-Tool과 가장 큰 차이점은 Rectangle 뿐만 아니라 Polygon, Line, Point등 다양한 형태의 도형을 labeling 할 수 있으며 Classification, Segmentation 등 다양한 task의 labeling도 지원을 하고 있습니다. 
이렇게 기능이 다양한 만큼 사용법을 익히기 위해 tutorial을 통해 약간의 사용법을 익힐 필요가 있습니다. 
자세한 사용법은 
<a href="https://github.com/wkentaro/labelme/tree/master/examples/tutorial" target="_blank"> 튜토리얼 </a>
을 통해 확인하실 수 있습니다.

<blockquote> Labelbox </blockquote>  

Labelbox: https://www.labelbox.com/
다음 소개드릴 프로그램은 마찬가지로 다양한 형태의 도형을 지원하고 segmentation labeling 또한 지원을 하고 있습니다. 

<figure>
	<img src="{{ '/assets/img/object_detection_eighth/8.PNG' | prepend: site.baseurl }}" alt=""> 
	<figcaption> [Labelbox UI 및 기능 예시] </figcaption>
</figure> 

웹을 통해 데이터를 업로드하고 annotation을 제작하는 방식이라 인터넷이 연결 되어있는 상황에서만 사용이 가능하며, 특히 데이터 보안 issue가 존재하는 경우에는 사용하기 어려운 단점이 있습니다. 

다만 labeling 결과를 csv, json 뿐만 아니라 일반적으로 많이 사용하는 데이터셋의 format(COCO, VOC, TFRecord) 등으로 export할 수 있어서 사용 중인 코드가 만약 저러한 format을 input으로 사용하도록 짜여 있는 경우 별도의 변환 과정 없이 쉽게 사용이 가능하다는 장점이 있습니다. 
장단점이 확실한 만큼 선택을 하실 때 고려하시면 좋을 것 같습니다.

<blockquote> RectLabel </blockquote>  
RectLabel: https://rectlabel.com/

<figure>
	<img src="{{ '/assets/img/object_detection_eighth/9.PNG' | prepend: site.baseurl }}" alt=""> 
	<figcaption> [RectLabel UI 및 기능 예시] </figcaption>
</figure> 

마지막으로 소개드릴 프로그램도 다양한 형태의 도형을 지원하며 오로지 **Mac OS** 에서만 사용 가능한 프로그램입니다. 
Object Detection과 Segmentation labeling을 지원하며 labeling 결과를 PASCAL VOC, COCO, KITTI, csv 등 다양한 format로 export 할 수 있습니다. 
무엇보다 디자인이 깔끔하고 App store에서 무료로 다운로드가 가능하여 맥북 등을 이용하시는 분들께 추천 드리는 프로그램입니다. 
저도 맥북에서 라벨링을 하는 경우 이 프로그램을 사용합니다. 

<blockquote> 결론 </blockquote>  
이번 포스팅에서는 Object Detection을 실제로 사용할 때 반드시 수행해야 하는 labeling에 대해서 설명을 드리고, labeling을 위해 제작된 여러 프로그램들을 간략하게 리뷰하였습니다. 
각 프로그램마다 특징이 있으니 업무를 수행하실 때 상황에 맞게 사용하시면 좋을 것 같습니다. 읽어 주셔서 감사합니다!

<blockquote> 참고 문헌 </blockquote>  
- <a href="https://github.com/puzzledqs/BBox-Label-Tool" target="_blank"> BBox-Label-Tool single class </a>  
- <a href="https://github.com/jxgu1016/BBox-Label-Tool-Multi-Class" target="_blank"> BBox-Label-Tool multi class </a>  
- <a href="https://github.com/wkentaro/labelme" target="_blank"> LabelMe </a>  
- <a href="https://www.labelbox.com/" target="_blank"> Labelbox  </a>  
- <a href="https://rectlabel.com/" target="_blank"> RectLabel </a>  
