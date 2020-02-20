---
layout: post
title:  “Tutorials of Object Detection using Deep Learning [3] The application of Object Detectio
date:   2018-10-26
description: Deep Learning을 이용한 Object detection Tutorial - [3] The application of Object Detection
comments: true
---

안녕하세요, Tutorials of Object Detection using Deep Learning 세번째 포스팅입니다. 
하나의 주제로 연재를 하니까 뭔가 색다르고 재미있네요 😊. 
이번 포스팅에서는 소제목에도 나와있듯이 앞서 다뤄왔던 Object detection이 실제로 어떻게 사용이 될 수 있는지에 대해 설명을 드릴 예정입니다. 
글을 작성하면서 다른 블로그 포스팅들을 참고하고 그림 자료를 인용하였으며, 제 개인적으로 생각해왔던 응용 사례들을 글로 정리해보았습니다.
**이 주제 만큼은 독자분들도 좋은 아이디어가 있으시면 언제든 댓글을 남겨 주시면 좋을 것 같습니다!**

이전 포스팅들은 다음과 같습니다.  

<a href="https://hoya012.github.io/blog/Tutorials-of-Object-Detection-Using-Deep-Learning-what-is-object-detection/" target="_blank"><b> Tutorials of Object Detection using Deep Learning: [1] What is object detection </b></a>  
<a href="https://hoya012.github.io/blog/Tutorials-of-Object-Detection-Using-Deep-Learning-first-object-detection-using-deep-learning/" target="_blank"><b> Tutorials of Object Detection using Deep Learning: [2] First Object Detection using Deep Learning </b></a>

<blockquote> 대표적인 Object detection 적용 사례 </blockquote>

### 자동차

Object detection 사례를 생각하면 가장 먼저 떠오르는 것이 바로 자동차입니다. 
자율주행 자동차에 대한 관심도 높아지고 있고, 최근 나오는 신차들을 보면 대부분 ADAS(Advanced Driver Assistance System)라는 운전자 보조 시스템을 탑재하여 출시되고 있습니다. 
이 ADAS에도 Object Detection이 적용이 될 수 있습니다. 

<figure>
	<img src="{{ '/assets/img/object_detection_third/fig1.PNG' | prepend: site.baseurl }}" alt=""> 
	<figcaption> [그림 1. Object Detection 적용 사례 1. 자동차] </figcaption>
</figure> 

자동차에서 인식하는 대상은 보행자, 다른 차량, 표지판 등이 있으며 해당 인식 정보를 이용하여 자율 주행, ADAS 등에 활용합니다. 
다른 Object Detection 사례 들에서도 중요하겠지만 특히 자동차에서 중요하게 여겨지는 점은 미검(False Negative), **즉 객체가 존재하지만 인식하지 못하는 경우를 최소화해야 한다는 점** 입니다. 
만약에 보행자가 존재하는데 몇 만분의 1 확률로 이를 놓친다면 바로 인명 사고로 이어질 가능성이 높기 때문입니다. 
그러므로 정확도가 굉장히 중요하게 여겨집니다.  

또한 저희가 주로 사용하는 컴퓨팅 파워가 보장되는 고사양의 서버 환경이 아닌, 자동차 내부의 탑재된 **Embedded Device에서 수행이 되어야하기 때문에 연산 량이 큰 모델을 사용하기 어렵습니다.**
그러므로 굉장히 어려운 적용 사례 중에 하나이며 많은 연구가 진행이 되고 있습니다.

### CCTV Surveillance
다음 사례로는 CCTV 감시가 있습니다. 
CCTV는 보통 고정된 위치만을 촬영하기도 하지만 각도를 바꿔가며 주변 환경까지 감시가 가능합니다. 
CCTV에서 촬영하는 영상을 토대로 Object Detection을 통해 이상 현상을 감지하는데 사용이 될 수 있습니다. 

<figure>
	<img src="{{ '/assets/img/object_detection_third/fig2.PNG' | prepend: site.baseurl }}" alt=""> 
	<figcaption> [그림 2. Object Detection 적용 사례 2. CCTV] </figcaption>
</figure> 

예를 들면, 지하철 같은 곳에서 갑자기 사람이 쓰러지거나 싸움이 일어나는 등 이상 현상이 발생하는 경우를 감지해야 하는 경우에 Object Detection을 통해 먼저 사람을 인식하는 과정이 선행이 될 수 있습니다. 
즉, 이상 현상 감지의 전 처리 역할로 사용이 될 수 있습니다. 
또한 백화점 같은 곳에서는 재방문 고객을 감지하는 **re-identification** 등에 마찬가지로 사람을 인식하는 전 처리 역할로 사용될 수 있습니다. 
이 외에도 주차 구역이 아닌 곳에 차량이 세워져 있거나, 갑자기 화재가 발생하는 것을 감지하는 등 다양하게 응용이 가능할 수 있습니다. 

### OCR(Optical Character Recognition)
다음 사례는 OCR입니다. OCR이란 이미지에서 글자를 인식하여 출력하는 것을 의미합니다. 

<figure>
	<img src="{{ '/assets/img/object_detection_third/fig3.PNG' | prepend: site.baseurl }}" alt=""> 
	<figcaption> [그림 3. Object Detection 적용 사례 3. OCR] </figcaption>
</figure> 

이 때 글자 영역을 탐색하는 데 Object Detection이 사용될 수 있습니다. 
OCR은 핀테크(체크카드, 신용카드) 등에서 카드 번호 등을 추출하는 데 사용이 될 수도 있고, 차량 번호판 인식에도 사용이 될 수 있습니다.  

### Aerial Image 분석
다음 사례는 상공에서 촬영한 이미지를 기반으로 유의미한 정보를 추출하는 Aerial Image 분석입니다. 

<figure>
	<img src="{{ '/assets/img/object_detection_third/fig4.PNG' | prepend: site.baseurl }}" alt=""> 
	<figcaption> [그림 4. Object Detection 적용 사례 4. Aerial Image] </figcaption>
</figure> 

주로 드론이나 헬리콥터 등을 이용해 상공에서 원하는 지역을 촬영한 뒤, 해당 지역의 차량, 집, 혹은 나무 등을 검출하는데 Object Detection 기술이 사용될 수 있습니다. 
해당 정보들을 토대로 죽은 나무들의 수를 파악하는 데 사용이 될 수 있고, 해당 지역의 집이 몇 채 있는지, 주차장을 예로 들면 차량이 몇 대 있는지 등을 파악할 수 있습니다.  

### 신체 인식
다음 사례는 흔히 볼 수 있는 신체 인식입니다. 
다들 한번쯤은 경험 해보셨을 얼굴 인식을 비롯해 손을 인식하는 등 신체 인식에서 Object Detection이 사용될 수 있습니다.  

<figure>
	<img src="{{ '/assets/img/object_detection_third/fig5.PNG' | prepend: site.baseurl }}" alt=""> 
	<figcaption> [그림 5. Object Detection 적용 사례 5. 신체 인식] </figcaption>
</figure> 

얼굴 인식을 통해 감정 분석을 하기도 하고 스마트폰의 잠금 화면을 해제하기도 합니다. 
또한 디즈니에서는 영화를 시청하는 관객들의 표정을 분석하기도 합니다. 
얼굴 뿐만 아니라 손을 인식하기도 하는데, 손을 인식하여 제스처를 인식하는데 사용이 되기도 하는 등 신체 인식에도 활발하게 Object Detection이 활용되고 있습니다.

### 제조업
다음 사례는 제조업입니다. 제조업 현장에서 제품의 결함이 존재하는지 검사하는 일을 Vision Inspection이라 부릅니다. 
수아랩을 비롯하여 다양한 회사에서 Vision Inspection 연구를 하고 있습니다.  

<figure>
	<img src="{{ '/assets/img/object_detection_third/fig6.PNG' | prepend: site.baseurl }}" alt=""> 
	<figcaption> [그림 6. Object Detection 적용 사례 6. 제조업] </figcaption>
</figure> 


제조업 현장에서 컨베이어벨트 상으로 제품이 지나갈 때, 제품 상에 부품이 올바르게 놓여 있는지, 혹은 부품이 누락이 되지 않았는지 등을 확인하는 데 Object Detection이 활용될 수 있습니다. 
제조업 특성 상 결함이 굉장히 드물게 발생하지만, 만약 결함을 잡아내지 못한다면 큰 타격을 입을 수 있기 때문에 앞서 자동차의 사례와 같이 미검을 줄이는 것이 굉장히 중요한 분야 중 하나입니다. 

### 스포츠 경기 분석
다음 설명드릴 사례는 스포츠 경기 분석입니다. 
스포츠 경기가 진행되는 동안 선수들의 위치를 분석하는 데 Object Detection이 사용될 수 있습니다.  

<figure>
	<img src="{{ '/assets/img/object_detection_third/fig7.PNG' | prepend: site.baseurl }}" alt=""> 
	<figcaption> [그림 7. Object Detection 적용 사례 7. 스포츠 경기 분석] </figcaption>
</figure> 

축구를 예로 들면, 90분이라는 시간 동안 각 선수들은 굉장히 바삐 움직이며 플레이를 하게 됩니다. 
만약 드론을 이용하거나 축구장 위쪽에 카메라를 설치하여 경기장 전체를 촬영할 수 있다면, 매시간마다 선수들의 위치를 Object Detection을 통해 찾을 수 있고, 이를 통해 경기 내용을 분석하거나 전술을 짜는데 활용할 수 있습니다. 

### 무인 점포
마지막으로 설명드릴 사례는 마트에서 계산을 할 때 사람이 일일이 바코드를 찍는 대신 가판대 위에 물건을 올려 두면 자동으로 인식하여 빠르게 계산할 수 있는 무인 점포입니다. 

<figure>
	<img src="{{ '/assets/img/object_detection_third/fig8.PNG' | prepend: site.baseurl }}" alt=""> 
	<figcaption> [그림 8. Object Detection 적용 사례 8. 무인 점포] </figcaption>
</figure> 

실제로 아마존에서는 **Amazon Go** 라는 이름으로 무인 점포 시스템을 출시하였으며 빠른 시일내에 상용화가 되었으면 하는 기술 중에 하나입니다. 
아직까지 완전히 무인으로 이루어지고 있지는 않지만 기술 발전과 함께 빠르게 성장할 것으로 기대되는 분야입니다. 

<blockquote> 결론 </blockquote>  

이번 포스팅에서는 현실에서 Object Detection이 어떤 방식으로 적용이 될 수 있는지 정리를 하였습니다. 
총 8가지의 사례를 설명 드렸는데요, 이 외에도 더 많은 사례가 존재할 것이라 생각합니다. 
서두에 말씀드린 것처럼 혹시 독자분들께서도 좋은 아이디어가 생기시면 공유해주시면 반영을 하도록 하겠습니다. 
다음 글은 Object Detection의 성능에 대해 설명을 드릴 예정이며 성능을 측정하는 방법들이 무엇이 있는지, 최근 Object Detection 연구들의 성능이 어느 정도인지 등을 다룰 예정입니다.

혹시 글을 읽으시다가 잘 이해가 되지 않는 부분은 편하게 댓글에 질문을 주시면 답변 드리겠습니다.

<blockquote> 참고 문헌 </blockquote>  
- <a href="https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b" target="_blank"> 자동차 관련 object detection 사례 그림 자료 </a>
- <a href="https://www.2020cctv.com/security-solutions/video-analytics/" target="_blank"> CCTV 관련 object detection 사례 그림 자료 </a>
- <a href="https://www.pyimagesearch.com/2017/07/17/credit-card-ocr-with-opencv-and-python/" target="_blank"> OCR 관련 object detection 사례 그림 자료 </a>
- <a href="https://tryolabs.com/blog/2017/08/30/object-detection-an-overview-in-the-age-of-deep-learning/" target="_blank"> Aerial Image 관련 object detection 사례 그림 자료 </a>
- <a href="https://medium.com/@klin505/objective-c-image-face-detection-1f634215311c" target="_blank"> 신체 인식 관련 object detection 사례 그림 자료 </a>
- <a href="https://www.bannerengineering.com/sg/en/solutions/part-quality-inspection.html#all" target="_blank"> 제조업 관련 object detection 사례 그림 자료 </a>
- <a href="https://www.researchgate.net/figure/Moving-object-detection-as-Quy-Tram-2014_fig1_2734373" target="_blank"> 스포츠 경기 분석 관련 object detection 사례 그림 자료 </a>
- <a href="https://medium.com/@madanram/you-only-look-once-aaf841df2c7b" target="_blank"> 무인 점포 관련 object detection 사례 그림 자료 </a>
