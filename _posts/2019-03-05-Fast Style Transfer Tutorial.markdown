---
layout: post
title:  “Fast Style Transfer PyTorch Tutorial”
date:   2019-03-05
description: Fast Style Transfer를 PyTorch로 쉽게 돌려볼 수 있는 tutorial 입니다. 
comments: true
---

안녕하세요, 오늘은 Style Transfer을 PyTorch로 실습하는 과정을 설명드릴 예정입니다. 
고흐풍을 다른 그림에 입히는 예제는 다들 인터넷에서 한번쯤은 보셨을 것입니다. 
저 또한 인터넷으로만 보다가 직접 학습시키고 test를 해보고 싶어서 코드를 찾다가 
<a href="https://github.com/pytorch/examples" target="_blank"> pytorch의 example repository </a>
에 잘 짜여진 code가 있어서 이전 포스팅들과 마찬가지로 효율적인 실습을 위해 **ipynb** 형태로 가공하였습니다.

실습 코드는 
<a href="https://github.com/hoya012/fast-style-transfer-tutorial-pytorch" target="_blank"> 해당 github repository </a>
에 업로드 해두었으니 다운 받으셔서 사용하시면 됩니다.

<blockquote> 논문 간단 소개 </blockquote>
오늘 다룰 논문은 
<a href="https://arxiv.org/pdf/1603.08155.pdf" target="_blank"> Perceptual Losses for Real-Time Style Transfer and Super-Resolution (2016, ECCV)</a>
라는 논문이며, 논문에 제목에서 알 수 있듯이 **Perceptual loss**라는 것을 제안하였고 Real-Time으로 동작할 만큼 빠른 방법을 제안하였습니다.

### 기존 방법과 차이점
Style Transfer의 초기 논문이라 부를 수 있는 
<a href="https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf" target="_blank"> Image Style Transfer Using Convolutional Neural Networks (2016, CVPR) </a>
은 pretrained된 network에 content image와 style image를 쌍으로 넣어줘서 style transfer를 하는 방식이다보니 매번 content image가 바뀔 때 마다 많은 연산을 수행하여야 하는 단점이 있습니다.

본 논문은 이러한 문제를 해결하기 위해 network에 style image 1장을 학습시키고 그 network를 그대로 이용하는 방법을 제안하였습니다.
즉 여러 장의 content image로 style transfer(inference)를 할 때 기존 방법처럼 매번 재학습을 시키지 않고 단순히 inference만 하면 되기 때문에 Real-Time으로 동작이 가능하다는 장점이 있습니다.

<figure>
	<img src="{{ '/assets/img/fast_style_transfer/1.PNG' | prepend: site.baseurl }}" alt="" width="500"> 
</figure>

위의 그림은 본 논문의 transformation network 구조를 보여주고 있습니다. 오늘 실습에서는 위의 구조를 구현하고 학습을 돌리고 테스트를 해볼 예정입니다.

논문의 자세한 내용이 궁금하신 분들은 본 논문을 읽어보시거나, 본 논문을 리뷰해놓은 다른 blog 글들을 참고하시면 더 이해가 잘 되실 것이라 생각합니다.

<blockquote> Requirements </blockquote>
지난 PyTorch ipynb 실습과 마찬가지로 
<a href=" https://colab.research.google.com/" target="_blank"> google colab </a>
을 이용하여 실습을 진행할 예정이며 **ipynb**를 google drive에 업로드한 뒤 colab으로 실행하시면 아무런 셋팅 없이 바로 코드를 실행해볼 수 있습니다.

또한 이번에는 이전 실습들과는 다르게 준비해야 할 것들이 많습니다. 
그만큼 컨텐츠가 다양해졌다는 것을 의미하며, 이번 실습을 통해 얻어가실 수 있는 것들은 다음과 같습니다.

- Google Drive 연동 (2GB 이상의 용량 필요)
- COCO Dataset 다운로드 및 loading
- Transfer Learning을 위한 checkpoint 저장
- Style Transfer 결과를 이미지 혹은 동영상으로 저장

이번 실습에는 제가 즐겨하는 게임인 배틀그라운드의 플레이 영상을 이용할 예정입니다.
사실 이 포스팅을 작성해야겠다고 생각한 계기도 게임을 하다가 문득 떠오른 아이디어에서 출발하였으며, 혹시 이 게임을 잘 모르시는 분들을 위해 간략하게 소개를 드리면 다음과 같습니다.

<figure>
	<img src="{{ '/assets/img/fast_style_transfer/2.PNG' | prepend: site.baseurl }}" alt="" width="500"> 
</figure>

100인의 플레이어가 전투에 필요한 물자들을 얻고 최종 1인을 향해 플레이하는 생존 게임이며 에란겔(도심, 산), 미라마(사막), 사녹(열대우림), 비켄디(설원) 크게 4가지 테마의 맵이 존재합니다.
그래서 저는 각 맵 간의 style transfer를 해보면 재미있겠다는 호기심 하나로 이번 실습 코드를 준비해보았으며, 말미에 데모를 통해 얼마나 style이 잘 transfer 되는지 보여드릴 예정입니다.

<blockquote> Code Implementation </blockquote>
pytorch example 코드를 기반으로 여러분들이 쉽게 실습을 해보실 수 있도록 정리한 **ipynb** 코드를 하나하나 설명드리도록 하겠습니다.

### 1. Google Drive 연동
```python
from google.colab import drive
drive.mount("/content/gdrive")
```

google drive는 단 두줄로 연동이 가능하며 해당 code block을 실행하시고 권한 요청을 승인하시면 저희 코드에서 google drive에 접근이 가능하게 됩니다.
즉, google drive에 있는 파일을 read 할 수도 있고, 실습 결과물들을 google drive에 저장을 할 수도 있게 됩니다. 

### 2. COCO dataset 다운로드 & Style Image 준비
본 논문에서는 network 학습을 위해 COCO 2014 training dataset을 사용하였는데, 용량이 13GB로 큰 편이라 대부분 Google Drive를 무료로 사용 중이신 분들은 용량이 15GB로 제한되기 때문에 실습에 무리가 있을 수 있습니다.
그래서 저는 비교적 용량이 적은 COCO 2017 validation dataset을 이용하였으며, 대신 training epoch을 키워주는 방식을 사용하였습니다. 용량이 많으신 분들은 원 논문처럼 COCO 2014 training set을 사용하시는 것을 권장합니다.

- COCO 2014 training: 약 80000장 / 13GB
- COCO 2017 validation: 약 5000장 / 1GB --> epoch을 16배 키워서 사용할 예정

COCO 2017 validation set은
<a href="http://images.cocodataset.org/zips/val2017.zip" target="_blank"> 해당 링크 </a>
를 클릭하시면 다운받으실 수 있으며, 다운 받으셔서 압축을 해제하신 후 google drive에 업로드하시면 됩니다.
혹은 압축파일 자체를 업로드하시고 google drive 내에서 압축 해제를 하셔도 무방합니다.

학습에 필요한 COCO dataset이 준비가 되셨다면, 이제는 style image를 준비하시면 됩니다.
저는 배틀그라운드의 4가지 테마의 맵 중에 설원 테마인 비켄디의 플레이 이미지 1장을 준비하였습니다. 

<figure>
	<img src="{{ '/assets/img/fast_style_transfer/3.PNG' | prepend: site.baseurl }}" alt="" width="500"> 
</figure>

마찬가지로 style image도 google drive에 업로드를 하신 뒤에 잘 업로드가 되었는지 확인하실 수 있습니다.

```python
style_image_location = "/content/gdrive/My Drive/Colab_Notebooks/data/vikendi.jpg"

style_image_sample = Image.open(style_image_location, 'r')
display(style_image_sample)
```

style image가 제대로 출력이 되지 않으면 아마 경로가 잘못되었을 가능성이 높으므로 경로를 잘 확인해주시면 됩니다.



