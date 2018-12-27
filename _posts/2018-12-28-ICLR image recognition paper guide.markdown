---
layout: post
title:  “ICLR 2019 image recognition paper list guide”
date:   2018-12-28
description: ICLR 2019 논문 중 이미지 인식과 관련있는 논문 리스트에 대해 제 주관적으로 정리하였습니다.
comments: true
---

안녕하세요, 이번 포스팅에서는 2019년 5월 6일 ~ 9일 미국 뉴올리언스에서 개최될  
 <a href="https://nips.cc/Conferences/2018/Dates" target="_blank"><b> ICLR 2019 </b></a> 
학회의 논문 중에 이미지 인식과 관련이 있는 논문 28편에 대해 제 주관적으로 리스트를 정리해보았습니다. 
아직 학회가 많이 남았지만 미리 읽어 보기 좋도록 리스트를 추리는 것을 목표로 글을 작성하였으며,
전체 accepted paper가 500편이다보니 하나하나 읽어보는 것은 불가능하여서, 제가 제목만 보고 재미있을 것 같은 논문 위주로 정리를 해보았습니다. 

**당부드리는 말씀은 제가 정리한 논문 리스트에 없다고 재미 없거나 추천하지 않는 논문은 절대 아니고 단지 제 주관에 의해 정리된 것임을 강조드리고 싶습니다.**

<blockquote> ICLR 2019 Paper Statistics </blockquote>
지난번 소개드렸던 NeurIPS 처럼 ICLR 도 굉장히 인기있는 학회인데요, 이 학회에는 매년 몇 편의 논문이 accept되는 지 조사를 해보았습니다. 

<figure>
	<img src="{{ '/assets/img/iclr_2019/iclr_acceptance.PNG' | prepend: site.baseurl }}" alt=""> 
	<figcaption> [최근 3년간 NIPS acceptance rate 비교] </figcaption>
</figure> 

매년 제출되는 논문 편수도 증가하고 있고, 그에 따라서 accept되는 논문들의 편수도 증가를 하고 있습니다. 
불과 2년전에 비해 규모가 약 3배가량 커졌으며 약 30% 대의 acceptance rate를 보이고 있는 것을 확인할 수 있습니다.

또한 ICLR는 특이하게 Open-Review 방식으로 review가 진행되어서 각 논문마다 reviewer로부터 몇 점을 받았는지 확인할 수 있습니다.
이를 잘 정리해놓은 자료를 발견하여서 
<a href="https://github.com/shaohua0116/ICLR2019-OpenReviewData" target="_blank"><b> 이 자료  </b></a> 
를 토대로 ICLR 2019를 분석하였습니다. 

우선 10점 만점의 점수 중에 accepted paper는 평균 6.6점 정도의 rating을, rejected paper는 평균 4.7점 정도의 rating을 보이고 있으며, 오늘 소개드릴 논문마다 평균 점수도 같이 기재할 예정입니다.
또한 **theory**, **robustness**, **graph neural network** 등의 키워드를 가진 논문들이 평균적으로 점수가 높았다고 합니다. 

참고로 올해는 총 24편의 oral paper와 476편의 poster 총 500편 논문이 accept되었으며, 저는 오늘 그 중 28편의 논문을 소개드리고자 합니다.

<blockquote> Image Recognition 관련 논문 소개 </blockquote>  

앞서 말씀드렸듯이 ICLR 2019에 accept된 논문을 모두 다 확인하기엔 시간과 체력이 부족하여서, 간단하게 제목만 보면서 제가 느끼기에 재미가 있을 것 같은 논문들을 추려보았습니다.
총 28편의 논문이며, 6편의 oral paper, 22편의 poster paper로 준비를 해보았습니다. 또한 각 논문마다 abstract를 읽고 논문을 간단히 정리해보았습니다.

##  <a href="https://openreview.net/pdf?id=B1xsqj09Fm" target="_blank"><b> 1.	Large Scale GAN Training for High Fidelity Natural Image Synthesis (Oral, )  </b></a>  
- 512x512 크기의 이미지와 같이 high resolution 이미지를 생성하는 generative model “BigGAN” 제안. 

<figure>
	<img src="{{ '/assets/img/iclr_2019/1_1.PNG' | prepend: site.baseurl }}" alt=""> 
	<figcaption> [본 논문의 결과 그림 예시] </figcaption>
</figure> 

##  <a href="https://openreview.net/pdf?id=Bygh9j09KX" target="_blank"><b> 2.	ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness (Oral, )  </b></a>  
- ImageNet으로 pretrain된 CNN은 object의 texture에 bias되어있음을 보이며, global object shape 정보를 이용하면 robust한 CNN을 만들 수 있음을 보임.

<figure>
	<img src="{{ '/assets/img/iclr_2019/2_1.PNG' | prepend: site.baseurl }}" alt=""> 
	<figcaption> [본 논문의 그림 예시 1] </figcaption>
</figure> 

<figure>
	<img src="{{ '/assets/img/iclr_2019/2_2.PNG' | prepend: site.baseurl }}" alt=""> 
	<figcaption> [본 논문의 그림 예시 2] </figcaption>
</figure> 

##  <a href="https://openreview.net/pdf?id=HkNDsiC9KQ" target="_blank"><b> 3.	Learning Unsupervised Learning Rules (Oral, )  </b></a>  
- Meta-learning 관련 논문이며 unsupervised representation learning update rule을 다룬 논문. 

<figure>
	<img src="{{ '/assets/img/iclr_2019/3_1.PNG' | prepend: site.baseurl }}" alt=""> 
	<figcaption> [본 논문의 그림 예시] </figcaption>
</figure> 

##  <a href="https://openreview.net/pdf?id=HygBZnRctX" target="_blank"><b> 4.	Transferring Knowledge across Learning Processes (Oral, )  </b></a>  
- Transfer learning에 대한 논문이며 meta learning 관점에서 학습을 통해 knowledge를 잘 transfer하도록 하는 “Leap” 라는 방법론 제안. 

<figure>
	<img src="{{ '/assets/img/iclr_2019/4_1.PNG' | prepend: site.baseurl }}" alt=""> 
	<figcaption> [본 논문의 방법론 예시] </figcaption>
</figure> 

##  <a href="https://openreview.net/pdf?id=rJEjjoR9K7" target="_blank"><b> 5.	Learning Robust Representations by Projecting Superficial Statistics Out (Oral, )  </b></a>  
- 학습에 사용하지 않은 배경이나 texture 등 작은 변화에 취약한 classifier를 개선하기 위해 unguided domain generalization 라는 문제 상황을 설정하고 이를 해결하기 위한 gray-level co-occurrence matrix(GLCM) 방법을 제안함. 

<figure>
	<img src="{{ '/assets/img/iclr_2019/5_1.PNG' | prepend: site.baseurl }}" alt=""> 
	<figcaption> [본 논문의 그림 예시] </figcaption>
</figure> 

##  <a href="https://openreview.net/pdf?id=rJl-b3RcF7" target="_blank"><b> 6.	The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks (Oral, )  </b></a>  
- pruning 관련 논문이며 기존의 pruning을 적용한 architecture 기반으로 학습을 수행할 때 학습이 잘 되지 않는 문제를 해결하는 방법을 제안함. 논문의 제목에서도 알 수 있듯이 Trainable한 pruned network를 찾는(winning tickets) 방법을 다룸. 

<figure>
	<img src="{{ '/assets/img/iclr_2019/6_1.PNG' | prepend: site.baseurl }}" alt=""> 
	<figcaption> [본 논문의 그림 예시] </figcaption>
</figure> 



<blockquote> 결론 </blockquote>
이번 포스팅에서는 ICLR 2019에 accept된 논문 중에 이미지 인식 분야와 관련있는 28편에 대해 정리를 해보았습니다. 
제가 정리한 논문 외에도 양질의 논문들이 많이 있으니 관심있으신 분들은 다른 논문들도 읽어보시고, 추천을 해주시면 감사하겠습니다!

<blockquote> 참고 문헌 </blockquote>  
- <a href="https://github.com/lixin4ever/Conference-Acceptance-Rate" target="_blank"> Statistics of acceptance rate for the main AI conferences </a>  

- <a href="https://github.com/shaohua0116/ICLR2019-OpenReviewData" target="_blank"> Statistics of ICLR 2019 </a>
