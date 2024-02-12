# 이미지 (3) - 객체 인식이란? R-CNN, YOLO

태그: R-CNN, YOLO, 이미지
No.: 14

참고 서적

- [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)
- [https://colab.research.google.com/drive/11-_7wwnmDOIohwJXE8ykPp0z4j04Evgd#scrollTo=tey2J0P7eh6n](https://colab.research.google.com/drive/11-_7wwnmDOIohwJXE8ykPp0z4j04Evgd#scrollTo=tey2J0P7eh6n)
- [https://discuss.pytorch.kr/t/d2l-ai-dive-into-deep-learning/308](https://discuss.pytorch.kr/t/d2l-ai-dive-into-deep-learning/308)

## Introduction of Object Detection

### Computer Vision Tasks

![Untitled](%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5%20(3)%20-%20%E1%84%80%E1%85%A2%E1%86%A8%E1%84%8E%E1%85%A6%20%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%89%E1%85%B5%E1%86%A8%E1%84%8B%E1%85%B5%E1%84%85%E1%85%A1%E1%86%AB%20R-CNN,%20YOLO%20bdf6776c3c364eefa33d3bbb71d80bba/Untitled.png)

### Object Detection: Multiple Objects

- 영역(위치 정보) 찾기 + 그 영역 안의 대상을 분류(classification)하기
    
    ![Untitled](%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5%20(3)%20-%20%E1%84%80%E1%85%A2%E1%86%A8%E1%84%8E%E1%85%A6%20%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%89%E1%85%B5%E1%86%A8%E1%84%8B%E1%85%B5%E1%84%85%E1%85%A1%E1%86%AB%20R-CNN,%20YOLO%20bdf6776c3c364eefa33d3bbb71d80bba/Untitled%201.png)
    
    - 2 stage detector: 영역 탐지 -> 이미지 특징 추출 -> 분류/location찾기
        - Regional Proposal과 Classification을 순차적으로 해결
            - 이미지 -> 후보 영역 자르기 -> 그 영역에서 피쳐 추출 (Conv) -> 분류
    - 1 stage detector: 이미지 특징 추출 -> (영역 탐지 + )분류/location 찾기
        - Regional Proposal과 Classification을 동시에 해결
            - 이미지 -> 전체 이미지에서 피쳐 추출 (Conv) -> 분류
        - 예) YOLO, SSD, RetinaNet
- Achor Based와 Anchor Free로 네트워크를 구분할 수도 있음
- Apply a CNN to many different crops of the image, CNN classifies each crop as object or background
    - Dog? Cat? Background?
    - 문제점: crop이 많아질수록 연산량이 올라감

### Region Proposals: Selective Search

- Region Proposals이란?
    - bounding box의 후보들을 제시하는 것 -> background 정도를 구분
- $S(r_i,r_j)$ : S는 두 영역 r_i와 r_j의 유사도

![Untitled](%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5%20(3)%20-%20%E1%84%80%E1%85%A2%E1%86%A8%E1%84%8E%E1%85%A6%20%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%89%E1%85%B5%E1%86%A8%E1%84%8B%E1%85%B5%E1%84%85%E1%85%A1%E1%86%AB%20R-CNN,%20YOLO%20bdf6776c3c364eefa33d3bbb71d80bba/Untitled%202.png)

## Networks for Object Detection

### (Slow) R-CNN

- [논문 링크](https://arxiv.org/abs/1311.2524)
- Region proposal -> Feature Extraction -> Classification/Regression

1) 2000개 정도 region 잡기

2) 특징 추출 using CNN (AlexNet 사용)

3) Warped image regions: AlexNet에 넣기 위해 각 영역의 이미지 크기를 보정 (227x227x3) -> 원래 이미지의 비율은 깨질 수 있음 -> 이미지 정보 왜곡 -> 성능 문제가 생길 수 있다.

4) 각 영역(image regions)을 ConVNet에 집어 넣기

- Fine tuning
    
    ![Untitled](%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5%20(3)%20-%20%E1%84%80%E1%85%A2%E1%86%A8%E1%84%8E%E1%85%A6%20%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%89%E1%85%B5%E1%86%A8%E1%84%8B%E1%85%B5%E1%84%85%E1%85%A1%E1%86%AB%20R-CNN,%20YOLO%20bdf6776c3c364eefa33d3bbb71d80bba/Untitled%203.png)
    
    - ImageNet 학습 또는 pre-trained 모델 사용
    - 이렇게 만들어진 모델을 원하는 Dataset에 맞춰서 fine-tuning 시킨다.
        - 이 과정에서 원하는 class의 수 (20개) + Background class를 하나 추가
        - 이때 Positive sample(IoU>=0.5)과 Negative sample의 비율을 1:3으로 하여 SGD방법으로 학습

5) Classification

![Untitled](%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5%20(3)%20-%20%E1%84%80%E1%85%A2%E1%86%A8%E1%84%8E%E1%85%A6%20%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%89%E1%85%B5%E1%86%A8%E1%84%8B%E1%85%B5%E1%84%85%E1%85%A1%E1%86%AB%20R-CNN,%20YOLO%20bdf6776c3c364eefa33d3bbb71d80bba/Untitled%204.png)

- 흔히 사용하는 softmax가 아닌 SVMs사용(Pascal VOC 데이터셋에 한함)
    - 원래 위치 - proposed 위치
- (Slow) R-CNN의 문제점
    - Regions of interest(RoI) from a proposal method 가 너무 많음 (2k) -> need to do ~2k independent forward passes for each image -> 너무 느림
    - 학습을 3번 해야 함 -> 메모리 차지가 크다
        - fine tuning
        - SVM classification
        - bounding box regression
            - true - predicted 에러가 가장 적게 만들기 위해 학습시킨다.
            - 최적화한 box가 바로 bounding box -> 예측값

### Fast R-CNN

- 이미지에서 바로 특징 추출 -> Backbone network 활용
- > feature map에서 Regions of Interest 찾기 (사이즈 조정 필요 x)
- softmax 이용
- Cropping Features: RoI pool

### Faster R-CNN👑

- Two-stage object detector
    - 1st stage: 이미지 당 한 번씩 돌아간다.
        - backbone network
        - region proposal network
    - 2nd stage: 영역(region) 당 한 번씩 돌아간다.
        - Crop features: RoI pool/align
            - Predict object class
            - Prediction bounding box offset
- **Region Proposal -> 이것도 신경망으로 하겠다**! GPU
    - **Insert Region Proposal Network(RPN**) to predict proposals from features
- RPN + Fast RCNN
    
    ![Untitled](%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5%20(3)%20-%20%E1%84%80%E1%85%A2%E1%86%A8%E1%84%8E%E1%85%A6%20%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%89%E1%85%B5%E1%86%A8%E1%84%8B%E1%85%B5%E1%84%85%E1%85%A1%E1%86%AB%20R-CNN,%20YOLO%20bdf6776c3c364eefa33d3bbb71d80bba/Untitled%205.png)
    

### Single-stage detector: YOLO/SSD

- 나쁘지 않은 성능과 꽤 빠른 속도! (실시간으로 가능한 정도를 보여준다)

### FPN (Feature Pyramid Network)

- Backbone에서 가져온 feature map을 보완하기 위함
- (a) featurized image pyramid
    - image resizing -> feature map 생성에 오래 걸린다.
- (b) single feature map
    - 단일 이미지에서 바로 feature map을 뽑는다. 빠른 대신 성능이 조금 낮음.
- (c) Pyramidal feature hierarchy
    - layer마다 피쳐를 뽑는데, low-level에서 뽑은 피쳐는 해상도가 낮을 것. layer마다 해상도 차이가 다르기 때문에 성능이 저하되는 문제가 있음.
- (d) **Feature Pyramid Network**
    - 위쪽 layer 로 갈수록 큰 대상, 아래쪽 layer로 갈수록 작은 대상을 본다
    - 화질 저하 문제를 해결하기 위해 위쪽 layer를 2배 해서 밑 layer에 그냥 더해준다(concat x, 그냥 덧셈)-> updated feature
        - 기존 Conv Net에서 지정한 layer마다 feature map을 추출하여 수정하기
        ![[Pasted image 20240124111839.png]]
- EfficientDet

### Anchor-Free Object Detection

- Anchor-Box 문제를 해결하기 위함
    - box 크기
    - positive/negative 자료 불균형 문제
- FCOS
    - Directly predict 4 box boundaries at every pixels
    - 중앙에 위치할수록 (center-ness) 가중치를 크게 준다

### Instance Segmentation

- Mask R-CNN
    - Faster R-CNN에다가 Masking만 추가한 것
- Panoptic Segmentation: Semantic + Instance Segmentation

### 객체 인식의 label은 어떻게 생겼을까?

- 패치마다 object 중심점을 x,y로 표현
- w, h는 object의 비율이기 때문에 만약 object가 패치보다 더 크다면 1을 넘을 수도 있음
- $label_{cell}=[c_1,c_2,\dots,c_{20},p_c,x,y,w,h]$
    - c : 카테고리, 이 중 하나만 1, 나머지는 0
        - 한 셀은 하나의 객체만 인식할 수 있음. -> 근접한 작은 물체에 대한 감지에 어려움이 있음.
    
    ![Untitled](%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5%20(3)%20-%20%E1%84%80%E1%85%A2%E1%86%A8%E1%84%8E%E1%85%A6%20%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%89%E1%85%B5%E1%86%A8%E1%84%8B%E1%85%B5%E1%84%85%E1%85%A1%E1%86%AB%20R-CNN,%20YOLO%20bdf6776c3c364eefa33d3bbb71d80bba/Untitled%206.png)
    

### Yolo Architecture

[YOLO: You Only Look Once](https://www.notion.so/YOLO-You-Only-Look-Once-6fcb2f2d5bb14b388f32e00e821cd5d7?pvs=21) 

- 실습 ([colab](https://colab.research.google.com/drive/11-_7wwnmDOIohwJXE8ykPp0z4j04Evgd#scrollTo=c8aXCyjTMBCX))
- 총평: 이전 모델들 보다 굉장히 빠른 속도를 보
- 24 layers

![Untitled](%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5%20(3)%20-%20%E1%84%80%E1%85%A2%E1%86%A8%E1%84%8E%E1%85%A6%20%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%89%E1%85%B5%E1%86%A8%E1%84%8B%E1%85%B5%E1%84%85%E1%85%A1%E1%86%AB%20R-CNN,%20YOLO%20bdf6776c3c364eefa33d3bbb71d80bba/Untitled%207.png)

- loss function: 위치, 클래스 등을 한 번에 다 업데이트 해주어야 함

> 기본 NN 프로세스
> 
> 1. Network Architecture
> 2. Loss & datasetup
> 3. Train (learning)
> 4. Prediction

### 네트워크를 비교하는 방법

![Untitled](%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5%20(3)%20-%20%E1%84%80%E1%85%A2%E1%86%A8%E1%84%8E%E1%85%A6%20%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%89%E1%85%B5%E1%86%A8%E1%84%8B%E1%85%B5%E1%84%85%E1%85%A1%E1%86%AB%20R-CNN,%20YOLO%20bdf6776c3c364eefa33d3bbb71d80bba/Untitled%208.png)

- 가장 최신의 모델과 자신을 비교, 속도와 성능 비교