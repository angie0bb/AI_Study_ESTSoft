# 이미지 (1) - CNN, Transfer Learning

태그: CNN, 이미지, 전이학습
No.: 12

### Introduction to Convolutional Neural Network

- Classification (이미지 분류), Retrieval (이미지 검색), Detection, Segmentation 등에 자주 쓰인다.

![혼공머신 책 참고 ](%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5%20(1)%20-%20CNN,%20Transfer%20Learning%20b0c8b7828a6e411f97c5ccd479c33f5e/Untitled.png)

혼공머신 책 참고 

### 이미지에 어떻게 가중치를 정할까?

- Convolution Filter: 이미지의 특징(feature)을 추출할 수 있다.
    - 예) 수직 경계선을 찾는 컨볼루션 필터 -> 다양한 필터를 이용해 다양한 특징을 추출할 수 있음.
        - 그림 크기: 5x5
        - conv 필터 크기: 3x3
        - 결과물 크기: 3x4
        
        ![Untitled](%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5%20(1)%20-%20CNN,%20Transfer%20Learning%20b0c8b7828a6e411f97c5ccd479c33f5e/Untitled%201.png)
        
    - 예) 수평 엣지를 찾는 컨볼루션 필터?
- CNN는 컨볼루션 필터를 학습시킨다.

### CNN

- convolution layer
    - 필터가 이미지를 돌면서 얻은 값을 모아놓은 면 -> activation map
    - 필터가 여러개면 activation map도 여러개의 '채널'(=depth, dimension)로 이루어지게 됨
    - `pytorch (N=batch,C=channel,H=height,W=width)`
- pooling layer
    - height, width, depth를 줄일 수 있음
- fully connected layer 로 주로 이루어져 있음.
    - 이미지를 1짜리 벡터로 flatten 해주기

### Convolution Neural Networks are just Neural Networks BUT...

Local connectivity
before: full conncectivity - 32x32x3 weights
now: one neuron will conncect to 5x5x3 chunk and only have 5x5x3 weights

- **local in space** (5x5 inside 32x32)
- but **full in depth** (all 3 depth channels)

### Convolution Layer (합성곱 층)

> 합성곱 신경망에서 필터는 이미지에 있는 어떤 특징을 찾는다고 생각할 수 있다. 처음에는 간단한 기본적인 특징 (직선, 곡선 등)을 찾고 층이 깊어질수록 다양하고 구체적인 특징을 감지할 수 있도록 필터의 개수를 늘린다. 또한 어떤 특징이 이미지의 어느 위치에 놓이더라도 쉽게 감지할 수 있도록 너비와 높이 차원을 점점 줄여나가는 것
> 
- **Convolve** the filter with the image: "slide over the image spatially, while computing dot products"
    - Convolution filter: 이미지의 특징(feature) 추출기
- **패딩(padding), 스트라이드(stride)**
    - 패딩: 패딩이 없다면 이미지의 가장자리 값, 모서리 값들은 안쪽 값에 비해 필터에 찍히는 횟수가 적을 것. 모서리에 있는 (중요한) 정보도 feature map에 잘 전달하기 위해 사용함.
    - 스트라이드: 필터의 이동 칸수, 가로 세로 다르게 지정할 수도 있지만 일반적이진 않다.
- Convolution layer를 거친 뒤, output size는?
    - $\frac{N+Padding*2-channel}{stride} + 1$
    - $\frac{N-F}{s}+1$
        
        ![Untitled](%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5%20(1)%20-%20CNN,%20Transfer%20Learning%20b0c8b7828a6e411f97c5ccd479c33f5e/Untitled%202.png)
        
        ![Untitled](%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5%20(1)%20-%20CNN,%20Transfer%20Learning%20b0c8b7828a6e411f97c5ccd479c33f5e/Untitled%203.png)
        
- 이때 학습시켜야할 parameter 개수는?
- 학습의 주체는 filter!
- (5x5x3 + bias(1)) x 6 = 456
    
    ![Untitled](%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5%20(1)%20-%20CNN,%20Transfer%20Learning%20b0c8b7828a6e411f97c5ccd479c33f5e/Untitled%204.png)
    
- Convolution Layer 실행을 위해서 세팅해주어야 하는 값들은?
    - depth x Width x Height
    - filter size (=kernel size라고도 함)
    - filter 갯수

### Pooling Layer

- 풀링(Pooling): convolution layer에서 만든 feature map의 가로세로 크기를 줄이는 역할, feature map 개수를 줄이는 것은 아님! 필터와는 다르게 가중치를 곱하지 않고 크기만 조정한다. 대신 최댓값이나 평균값을 계산하는 역할을 수행한다.
    - convolution layer에서 스트라이드를 크게 하여 feature map크기를 줄이는 것보다, 풀링 층에서 크기를 줄이는 것이 경험적으로 더 나은 성능을 낸다.
    - Max Pooling
    - Average pooling

### Fully Connected Layer

- 32x32x3 크기의 image -> flatten -> stretch to 3072x1
    - width 32, height 32, depth(RGB) 3
        - all neural net activations arranged in 3 dimensions: height, wideth, depth
        - ==pytorch Tensor dimension: NxCxHxW==
            - N: batch size, C: Channel size (depth), H: Height, W: Width
            
            ![Untitled](%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5%20(1)%20-%20CNN,%20Transfer%20Learning%20b0c8b7828a6e411f97c5ccd479c33f5e/Untitled%205.png)
            
            ![Untitled](%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5%20(1)%20-%20CNN,%20Transfer%20Learning%20b0c8b7828a6e411f97c5ccd479c33f5e/Untitled%206.png)
            

## 대표 CNN 네트워크 예시

- CNN Architectures
    - Large-Scale CNN
    - Efficient (mobile-target) CNN: 속도, 효율 중시

### 1. AlexNet (8 layers)

![Untitled](%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5%20(1)%20-%20CNN,%20Transfer%20Learning%20b0c8b7828a6e411f97c5ccd479c33f5e/Untitled%207.png)

- layer 개수를 셀 때에는 convolution layer만 세는 것!
- 최초의 CNN-based ImageNet challenge winner

### 2. VGG (Deeper Networks 시작)

- Early stopping으로 overfitting 막기
    - 작은 filter 사용 (**3x3 conv**) why?
        - stack of two 3x3 conv (stride 1) layers has same effective receptive field as one 5x5 conv layer -> 파라미터(필터 + 편향)가 적어지니까 좋다.
        - NVIDIA CUDA도 3x3 에 최적화되어있었음.
        - but deeper and more non-linearities
            
            ![Untitled](%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5%20(1)%20-%20CNN,%20Transfer%20Learning%20b0c8b7828a6e411f97c5ccd479c33f5e/Untitled%208.png)
            

### 3. GoogLeNet (22 layers)

- 사실 원래 레이어가 많으면 학습이 잘 되기 어려움. (높게만 쌓으면) -> Batch normalization으로 보완
    - layer가 많을수록 기울기가 0이 되어감(vanishing gradient)
- Auxilliary Classifiers: 사라져가는 gradient 문제를 해결하기 위한 장치
- Inception module
    - feature concatenate operation
    - 여러 스케일의 convolution 연산 (concatenate) -> 효율적으로 특징을 뽑아낼 수 있음.
        - 예) 원근에 따른 크기도 고려됨
            - 5x5는 가장 가까이에 있는 이미지, 3x3은 멀리있는 이미지
            
            ![Untitled](%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5%20(1)%20-%20CNN,%20Transfer%20Learning%20b0c8b7828a6e411f97c5ccd479c33f5e/Untitled%209.png)
            

### 4. ResNet

- Basic block
- Bottleneck block
    - 50 layer부터는 basic block으로 구성하면 parameter가 너무 많이 생김 -> 메모리 부하 커진다.
    - 3x3 2개가 아니라, 1x1, 3x3, 1x1 으로 3번 거쳐서 이미지 크기는 유지하되 parameter개수만 줄임.
    
    ![Untitled](%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5%20(1)%20-%20CNN,%20Transfer%20Learning%20b0c8b7828a6e411f97c5ccd479c33f5e/Untitled%2010.png)
    
- 기울기 소실 문제는? Residual로 해결
- 미분에서 0이 나오지 않기 위해, 자기 자신 x를 더해서 무조건 1이상이 나오게 처리함.
- Residual Block (Basic, Bottleneck)을 생성

![Untitled](%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5%20(1)%20-%20CNN,%20Transfer%20Learning%20b0c8b7828a6e411f97c5ccd479c33f5e/Untitled%2011.png)

- 마지막에 8x8 -> 1x1 (Average pooling) 줄어드는 것이 큰 상관이 없는 이유?
    - 첫번째 layer에서 하는 것도 아니고, 이미 32 -> 8까지 줄어들어서 특징만 추출된 상태이기 때문에 큰 이상 없음.
- deep residual learning에서 identity를 더할 때 convolution layer를 거치면 x가 아니라 가중치가 고려된 wX로 변경되고 미분하면 1이 아니라 w가 되는 것이 아닌가?
    - 제안: 1x1 conv, s=2 -> 1x1 pooling, s=2

### 5. ResNeXt

- GoogLeNet과 비슷하지만, 조금 더 세밀하게 쪼개고, residual block도 넣어줌!

![Untitled](%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5%20(1)%20-%20CNN,%20Transfer%20Learning%20b0c8b7828a6e411f97c5ccd479c33f5e/Untitled%2012.png)

### 6. SENet

- channel relationship

![Untitled](%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5%20(1)%20-%20CNN,%20Transfer%20Learning%20b0c8b7828a6e411f97c5ccd479c33f5e/Untitled%2013.png)

### 7. DenseNet

![Untitled](%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5%20(1)%20-%20CNN,%20Transfer%20Learning%20b0c8b7828a6e411f97c5ccd479c33f5e/Untitled%2014.png)

- ResNet과 유사한지만...차이점은?
    - low-level feature 유지
    - channel-wise concatenation -> 동시 추적 가능 (과거로 가지 않고도)
    - backpropgation 계산이 빠르다.

### 8. EfficientNet

![Untitled](%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5%20(1)%20-%20CNN,%20Transfer%20Learning%20b0c8b7828a6e411f97c5ccd479c33f5e/Untitled%2015.png)

- width, depth, resolution을 동시에 고려해 보는 모델
    - width = conv layer의 채널 수
    - depth = conv layer의 레이어 수
- flops -> 얼마나 시간이 오래걸리는지, 복잡도

## Backbone Network란?

- CNN은 feature extractor로 사용(이미지의 특징 추출)

![Untitled](%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5%20(1)%20-%20CNN,%20Transfer%20Learning%20b0c8b7828a6e411f97c5ccd479c33f5e/Untitled%2016.png)

## Transfer Learning (전이 학습)

- 왜 필요할까?
    - 주어진 데이터(target data)가 너무 적을 때 -> 학습이 잘 안 됨
    - 누군가가 이미 정해둔 가중치를 가져오면 되지 않을까?
        - 원래 initial 가중치는 랜덤임. -> 이걸 랜덤을 주는게 아니라 다른 사람이 학습해둔 가중치를 가져오는 것
- pre-training이 필요
    - 이때 얻은 pre-trained weights를 가져와서 target task를 진행
    - 일본어, 영어 배운 사람은 다른 언어도 쉽게 배운것처럼, 도메인이 아주 달라도 랜덤 초기 가중치보다는 좋은 결과를 도출할 수 있음.
- shape을 잘 맞춰서 넣어주기만 하면 됨.
- 데이터가 적은 상황에서는 유용하게 사용할 수 있음.