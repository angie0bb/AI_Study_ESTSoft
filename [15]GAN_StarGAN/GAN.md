# 이미지 (4) - 생성 모델 GAN과 친구들

태그: GAN, StyleGAN, 이미지
No.: 15

### 생성모델 (Generative Models)

- 생성 모델은 실존하지 않지만, 있을 법한 이미지, 텍스트 등을 생성할 수 있는 모델을 의미한다.
- 생성 모델의 목표: 이미지 데이터의 분포를 근사하는 모델 G를 만드는 것
    - 모델 G가 잘 동작한다는 의미는 원래 이미지들의 분포를 잘 모델링할 수 있다는 것을 의미한다. 학습이 잘 되었다면 통계적으로 평균적인 특징을 가지는 데이터를 쉽게 생성할 수 있다.
    
    ![Untitled](%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5%20(4)%20-%20%E1%84%89%E1%85%A2%E1%86%BC%E1%84%89%E1%85%A5%E1%86%BC%20%E1%84%86%E1%85%A9%E1%84%83%E1%85%A6%E1%86%AF%20GAN%E1%84%80%E1%85%AA%20%E1%84%8E%E1%85%B5%E1%86%AB%E1%84%80%E1%85%AE%E1%84%83%E1%85%B3%E1%86%AF%20e1373080c5694738b40c76573c897685/Untitled.png)
    

### GAN(Generative Adversarial Network)

- 실습 ([colab](https://colab.research.google.com/drive/1zIjW9SFaG7531MPsnpcxfYxPJ8u1XLPV#scrollTo=srQI5xI6ar-X))
- 참고 ([나동빈 youtube](https://www.youtube.com/watch?v=AVvlDmhHgC4))
- 논문 ([GAN](https://arxiv.org/abs/1406.2661))
    - by Goodfellow
- 1st Player: Generative Model
    - 분포를 학습, Generator (이미지 생성)
- 2nd Player: Dscriminator
    - DIscriminator (진짜인지, 아닌지 판별하는 확률 값)
- GAN = learn to generate from **training data distribution** through **2-player game**

### 확률분포

- 확률분포란 확률 변수(X)가 특정한 값을 가질 확률을 나타내는 함수, 확률의 합은 1이어야 함.
- P(X=1) = 1/6
- 확률변수 X의 개수를 정확히 셀 수 있을 때에는 이산확률변수, 없을 때에는 연속확률분포라고 칭함 (확률 밀도 함수를 이용해 영역으로 분포를 표시)
    - 예) 정규분포(normal distribution) $N(\mu,\sigma^2)$
    
    ![Untitled](%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5%20(4)%20-%20%E1%84%89%E1%85%A2%E1%86%BC%E1%84%89%E1%85%A5%E1%86%BC%20%E1%84%86%E1%85%A9%E1%84%83%E1%85%A6%E1%86%AF%20GAN%E1%84%80%E1%85%AA%20%E1%84%8E%E1%85%B5%E1%86%AB%E1%84%80%E1%85%AE%E1%84%83%E1%85%B3%E1%86%AF%20e1373080c5694738b40c76573c897685/Untitled%201.png)
    
- 기댓값: (단순 평균과는 다르다)
    - 이산확률변수: $E[X]=\Sigma_i x_i*f(x_i)$
    - 연속확률변수: $E[x]=\int x*f(x)dx$
- 이미지 데이터에 대한 확률 분포
    - 이미지 데이터 또한 분포로 나타낼 수 있기 때문에, 그 분포를 근사하는 모델을 학습할 수 있다.
    - 이미지 데이터는 다차원 특징 공간(주로 RGB)의 한 점으로 표시가 된다.
    - 사람의 얼굴의 특징들은 통계적인 평균치가 존재한다.
    - hidden dimension = 2 (2 features)이라면?
    
    ![Untitled](%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5%20(4)%20-%20%E1%84%89%E1%85%A2%E1%86%BC%E1%84%89%E1%85%A5%E1%86%BC%20%E1%84%86%E1%85%A9%E1%84%83%E1%85%A6%E1%86%AF%20GAN%E1%84%80%E1%85%AA%20%E1%84%8E%E1%85%B5%E1%86%AB%E1%84%80%E1%85%AE%E1%84%83%E1%85%B3%E1%86%AF%20e1373080c5694738b40c76573c897685/Untitled%202.png)
    

### GAN의 Objective Function(목적 함수)

- x 분포에서 나온 확률의 기대값 + z(latent space) 분포에서 나온 확률의 기댓값
- 사람인지 아닌지 오락가락해야 생성 모델이 만든 이미지를 잘 만든 것, 그래서 목표가 1/2
    - D(G(z)) = 1/2 (내가 만들어 낸 이미지가 1/2)
- 업데이트 할 때에
    - Fake image: 더 진짜처럼
    - Real image: 더 오락가락하게
- 학습 시에는 D, G를 미니 배치마다 k번씩 학습해서 2개가 optimal한 값으로 나올 수 있게 학습을 진행한다.
    - 🤔왜?

> Optimizing D to completion in the inner loop of training is computationally prohibitive, and on finite datasets would result in overfitting. Instead, we alternate between k steps of optimizing D and one step of optimizing G
> 
- G: Generator, 생성자, generate fake samples that can fool D, 새로운 이미지 생성
    - $G(z)$: new data instance(instance=이미지 한 장)
    - **생성자인 G는 D(G(z)) 값이 1을 내뱉을 수 있도록 학습(그럴싸하게 보이게끔!)**
- D: Discriminator, 판별자, classify fake samples vs. real images, 얼마나 진짜 같은 지에 대한 판별값
    - $D(x)$: Probability of a sample from the real distribution (학습 데이터의 분포)
    - **판별자는 원본 데이터에 대해서는 1을 뱉을 수 있게 학습, 가짜 이미지에 대해서는 0을 뱉을 수 있도록 학습**
    - x: real -> D(x) = 1, x:fake -> D(x)=0
    - $D(G(z))$: 생성한 fake image를 D에 넣어서 판별
    - 실제 학습을 하다 보면 초기 학습 때, 아래 minmaxV(D,G) 공식이 G에 대해서 충분한 gradient를 주지 않을 수 있음. (너무 작게 줘서 학습이 잘 안 되게끔) 만약, G(z)를 너무 못 만들으면 D(x)가 바로 fake data로 판별해 버릴 것임. 이렇게 되면 $log(1-D(G(z)))$ 가 saturates할 수 있음. (너무 작아져서 이후 값이 거의 변하지 않게 되는 현상, vanishing gradient 문제 발생할 수 있다) 이 문제를 해결하기 위해 $log(1-D(G(z)))$를 최소화하는 것보다, $log(D(G(z)))$ 를 최대화시키는 방향으로 학습할 수 있음. 이렇게 하면 초기 학습 때 stronger gradients를 줄 수 있음.

> In practice, equation 1 may not provide sufficient gradient for G to learn well. Early in learning, when G is poor, D can reject samples with high confidence because they are clearly different from the training data. In this case, log(1 − D(G(z))) saturates. Rather than training G to minimize log(1 − D(G(z))) we can train G to maximize log D(G(z)). This objective function results in the same fixed point of the dynamics of G and D but provides much stronger gradients early in learning
> 

![Untitled](%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5%20(4)%20-%20%E1%84%89%E1%85%A2%E1%86%BC%E1%84%89%E1%85%A5%E1%86%BC%20%E1%84%86%E1%85%A9%E1%84%83%E1%85%A6%E1%86%AF%20GAN%E1%84%80%E1%85%AA%20%E1%84%8E%E1%85%B5%E1%86%AB%E1%84%80%E1%85%AE%E1%84%83%E1%85%B3%E1%86%AF%20e1373080c5694738b40c76573c897685/Untitled%203.png)

### GAN의 수렴 과정

![Untitled](%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5%20(4)%20-%20%E1%84%89%E1%85%A2%E1%86%BC%E1%84%89%E1%85%A5%E1%86%BC%20%E1%84%86%E1%85%A9%E1%84%83%E1%85%A6%E1%86%AF%20GAN%E1%84%80%E1%85%AA%20%E1%84%8E%E1%85%B5%E1%86%AB%E1%84%80%E1%85%AE%E1%84%83%E1%85%B3%E1%86%AF%20e1373080c5694738b40c76573c897685/Untitled%204.png)

- GAN의 목표: 생성자의 분포가 원본 학습 데이터셋의 분포를 잘 따라할 수 있게 만드는 것
    - 학습이 잘 되었다면 통계적으로 평균적인 특징을 가지는 데이터를 생성할 수 있음
    - $P_g$ -> $P_{data}$,, $D(G(z))$ -> 1/2
        - 🤔왜 판별자의 분포가 1/2로 수렴하도록 만드는 지?
            - 학습이 이루어진 이후에는 생성자가 만든 가짜 이미지와 진짜 이미지를 구분할 수 없음. 결국, D(G(z)) = 1/2로 나온다는 의미는 학습이 잘 되었다는 의미.
    - $P_g$ -> $P_{data}$,에 대한 증명
        - Global Optimality
- Gradient check
    
    ![https://i.imgur.com/p0WWrLU.png](https://i.imgur.com/p0WWrLU.png)
    

### pix2pix: Image-to-Image Translation

- 어느정도 테두리, sketch가 되어 있는 이미지를 인풋으로 넣는다.
- 테두리를 면으로, 복원용

### Cycle-GAN

> 생성적 적대 신경망(GAN)의 변형 중 하나로, 주로 두 가지 도메인 간에 존재하는 이미지의 스타일을 변환하는 데 사용됩니다. 예를 들어, 말과 얼룩말의 이미지를 서로 변환할 수 있습니다. 이때, 얼룩말에서 말로 변환하는 것과 말에서 얼룩말로 변환하는 두 가지 네트워크가 함께 사용됩니다.
> 
- 특징
    1. 짝을 이루지 않는 이미지 to 이미지 변환
    2. GAN 2개로 구성
- 같은 pairs가 아니라 Content, style이미지가 서로 다름, 그렇다면 True, predict 2개 비교 대상은 어떻게 고를까?
    - 얼룩말을 말로 만들고, 그 만들어진 말을 다시 얼룩말로 만들어서 얼룩말끼리 비교
- Cycle Consistency
    - 역함수 이용
        
        ![Untitled](%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5%20(4)%20-%20%E1%84%89%E1%85%A2%E1%86%BC%E1%84%89%E1%85%A5%E1%86%BC%20%E1%84%86%E1%85%A9%E1%84%83%E1%85%A6%E1%86%AF%20GAN%E1%84%80%E1%85%AA%20%E1%84%8E%E1%85%B5%E1%86%AB%E1%84%80%E1%85%AE%E1%84%83%E1%85%B3%E1%86%AF%20e1373080c5694738b40c76573c897685/Untitled%205.png)
        
        ![https://i.imgur.com/0QOAs87.png](https://i.imgur.com/0QOAs87.png)
        
        ![https://i.imgur.com/E5aksSy.png](https://i.imgur.com/E5aksSy.png)
        
    
    ![https://i.imgur.com/T4ZOPrz.png](https://i.imgur.com/T4ZOPrz.png)
    

![https://i.imgur.com/UgTotMO.png](https://i.imgur.com/UgTotMO.png)

![https://i.imgur.com/vbb9rCU.png](https://i.imgur.com/vbb9rCU.png)

### 성능지표

- FID: 그림의 세밀도 (작을수록 좋음)
    - 해상도가 좋고 사실적인 이미지일수록 FID가 낮은 경향이 있음. 낮은 dimension의 피쳐가 실제 이미지와 비슷할수록 FID가 낮음!

> FID compares the mean and standard deviation of the gaussian distributions containing feature vectors obtained from the deepest layer in Inception v3. High-quality and highly realistic images tend to have low FID scores meaning that their low dimension feature vectors are more similar to those of real images of the same type e.g faces to faces, birds to birds, etc. The calculation is:
> 
- PIPS: diversity of generated images 생성한 이미지의 다양성 (measures perceptual similarity)
    - Evaluate the distance between image patches by using trained network (alexnet, vgg...)

### StyleGAN

![Untitled](%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5%20(4)%20-%20%E1%84%89%E1%85%A2%E1%86%BC%E1%84%89%E1%85%A5%E1%86%BC%20%E1%84%86%E1%85%A9%E1%84%83%E1%85%A6%E1%86%AF%20GAN%E1%84%80%E1%85%AA%20%E1%84%8E%E1%85%B5%E1%86%AB%E1%84%80%E1%85%AE%E1%84%83%E1%85%B3%E1%86%AF%20e1373080c5694738b40c76573c897685/Untitled%206.png)

- 논문: [link](https://arxiv.org/abs/1812.04948)
- 참고자료: [link](https://paperswithcode.com/method/stylegan)
- 아이디어: 눈동자 색, 머리카락 색 등만 다른 색으로 주고 싶어서 원하는 style을 수치화 시키고 여러 scale에 가중치와 편향을 넣어서 학습시키는 방법으로 GAN에 적용
- AdaIN

$$
AdaIN(x_i,y)=y_{s,i} \frac{x_i-\mu(x_i)}{\sigma (x_i)}+y_{b,i}
$$

- NN에서도 각 layer를 지나가며 scale, variance의 변화가 자주 일어나며 이는 학습이 불안정해지는 문제를 일으킨다. 따라서 Batch Normalization 과 같은 기법을 각 layer에 사용하면서 이러한 문제를 해소하려 한다.
    - StyleGAN에서도 layer를 거치며 학습이 불안정해지기 때문에 normalization을 각 layer에다 추가해야하는데, 그 역할을 AdaIN이 한다.
    - AdaIN에서는 normalization을 할 때마다 한 번에 하나씩만 W가 기여하므로, 하나의 style이 각각의 scale에서만 영향을 끼칠 수 있도록 분리를 해주는 효과를 갖고 있다.

###