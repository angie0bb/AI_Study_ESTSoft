# Pytorch 튜토리얼 (1) - 이미지 분류

태그: MNIST, Pytorch, 분류
No.: 5

### Tip

- ❗파일 구조 잘 짜기! (모듈화)
    - ipynb 파일에는 결과만 보여주고, 폴더도 이름 신경 써서 필요한 파일들을 모아서 넣어주자.
    - 노트북 안에서도 다른 사람이 볼 때 이해하기 쉽게 명시해주기
- Parameters나 모델 바꿔가면서 실험할 때에 기록은 어떻게? (실험 신뢰도 문제)
    - ❗노트북에서는 결과만 보여줘서 헷갈리지 않게 진행, 꼭 restart 해줘야 함.
    - ❗가능하면 매 epoch마다 저장하기
    - ❗랜덤 시드 3개 정도를 같은 조건으로 실험해서, 마지막 epoch의 평균값, 표준편차값 등을 같이 제공하기

### 틈새 공부

- Batch Size
    - Batch 크기는 모델 학습 중 parameter를 업데이트할 때 사용할 데이터 개수를 의미한다. Batch 크기만큼 데이터를 활용해 모델이 예측한 값과 실제 정답 간의 오차(conf. [손실함수](https://heytech.tistory.com/361))를 계산하여 [Optimizer](https://heytech.tistory.com/380)가 parameter를 업데이트합니다
    - Batch 크기는 몇 개의 문제를 한 번에 쭉 풀고 채점할지를 결정하는 것과 같습니다. 예를 들어, 총 100개의 문제가 있을 때, 20개씩 풀고 채점한다면 Batch 크기는 20입니다
    

![Untitled](Pytorch%20%E1%84%90%E1%85%B2%E1%84%90%E1%85%A9%E1%84%85%E1%85%B5%E1%84%8B%E1%85%A5%E1%86%AF%20(1)%20-%20%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5%20%E1%84%87%E1%85%AE%E1%86%AB%E1%84%85%E1%85%B2%20e77652769f5b41a58de3b1e7bd0b3ab9/Untitled.png)

- Iteration
    - Iteration는 전체 데이터에 대해 총 Batch의 수를 의미하며, '이터레이션'이라고 읽고 Step이라고 부르기도 합니다.
    - Batch 크기가 300이고 전체 데이터 개수가 3,000이라면 전체 데이터셋을 학습시키기 위해서는 총 10개의 Batch가 필요합니다. 10번에 걸쳐 파라미터를 업데이트해야 되니까 말이죠. 즉, Iteration의 수는 10입니다.
    
- Epoch
    - Epoch는 '에포크'라고 읽고 전체 데이터셋을 학습한 횟수를 의미합니다.

![Untitled](Pytorch%20%E1%84%90%E1%85%B2%E1%84%90%E1%85%A9%E1%84%85%E1%85%B5%E1%84%8B%E1%85%A5%E1%86%AF%20(1)%20-%20%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5%20%E1%84%87%E1%85%AE%E1%86%AB%E1%84%85%E1%85%B2%20e77652769f5b41a58de3b1e7bd0b3ab9/Untitled%201.png)

### 실습

- Dataset: Fashion.MNIST → MNIST 로 바꿔서 손글씨 분류 모델 만들기
    - [pytorch 공식문서 link](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)