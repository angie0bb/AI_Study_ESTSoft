# 이미지 (5) - Data Augmentation

태그: 데이터전처리, 이미지
No.: 16

### Data Augmentation

- 실습([colab](https://colab.research.google.com/drive/12KANLZTeKCZf077efoeby0M6NhlahkRy#scrollTo=u7BQ3bqHXV08))
- [샘플이 적다 (2) - 데이터 증강(Data Augmentation)](https://www.notion.so/2-Data-Augmentation-e72b2c11e3a644db8f2b79c15f3c29e3?pvs=21)

![https://i.imgur.com/pwZEFQN.png](https://i.imgur.com/pwZEFQN.png)

- 회전, 좌우반전, 상하반전, 확대 축소
- 기존 데이터를 이용하여 인위적으로 변화된(modified) 데이터를 training set에 추가하는 것
- 왜?
    - **과적합 방지를 위해**
    - 초기 데이터 수가 너무 적어서
    - 더 다양한 피쳐 학습을 가능케 해서 성능 향상에 도움을 주기 위해

### Image Data Augmentation 방법

1. Basic Image Manipulations: RGBShift, ToGray, ChannelShuffle, Blur, VerticalFlip RandomRotate90
    - Kernel Filters
    - Color Space Transformations
    - Random Erasing
    - Geometric Transformations
    - Mixing Images
        
        ![https://i.imgur.com/HHx3N7h.png](https://i.imgur.com/HHx3N7h.png)
        
        ![https://i.imgur.com/HHRaCIZ.png](https://i.imgur.com/HHRaCIZ.png)
        
2. Deep Learning Approaches: 이미지 생성, 스타일 바꾸기 등
    - Adversarial Training
    - Neural Style Transfer
    - GAN Data Augmentation
        - 의료 자료는 자료 뷸군형이 크기 때문에 특히 많이 쓰임.
3. Then, Meta Learning!

### Data Augmentation의 주의점

- augmentation 후 이미지가 괜찮은지 눈으로 확인하기
- 본질적인 특징이 바뀌거나 삭제될 수 있음
    - ㅏ -> ㅓ
    - 심장의 모양 좌우 반전
1. 이미지 한개가 transforms을 지나서 변화된 New이미지 한장이 됨
2. 잊지말고 New이미지 저장해 놓기
3. 덧. transforms.Compose(...)에 ...이 많아도 한개의 transforms 임