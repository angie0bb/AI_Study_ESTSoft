# 이미지 (2) Vision Transformer, Swin Transformer

태그: SWIN, Transformer, ViT, 이미지
No.: 13

관련 논문

- Attention is all you need: Transformer
- **An image is worth 16x16 words ~ : Vision Transformer**
- Swin transformer: Swin Transformer

## Vision Transformer (ICLR '21)

[**ViT: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**](https://www.notion.so/ViT-An-Image-is-Worth-16x16-Words-Transformers-for-Image-Recognition-at-Scale-3490a2d0b0194eaea126d09ab509f80c?pvs=21) 

### 특징

1. Image task를 Transformer 구조 모델(NLP task)에서 구현하였다.
2. Vision task에서 CNN을 사용하지 않고 Transformer의 Multi-head attention 구조를 사용하여 충분히 좋은 결과를 냄
    1. Transformer의 Encoder부분을 변형해서 사용함.
- 단, ViT는 해상도와 물체의 크기를 고려하지 못함

## Swin Transformer

- multi-scale feature map이 많아짐.
- ViT는 동일한 크기로 특징을 자르다 보니 원근감이나 해상도를 고려하지 못함.
- Swin transformer는 다양한 window 사이즈로 자르고 shift함

### 특징

1. 다양한 목적의 backbone으로 사용가능함
2. 기존 ViT 모델보다 더 적은 연산량을 가짐