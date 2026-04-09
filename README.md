# U-Net
U-Net 구현
## U-Net: Semantic Segmentation for Biomedical Images

이 저장소는 논문 **"U-Net: Convolutional Networks for Biomedical Image Segmentation"**의 핵심 아키텍처를 PyTorch를 사용하여 구현한 프로젝트입니다. U-Net은 적은 양의 학습 데이터로도 정밀한 세그멘테이션 성능을 보여주는 모델로, 의료 영상 처리뿐만 아니라 다양한 컴퓨터 비전 분야에서 널리 활용되고 있습니다.

---

### 1. 개요 (Overview)
U-Net은 **Contracting Path(수축 경로)**와 **Symmetric Expanding Path(확장 경로)**가 결합된 U자형 구조를 가집니다. 수축 경로에서 이미지의 컨텍스트(Context)를 포착하고, 확장 경로에서는 캡처된 특징을 복원하여 정밀한 위치 정보(Localization)를 결합하는 것이 특징입니다.

---

### 2. 주요 특징 (Key Features)
* **Symmetry Structure:** 인코더와 디코더가 대칭을 이루어 정교한 특징 추출 및 복원 가능.
* **Skip Connection:** 수축 경로의 특징 맵을 확장 경로에 직접 결합하여 저수준의 위치 정보 손실을 방지.
* **Overlap-tile Strategy:** 큰 이미지를 타일 단위로 나누어 처리함으로써 해상도 제한을 극복.


---

### 3. 기술 스택 (Tech Stack)
* **Language:** Python
* **Framework:** PyTorch
* **Libraries:** NumPy, Matplotlib, OpenCV (또는 PIL)

---

### 4. 모델 구조 (Architecture)

1.  **Contracting Path (Encoder):**
    * $3 \times 3$ Convolution (Padding 0, ReLU)
    * $2 \times 2$ Max Pooling (Stride 2)
    * 채널 수를 2배씩 증가시키며 이미지 크기 축소.

2.  **Expanding Path (Decoder):**
    * $2 \times 2$ Up-convolution (Transposed Convolution)
    * Skip connection을 통한 특징 맵 Concatenation.
    * $3 \times 3$ Convolution (ReLU)

3.  **Final Layer:**
    * $1 \times 1$ Convolution을 사용하여 원하는 클래스 개수만큼 채널 조정.

---


### 6. 참고 문헌 (References)
* Ronneberger, O., Fischer, P., & Brox, T. (2015). [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597). MICCAI.
