# MusicVAE
Drum Sampling 4 bars by using Google Magenta MusicVAE Model.  
Google Magenta MusicVAE를 사용하여 4마디에 해당하는 드럼 샘플 Generate.
  
- Google Magenta: https://github.com/magenta/magenta [[Github]](https://github.com/magenta/magenta)
- MusicVAE: https://github.com/magenta/magenta/tree/main/magenta/models/music_vae [[Github]](https://github.com/magenta/magenta/tree/main/magenta/models/music_vae)
- Groove MIDI Dataset: https://magenta.tensorflow.org/datasets/groove [[download]](https://magenta.tensorflow.org/datasets/groove)
- A Hierarchical Latent Vector Model for Learning Long-Term Structure in Music (2018, Google Magenta) [[PDF]](https://arxiv.org/pdf/1803.05428.pdf)


---
## Paper: A Hierarchical Latent Vector Model for Learning Long-Term Structure in Music (2018, Google Magenta)
- MIDI 데이터에 대해 VAE 학습
- 학습된 Latent Vector와 Decoder를 활용
- Interpolation, Sampling 등 음악 창작도구로 활용

---
### VAE(Variational Auto Encoder)
<p align="center"><img width=50% src="https://user-images.githubusercontent.com/59362257/172065378-c11168a8-4000-44f2-a504-4eed2a93d0b1.png"></p>
<p align="center">Basic Architecture of a Single Layer AutoEncoder</p>
<p align="center"><img width=50% src="https://user-images.githubusercontent.com/59362257/172065954-4e1341ca-894c-4496-bbec-7613486b4ad8.png"></p>
<p align="center">VAE Architecture</p>

- AE(Auto Encoder)와 유사
- Training Data를 Bottleneck 구조를 통해 압축, 저차원의 Latent Code 생성
- Latent Space를 통해 입력 데이터간 유사성과 차이를 학습
- Encoder는 입력데이터를 평균과 표준편차 Vector로 Encoding, 두 Vector에 대응하는 분포에서 샘플링 수행
- 쿨백 라이블러 발산(Kullback-Leibler Divergence, KLD)을 손실함수로 사용, 해당분포가 표준정규분포에 가까워지도록 학습
- 표준정규분포에 근사한 분포에서 샘플링된 VAE의 잠재공간분포(Latent Space Distribution)는 원점을 기준으로 대칭적이고 고른 형태의 분포 

<p align="center"><img width=75% src="https://user-images.githubusercontent.com/59362257/172064964-742ea810-a43b-4714-86db-301a8207ec13.png"></p>
<p align="center">Latent Distribution for Label</p>

- 같은공간을 기준으로 AE는 이산적(Discrete, 듬성듬성한 형태)
- 같은공간을 기준으로 VAE는 연속적(Continuous, 밀집된 형태)
- AE는 VAE에 비해 한쪽으로 치우치고 넓게 분포
- 이 분포 사이의 빈틈이 데이터 재구성시 저품질의 데이터를 구성하는 원인
- AE: 입력데이터 재구성은 가능하나, 새로운 데이터 생성에 부적합
- VAE: Generative Model로 새로운 데이터 생성에 적합

---
### Recurrent VAEs
- MusicVAE는 순환신경망 RNN 모델을 적용(Recurrent), 2가지 문제 발생
- 순환신경망 RNN의 특성상 Decoder는 학습과정에서 Latent Code를 무시할 수도 있음
- 전체 시퀀스를 단일 잠재 벡터로 압축, 정보의 손실 발생
- 이러한 문제들을 해결하기 위해 계층적(Hierarchical) RNN을 Decoder에 사용

---
### Bidirectional Encoder
- Encoder로 2 Layer-Bidirectional LSTM 사용
- Input Sequence의 Long-Term Context 모델링에 적합
- Encoder Output을 Concat하여 2개의 FC Layer를 거침
- Latent Space의 파라미터인 평균과 분산 추정

---
### Hierarchical Decoder
- Decoder에 Conductor 추가하여 Hierarchical하게 구성
- 음악이 갖는 긴 Data Sequence로 인한 Vanishing Influence 문제 해결
- Data Sequence X가 U개의 Subsequence로 분리될 수 있다는 가정 기반
- Conductor는 Latent Vector Z를 입력받아, U차원으로 임베딩
- Decoder는 U개의 Vector를 받아 최종결과값 출력

---
### Multi-Stream Modeling
- 일반적으로 Sequential Data는 텍스트와 같이 Single Stream으로 구성
- 하지만, 음악은 기본적으로 Multi Stream인 경우가 많음
- Trio Model을 도입, 출력 토큰에 대해 3가지 악기(드럼, 베이스, 멜로디) 각각의 개별 분포를 생성
- 각 개별스트림을 직교 차원으로 간주, 각 악기에 대해 별도의 Decoder RNN 사용

---
### MusicVAE Model
<p align="center"><img width=50% src="https://user-images.githubusercontent.com/59362257/172065050-8653db32-e4dc-4b96-8a9c-7aae92121873.png"></p>

- `Input` - `2 Layer Bi-LSTM` - `2 FC Layer` - `Latent Space` - `FC Layer` - `2 Layer Uni-LSTM` - `2 Layer LSTM` - `Output`
- 정규분포와 Latent Vector와의 쿨백 라이블러 발산(Kullback-Leibler Divergence, KLD) 손실함수를 최소화하는 방향으로 학습
- Decoder를 통과한 Output과 Input의 Log Loss(on/off), MSE(velocity, offset)를 최소화하는 방향으로 학습


## Environments
- Google Colab GPU Runtime
- Clone `MusicVAE_Drum_Sampling_4bars.ipynb` and put it in Google Drive like down below.
- Clone [magenta](https://github.com/magenta/magenta) and put it in Google Drive like down below.
```
Google Drive
└── Colab Notebooks
    ├── magenta*
    │   ├── demos
    │   └── magenta
    └── MusicVAE_Drum_Sampling_4bars.ipynb*
```


## Train
- `preprocess_tfrecord.py`의 `flags.DEFINE_bool` flags 수정
- `is_drum` & `drums_only`를 `True`로 수정
- Preprocess 과정에서 Drum NoteSequences만 남도록 하기 위함
- Colab 환경에서 수정할 수 없기 때문에 파일 직접 수정

```
Google Drive
└── Colab Notebooks
    ├── magenta
    │   ├── demos
    │   └── magenta
    │       └── models
    │           └── music_vae
    │               ├── js
    │               ├── config.py*
    │               ├── music_vae_train.py*
    │               └── preprocess_tfrecord.py*
    └── MusicVAE_Drum_Sampling_4bars.ipynb
```



## Test
```
Google Drive
└── Colab Notebooks
    ├── magenta
    │   ├── demos
    │   └── magenta
    │       └── models
    │           └── music_vae
    │               ├── js
    │               ├── weights
    │               │   └── groovae4bar
    │               │       └── train*
    │               │           ├── model.ckpt-2721.data-00000-of-00001
    │               │           ├── model.ckpt-2721.index
    │               │           ├── model.ckpt-2721.meta
    │               │           ├── model.ckpt-35026.data-00000-of-00001
    │               │           ├── model.ckpt-35026.index
    │               │           ├── model.ckpt-35026.meta
    │               │           ├── model.ckpt-40106.data-00000-of-00001
    │               │           ├── model.ckpt-40106.index
    │               │           ├── model.ckpt-40106.meta
    │               │           ├── model.ckpt-45182.data-00000-of-00001
    │               │           ├── model.ckpt-45182.index
    │               │           ├── model.ckpt-45182.meta
    │               │           ├── model.ckpt-50058.data-00000-of-00001
    │               │           ├── model.ckpt-50058.index
    │               │           └── model.ckpt-50058.meta
    │               ├── config.py
    │               ├── music_vae_train.py
    │               └── preprocess_tfrecord.py
    └── MusicVAE_Drum_Sampling_4bars.ipynb
```
## Conclusion




## Reference
- A Hierarchical Latent Vector Model for Learning Long-Term Structure in Music (2018, Google Magenta) [[PDF]](https://arxiv.org/pdf/1803.05428.pdf)
- Google Magenta: https://github.com/magenta/magenta [[Github]](https://github.com/magenta/magenta)
- MusicVAE: https://github.com/magenta/magenta/tree/main/magenta/models/music_vae [[Github]](https://github.com/magenta/magenta/tree/main/magenta/models/music_vae)
