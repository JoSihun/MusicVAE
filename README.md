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
<p align="center"><img width=75% src="https://user-images.githubusercontent.com/59362257/172065378-c11168a8-4000-44f2-a504-4eed2a93d0b1.png"></p>
<p align="center">Basic Architecture of a Single Layer AutoEncoder</p>
<p align="center"><img width=75% src="https://user-images.githubusercontent.com/59362257/172065954-4e1341ca-894c-4496-bbec-7613486b4ad8.png"></p>
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
<p align="center"><img width=75% src="https://user-images.githubusercontent.com/59362257/172065050-8653db32-e4dc-4b96-8a9c-7aae92121873.png"></p>

- `Input` - `2 Layer Bi-LSTM` - `2 FC Layer` - `Latent Space` - `FC Layer` - `2 Layer Uni-LSTM` - `2 Layer LSTM` - `Output`
- 정규분포와 Latent Vector와의 쿨백 라이블러 발산(Kullback-Leibler Divergence, KLD) 손실함수를 최소화하는 방향으로 학습
- Decoder를 통과한 Output과 Input의 Log Loss(on/off), MSE(velocity, offset)를 최소화하는 방향으로 학습


---
## Environments
- Google Colab GPU Runtime
- Clone `samples` and put it in Google Drive like down below.
- Clone `Preprocess.ipynb` and put it in Google Drive like down below.
- Clone `Visualization.ipynb` and put it in Google Drive like down below.
- Clone `MusicVAE_Drum_Sampling_4bars.ipynb` and put it in Google Drive like down below.
- Clone [magenta](https://github.com/magenta/magenta) and put it in Google Drive like down below.
- Decompress `groove-v1.0.0-midionly.zip` from `datasets` directory and put `groove` in Google Drive like down below.
```
Google Drive
└── Colab Notebooks
    ├── groove*
    ├── magenta*
    │   ├── demos
    │   └── magenta
    ├── samples*
    ├── Preprocess.ipynb*
    ├── Visualization.ipynb*
    └── MusicVAE_Drum_Sampling_4bars.ipynb*
```


---
## Preprocess
- `Preprocess.ipynb` 참고
- 아래와 같이 proto 형식의 sequence로 변환
- `pitch`: Roland Mapping 기반 9차원 Mapping
- `mapping`: Bass, Snare, Closed Hi-Hat, High Floor Tom, Open Hi-Hat, Low-Mid Tom, High Tom, Crash, Ride
- `samples`의 midi 파일을 사용하여 proto 변환 및 tfrecord 변환 전처리 확인
- 실제 학습에서는 매개변수로 Groove MIDI Dataset 사용, 자동으로 Preprocessing 진행
```
notes {
  pitch: 42
  velocity: 17
  end_time: 0.12272727272727273
  is_drum: true
}
```
- `convert_dir_to_note_sequences.py`: 폴더 내 모든 midi 파일을 tfrecord 형식으로 변환
- /content/drive/MyDrive/Colab Notebooks/magenta/magenta/scripts/convert_dir_to_note_sequences.py
```linux
!python convert_dir_to_note_sequences.py \
  --input_dir="/content/drive/MyDrive/Colab Notebooks/samples" \
  --output_file=SEQUENCES_TFRECORD \
  --recursive
```


---
## Train
- `MusicVAE_Drum_Sampling_4bars.ipynb`의 `Train` 파트 참고
- `preprocess_tfrecord.py`의 `flags.DEFINE_bool` flags 수정필요
- Colab 환경에서 수정할 수 없기 때문에 파일 직접 수정 후 Google Drive로 파일 이동

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


---
## Test(Generate Samples)
- `MusicVAE_Drum_Sampling_4bars.ipynb`의 `Generate Samples` 파트 참고
- Generate된 4마디 Drum Sample은 `samples` 폴더에 존재
- 학습이 잘 되었다면 Decoder는 정규분포에 근사한 Latent Vector를 입력받아 기존 Input Data를 fitting
- 즉, 정규분포의 랜덤벡터를 Decoder에 통과시키면, 새로운 샘플을 무한히 생성 가능
- Pretrained Model을 사용하고 싶다면 `models` 폴더 내 모든 파일을 아래와 같이 `train` 폴더에 위치
```
Google Drive
└── Colab Notebooks
    ├── magenta
    │   ├── demos
    │   └── magenta
    │       └── models
    │           └── music_vae
    │               ├── js
    │               ├── weights*
    │               │   └── groovae4bar*
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


---
## Visualization
- `Visualization.ipynb` 참고
- proto 형식의 sequence 시각화
- `groove-v1.0.0-midionly.zip` dataset 사용 (4/4박자 midi만 사용)
<p align="center"><img width=75% src="https://user-images.githubusercontent.com/59362257/172072025-0227252a-f4b7-40e1-8c8e-f375388e3a3b.png"></p>
<p align="center">groove10_102_beat_4-4</p>
<p align="center"><img width=75% src="https://user-images.githubusercontent.com/59362257/172072057-7bb77b7d-4312-4b5f-bdf0-826fef3993f9.png"></p>
<p align="center">7_hiphop_100_beat_4-4</p>
<p align="center"><img width=75% src="https://user-images.githubusercontent.com/59362257/172072070-6a9124a8-0c59-44fe-960a-2e32b56422be.png"></p>
<p align="center">32_hiphop_92_beat_4-4</p>


---
## Conclusion
- Generated Samples with Trained Model 비교
- Generated Samples with Pretrained Model 비교
- Pretrained, Trained 모두 성공적으로 드럼 사운드만을 추출
- Trained Epochs 가 증가할수록 복잡하고 다양한 패턴의 드럼 샘플 생성
- Pretrained보다 Trained 모델이 직관적으로 더 복잡하고 다양한 패턴의 드럼 샘플 생성  
  -> 논문에서 언급한대로 학습한 Epoch가 높기 때문으로 추측


---
## Reference
- A Hierarchical Latent Vector Model for Learning Long-Term Structure in Music (2018, Google Magenta) [[PDF]](https://arxiv.org/pdf/1803.05428.pdf)
- Google Magenta: https://github.com/magenta/magenta [[Github]](https://github.com/magenta/magenta)
- MusicVAE: https://github.com/magenta/magenta/tree/main/magenta/models/music_vae [[Github]](https://github.com/magenta/magenta/tree/main/magenta/models/music_vae)
