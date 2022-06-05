# MusicVAE
Drum Sampling 4 bars by using Google Magenta MusicVAE Model.  
Google Magenta MusicVAE를 사용하여 4마디에 해당하는 드럼 샘플 Generate.
  
- Google Magenta: https://github.com/magenta/magenta [[Github]](https://github.com/magenta/magenta)
- MusicVAE: https://github.com/magenta/magenta/tree/main/magenta/models/music_vae [[Github]](https://github.com/magenta/magenta/tree/main/magenta/models/music_vae)
- A Hierarchical Latent Vector Model for Learning Long-Term Structure in Music (2018, Google Magenta) [[PDF]](https://arxiv.org/pdf/1803.05428.pdf)


## Paper: A Hierarchical Latent Vector Model for Learning Long-Term Structure in Music (2018, Google Magenta)


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
