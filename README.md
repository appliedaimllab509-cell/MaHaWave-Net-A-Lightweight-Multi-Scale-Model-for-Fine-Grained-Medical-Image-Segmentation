# MaHaWave-Net-A-Lightweight-Multi-Scale-Model-for-Fine-Grained-Medical-Image-Segmentation
Code will be resealing soon..!!

## Abstract
In recent years, U-Net–based transformer models have achieved remarkable success in medical image segmentation by effectively capturing hierarchical features. Visual state space models have recently emerged as an efficient alternative, offering competitive accuracy with linear complexity. However, both transformer-based and Mamba-based U-Nets suffer from high computational complexity in terms of model parameters and floating-point operations. Inspired by the strengths of both approaches, we propose MaHaWave-Net, a novel and lightweight U-Net-like architecture combining Mamba state space blocks, newly designed learnable Haar Wavelet layers, and MLP layers. The proposed learnable wavelet layers efficiently capture fine-grained information by extracting approximation and detail coefficients across multiple levels, mitigating information loss and serving as a linear-complexity alternative module to the transformer self-attention module. Extensive experiments on ISIC 2017 and ISIC 2018 datasets illustrate that MaHaWave-Net consistently outperforms existing methods. Compared to U-Net, it achieves Dice/IoU improvements of 2.09\% and 3.33\% on ISIC 2017 and 1.78\% and 2.88\% on ISIC 2018, with significantly reduced computational cost, establishing its effectiveness as a lightweight medical image segmentation model.

## 0. Main Environments
```bash
conda create -n vmunet python=3.8
conda activate vmunet
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm==1.0.1  # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs
```
The .whl files of causal_conv1d and mamba_ssm could be found here. {[Baidu](https://pan.baidu.com/s/1Tibn8Xh4FMwj0ths8Ufazw?pwd=uu5k) or [GoogleDrive](https://drive.google.com/drive/folders/1ZJjc7sdyd-6KfI7c8R6rDN8bcTz3QkCx?usp=sharing)}

## 1. Prepare the dataset

### ISIC datasets
- The ISIC17 and ISIC18 datasets, divided into a 7:3 ratio, can be found here {[Baidu](https://pan.baidu.com/s/1Y0YupaH21yDN5uldl7IcZA?pwd=dybm)}. 

- After downloading the datasets, you are supposed to put them into './data/isic17/' and './data/isic18/', and the file format reference is as follows. (take the ISIC17 dataset as an example.)

- './data/isic17/'
  - train
    - images
      - .png
    - masks
      - .png
  - val
    - images
      - .png
    - masks
      - .png

## 2. Proposed Framework
![MaHaWave-Net Architecture](/home/user/Downloads/abhi_sachin/framework_architec/MaHaWaveNet)

*Figure 1: Overview of the proposed MaHaWave-Net architecture.*


## 3. Result 
### Performance Comparison with SOTA on ISIC-17 Dataset 
| **Model**                 | **mIoU ↑ (%)** | **DSC ↑ (%)** | **Acc ↑ (%)** | **Spe ↑ (%)** | **Sen ↑ (%)** |
| :------------------------ | :------------: | :-----------: | :-----------: | :-----------: | :-----------: |
| U-Net                     |      76.98     |     86.99     |     95.65     |     97.43     |     86.82     |
| UT-NetV2                  |      77.35     |     87.23     |     95.84     |     98.05     |     84.85     |
| TransFuse                 |      79.21     |     88.40     |     96.17     |     97.98     |     87.14     |
| MALUNet                   |      78.78     |     88.13     |     96.18     |     98.47     |     84.78     |
| VM-UNet                   |    **80.23**   |   **89.03**   |   **96.29**   |     97.58     |   **89.90**   |
| **MaHaWave-Net (Ours)** |    **79.88**   |   **88.82**   |   **96.26**   |   **97.80**   |   **88.61**   |

### Performance Comparison with SOTA on ISIC-18 Dataset 
| **Model**                 | **mIoU ↑ (%)** | **DSC ↑ (%)** | **Acc ↑ (%)** | **Spe ↑ (%)** | **Sen ↑ (%)** |
| :------------------------ | :------------: | :-----------: | :-----------: | :-----------: | :-----------: |
| UNet++                    |      78.31     |     87.83     |     94.02     |     95.75     |     88.65     |
| UT-NetV2                  |      78.97     |     88.25     |     94.32     |     96.48     |     87.60     |
| SANet                     |      79.52     |     88.59     |     94.39     |     95.97     |     89.46     |
| TransFuse                 |      80.63     |     89.27     |     94.66     |     95.74     |   **91.28**   |
| MALUNet                   |      80.25     |     89.04     |     94.62     |     96.19     |     89.74     |
| VM-UNet                   |    **81.35**   |   **89.71**   |     94.21     |     96.13     |     91.12     |
| **MaHaWave-Net (Ours)** |    **80.88**   |   **89.43**   |   **94.84**   |   **96.50**   |   **89.66**   |

The results, summarized in Table~\ref{SOTA_Performance}, show that MahaWave-Net achieves competitive performance on ISIC 2017 across four key metrics, Viz. mIoU, Accuracy (Acc), Precision (Pr), and Specificity (Spe), while surpassing existing methods on ISIC 2018 in terms of Accuracy and Specificity. In particular, the model attains 94.84\% accuracy and 96.50\% specificity on ISIC 2018, slightly outperforming the more complex VM-UNet (94.21\% accuracy, 96.13\% specificity) with improvements of 0.63\% in accuracy and 0.37\% in specificity. The performance improvements can be attributed to the use of multi-scale learnable wavelet layers, which enhance fine-grained feature extraction and preserve contextual information across multiple scales. Notably, MahaWave-Net delivers these results with substantially fewer parameters and reduced FLOPs compared to transformer- and mamba-based models. This synergy of accuracy and efficiency makes MahaWave-Net highly effective for medical image segmentation and particularly well-suited for deployment on resource-limited edge devices, where many SOTA models face challenges due to their complexity.

## 4. Ablation Study
### 1. Computational Complexity 
Table below of Computational_Cost presents a comparison of the trainable parameters and floating-point operations (FLOPs) required by the proposed model and several state-of-the-art methods for medical image segmentation.
| **Method**                | **Parameters (M)** | **FLOPs (G)** |
| :------------------------ | :----------------: | :-----------: |
| VM-UNet                   |        27.43       |      4.11     |
| **MaHaWave-Net (Ours)** |      **4.98**      |    **1.73**   |
| U-Net                     |        7.77        |     13.78     |
| TransFuse                 |        26.27       |     11.53     |
| UTNetV2                   |        12.80       |     15.50     |
| SANet                     |        23.90       |      5.99     |
| UNet++                    |        9.16        |     34.90     |

Among the compared models, VM-UNet exhibits the highest parameter count and subsequently, UNet++ exhibits the highest FLOPs count. In contrast, the proposed model requires the least computational resources, with only 4.98 million trainable parameters and 1.73 Giga FLOPs. It specifically reduces the number of parameters and FLOPs by factors of 5.55 and 2.37, respectively, compared to the best-performing VM-UNet model. 

### 2. Impact  of Levels 
We evaluated six different values of this parameter, Viz.~12, 18, 24, 30, 36, and 42. The results clearly indicate that increasing the number of levels enhances the performance of MaHaWave-Net, as it enables the model to capture richer multi-scale features for fine-grained image segmentation. The best results are achieved when the level parameter is set to 42 for both datasets.
#### I. on ISIC-17 Dataset
| **Level** | **mIoU (%)** | **DSC (%)** | **Acc (%)** | **Spe (%)** | **Sen (%)** |
| :-------: | :----------: | :---------: | :---------: | :---------: | :---------: |
|     12    |     77.65    |    87.42    |    95.71    |    97.09    |    88.88    |
|     18    |     78.38    |    87.88    |    95.99    |    97.86    |    86.72    |
|     24    |     78.71    |    88.08    |    96.05    |    97.83    |    87.18    |
|     30    |     78.98    |    88.25    |    96.12    |    97.93    |    87.08    |
|     36    |     79.37    |    88.50    |  **96.27**  |  **98.43**  |    85.56    |
|   **42**  |   **79.88**  |  **88.82**  |    96.26    |    97.80    |  **88.61**  |
#### II. on ISIC-18 Dataset
| **Level** | **mIoU (%)** | **DSC (%)** | **Acc (%)** | **Spe (%)** | **Sen (%)** |
| :-------: | :----------: | :---------: | :---------: | :---------: | :---------: |
|     12    |     80.44    |    89.16    |    94.76    |  **96.82**  |    88.38    |
|     18    |     79.68    |    88.69    |    94.40    |    95.76    |  **90.16**  |
|     24    |     79.91    |    88.83    |    94.59    |    96.60    |    88.34    |
|     30    |     79.39    |    88.51    |    94.34    |    95.91    |    89.48    |
|     36    |     79.18    |    88.38    |    94.25    |    95.69    |    89.77    |
|   **42**  |   **80.88**  |  **89.43**  |  **94.84**  |    96.50    |    89.66    |

### 2. Impact  of Encoder-Decoder Depth 
To examine variations in performance metrics, we employ a strong encoder–decoder configuration in the asymmetric MaHaWave-Net design. Following \cite{ruan2024vm}, an ablation study is conducted, with results summarized in Table \ref{Ablation_Encoder}.~The model achieves optimal performance with the encoder–decoder configuration $[2,2,2,1]$ across both datasets.~However, when scaled to a larger asymmetric configuration of $[2,2,9,2]-[2,9,2,2]$, performance consistently declines, indicating that excessive scaling reduces model effectiveness. For the ablation study, the number of heads and levels is fixed at 16 and 42, respectively, and BceDice Loss is employed as the loss function.
#### I. on ISIC-17 Dataset
| **Metric**      | **[2,2,2,1]** | **[2,2,2,2]** | **[2,2,4,2]** | **[2,2,9,2]** |
| :-------------- | :-----------: | :-----------: | :-----------: | :-----------: |
| mIoU (%)        |   **79.88**   |     78.88     |     78.90     |     79.01     |
| DSC (%)         |   **88.82**   |     88.19     |     88.21     |     88.27     |
| Acc (%)         |   **96.26**   |     96.09     |     96.03     |     96.09     |
| Precision (%)   |     89.02     |   **89.32**   |     87.89     |     88.80     |
| Specificity (%) |     97.80     |   **97.90**   |     97.54     |     97.77     |
| Sensitivity (%) |   **88.61**   |     87.09     |     88.52     |     87.75     |
| HD95 ↓          |      747      |      810      |      795      |    **663**    |

#### II. on ISIC-18 Dataset
| **Metric**      | **[2,2,2,1]** | **[2,2,2,2]** | **[2,2,4,2]** | **[2,2,9,2]** |
| :-------------- | :-----------: | :-----------: | :-----------: | :-----------: |
| mIoU (%)        |   **80.88**   |     79.50     |     79.92     |     80.33     |
| DSC (%)         |   **89.43**   |     88.57     |     88.84     |     89.09     |
| Acc (%)         |   **94.84**   |     94.67     |     94.51     |     94.77     |
| Precision (%)   |     89.20     |   **92.63**   |     87.95     |     90.53     |
| Specificity (%) |     96.50     |   **97.82**   |     96.04     |     97.04     |
| Sensitivity (%) |     89.66     |     84.86     |   **89.74**   |     87.69     |
| HD95 ↓          |    **282**    |      516      |      531      |      298      |


## 5. Prepare the pre_trained weights

- The weights of the pre-trained VMamba could be downloaded from [Baidu](https://pan.baidu.com/s/1ci_YvPPEiUT2bIIK5x8Igw?pwd=wnyy) or [GoogleDrive](https://drive.google.com/drive/folders/1ZJjc7sdyd-6KfI7c8R6rDN8bcTz3QkCx?usp=sharing). After that, the pre-trained weights should be stored in './pretrained_weights/'.



## 6. Train the MaHaWave-Net
```bash
cd MaHaWave-Net
python train.py  # Train and test MaHaWave-Net on the ISIC17 or ISIC18 dataset.
```

**NOTE**: If you want to use the trained checkpoint for inference testing only and save the corresponding test images, you can follow these steps:  

- **In `config_setting`**:  
   - Set the parameter `only_test_and_save_figs` to `True`.  
   - Fill in the path of the trained checkpoint in `best_ckpt_path`.  
   - Specify the save path for test images in `img_save_path`.  

- **Execute the script**:  
   After setting the above parameters, you can run `train.py`.

## 7. Obtain the outputs
- After trianing, you could obtain the results in './results/'



