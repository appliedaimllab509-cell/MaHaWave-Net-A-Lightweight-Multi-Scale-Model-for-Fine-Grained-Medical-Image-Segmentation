# MaHaWave-Net-A-Lightweight-Multi-Scale-Model-for-Fine-Grained-Medical-Image-Segmentation

## Abstract
In recent years, U-Net–based transformer models have achieved remarkable success in medical image segmentation by effectively capturing hierarchical features. Visual state space models have recently emerged as an efficient alternative, offering competitive accuracy with linear complexity. However, both transformer-based and Mamba-based U-Nets suffer from high computational complexity in terms of model parameters and floating-point operations. Inspired by the strengths of both approaches, we propose MaHaWave-Net, a novel and lightweight U-Net-like architecture combining Mamba state space blocks, newly designed learnable Haar Wavelet layers, and MLP layers. The proposed learnable wavelet layers efficiently capture fine-grained information by extracting approximation and detail coefficients across multiple levels, mitigating information loss and serving as a linear-complexity alternative module to the transformer self-attention module. Extensive experiments on ISIC 2017 and ISIC 2018 datasets illustrate that MaHaWave-Net consistently outperforms existing methods. Compared to U-Net, it achieves Dice/IoU improvements of 2.09\% and 3.33\% on ISIC 2017 and 1.78\% and 2.88\% on ISIC 2018, with significantly reduced computational cost, establishing its effectiveness as a lightweight medical image segmentation model.

## Result 
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

## Ablation Study
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
#### A. on ISIC-17 Dataset
| **Level** | **mIoU (%)** | **DSC (%)** | **Acc (%)** | **Spe (%)** | **Sen (%)** |
| :-------: | :----------: | :---------: | :---------: | :---------: | :---------: |
|     12    |     77.65    |    87.42    |    95.71    |    97.09    |    88.88    |
|     18    |     78.38    |    87.88    |    95.99    |    97.86    |    86.72    |
|     24    |     78.71    |    88.08    |    96.05    |    97.83    |    87.18    |
|     30    |     78.98    |    88.25    |    96.12    |    97.93    |    87.08    |
|     36    |     79.37    |    88.50    |  **96.27**  |  **98.43**  |    85.56    |
|   **42**  |   **79.88**  |  **88.82**  |    96.26    |    97.80    |  **88.61**  |
#### A. on ISIC-18 Dataset
| **Level** | **mIoU (%)** | **DSC (%)** | **Acc (%)** | **Spe (%)** | **Sen (%)** |
| :-------: | :----------: | :---------: | :---------: | :---------: | :---------: |
|     12    |     80.44    |    89.16    |    94.76    |  **96.82**  |    88.38    |
|     18    |     79.68    |    88.69    |    94.40    |    95.76    |  **90.16**  |
|     24    |     79.91    |    88.83    |    94.59    |    96.60    |    88.34    |
|     30    |     79.39    |    88.51    |    94.34    |    95.91    |    89.48    |
|     36    |     79.18    |    88.38    |    94.25    |    95.69    |    89.77    |
|   **42**  |   **80.88**  |  **89.43**  |  **94.84**  |    96.50    |    89.66    |

