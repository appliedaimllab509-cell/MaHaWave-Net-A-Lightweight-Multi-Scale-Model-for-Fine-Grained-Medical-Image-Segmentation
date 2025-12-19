# MaHaWave-Net-A-Lightweight-Multi-Scale-Model-for-Fine-Grained-Medical-Image-Segmentation

## Abstract
In recent years, U-Net–based transformer models have achieved remarkable success in medical image segmentation by effectively capturing hierarchical features. Visual state space models have recently emerged as an efficient alternative, offering competitive accuracy with linear complexity. However, both transformer-based and Mamba-based U-Nets suffer from high computational complexity in terms of model parameters and floating-point operations. Inspired by the strengths of both approaches, we propose MaHaWave-Net, a novel and lightweight U-Net-like architecture combining Mamba state space blocks, newly designed learnable Haar Wavelet layers, and MLP layers. The proposed learnable wavelet layers efficiently capture fine-grained information by extracting approximation and detail coefficients across multiple levels, mitigating information loss and serving as a linear-complexity alternative module to the transformer self-attention module. Extensive experiments on ISIC 2017 and ISIC 2018 datasets illustrate that MaHaWave-Net consistently outperforms existing methods. Compared to U-Net, it achieves Dice/IoU improvements of 2.09\% and 3.33\% on ISIC 2017 and 1.78\% and 2.88\% on ISIC 2018, with significantly reduced computational cost, establishing its effectiveness as a lightweight medical image segmentation model.

## Result 
### Table-1: Performance Comparison with SOTA on ISIC-17 Dataset 
| **Model**                 | **mIoU ↑ (%)** | **DSC ↑ (%)** | **Acc ↑ (%)** | **Spe ↑ (%)** | **Sen ↑ (%)** |
| :------------------------ | :------------: | :-----------: | :-----------: | :-----------: | :-----------: |
| U-Net                     |      76.98     |     86.99     |     95.65     |     97.43     |     86.82     |
| UT-NetV2                  |      77.35     |     87.23     |     95.84     |     98.05     |     84.85     |
| TransFuse                 |      79.21     |     88.40     |     96.17     |     97.98     |     87.14     |
| MALUNet                   |      78.78     |     88.13     |     96.18     |     98.47     |     84.78     |
| VM-UNet                   |    **80.23**   |   **89.03**   |   **96.29**   |     97.58     |   **89.90**   |
| **MaHaWave-Net (Ours)** |    **79.88**   |   **88.82**   |   **96.26**   |   **97.80**   |   **88.61**   |

### Table-2: Performance Comparison with SOTA on ISIC-18 Dataset 
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

