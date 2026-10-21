# Results

All numbers are on the held-out test split (95 scans, 36 patients), which was not used for training or for choosing the referral threshold.

## Detection

| Model | AUC | Balanced accuracy | Sensitivity | Specificity | MCC | ECE |
|---|---|---|---|---|---|---|
| ResNet-34 | 0.990 [0.971, 1.000] | 0.964 [0.925, 0.992] | 1.000 | 0.928 [0.857, 0.978] | 0.882 | 0.048 |
| MobileNetV3-Large | 1.000 [1.000, 1.000] | 0.957 [0.922, 0.984] | 1.000 | 0.913 [0.840, 0.970] | 0.861 | 0.111 |
| MC-dropout ensemble | 1.000 [1.000, 1.000] | 0.964 [0.926, 0.989] | 1.000 | 0.928 [0.851, 0.978] | 0.882 | 0.117 |

![ROC and reliability](figures/fig_roc_reliability.png)

![Confusion matrices](figures/fig_confusion.png)

Every ERM scan received a higher ERM probability than every normal scan (AUC 1.000). The 5 errors at the 0.5 threshold are all false positives.

## Referral

| Coverage | Auto-reported | Referred | Balanced accuracy |
|---|---|---|---|
| 100% | 95 | 0 | 0.964 |
| 95% | 90 | 5 | 0.961 |
| 90% | 86 | 9 | 0.983 |
| 85% | 81 | 14 | 1.000 |
| 80% | 76 | 19 | 1.000 |
| Calibrated τ | 83 | 12 | 1.000 |

![Referral curve and uncertainty](figures/fig_selective.png)

Mean variance was 0.0291 on misclassified scans and 0.0065 on correct ones (error-detection AUROC 0.936). The two backbones agreed on 94.7% of test scans.

## Grad-CAM

![Grad-CAM examples](figures/fig_gradcam.png)

| | Tissue attention ratio |
|---|---|
| ERM scans | 1.37 |
| Normal scans | 0.50 |

A ratio above 1 means attribution falls on retinal tissue more than its area share. For normal scans the map explains the ERM logit, which has little to point at, so attribution drifts to the choroid and borders.

## Data

![Example scans](figures/fig_examples.png)

OCTDL, Kulyabin M. et al., "OCTDL: Optical coherence tomography dataset for image-based deep learning methods," *Sci. Data* 11, 365 (2024), CC BY 4.0.
