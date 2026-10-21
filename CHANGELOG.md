# Changelog

## 1.0.0

- MC-dropout ensemble of ResNet-34 and MobileNetV3-Large trained on real OCTDL scans with a patient-level train/validation/test split.
- Referral threshold calibrated on validation data; all metrics reported on the held-out test split with patient-level bootstrap intervals.
- Dashboard with Analyse, Model performance, Methods and Paper tabs, eight held-out sample scans and a printable per-scan report.
- Grad-CAM attribution with a tissue attention measurement.
