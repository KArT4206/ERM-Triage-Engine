# Architecture

```
OCT B-scan (any size)
   │  grayscale → 224×224 → p1–p99 normalisation → 3 channels → ImageNet normalisation
   ▼
┌──────────────────────────┐    ┌──────────────────────────────┐
│ ResNet-34                │    │ MobileNetV3-Large            │
│ 50% frozen, Dropout 0.3  │    │ 50% frozen, Dropout 0.3      │
│ 15 MC-dropout passes     │    │ 15 MC-dropout passes         │
└────────────┬─────────────┘    └──────────────┬───────────────┘
             └───────────── 30 probabilities ──┘
                              │
              mean p̄(x), variance u(x), 95% interval
                              │
            triage: u(x) ≤ τ, backbones agree, p̄ outside 0.35–0.65
                    │                         │
               Auto-report            Refer to specialist
```

![Pipeline](figures/fig_pipeline.png)

## Components

| File (private repo) | Role |
|---|---|
| `prepare_octdl.py` | Builds the patient-disjoint train/val/test folders from the OCTDL zip and label CSV |
| `erm_engine.py` | Dataset, preprocessing, both backbones, training loop, MC-dropout ensemble, metrics, threshold calibration, report |
| `app.py` | Flask server: `/predict`, `/model-card`, `/samples`, `/paper/*`, `/report` |
| `templates/index.html` | Dashboard (Analyse, Model performance, Methods, Paper) |
| `templates/report.html` | Printable A4 per-scan report |
| `make_samples.py` | Picks the eight demo scans from the test split |
| `paper_analysis.py` | Bootstrap intervals, referral analysis, Grad-CAM tissue ratio and the figures |

## MC dropout

At inference, batch-normalisation layers stay in evaluation mode while dropout stays active. Each backbone runs 15 passes; the 30 pooled probabilities give

- p̄(x), the mean, which decides the class at 0.5;
- u(x), the variance, which reflects both dropout noise within a model and disagreement between the two models;
- the 2.5th and 97.5th percentiles, shown as the 95% interval.

## Referral threshold

τ is the variance below which the 90% most certain validation scans fall. It is fixed after calibration and never tuned on the test split. The current value is 0.0219.

## Privacy

The app keeps nothing. An uploaded image is processed in memory for one request and discarded; only its SHA-256 hash appears on the report.
