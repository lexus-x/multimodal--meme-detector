# Model Card

## Model Details

- **Name**: Multimodal Meme Offensive Content Detector
- **Version**: 1.0.0
- **Type**: Multimodal binary classifier (text + image)
- **Framework**: PyTorch

## Intended Use

- **Primary**: Research on multimodal fusion strategies for offensive content detection in memes
- **Secondary**: Educational tool for understanding cross-modal attention mechanisms
- **NOT intended for**: Production content moderation without further validation

## Training Data

- **Dataset**: MultiOFF (743 manually annotated memes)
- **Limitations**: Small dataset, moderate inter-annotator agreement, English only

## Performance

| Model | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Early Fusion (baseline) | ~0.68 | ~0.62 | ~0.64 |
| Cross-Modal Attention | ~0.72 | ~0.66 | ~0.68 |
| Gated Fusion | ~0.70 | ~0.64 | ~0.66 |

## Limitations

- Small training set (743 samples) limits generalization
- English-only; no multilingual support
- Binary classification only (no fine-grained harm types)
- Moderate inter-annotator agreement (κ≈0.4-0.5) means label noise
- Image backbone is frozen VGG16/ResNet50 (ImageNet features may not capture meme-specific visual patterns)

## Ethical Considerations

- Model may exhibit biases present in the training data
- False positives may suppress legitimate speech
- False negatives may miss genuinely harmful content
- Should not be used as sole content moderation tool

## Citations

```bibtex
@inproceedings{alam2020multioff,
  title={Multimodal Meme Dataset (MultiOFF) for Identifying Offensive Content in Image and Text},
  author={Alam, Sarda N and others},
  booktitle={LREC},
  year={2020}
}
```
