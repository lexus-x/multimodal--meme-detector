# Datasets

## Primary: MultiOFF

**Multimodal Meme Classification — Identifying Offensive Content in Image and Text**

| Property | Value |
|----------|-------|
| Total samples | 743 |
| Train / Val / Test | 445 / 149 / 149 |
| Offensive / Non-offensive | ~200 / ~543 |
| Annotators | 8 human volunteers |
| Inter-annotator agreement | κ ≈ 0.4–0.5 (moderate) |
| Paper | Alam et al., LREC 2020 |

### Download
```bash
python -m research.data.download
```

### Format
CSV with columns: `image_name`, `sentence`, `label` (0=non-offensive, 1=offensive)

## Alternative Datasets

| Dataset | Samples | Task | Source |
|---------|---------|------|--------|
| Hateful Memes | 10,000 | Binary hateful/not | Kiela et al., NeurIPS 2020 |
| Memotion 1.0 | 8,898 | Multi-class | Sharma et al., SemEval 2020 |
| Harm-C | 3,035 | Fine-grained harmful | Cao et al., 2022 |
| MAMI | 12,000 | Misogynous memes | SemEval 2022 |
| HarMeme | 3,398 | COVID-19 harmful | Qazi et al., 2022 |

## Using Custom Data

Prepare a CSV with columns:
```csv
image_name,sentence,label
img_001.jpg,"This is the meme text",0
img_002.jpg,"Offensive meme text here",1
```

Place images in a directory and point to them:
```bash
python -m core.train --train_csv my_data/train.csv --img_dir my_data/images
```
