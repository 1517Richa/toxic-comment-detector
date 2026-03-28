# Data Directory

Place the Jigsaw Toxic Comment Classification dataset here.

## How to get the data

1. Go to the Kaggle competition page:
   https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
2. Accept the competition rules and download `train.csv`.
3. Place it in this folder as `data/train.csv`.

## No Kaggle account?

If `data/train.csv` is not found, `train.py` automatically generates synthetic demo data.

```bash
python train.py --fast
```
