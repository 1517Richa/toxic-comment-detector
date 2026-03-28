# Data Directory

Place the Jigsaw Toxic Comment Classification dataset here.

## How to get the data

1. Go to the Kaggle competition page:  
   https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data

2. Accept the competition rules and download **`train.csv`**

3. Place it in this folder:
   ```
   data/train.csv
   ```

## No Kaggle account?

The project works without the real dataset.  
`train.py` automatically generates synthetic demo data if `train.csv` is not found.

```bash
python train.py --fast    # uses synthetic data, trains in ~5 minutes
```

## File size note

`train.csv` is ~52 MB and is excluded from this repository via `.gitignore`.
