# Coffee bean defect percentage

This is a binary bean-level model. It learns `Defect` from the 17 folders in
the defect ZIP and `Good` from the `Dark`, `Green`, `Light`, and `Medium` image
folders in the healthy ZIP. For a photo containing several beans, OpenCV first
finds individual beans, then the model classifies each crop.

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Train with the downloaded archives

```bash
python3 app.py train \
  "/Users/apple/Downloads/archive (1).zip" \
  "/Users/apple/Downloads/archive (2).zip"
```

The command prints a held-out classification report and writes
`coffee_defect_model.joblib`.

## Score a photo

```bash
python3 app.py predict "/Users/apple/Downloads/Normales.jpg"
```

The result includes the number of beans detected, defect count, and:

`defect_percentage = defect_beans / beans_detected * 100`

Use a plain, contrasting background and spread beans apart for the most
reliable count. This is an image-level classifier, not a certified coffee
grading instrument; test it on new labeled photos before using it for quality
decisions.
# ethiopian-coffee-defect-detection
Machine Learning and Computer Vision system for detecting and classifying visible defects in Ethiopian green coffee beans.
