# PAN26 Generated Text Detection

Ensemble AI-generated text detector developed for the **PAN 2026 Voight-Kampff AI Detection** shared task.

The system combines multiple transformer classifiers with handcrafted statistical features and fuses them using an XGBoost meta-classifier.

## Architecture

- **ModernBERT** – long-context transformer scorer
- **DeBERTa-v3** – sliding-window transformer scorer
- **RoBERTa** – transformer baseline scorer
- **17 statistical features**
  - GPT-2 perplexity features
  - lexical diversity
  - structural text properties
- **XGBoost meta-classifier** with isotonic calibration
- **Rejection mechanism** for uncertain predictions (`0.5` score)

## Repository structure

```text
models/              Fine-tuned transformer models
features/            Statistical feature extraction
meta_classifier/     XGBoost ensemble
inference.py         Final prediction pipeline
train_*.py           Training scripts
```

## Method

Each transformer outputs a probability of AI-generated text. These scores are combined with 17 statistical features into a 20-dimensional feature vector, which is processed by an XGBoost meta-classifier to produce the final prediction. A confidence-based rejection mechanism abstains on highly uncertain samples.

## Citation

## Citation

Švihura, L. (2026). *Team original at PAN 2026: Ensemble Detection of AI-Generated Text via Transformer Scorers and Statistical Features*. Notebook for the PAN Lab at CLEF 2026.
