# Emotion Analysis Project
<a target="_blank" href="https://colab.research.google.com/github/MMahdiSetak/Emotion-Analysis/blob/main/go-emotions/go-emotions.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/MMahdiSetak/Emotion-Analysis/blob/main/LICENSE)

## Overview
This project focuses on emotion analysis using a neural network model built on top of BERT (Bidirectional Encoder Representations from Transformers). The goal is to accurately classify text into various emotion categories such as admiration, amusement, anger, etc.

## Features
- Utilizes the BERT model for sequence classification.
- Emotion analysis on 28 different emotions.
- Training and evaluation datasets from the GoEmotions dataset.
- Detailed performance metrics including accuracy, precision, recall, F1 score, and MCC (Matthews correlation coefficient).

## Results
The model was evaluated on various emotion categories with the following results (threshold set at 0.5):

| Emotions       | Accuracy | Precision | Recall | F1     | MCC    | Support |
|----------------|----------|-----------|--------|--------|--------|---------|
| admiration     | 0\.943   | 0\.699    | 0\.681 | 0\.689 | 0\.658 | 504\.0  |
| amusement      | 0\.982   | 0\.799    | 0\.83  | 0\.814 | 0\.805 | 264\.0  |
| anger          | 0\.969   | 0\.623    | 0\.384 | 0\.475 | 0\.474 | 198\.0  |
| annoyance      | 0\.942   | 0\.549    | 0\.088 | 0\.151 | 0\.203 | 320\.0  |
| approval       | 0\.943   | 0\.65     | 0\.259 | 0\.371 | 0\.387 | 351\.0  |
| caring         | 0\.975   | 0\.521    | 0\.185 | 0\.273 | 0\.301 | 135\.0  |
| confusion      | 0\.975   | 0\.614    | 0\.281 | 0\.386 | 0\.405 | 153\.0  |
| curiosity      | 0\.951   | 0\.547    | 0\.412 | 0\.47  | 0\.45  | 284\.0  |
| desire         | 0\.987   | 0\.719    | 0\.277 | 0\.4   | 0\.441 | 83\.0   |
| disappointment | 0\.973   | 0\.667    | 0\.053 | 0\.098 | 0\.183 | 151\.0  |
| disapproval    | 0\.952   | 0\.516    | 0\.243 | 0\.331 | 0\.333 | 267\.0  |
| disgust        | 0\.98    | 0\.638    | 0\.301 | 0\.409 | 0\.43  | 123\.0  |
| embarrassment  | 0\.993   | 1\.0      | 0\.027 | 0\.053 | 0\.164 | 37\.0   |
| excitement     | 0\.983   | 0\.658    | 0\.243 | 0\.355 | 0\.393 | 103\.0  |
| fear           | 0\.991   | 0\.708    | 0\.59  | 0\.643 | 0\.641 | 78\.0   |
| gratitude      | 0\.99    | 0\.946    | 0\.898 | 0\.921 | 0\.916 | 352\.0  |
| grief          | 0\.999   | 0\.0      | 0\.0   | 0\.0   | 0\.0   | 6\.0    |
| joy            | 0\.98    | 0\.732    | 0\.509 | 0\.601 | 0\.601 | 161\.0  |
| love           | 0\.983   | 0\.8      | 0\.824 | 0\.812 | 0\.803 | 238\.0  |
| nervousness    | 0\.996   | 0\.0      | 0\.0   | 0\.0   | 0\.0   | 23\.0   |
| optimism       | 0\.975   | 0\.719    | 0\.441 | 0\.547 | 0\.552 | 186\.0  |
| pride          | 0\.997   | 0\.0      | 0\.0   | 0\.0   | 0\.0   | 16\.0   |
| realization    | 0\.975   | 1\.0      | 0\.055 | 0\.105 | 0\.232 | 145\.0  |
| relief         | 0\.998   | 0\.0      | 0\.0   | 0\.0   | 0\.0   | 11\.0   |
| remorse        | 0\.992   | 0\.611    | 0\.589 | 0\.6   | 0\.596 | 56\.0   |
| sadness        | 0\.978   | 0\.728    | 0\.378 | 0\.498 | 0\.515 | 156\.0  |
| surprise       | 0\.979   | 0\.655    | 0\.404 | 0\.5   | 0\.505 | 141\.0  |
| neutral        | 0\.782   | 0\.721    | 0\.55  | 0\.624 | 0\.482 | 1787\.0 |
| macro\_avg     | NaN      | 0\.601    | 0\.339 | 0\.397 | 0\.41  | NaN     |


### Reference Results from GoEmotions Paper
For comparison, here are the results reported in the GoEmotions paper:

| Emotion       | Precision | Recall | F1   |
|---------------|-----------|--------|------|
| admiration    | 0.53      | 0.83   | 0.65 |
| amusement     | 0.70      | 0.94   | 0.80 |
| anger         | 0.36      | 0.66   | 0.47 |
| annoyance     | 0.24      | 0.63   | 0.34 |
| approval      | 0.26      | 0.57   | 0.36 |
| caring        | 0.30      | 0.56   | 0.39 |
| confusion     | 0.24      | 0.76   | 0.37 |
| curiosity     | 0.40      | 0.84   | 0.54 |
| desire        | 0.43      | 0.59   | 0.49 |
| disappointment| 0.19      | 0.52   | 0.28 |
| disapproval   | 0.29      | 0.61   | 0.39 |
| disgust       | 0.34      | 0.66   | 0.45 |
| embarrassment | 0.39      | 0.49   | 0.43 |
| excitement    | 0.26      | 0.52   | 0.34 |
| fear          | 0.46      | 0.85   | 0.60 |
| gratitude     | 0.79      | 0.95   | 0.86 |
| grief         | 0.00      | 0.00   | 0.00 |
| joy           | 0.39      | 0.73   | 0.51 |
| love          | 0.68      | 0.92   | 0.78 |
| nervousness   | 0.28      | 0.48   | 0.35 |
| neutral       | 0.56      | 0.84   | 0.68 |
| optimism      | 0.41      | 0.69   | 0.51 |
| pride         | 0.67      | 0.25   | 0.36 |
| realization   | 0.16      | 0.29   | 0.21 |
| relief        | 0.50      | 0.09   | 0.15 |
| remorse       | 0.53      | 0.88   | 0.66 |
| sadness       | 0.38      | 0.71   | 0.49 |
| surprise      | 0.40      | 0.66   | 0.50 |
| macro-average | 0.40      | 0.63   | 0.46 |
| std           | 0.18      | 0.24   | 0.19 |

## Usage
1. **Setup Environment**: Ensure you have a Python environment with necessary libraries such as `torch`, `transformers`, `datasets`, `numpy`, `pandas`, and `matplotlib`.
2. **Load and Preprocess Data**: Data is loaded from the GoEmotions dataset and preprocessed for the BERT model.
3. **Model Training and Evaluation**: The model is trained on the training dataset and evaluated on the test set.
4. **Inference**: Use the model to predict the emotion of new text data.



## Reference
- Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020). "GoEmotions: A Dataset of Fine-Grained Emotions". ACL 2020. available: https://doi.org/10.48550/arXiv.2005.00547


## Contributions
Feedback, bug reports, and contributions are welcome! Please feel free to submit issues and enhancement requests.
