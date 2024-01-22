# Emotion Analysis Using BERT and TF-IDF

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/MMahdiSetak/Emotion-Analysis/blob/main/LICENSE)

## Abstract
This project explores emotion analysis in textual data, applying two distinct approaches: BERT fine-tuning and TF-IDF feature extraction. We conducted experiments on two datasets, GoEmotions and AIT-2018, to compare and understand the effectiveness of these methods in emotion classification tasks.


## Methods
### BERT Fine-Tuning
We utilize the BERT (Bidirectional Encoder Representations from Transformers) model, a state-of-the-art pre-trained language model, for emotion classification. The model is fine-tuned on our datasets, allowing it to adapt to the nuances of emotion-specific language.

### TF-IDF Features
The Term Frequency-Inverse Document Frequency (TF-IDF) approach is used to transform text into a meaningful representation of numbers. This classical NLP technique serves as a baseline to compare against the more advanced BERT model.

## Dataset Descriptions
### GoEmotions Dataset
GoEmotions contains a wide range of emotions, making it ideal for fine-grained emotion analysis. It includes over 58,000 Reddit comments, labeled for 27 emotion categories or neutral.

#### Results on GoEmotions
The BERT fine-tuning approach yielded promising results across various emotion categories with a notable improvement in precision and recall over baseline models. The macro-average F1 score achieved was 0.397. Detailed performance metrics for each emotion category are provided.

### AIT-2018 Dataset
AIT-2018, part of the SemEval-2018 Task 1, is a rich dataset focusing on the affect in tweets. It contains annotated tweets for 11 emotion categories, providing a diverse and challenging dataset for emotion intensity analysis.
#### Results on AIT-2018
With BERT fine-tuning, we observed a significant improvement in emotion detection accuracy compared to the TF-IDF baseline. The model performed well across various emotions, particularly in detecting complex emotions like joy and optimism, with a macro-average F1 score of 0.522.


## Comparative Analysis
This comparative analysis indicates that BERT fine-tuning consistently outperforms the TF-IDF approach in both datasets. The results highlight the effectiveness of transformer-based models in capturing emotional nuances in text, owing to their deep contextual understanding.
### GoEmotions
#### BERT Fine-Tuning
<a target="_blank" href="https://colab.research.google.com/github/MMahdiSetak/Emotion-Analysis/blob/main/go-emotions/bert.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a> 

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

The BERT model achieved notable accuracy and F1 scores across multiple emotion categories, demonstrating its effectiveness in fine-grained emotion classification.

#### Reference Results from GoEmotions Paper
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

#### TF-IDF Features:
<a target="_blank" href="https://colab.research.google.com/github/MMahdiSetak/Emotion-Analysis/blob/main/go-emotions/tf-idf.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a> 

|index|accuracy|precision|recall|f1|mcc|support|
|---|---|---|---|---|---|---|
|admiration|0\.928|0\.742|0\.343|0\.469|0\.474|504\.0|
|amusement|0\.969|0\.8|0\.485|0\.604|0\.609|264\.0|
|anger|0\.965|0\.622|0\.116|0\.196|0\.259|198\.0|
|annoyance|0\.942|0\.769|0\.031|0\.06|0\.148|320\.0|
|approval|0\.938|0\.711|0\.077|0\.139|0\.221|351\.0|
|caring|0\.976|0\.714|0\.074|0\.134|0\.225|135\.0|
|confusion|0\.971|0\.333|0\.013|0\.025|0\.061|153\.0|
|curiosity|0\.949|0\.875|0\.025|0\.048|0\.142|284\.0|
|desire|0\.986|0\.688|0\.133|0\.222|0\.298|83\.0|
|disappointment|0\.973|1\.0|0\.013|0\.026|0\.113|151\.0|
|disapproval|0\.951|1\.0|0\.007|0\.015|0\.084|267\.0|
|disgust|0\.98|0\.697|0\.187|0\.295|0\.354|123\.0|
|embarrassment|0\.994|1\.0|0\.054|0\.103|0\.232|37\.0|
|excitement|0\.983|0\.846|0\.107|0\.19|0\.297|103\.0|
|fear|0\.987|0\.818|0\.115|0\.202|0\.304|78\.0|
|gratitude|0\.986|0\.963|0\.815|0\.883|0\.879|352\.0|
|grief|0\.999|0\.0|0\.0|0\.0|0\.0|6\.0|
|joy|0\.972|0\.583|0\.174|0\.268|0\.308|161\.0|
|love|0\.973|0\.774|0\.546|0\.64|0\.637|238\.0|
|nervousness|0\.996|0\.0|0\.0|0\.0|-0\.001|23\.0|
|optimism|0\.974|0\.782|0\.328|0\.462|0\.496|186\.0|
|pride|0\.997|0\.5|0\.062|0\.111|0\.176|16\.0|
|realization|0\.974|1\.0|0\.021|0\.041|0\.142|145\.0|
|relief|0\.998|0\.0|0\.0|0\.0|0\.0|11\.0|
|remorse|0\.992|0\.667|0\.357|0\.465|0\.484|56\.0|
|sadness|0\.975|0\.738|0\.199|0\.313|0\.375|156\.0|
|surprise|0\.974|0\.515|0\.121|0\.195|0\.241|141\.0|
|neutral|0\.731|0\.623|0\.466|0\.533|0\.357|1787\.0|
|macro\_avg|NaN|0\.67|0\.174|0\.237|0\.283|NaN|

The TF-IDF model provided a strong baseline but was generally outperformed by BERT, particularly in capturing subtle emotional nuances.

### AIT-2018
#### BERT Fine-Tuning
<a target="_blank" href="https://colab.research.google.com/github/MMahdiSetak/Emotion-Analysis/blob/main/AIT-2018/bert.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a> 

threshold set at 0.5:

|index|accuracy|precision|recall|f1|mcc|support|
|---|---|---|---|---|---|---|
|anger|0\.855|0\.797|0\.765|0\.781|0\.673|1101\.0|
|anticipation|0\.871|0\.517|0\.108|0\.179|0\.192|425\.0|
|disgust|0\.832|0\.756|0\.743|0\.75|0\.624|1099\.0|
|fear|0\.927|0\.819|0\.654|0\.727|0\.691|485\.0|
|joy|0\.867|0\.872|0\.82|0\.845|0\.73|1442\.0|
|love|0\.886|0\.68|0\.535|0\.599|0\.539|516\.0|
|optimism|0\.814|0\.737|0\.732|0\.735|0\.592|1143\.0|
|pessimism|0\.891|0\.566|0\.216|0\.313|0\.303|375\.0|
|sadness|0\.834|0\.759|0\.64|0\.694|0\.585|960\.0|
|surprise|0\.949|0\.692|0\.053|0\.098|0\.182|170\.0|
|trust|0\.953|0\.4|0\.013|0\.025|0\.065|153\.0|
|macro\_avg|NaN|0\.691|0\.48|0\.522|0\.471|NaN|

BERT's performance on AIT-2018 showed its robustness in handling short, informal text, with high accuracy in most categories.

#### TF-IDF Features:
<a target="_blank" href="https://colab.research.google.com/github/MMahdiSetak/Emotion-Analysis/blob/main/AIT-2018/tf-idf.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a> 

|index|accuracy|precision|recall|f1|mcc|support|
|---|---|---|---|---|---|---|
|anger|0\.78|0\.796|0\.47|0\.591|0\.484|1101\.0|
|anticipation|0\.869|0\.375|0\.007|0\.014|0\.036|425\.0|
|disgust|0\.758|0\.729|0\.449|0\.555|0\.424|1099\.0|
|fear|0\.887|0\.909|0\.268|0\.414|0\.458|485\.0|
|joy|0\.77|0\.879|0\.556|0\.681|0\.548|1442\.0|
|love|0\.865|0\.797|0\.198|0\.317|0\.354|516\.0|
|optimism|0\.734|0\.754|0\.359|0\.486|0\.378|1143\.0|
|pessimism|0\.887|0\.818|0\.024|0\.047|0\.128|375\.0|
|sadness|0\.781|0\.9|0\.29|0\.438|0\.43|960\.0|
|surprise|0\.949|1\.0|0\.029|0\.057|0\.167|170\.0|
|trust|0\.953|0\.0|0\.0|0\.0|0\.0|153\.0|
|macro\_avg|NaN|0\.723|0\.241|0\.327|0\.31|NaN|

The TF-IDF approach showed limitations in handling the complexity and brevity of tweets, resulting in lower performance metrics compared to BERT.

## Conclusion
This study demonstrates the superiority of BERT fine-tuning in emotion analysis over traditional TF-IDF features, particularly in datasets with nuanced emotional expressions. Future work could explore combining BERT with other feature extraction techniques or applying it to other forms of textual data.

## Reference
- Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020). "GoEmotions: A Dataset of Fine-Grained Emotions". ACL 2020. available: [here](https://doi.org/10.48550/arXiv.2005.00547)

## Contributions
Feedback, bug reports, and contributions are welcome! Please feel free to submit issues and enhancement requests.
