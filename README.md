# Social Media Sentiment Analysis — Project Report

## Overview
This project analyses **real, user-generated social media text data** - actual customer messages, comments, and complaints collected across social platforms (Facebook, Twitter, Instagram, LinkedIn, TikTok, and YouTube) rather than simulated data. Because the data is real, it carries the characteristics typical of authentic social media content: informal language, spelling inconsistencies, emojis, mixed languages, missing values, competition and spam-like entries. The end goal is to build a machine learning pipeline capable of automatically classifying the sentiment (Positive, Negative, or Neutral) of new, unseen messages of the same real-world nature.

---

## Objectives
Below are the objectives of the project:

1. Understand the distribution of sentiment across a social media message dataset spanning multiple social platforms.
2. Identify which platforms contribute most to each sentiment category.
3. Prepare the dataset for machine learning by cleaning, balancing, and vectorizing message text.
4. Train and compare multiple classical machine learning models to automatically predict sentiment from message text.
5. Establish a baseline pipeline that can be applied to new, unlabelled messages going forward.

---


## Tools I Used

For my deep dive in dealing with unstractured text data, I harnessed the power of several key tools: 

- Python: The backbone of my analysis, allowing me to analyse the data and find critical insights. The following libraries led to the success of my investigation: 
    - Pandas Library: For data manipulation
    - Matplotlib Library: For visualisation of the data
    - Seaborn Library: For advanced visuals
- Jupyter Notebook: The tool I used to run my Python scripts which let me easily include my notes and analysis.
- Visual Studio Code: My go-to for executing my Python scripts.
Git & GitHub: Essential for version control and sharing my Python code and analysis, ensuring collaboration and project tracking.

---
## The Analysis
---
### 1. Understanding the distribution of sentiment across a social media message dataset spanning multiple social platforms.
---
To fully create an ML pipeline, one has to understand message distribution across various social platforms. This is an important step as it will allow us to learn about different sentiment classes and their unique values, and how to later balance these classes to avoid model bias.

#### Visualise Data
```
platform_counts = df_sent['snTypeColumn'].value_counts()
platform_pct = (platform_counts / platform_counts.sum() * 100).reset_index()
platform_pct.columns = ['Platform', 'percentage']
platform_pct = platform_pct.sort_values('percentage', ascending=True)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

sns.barplot(data=platform_pct, x='percentage', y='Platform', ax=ax[0], hue='Platform', palette='dark:b_r', legend=False)
sns.despine()
ax[0].set_title('Message Source')
ax[0].set_xlabel('% of messages')
ax[0].set_ylabel('')
ax[0].set_xlim(0, platform_pct['percentage'].max() * 1.15)
ax[0].invert_yaxis()

for n, v in enumerate(platform_pct['percentage']):
    ax[0].text(v + 0.5, n, f'{v:.1f}%', va='center')

df_sent['Sentiment'].value_counts().plot(kind='pie', startangle=90, autopct='%1.1f%%', ax=ax[1], ylabel='', title='Sentiment Proportion')

fig.tight_layout()
plt.show()
```

#### Results

![message source & sentiment split](images\sentiment_split.png)
*Bar graph represents the platforms with the highest volume of conversations. Pie chart gives us the breakdown of the customer sentiment.*

### Insights: 

Volume
- Facebook dominates overwhelmingly, Twitter is a distant second at 27.5%. 
- Together, Facebook and Twitter make up over 96% of all messages. 
- Instagram (3.2%) is a minor contributor, while LinkedIn, TikTok, and YouTube are essentially negligible.
Takeaway: Any analysis or strategy based on this dataset is really a Facebook/Twitter story — conclusions may not generalize well to platforms like TikTok or YouTube given their tiny sample sizes.

Sentiment Proportion

- Neutral sentiment is the largest segment at 50.4%, meaning half the conversation don't lean clearly positive or negative.
- Positive sentiment (32.6%) nearly doubles negative sentiment (17.0%), suggesting the overall tone skews favorable when sentiment is expressed.

### 2. Identify which platforms contribute most to each sentiment category.

---

## Data Cleaning & Preparation

This section explains how raw, messy data was transformed into a usable, trustworthy dataset.

### 3.1 Removing irrelevant categories
The `Uncategorized` sentiment class (1,632 rows) was removed entirely, since it does not represent a genuine sentiment label and would confuse a classifier trained to distinguish Positive/Negative/Neutral.

### 3.2 Column renaming
Columns were renamed for clarity and ease of coding: `Conversation Stream` → `message`, `snTypeColumn` → `channel`, `Created Time` → `date`.

### 3.3 Date conversion
`date` was converted from a raw string to a proper `datetime` type using `pd.to_datetime()`. This column was retained as metadata (useful for future time-based analysis, e.g. sentiment shifts around specific events) but was **not** used as a model input feature, since a raw timestamp carries no direct sentiment signal.

### 3.4 Text cleaning (NLTK pipeline)
Each message was processed through a custom cleaning function prior to modeling:

1. HTML entity decoding (e.g. `&#39;` → `'`)
2. URL and @mention removal
3. Lowercasing
4. Removal of punctuation/numbers (accented characters preserved, to avoid corrupting non-English text such as French messages present in the data)
5. Tokenization (`nltk.word_tokenize`)
6. English stopword removal
7. Lemmatization (`WordNetLemmatizer`)

**Known limitation:** this pipeline strips emojis rather than converting them to sentiment-bearing tokens, and is English-tuned — non-English messages (e.g. French) are only partially handled (stopwords/lemmatization do not apply correctly to non-English tokens). This is documented further in Section 8 (Limitations).

---

## 4. Class Balancing

**Purpose of this section:** Justify how class imbalance was addressed, since imbalanced training data biases a classifier toward the majority class regardless of the algorithm chosen.

The original distribution was imbalanced (Neutral nearly 3x the size of Negative). Two options were considered:

| Method | Description | Decision |
|---|---|---|
| Class weighting (`class_weight='balanced'`) | Keep all data, penalize minority-class errors more during training | Not used as primary method |
| **Random undersampling** | Sample an equal number of rows per class | **Selected approach** |

**Rationale:** the smallest class (Negative, 3,944 rows) was large enough that undersampling to 3,500 rows per class only sacrificed ~444 rows — a minor cost relative to the benefit of a genuinely balanced, easier-to-interpret training and evaluation set. Undersampling was implemented via a per-group `sample()` function, producing a balanced dataset of **10,500 rows** (3,500 each of Positive, Negative, Neutral).

---

## 5. Feature Engineering

**Purpose of this section:** Explain how cleaned text was converted into a numeric format suitable for machine learning algorithms, since models cannot operate on raw text directly.

**Method used:** TF-IDF (Term Frequency – Inverse Document Frequency) vectorization.

- **Term Frequency** captures how often a word appears in a given message.
- **Inverse Document Frequency** down-weights words that appear across most messages (less informative) and up-weights rarer, more discriminative words.
- **Configuration:** `max_features=5000`, `ngram_range=(1,2)` (captures both single words and two-word phrases, e.g. "not good"), `min_df=2` (filters out words appearing in only one document, reducing noise from typos/one-off tokens).

**CountVectorizer note:** TF-IDF was used as a complete, standalone alternative to CountVectorizer — it does not require a separate counting step, as it performs counting internally before applying the IDF weighting.

---

## 6. Modeling Approach

**Purpose of this section:** Document which algorithms were tested and why, so future readers understand the reasoning behind model selection rather than just the final choice.

A stratified 80/20 train/test split was used throughout (`stratify=Sentiment`), ensuring both sets preserved equal class proportions. All preprocessing and modeling steps were combined into `scikit-learn` `Pipeline` objects to prevent data leakage (the vectorizer is fit only on training data) and to simplify future re-use.

Three models were trained and compared:

| Model | Rationale |
|---|---|
| **Logistic Regression** | Fast, interpretable baseline; strong general performance on TF-IDF text features |
| **Linear SVM (`LinearSVC`)** | Well-suited to high-dimensional, sparse TF-IDF data; historically a strong classical performer for text classification |
| **Random Forest** | Tested as a non-linear comparison point; tree-based models are generally less suited to sparse, high-dimensional text features, so this was included primarily to confirm that empirically rather than assume it |

**Random Forest tuning:** `GridSearchCV` (5-fold cross-validation, `f1_macro` scoring) was used to search over `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf`.

---

## 7. Results

**Purpose of this section:** Present findings objectively, separate from interpretation or recommendations (which follow in Sections 8–9).

- All three models (Logistic Regression, Linear SVM, Random Forest — including the tuned version) produced **broadly similar performance**, with no model showing a decisive advantage over the others.
- Confusion matrices (row-normalized, i.e. showing % of actual class predicted as each label) were used as the primary comparison visual across all three models, since they reveal *which* sentiment classes get confused with which — more diagnostic than accuracy alone.
- A 2D visualization of the TF-IDF feature space (via `TruncatedSVD` dimensionality reduction) was also explored, colored by actual sentiment and by prediction correctness — though this was noted as a simplification that loses substantial information when compressing thousands of TF-IDF dimensions down to two, and should not be over-interpreted as the model's true decision boundary.

*(Insert final numeric results — accuracy, per-class precision/recall/F1, and confusion matrix images — here once finalized, so this report reflects concrete figures rather than qualitative description alone.)*

---

## 8. Limitations

**Purpose of this section:** Transparently document weaknesses in the current approach, since scientific reporting requires acknowledging what could affect the validity or generalizability of results.

1. **Emoji handling:** the NLTK-based cleaning pipeline strips emojis rather than preserving their sentiment meaning. Some messages (e.g. entirely emoji-based) may lose all usable signal after cleaning.
2. **Multilingual content:** the dataset contains non-English text (e.g. French). Current preprocessing (stopwords, lemmatization) is English-only, meaning non-English messages are incompletely processed.
3. **Undersampling discards data:** roughly 14,000 real Neutral/Positive messages were excluded from training to achieve class balance. These were not used to validate the model against the natural, real-world class distribution.
4. **Feature ceiling:** the similarity in performance across three different algorithm types (linear, and tree-based) suggests the limiting factor is more likely the **feature representation (TF-IDF) and data characteristics** (short, noisy, informal social text) rather than model choice.
5. **Possible near-duplicate/spam content:** some rows appeared to be repeated or bot-like (e.g. emoji-only spam), which was noted but not formally deduplicated before splitting data.

---

## 9. Recommendations

**Purpose of this section:** Translate findings and limitations into concrete next steps.

1. **Validate against the natural distribution:** evaluate the trained model against the full, untouched imbalanced dataset (not just the balanced subset) to understand real-world performance.
2. **Test class weighting as an alternative to undersampling**, to compare against the current approach without discarding data.
3. **Consider a pretrained transformer model** (e.g. `cardiffnlp/twitter-roberta-base-sentiment`), purpose-built for informal social media text, as the next step given that classical TF-IDF models have plateaued at similar performance regardless of algorithm.
4. **Improve emoji handling** by converting emojis to descriptive text tokens (e.g. via the `emoji` library) rather than stripping them, to preserve sentiment signal in emoji-heavy or emoji-only messages.
5. **Add language detection and routing** to properly handle non-English messages, either via translation, multilingual stopword lists, or a multilingual model.
6. **Deduplicate near-identical/spam messages** prior to train/test splitting to avoid inflated performance from repeated content appearing in both sets.



## USE FOR INSIGHTS

**Purpose of this section:** Document what the raw data looked like, so any cleaning or modeling decisions later can be traced back to a specific data characteristic.

| Column | Description |
|---|---|
| `Conversation Stream` (later renamed `message`) | Raw text of the social media message/comment |
| `Sender Profile Image Url` | Dropped — not relevant to sentiment |
| `Created Time` (later renamed `date`) | Timestamp of when the message was posted |
| `Permalink` | Dropped — not relevant to sentiment |
| `snTypeColumn` (later renamed `channel`) | Social media platform the message originated from |
| `Associated Cases` | Dropped — not relevant to sentiment |
| `Sentiment` | Target label: Positive, Negative, Neutral, or Uncategorized |
| `Mentions (SUM)` | Dropped — not relevant to sentiment |

**Original dataset size:** 24,903 rows, 8 columns.

**Missing data:** `Conversation Stream` had 1,636 missing (NaN) values overall. Missingness was checked *per sentiment group* to determine whether missing message text was concentrated in a particular category (relevant because `Uncategorized` rows plausibly lack message text as the reason they were never categorized in the first place).

**Original sentiment distribution:**

| Sentiment | Count |
|---|---|
| Neutral | 11,735 |
| Positive | 7,592 |
| Negative | 3,944 |
| Uncategorized | 1,632 |

**Platform (channel) distribution:** Facebook (68.6%) and Twitter (27.5%) dominate the dataset, with Instagram, LinkedIn, TikTok, and YouTube together making up under 4% combined.

---