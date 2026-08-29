# Social Media Sentiment Analysis — Project Report

## Overview
This project analyses **real, user-generated social media text data** - actual customer messages, comments, and complaints collected across social platforms (Facebook, Twitter, Instagram, LinkedIn, TikTok, and YouTube) rather than simulated data. Because the data is real, it carries the characteristics typical of authentic social media content: informal language, spelling inconsistencies, emojis, mixed languages, missing values, competition and spam-like entries. The end goal is to build a machine learning pipeline capable of automatically classifying the sentiment (Positive, Negative, or Neutral) of new, unseen messages of the same real-world nature.

---

## Objectives
Below are the objectives of the project:

1. Understand the distribution of sentiment across a social media message dataset spanning multiple social platforms.
2. Identify which platforms contribute most to each sentiment category.
3. Prepare the dataset for machine learning by cleaning, balancing, and vectorizing message text.
4. Model selection and pipeline setup.

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

### 1. Understanding the distribution of sentiment across a social media message dataset spanning multiple social platforms.

To fully create an ML pipeline, one has to understand message distribution across various social platforms. This is an important step as it will allow us to learn about different sentiment classes and their unique values, and how to later balance these classes to avoid model bias.

#### Visualise data

```python
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

#### Insight:

Volume
- Facebook dominates overwhelmingly, Twitter is a distant second at 27.5%. 
- Together, Facebook and Twitter make up over 96% of all messages. 
- Instagram (3.2%) is a minor contributor, while LinkedIn, TikTok, and YouTube are essentially negligible.

Sentiment Proportion

- Neutral sentiment is the largest segment at 50.4%, meaning half the conversation don't lean clearly positive or negative.
- Positive sentiment (32.6%) nearly doubles negative sentiment (17.0%), suggesting the overall tone skews favorable when sentiment is expressed.

### 2. Identify which platforms contribute most to each sentiment category.

Exploring message distribution further, we now look at the different sentiment classes and how social platforms fare. This is a critical piece as it informs us of the different platform strengths, and highlights composition differences to account for during preprocessing. Raw class counts alone don't reveal whether a class is dominated by a single platform's writing style, which risks the model learning platform artifacts instead of genuine sentiment signal.

#### Results

![platform share of messages](images\2.png)
*Figure: Platform share of messages by sentiment category - percentages reflect each platform's contribution within a given sentiment class.*

#### Insights:

**Negative sentiment** — driven almost entirely by Facebook.

- Facebook contributes 79% of the negative chatter, by far the dominant source.
- Twitter follows at 17%, and Instagram a minor 4%.
- LinkedIn, TikTok, and YouTube are negligible (<1%).

**Neutral sentiment** — Facebook Dominates.

- Facebook accounts for 87% of neutral messages — the highest concentration of any category.
- Twitter (10%) and Instagram (3%) are minor by comparison.

**Positive sentiment** — Twitter leads

- Twitter contributes 60% of positive chatter, making it the dominant platform. The only sentiment category where Facebook isn't the top contributor.
- Facebook still contributes a substantial 35%. Instagram is a minor 4%.

Key takeaway: Facebook drives complaints and routine chatter, Twitter drives the positive buzz (tied to campaign hype & competition entries).

### 3. Preparing the data through cleaning, class-balancing & text vectorisation.

Earlier, we found that sentiment classes were not evenly distributed - neutral carries significantly more weight than positive or negative. Passing the data through as-is risks a model that is doubly biased.

To address this, we identified the sentiment class with the lowest count and used it as the target size for resampling, penalising the over-represented classes. The negative class had the lowest count at 3,944 messages, so this became our benchmark for building a balanced dataset:

```python
# creating a balanced dataset
def balanced_sample(df, target_n):
    group_col= 'Sentiment'
    
    parts = []
    for value, group in df.groupby(group_col):
        parts.append(group.sample(n=min(len(group), target_n)))
    return pd.concat(parts).reset_index(drop=True)

df_balanced = balanced_sample(df_ML, 3500)
print(df_balanced['Sentiment'].value_counts())

```

With a balanced class distribution in place, the next step was preparing the raw text itself. We used the NLTK library to apply standard text-processing techniques to strip noise from the messages before vectorisation.

**What NLP pipeline does** — lowercasing, removing URLs/mentions/hashtags, stripping punctuation and extra whitespace, tokenising, removing stopwords, and lemmatising.


### 4. Model Selection & Pipeline Setup

With a balanced, cleaned dataset in hand, the next step was splitting the data and defining the classification pipeline.

We split the cleaned messages and their sentiment labels into training and test sets using an 80/20 split.

```python
X_train, X_test, y_train, y_test = train_test_split(
    df_balanced['message_clean'],
    df_balanced['Sentiment'],
    test_size=0.2,
    stratify=df_balanced['Sentiment'],
    random_state=42
)
```

For classification, we chose **Logistic Regression** and **TF-IDF** as a classifier:

```python
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])
```

**TF-IDF (Term Frequency–Inverse Document Frequency)** converts each cleaned message into a numerical vector, weighting words by how important they are to a given message relative to the full dataset — common words across all messages are down-weighted, while distinctive words that help distinguish sentiment are up-weighted.

#### Model fitting & predictions

```python
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

```
#### Results

![logistic_classifier](images\logistic_reg.png)
*Figure: Confusion matrix for the Logistic Regression sentiment classifier. Rows represent actual sentiment labels; columns represent predicted labels.*

#### Insights:

- Negative: 594 of 698 actual negative messages were correctly classified (~85% accuracy for this class).
- Neutral: 433 of 579 actual neutral messages were correctly classified (~75%). Neutral is most often confused with Negative (131 cases), and to a lesser extent Positive (15 cases). This is a common pattern in sentiment classification — neutral is inherently the "fuzziest" class, sitting on the boundary of both.
- Positive: 567 of 659 actual positive messages were correctly classified (~86% accuracy).
- **accuracy**: ~82% across all classes (1,594 correct out of 1,936 total test messages).

---
## What I Learned

- **Text cleaning is not one-size-fits-all.** Dealing with real, informal social data meant standard NLP steps (lowercasing, stopword removal, lemmatisation) had to be applied thoughtfully rather than blindly, since aggressive cleaning risks stripping signal along with noise.

- **Confusion matrices tell a more honest story than accuracy alone.** An 82% overall accuracy masked meaningfully different performance per class — Neutral (~75%) underperformed Negative and Positive (~85–86%). Reading the matrix, not just the headline number, revealed *where* the model struggles and *why* (Neutral being the boundary case between the other two).

---

## Challenges I Faced

- **Real, unstructured social data is messy by nature.** Informal language, spelling inconsistencies, emojis, mixed languages, and spam/competition entries made cleaning far less straightforward than working with curated or simulated datasets.

- **Neutral as an inherently fuzzy category.** Sentiment on the boundary between mildly negative and mildly positive is genuinely ambiguous even to a human reader in places, which likely sets a practical ceiling on how well any model can classify it without additional context (e.g. sarcasm, tone, conversation history).

---

## Conclusion

This project built an end-to-end sentiment classification pipeline on real, multi-platform social media data; from understanding sentiment and platform distribution, through addressing class imbalance and platform-driven bias, to a working Logistic Regression classifier achieving ~82% accuracy. The analysis surfaced two distinct insights with practical value beyond the model itself: Facebook and Twitter play structurally different roles in the conversation (support/complaints vs. positive engagement), and Neutral sentiment remains the hardest class to classify reliably.

Next steps would include exploring platform-stratified sampling more rigorously, testing whether a more expressive model (e.g. an ensemble method or transformer-based embedding) improves Neutral-class performance.