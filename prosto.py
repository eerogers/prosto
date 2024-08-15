#import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, plot_confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from string import punctuation
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re  # Regular expressions
#warnings.filterwarnings("ignore")
np.random.seed(123)

data = pd.read_csv("data/labeledTrainData.tsv", sep='\t')
#print(data.head())
#print(data.shape)
#print(data.isnull().sum())
#print(data.sentiment.value_counts())
#print("hello")
stop_words = stopwords.words('english')

def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r'http\S+',' link ', text)
    text = re.sub(r'\b\d+(?:\.\d+)?\s+', '', text)  # Remove numbers
    text = ''.join([c for c in text if c not in punctuation])
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)

    return text

data["cleaned_review"] = data["review"].apply(text_cleaning)
print(data.head())
X = data["cleaned_review"]
y = data.sentiment.values
# split data into train and validate
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42,
    shuffle=True,
    stratify=y,
)
# Create a classifier in pipeline
sentiment_classifier = Pipeline(steps=[('pre_processing',TfidfVectorizer(lowercase=False)),('naive_bayes',MultinomialNB())])
# train the sentiment classifier 
sentiment_classifier.fit(X_train,y_train)
# test model performance on valid data 
y_preds = sentiment_classifier.predict(X_valid)
ac = accuracy_score(y_valid,y_preds)
print(ac)
#save model 
import joblib 
joblib.dump(sentiment_classifier, 'models/sentiment_model_pipeline.pkl')
