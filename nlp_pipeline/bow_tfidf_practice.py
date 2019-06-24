import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

corpus = ["The first time you see The Second Renaissance it may look boring.",
        "Look at it at least twice and definitely watch part 2.",
        "It will change your view of the matrix.",
        "Are the human people the ones who started the war?",
        "Is AI a bad thing ?"]

stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

def tokenize(text):
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens

for sentence in corpus:
    print(tokenize(sentence))

# initialize count vectorizer object
vect = CountVectorizer(tokenizer=tokenize)

# get counts of each token (word) in text data
X = vect.fit_transform(corpus)

# convert sparse matrix to numpy array to view
print(X.toarray())

# view token vocabulary and counts
print(vect.vocabulary_)

# initialize tf-idf transformer object
transformer = TfidfTransformer(smooth_idf=False)

# use counts from count vectorizer results to compute tf-idf values
tfidf = transformer.fit_transform(X)

# convert sparse matrix to numpy array to view
print(tfidf.toarray())

