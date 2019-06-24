import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet') # download for lemmatization
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

text = "The first time you see The Second Renaissance it may look boring. Look at it at least twice and definitely watch part 2. It will change your view of the matrix. Are the human people the ones who started the war ? Is AI a bad thing ?"

# Normalize text
text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

# Tokenize text
words = text.split()
print(words)

# Remove stop words
words = [w for w in words if w not in stopwords.words("english")]
print(words)

# Reduce words to their stems
stemmed = [PorterStemmer().stem(w) for w in words]
print(stemmed)

# Reduce words to their root form
lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
print(lemmed)

# Lemmatize verbs by specifying pos
lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]
print(lemmed)