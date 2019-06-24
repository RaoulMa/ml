import nltk
nltk.download('punkt')
nltk.download('stopwords')
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

text = "The first time you see The Second Renaissance it may look boring. Look at it at least twice and definitely watch part 2. It will change your view of the matrix. Are the human people the ones who started the war ? Is AI a bad thing ?"
print(text)

# Normalize text
text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

# Tokenize text
words = word_tokenize(text)
print(words)

# Remove stop words
words = [w for w in words if w not in stopwords.words("english")]
print(words)

