import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

text = "Dr. Smith graduated from the University of Washington. He later started an analytics firm called Lux, which catered to enterprise customers."
print(text)

# Split text into words using NLTK
words = word_tokenize(text)
print(words)

# Split text into sentences
sentences = sent_tokenize(text)
print(sentences)

