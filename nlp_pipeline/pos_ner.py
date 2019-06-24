import nltk
from nltk.tokenize import word_tokenize
nltk.download('words')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')

from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize

text = "I always lie down to tell a lie."

# tokenize text
sentence = word_tokenize(text)

# tag each word with part of speech
print(pos_tag(sentence))

text = "Antonio joined Udacity Inc. in California."

# tokenize, pos tag, then recognize named entities in text
tree = ne_chunk(pos_tag(word_tokenize(text)))
print(tree)

