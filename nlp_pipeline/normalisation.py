import re

text = "The first time you see The Second Renaissance it may look boring. Look at it at least twice and definitely " \
       "watch part 2. It will change your view of the matrix. Are the human people the ones who started the war ? " \
       "Is AI a bad thing ?"
print(text)

# Convert to lowercase
text = text.lower()
print(text)

# Remove punctuation characters
text = re.sub(r"[^a-zA-Z0-9]", " ", text)
print(text)