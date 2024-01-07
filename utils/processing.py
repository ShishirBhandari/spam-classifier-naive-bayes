import nltk
nltk.download("punkt")
nltk.download("stopwords")
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk import word_tokenize


stemmer = PorterStemmer()


def word_stemmer(words):
    stems = [stemmer.stem(word) for word in words]
    return stems


def clean_text(text):
    text = word_tokenize(text.lower())
    
    # Remove punctuation and stop words
    new = []
    for token in text:
        if token in stopwords.words("english") or token in string.punctuation:
            continue
        
        new.append(token)

    # Stemming
    text = word_stemmer(new)

    return " ".join(text)
