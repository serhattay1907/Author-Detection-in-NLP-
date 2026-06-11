#!/usr/bin/env python3
"""
Author Detection using NLP
Notebook converted to Python script
Original: SERHAT_TAY_Author_Detection_using_NLP.ipynb
"""

# ============================================================================
# This script was converted from Jupyter Notebook
# Note: Some paths may need adjustment for your local environment
# Original paths used Google Colab format: /content/...
# ============================================================================


# Cell 1
#this is where we download the dataset
!gdown https://drive.google.com/uc?id=120-9PFzIvWy1SVHpTiOO6_7vOyMpXZEy

# Cell 2
# Plus this is where we unzip the content as below
!unzip /content/30Columnists.zip -d /content

# Cell 3
# Read sub-dirs as author names, i.e. labels
import os
data_path = "/content/30Columnists/raw_texts";

labels = [];
for author in os.listdir(data_path):
  labels.append(author);

labels.sort(key=lambda e : int(e));
print(labels)

# Cell 4
# Read articles of each author
import re
dataset = {}; # {author:[docs]}
for author in labels:
    docs = [];
    for file in os.listdir(os.path.join(data_path, author)):
      if file.endswith('.txt'):
        with open(os.path.join(data_path, author, file), 'rb') as f:
          # read the article line by line

          # "cp1254" solves \x92, \x93 etc issue
          article = f.read().decode('cp1254').replace('\n', '').replace('\r', '').replace('\f', '');
#          article = f.read().decode('iso-8859-9').replace('\n', '').replace('\r', '').replace('\f', '');
#          article = f.read().decode('cp1254');

          # make a basic cleaning, i.e., remove non printable chars like lf, cr, ff etc.
          article = re.sub(r"[\n\r\f]", "", article);
          docs.append(article);
#    print("# of files red for author {} is {}".format(author, len(docs)));
    dataset.update({author:docs});

# print the number of articles red for each author
for author in labels:
  print("{} \t {}".format(author, len(dataset[author])));

# Cell 5
for author in dataset:
  print("%s\t" % author, *[text[:30] for text in dataset[author][0:10]], sep=" | ");

# Cell 6
import pandas as pd

df = pd.DataFrame.from_dict(dataset); # doc-by-author table
#df.head()
df

# Cell 7
import nltk
nltk.download('punkt'); # NLTK uses punkt (punktuation) tokenizer, so it needs the data.

# Cell 8
import nltk
nltk.download('punkt'); # Ensure punkt tokenizer data is available
nltk.download('punkt_tab'); # Download the specific punkt_tab resource
from nltk import word_tokenize
# Word Tokenization for a single document, the first document of the first author
word_tokens = word_tokenize(dataset['1'][0])
word_tokens

# Cell 9

import nltk # Just import nltk library

word_tokens = nltk.word_tokenize(dataset['1'][0])
word_tokens

# Cell 10
# You can use nltk class to access word_tokenize function or other functions
import nltk # Just import nltk library

# Extra: to wrap text to next line in output
import textwrap

# Tokenize the text using "nltk.word_tokenize()" function
sentences = nltk.sent_tokenize(dataset['1'][0])
for no, sentence in enumerate(sentences):
  print(f"Sentence {no+1}")
  wrapped_text = textwrap.fill(sentence, width=70)
  print(wrapped_text)
  print()

# Cell 11
import nltk

# Tokenize all of the articles for each author at once
tokenized_dataset = {author: [nltk.word_tokenize(text) for text in texts] for author, texts in dataset.items()}

# Print the first 5 tokens of the first 5 articles for each author
for author in tokenized_dataset:
  print("%s\t" % author, *[text[:5] for text in tokenized_dataset[author][0:5]], sep=" | ");

# Cell 12
# First, create a long token list for author individually
author_token_list = {author: [x for X in article_tokens for x in X] for author, article_tokens in tokenized_dataset.items()}

# Print token list for each author
for author, tokens in author_token_list.items():
  print(f"Token list for {author}: {tokens[:100]}")

# Cell 13
# Then, create a long token list for all articles

token_list = [token for tokens in author_token_list.values() for token in tokens]

print(f"Token list for all articles: {token_list[:100]}")

# Cell 14
print(f"Total number of tokens: {len(token_list)} \n")

for author, tokens in author_token_list.items():
  print(f"Number of tokens for author {author}: {len(tokens)}")

# Cell 15
# Check frequency distribution of the tokens/words of the article
from nltk.probability import FreqDist
fdist = FreqDist(token_list);
fdist.most_common(20)

# Cell 16
# Accept only letter and number sequences as tokens
tokens_list_no_punc = [token for token in token_list if token.isalnum()];

print(f"Number of tokens after removing punctuation: {len(tokens_list_no_punc)}")

# Cell 17
from nltk.probability import FreqDist
fdist = FreqDist(tokens_list_no_punc);
fdist.most_common(20)

# Cell 18
token_list_lower = [token.lower() for token in tokens_list_no_punc]

print(f"Number of tokens after case folding: {len(token_list_lower)}")

# Cell 19
from nltk.probability import FreqDist
fdist = FreqDist(token_list_lower);
# Print total number of tokens
print(f"Total number of tokens: {len(token_list_lower)}")
print()
# Print most common tokens
fdist.most_common(20)

# Cell 20
import nltk
nltk.download('stopwords');

# Cell 21
# Remove stopwords
from nltk.corpus import stopwords

stopword_list = stopwords.words('english')

token_list_clean = [token for token in token_list_lower if token not in stopword_list]

print(f"Number of tokens after stopword removal: {len(token_list_clean)}")


# Cell 22
# Apply nltk stemmer to token_list_clean
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
token_list_stemmed = [stemmer.stem(token) for token in token_list_clean]

print(f"Number of tokens after stemming: {len(token_list_stemmed)}")

# Cell 23
from nltk.probability import FreqDist
fdist = FreqDist(token_list_stemmed);
# Print total number of tokens
print(f"Total number of tokens: {len(token_list_stemmed)}")
print()
# Print most common tokens
print("Most common 20 tokes")
fdist.most_common(20)

# Cell 24
import nltk
nltk.download('wordnet')

# Cell 25
# Apply nltk lemmatizer to token_list_clean
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
token_list_lemmatized = [lemmatizer.lemmatize(token) for token in token_list_clean]

print(f"Number of tokens after lemmatization: {len(token_list_lemmatized)}")

# Cell 26
from nltk.probability import FreqDist
fdist = FreqDist(token_list_lemmatized);
# Print total number of tokens
print(f"Total number of tokens: {len(token_list_lemmatized)}")
print()
# Print most common tokens
print("Most common 20 tokes")
fdist.most_common(20)

# Cell 27
# Print raw token list for each author
for author, tokens in author_token_list.items():
  print(f"Token list for {author}: {tokens[:100]}")

# Cell 28
# Preprocess token list of each author
# Just lowercase: no punctuation removal, no stopword elimination
author_token_list_clean = {author: [token.lower() for token in tokens] for author, tokens in author_token_list.items()}

# Cell 29
# Print token list for each author
for author, tokens in author_token_list_clean.items():
  print(f"Token list for {author}: {tokens[:100]}")

# Cell 30
# Number of tokens in all articles of each author
n_author = {author: len(tokens) for author, tokens in author_token_list_clean.items()}
n_author

# Cell 31
import nltk
from nltk.probability import FreqDist

for author, tokens in author_token_list_clean.items():
  fdist = FreqDist(tokens);
  print(f"Most frequent tokens for {author}: {fdist.most_common(10)}")

# Cell 32
import nltk
from nltk.probability import FreqDist

for author, tokens in author_token_list_clean.items():
  fdist = FreqDist(tokens);
  print(f"Most frequent tokens for {author}: {fdist.hapaxes()[:10]}")

# Cell 33
# Tokenization with handling issues like "word.word"

import nltk
import re

pattern = r"([a-zA-Z]+)\.([a-zA-Z]+)"

def tokenize_texts(texts):
  tokenized_texts = []
  for text in texts:
    tokens = nltk.word_tokenize(text)
    for token in tokens:
      match = re.match(pattern, token)
      if match:
        tokens.remove(token)
        tokens.extend(match.groups())
    tokenized_texts.append(tokens)
  return tokenized_texts

# Tokenize texts for each author
tokenized_dataset = {}
for author, texts in dataset.items():
  tokenized_dataset[author] = tokenize_texts(texts)

# Cell 34
# Print the first 5 tokens of the first 5 articles for each author
for author in tokenized_dataset:
  print("%s\t" % author, *[text[:5] for text in tokenized_dataset[author][0:5]], sep=" | ");

# Cell 35
# Read articles of each author
import re
dataset = {}; # {author:[article1, article2, ...]}
for author in labels:
    docs = [];
    for file in os.listdir(os.path.join(data_path, author)):
      if file.endswith('.txt'):
        with open(os.path.join(data_path, author, file), 'rb') as f:
          # read the article line by line

          # "cp1254" solves \x92, \x93 etc issue
          article = f.read().decode('cp1254');
#          article = f.read().decode('iso-8859-9');
#          article = f.read().decode('utf-8');

          # make a basic cleaning, i.e., remove non printable chars like lf, cr, ff etc.
          article = re.sub(r"[\n\r\f]", "", article);

          # translate "word.word" to "word. word"
          r"([a-zA-Z]+)\.([a-zA-Z]+)"
          article = re.sub(r"([a-zA-Z]+)\.([a-zA-Z]+)", r"\1. \2", article);
          docs.append(article);
#    print("# of files red for author {} is {}".format(author, len(docs)));
    dataset.update({author:docs});

# print the number of articles red for each author
for author in labels:
  print("{} \t {}".format(author, len(dataset[author])));

# Cell 36
from sklearn.feature_extraction.text import TfidfVectorizer

# Prepare data for TF-IDF: flatten articles and create labels
all_articles = []
all_authors = []

for author, articles in dataset.items():
    for article in articles:
        all_articles.append(article)
        all_authors.append(author)

print(f"Total number of articles: {len(all_articles)}")
print(f"Total number of author labels: {len(all_authors)}")

# Cell 37
# Initialize TfidfVectorizer for word n-grams (unigram and bigram)
# We'll use our preprocessing steps here (lowercase, stopword removal, handling word.word done in dataset loading)
word_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    ngram_range=(1, 2) # Unigrams and Bigrams
)

# Fit and transform the articles to get TF-IDF features
tfidf_word_features = word_vectorizer.fit_transform(all_articles)

print(f"Shape of TF-IDF word features matrix: {tfidf_word_features.shape}")
print(f"Number of unique word features: {len(word_vectorizer.get_feature_names_out())}")

# Cell 38
# Initialize TfidfVectorizer for character n-grams (range 3 to 5)
# Character n-grams are often less sensitive to misspellings and can capture stylistic elements
char_vectorizer = TfidfVectorizer(
    lowercase=True,
    analyzer='char_wb', # 'char_wb' for character n-grams within word boundaries
    ngram_range=(3, 5) # Character n-grams from length 3 to 5
)

# Fit and transform the articles to get TF-IDF features
tfidf_char_features = char_vectorizer.fit_transform(all_articles)

print(f"Shape of TF-IDF character features matrix: {tfidf_char_features.shape}")
print(f"Number of unique character features: {len(char_vectorizer.get_feature_names_out())}")

# Cell 39
import nltk

# Tokenize all of the articles for each author at once
tokenized_dataset = {author: [nltk.word_tokenize(text) for text in texts] for author, texts in dataset.items()}

# Cell 40
# First, create a long token list for author individually
author_token_list = {author: [x for X in article_tokens for x in X] for author, article_tokens in tokenized_dataset.items()}

# Print token list for each author
for author, tokens in author_token_list.items():
  print(f"Token list for {author}: {tokens[:100]}")

# Cell 41
# Visually check sentence boundiries
import nltk
import textwrap

sentences = nltk.sent_tokenize(dataset['1'][0])

for no, sentence in enumerate(sentences):
  print(f"Sentence {no+1}")
  wrapped_text = textwrap.fill(sentence, width=70)
  print(wrapped_text)
  print()

# Cell 42
import nltk

author_article_sentence_list = {author: [nltk.sent_tokenize(text) for text in texts] for author, texts in dataset.items()}

# Number of sentences for each article of the first author
print(f"# of sentences in the first article of the first author: {len(author_article_sentence_list['1'][0])}")
print()
print(author_article_sentence_list['1'][0])


# Cell 43
# First, create a long token list for authors individually
author_token_list = {author: [x for X in article_tokens for x in X] for author, article_tokens in tokenized_dataset.items()}

# Print token list for each author
for author, tokens in author_token_list.items():
  print(f"Token list for {author}: {tokens[:100]}")

# Cell 44
# Least frequent tokens, i.e., hapaxes
import nltk
from nltk.probability import FreqDist

for author, tokens in author_token_list.items():
  fdist = FreqDist(tokens);
  print(f"Least frequent tokens for {author}: {fdist.hapaxes()[:10]}")

# Cell 45
# Preprocess token list of each author
# Just lowercase: no punctuation removal, no stopword elimination
author_token_list_clean = {author: [token.lower() for token in tokens] for author, tokens in author_token_list.items()}

# Cell 46
# Number of tokens in all articles of each author
n_author = {author: len(tokens) for author, tokens in author_token_list_clean.items()}
n_author

# Cell 47
author_vocab = {author: set(tokens) for author, tokens in author_token_list_clean.items()}

# Cell 48
for author, vocab in author_vocab.items():
  print(f"Vocabulary of {author}: {vocab}")
  print(f"Size of vocabulary of {author}: {len(vocab)}")
  print()

# Cell 49
# For each word in the vocabulary of every author, unigram probabilities

unigram_probs = {}

for author, tokens in author_token_list_clean.items():
  fdist = FreqDist(tokens)
  unigram_probs[author] = {word: fdist[word]/n_author[author] for word in fdist}

  print(f"Unigram probs for {author}: {unigram_probs[author]}")
  print()

# Cell 50
# Long list of sentences for each author
# Prefix "_bos_" to each sentence
author_sentence_list = {author: ["_bos_ " + x.strip() for X in article_sentences for x in X] for author, article_sentences in author_article_sentence_list.items()}

# Print the first 5 sentences of the first 5 articles for each author
for author, sentences in author_sentence_list.items():
  print(f"Author {author}")
  print(f"Total # of sentences: {len(sentences)}")
  print(f"First 5 sentences: {sentences[:5]}")
  print()

# Cell 51
import nltk

# To speed-up
tokenizer = nltk.WordPunctTokenizer()

def bigrams(sentence):
  return [tuple(sentence[i:i+2]) for i in range(len(sentence)-1)]

# Tokenize each sentence into a list of bigrams
author_sentence_bigram_list = {author: [bigrams(tokenizer.tokenize(sentence)) for sentence in sentences] for author, sentences in author_sentence_list.items()}

# Cell 52
# Print bigrams for the first author
print(f"Author: {list(author_sentence_bigram_list.keys())[0]}")
print(f"Bigrams of the first sentence: {author_sentence_bigram_list[list(author_sentence_bigram_list.keys())[0]][0]}")

# Cell 53
# Print the total number of bigrams for each author
for author, bigrams_list in author_sentence_bigram_list.items():
  print(f"Author: {author}")
  print(f"Total # of bigrams: {sum([len(bigrams) for bigrams in bigrams_list])}")
  print()

# Cell 54
# Long list of bigrams in all articles for each author
author_bigram_list = {author: [bigram for X in article_bigrams for bigram in X] for author, article_bigrams in author_sentence_bigram_list.items()}

# Cell 55
# Print the first 50 bigrams of each author
for author, bigrams in author_bigram_list.items():
  print(f"Author: {author}")
  print(f"First 50 bigrams: {bigrams[:50]}")
  print()

# Cell 56
# Unigram counts in sentences for each author
import nltk
from nltk.probability import FreqDist

tokenizer = nltk.WordPunctTokenizer()

author_unigram_list_for_bigrams = {author: [tokenizer.tokenize(sentence) for sentence in sentences] for author, sentences in author_sentence_list.items()}
author_unigram_list_for_bigrams = {author: [x for X in article_unigrams for x in X] for author, article_unigrams in author_unigram_list_for_bigrams.items()}

author_unigram_counts_for_bigrams = {author: FreqDist(unigrams) for author, unigrams in author_unigram_list_for_bigrams.items()}


# Cell 57
# Most common 10 unigrams for each author
for author, unigram_counts in author_unigram_counts_for_bigrams.items():
  print(f"Author: {author}")
  print(f"Unigram counts: {unigram_counts.most_common(10)}")
  print()

# Cell 58
# Bigram counts in sentences for each author
import nltk
from nltk.probability import FreqDist

author_bigram_counts = {author: FreqDist(bigrams) for author, bigrams in author_bigram_list.items()}

# Cell 59
# Print most common 10 bigrams in sentences
for author, bigram_counts in author_bigram_counts.items():
  print(f"Author: {author}")
  print(f"Bigram counts: {bigram_counts.most_common(10)}")
  print()

# Cell 60
# Calculate bigram probabilities

bigram_probs = {}

for author, bigram_counts in author_bigram_counts.items():
  bigram_probs[author] = {bigram: count/author_unigram_counts_for_bigrams[author][bigram[0]] for bigram, count in bigram_counts.items()}

# Cell 61
# Print the bigram probabilities for first 10 bigrams for each author
for author, bigram_probs_dict in bigram_probs.items():
  print(f"Author: {author}")
  print(f"Bigram probabilities: {dict(list(bigram_probs_dict.items())[:10])}")
  print()

# Cell 62
# Long list of sentences for each author
# Prefix "_bos_ _bos_" to each sentence
author_sentence_list_for_trigrams = {author: ["_bos_ _bos_ " + x.strip() for X in article_sentences for x in X] for author, article_sentences in author_article_sentence_list.items()}

# Print the first 5 sentences of the first 5 articles for each author
for author, sentences in author_sentence_list_for_trigrams.items():
  print(f"Author {author}")
  print(f"Total # of sentences: {len(sentences)}")
  print(f"First 5 sentences: {sentences[:5]}")
  print()

# Cell 63
import nltk

# To speed-up
tokenizer = nltk.WordPunctTokenizer()

def trigrams(sentence):
  return [tuple(sentence[i:i+3]) for i in range(len(sentence)-2)]

# Tokenize each sentence into a list of bigrams
author_sentence_trigram_list = {author: [trigrams(tokenizer.tokenize(sentence)) for sentence in sentences] for author, sentences in author_sentence_list_for_trigrams.items()}

# Cell 64
# Print trigrams for the first author
print(f"Author: {list(author_sentence_trigram_list.keys())[0]}")
print(f"Trigrams of the first sentence: {author_sentence_trigram_list[list(author_sentence_trigram_list.keys())[0]][0]}")

# Cell 65
# Long list of tigrams for each author
author_trigram_list = {author: [bigram for X in sentence_trigrams for bigram in X] for author, sentence_trigrams in author_sentence_trigram_list.items()}

# Cell 66
# Print the first 50 bigrams of each author
for author, trigrams in author_trigram_list.items():
  print(f"Author: {author}")
  print(f"First 50 trigrams: {trigrams[:50]}")
  print()

# Cell 67
# Bigram list of sentences from sentence list for trigrams
import nltk

# To speed-up
tokenizer = nltk.WordPunctTokenizer()

def bigrams(sentence):
  return [tuple(sentence[i:i+2]) for i in range(len(sentence)-1)]

# Tokenize each sentence into a list of bigrams
author_sentence_bigram_list_for_trigrams = {author: [bigrams(tokenizer.tokenize(sentence)) for sentence in sentences] for author, sentences in author_sentence_list_for_trigrams.items()}

# Cell 68
# Long list of bigrams from sentences for trigrams
author_bigram_list_for_trigrams = {author: [bigram for X in sentence_bigrams for bigram in X] for author, sentence_bigrams in author_sentence_bigram_list_for_trigrams.items()}

# Cell 69
# Bigram counts in sentences for each author
import nltk
from nltk.probability import FreqDist

author_bigram_counts_for_trigrams = {author: FreqDist(bigrams) for author, bigrams in author_bigram_list_for_trigrams.items()}

# Cell 70
# Print most common 10 bigrams in sentences
for author, bigram_counts in author_bigram_counts_for_trigrams.items():
  print(f"Author: {author}")
  print(f"Bigram counts: {bigram_counts.most_common(10)}")
  print()

# Cell 71
# Trigram counts

author_trigram_counts = {author: FreqDist(trigrams) for author, trigrams in author_trigram_list.items()}

# Cell 72
# Calculate trigram probabilities

trigram_probs = {}

for author, trigram_counts in author_trigram_counts.items():
  trigram_probs[author] = {trigram: count/author_bigram_counts_for_trigrams[author][tuple(trigram[:2])] for trigram, count in trigram_counts.items()}

# Cell 73
# Print first 10 trigrams probs for each author

for author, trigram_probs_dict in trigram_probs.items():
  print(f"Author: {author}")
  print(f"Trigram probabilities: {dict(list(trigram_probs_dict.items())[:10])}")

# Cell 74
import random

def generate_sentence_bigram(seed, author_bigram_probs, num_words=100):
    current_word = seed
    sentence = []

    while len(sentence) < num_words:
        # Filter bigrams matching the current token
        # next_word_probs = [ (next_word_1, prob), (next_word_2, prob) ]
        next_word_probs = [ (key[1], value) for key, value in author_bigram_probs.items() if key[0] == current_word]

        # If no matching bigrams, break
        if not next_word_probs:
            print(f"No matching bigrams found for word: {current_word}.")
            break

        # Sort next words in increasing order of prob
        next_word_probs.sort(key=lambda x: x[1])

        # Normalize next word probabilities
        total_prob = sum(prob for _, prob in next_word_probs)
        next_word_probs = [(word, prob / total_prob) for word, prob in next_word_probs]

        # Create Cumulative Mass (Density) Function - CDF
        cumulative_prob = 0
        cdf = [(w, cumulative_prob := cumulative_prob + p) for w, p in next_word_probs]

        # Choose next word based on cdf
        r = random.random()
        next_word = next(w for w, p in cdf if r < p)

        # Append next word to sentence
        sentence.append(next_word)

        # Set current_word to next_word
        current_word = next_word

    return ' '.join(sentence)

# Generate a sentence starting with the word '<bos>'
seed = "_bos_"

# Chose an author to generate articles of the same tone
seed_author = "1"

# Generate a sentence using the bigram language model
generated_sentence = generate_sentence_bigram(seed, bigram_probs[seed_author], num_words=50)

import textwrap
wrapped_text = textwrap.fill(generated_sentence, width=70)
print(wrapped_text)


# Cell 75
# Generate sentences using trigram_probs
import random

def generate_sentence_trigram(seed, author_trigram_probs, num_words=100):
    current_bigram = tuple(seed.split())
    sentence = []

    while len(sentence) < num_words:
        w1 = current_bigram[0]
        w2 = current_bigram[1]

        # Filter trigrams matching the current tuple (w_1, w_2)
        # next_word_probs = [ (next_word_1, prob), (next_word_2, prob) ]
        next_word_probs = [ (keys[2], value) for keys, value in author_trigram_probs.items() if keys[0] == w1 and keys[1] == w2 ]

        # If no matching bigrams, break
        if not next_word_probs:
            print(f"No matching next word found for bigram: {current_bigram}.")
            break

        # Sort next words in increasing order of prob
        next_word_probs.sort(key=lambda x: x[1])

        # Normalize next word probabilities
        total_prob = sum(prob for _, prob in next_word_probs)
        next_word_probs = [(word, prob / total_prob) for word, prob in next_word_probs]

        # Create Cumulative Mass (Density) Function - CDF
        cumulative_prob = 0
        cdf = [(w, cumulative_prob := cumulative_prob + p) for w, p in next_word_probs]

        # Choose next word based on cdf
        r = random.random()
        next_word = next(w for w, p in cdf if r < p)

        # Append next word to sentence
        sentence.append(next_word)

        # Set current_bigram to next bigram
        current_bigram = (w2, next_word)

    return ' '.join(sentence)


# Generate a sentence starting with the word '<bos>'
seed = "_bos_ _bos_"

# Chose an author to generate articles of the same tone
seed_author = "1"

# Generate a sentence using the bigram language model
generated_sentence = generate_sentence_trigram(seed, trigram_probs[seed_author])

import textwrap
wrapped_text = textwrap.fill(generated_sentence, width=70)
print(wrapped_text)


# Cell 76
# Install Hugging Face libraries if not installed before
!pip install transformers datasets

# Cell 77
from transformers import pipeline
text_generator = pipeline("text-generation")

# Cell 78
# Generate sentences

context = u'This NLP course is awesome'
# text_generator("Great power comes great ", max_new_tokens=500, do_sample=False)
text = text_generator(context, max_new_tokens=50)

import textwrap
wrapped_text = textwrap.fill(text[0]['generated_text'], width=70)
print(wrapped_text)


# Cell 79
!pip install einops

# Cell 80
# Use a pipeline as a high-level helper
from transformers import pipeline

text_generator = pipeline("text-generation", model="microsoft/phi-1_5", trust_remote_code=True)

# Cell 81
# Generate sentences

context = u'This NLP course is awesome'
# text_generator("Great power comes great ", max_new_tokens=500, do_sample=False)
text = text_generator(context, max_new_tokens=50)

import textwrap
wrapped_text = textwrap.fill(text[0]['generated_text'], width=70)
print(wrapped_text)


# Cell 82
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Split data into training and testing sets for word TF-IDF features
X_train_word, X_test_word, y_train, y_test = train_test_split(
    tfidf_word_features, all_authors, test_size=0.2, random_state=42
)

print(f"Shape of X_train_word: {X_train_word.shape}")
print(f"Shape of X_test_word: {X_test_word.shape}")
print(f"Length of y_train: {len(y_train)}")
print(f"Length of y_test: {len(y_test)}")

# Cell 83
# Initialize and train a Logistic Regression model
# Using a small C value and 'saga' solver for multiclass classification with L1 regularization
# 'multi_class='multinomial'' handles multiple classes directly

logistic_model_word = LogisticRegression(
    C=1.0,  # Regularization strength
    solver='saga', # Solver that supports L1 regularization for multiclass
    max_iter=1000, # Increased iterations for convergence
    random_state=42,
    n_jobs=-1 # Use all available CPU cores
)

print("Training Logistic Regression model with Word TF-IDF features...")
logistic_model_word.fit(X_train_word, y_train)
print("Training complete.")

# Make predictions on the test set
y_pred_word = logistic_model_word.predict(X_test_word)

# Evaluate the model
accuracy_word = accuracy_score(y_test, y_pred_word)
f1_macro_word = f1_score(y_test, y_pred_word, average='macro')

print(f"\nAccuracy with Word TF-IDF features: {accuracy_word:.4f}")
print(f"F1-score (macro) with Word TF-IDF features: {f1_macro_word:.4f}")

# Display classification report
print("\nClassification Report (Word TF-IDF features):\n")
print(classification_report(y_test, y_pred_word, zero_division=0))

# Display Confusion Matrix
conf_matrix_word = confusion_matrix(y_test, y_pred_word)
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix_word, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(all_authors), yticklabels=np.unique(all_authors))
plt.title('Confusion Matrix for Author Attribution (Word TF-IDF)')
plt.xlabel('Predicted Author')
plt.ylabel('True Author')
plt.show()

# Cell 84
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

# Re-display Confusion Matrix for Word TF-IDF
# Assuming y_test and y_pred_word are available from previous execution
conf_matrix_word = confusion_matrix(y_test, y_pred_word)

plt.figure(figsize=(15, 12))
sns.heatmap(conf_matrix_word, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(all_authors), yticklabels=np.unique(all_authors))
plt.title('Confusion Matrix for Author Attribution (Word TF-IDF)')
plt.xlabel('Predicted Author')
plt.ylabel('True Author')
plt.show()

print("\n--- Analyzing Confused Authors ---")
# Iterate through the confusion matrix to identify significant off-diagonal elements
# A significant off-diagonal value means a true author was often predicted as another author
for i, true_author_label in enumerate(np.unique(all_authors)):
    for j, predicted_author_label in enumerate(np.unique(all_authors)):
        if i != j and conf_matrix_word[i, j] > 0: # Look for misclassifications
            print(f"True Author {true_author_label} was predicted as Author {predicted_author_label} {conf_matrix_word[i, j]} times.")


# Cell 85
feature_names = word_vectorizer.get_feature_names_out()
coefs = logistic_model_word.coef_

print("Top 10 Most Indicative Features for Each Author:")
print("---------------------------------------------------")

# Get unique author labels
author_labels = np.unique(all_authors)

# Loop through each author (class) and find the top features
for i, author_label in enumerate(author_labels):
    # Get the coefficients for the current author
    author_coefs = coefs[i]

    # Get the indices of features sorted by their coefficients in descending order
    top_feature_indices = author_coefs.argsort()[-10:][::-1]

    # Get the actual feature names using the indices
    top_features = [feature_names[j] for j in top_feature_indices]

    print(f"Author {author_label}: {', '.join(top_features)}")

