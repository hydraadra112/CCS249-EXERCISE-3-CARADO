import wikipedia
from nltk.tokenize import word_tokenize
import math
from nltk.util import bigrams
import string

#################################################
# SUBMISSION TO EXERCISE FOR UNIT 3
# PREPARED BY: JOHN MANUEL CARADO
# BSCS 3-A
# CCS 249 - NATURAL LANGUAGE PROCESSING
#################################################

### 1. (10 points) Use the Wikipedia python module and access any topic,
# as you will use that as your corpus, with a word limit of 1000 words.

# Fetch the deep learning wikipedia page
page = wikipedia.page('Deep_learning')
deep_learning = page.content[:1000]

### 2. Train 2 models: a Bigram and Trigram Language Model, 
# use the shared code as reference for bigram modeling, 
# and update it to support trigrams.

def preprocess_text(corpus: str):
    """
    Removes capitalization, punctuation, and converts the text into tokens
    
    Args:
        corpus (str): The string to preprocess
    
    """
    
    # Remove capitalization
    lowered_text = corpus.lower()
    
    # Remove punctuation
    normalized_text = lowered_text.translate(str.maketrans('', '', string.punctuation))
    
    # Convert to tokens
    tokens = word_tokenize(normalized_text)
    
    return tokens

def train_bigram(corpus: str, print_output: bool = False) -> dict:
    """
    Trains a bi-gram model with inputted corpus
    
    Args:
        corpus (str): The corpus to train the bi-gram model
        print_output (bool): Print the bigram or no
        
    Returns:
        dict: A dictionary containing the bigrams and its count
    """
    
    tokens = preprocess_text(corpus)
    
    bigram_counts = dict()
    
    # Create bigrams
    for i in range(len(tokens) - 1):
        bigram = (tokens[i], tokens[i+1])
        
        if bigram in bigram_counts:
            bigram_counts[bigram] += 1
            
        else:
            bigram_counts[bigram] = 1
            
    if print_output:
        for bigram, count in bigram_counts.items():
            print(f'{bigram}: {count}')
            
    return bigram_counts
    
def train_trigram(corpus: str, print_output: bool = False) -> dict:
    """
    Trains a tri-gram model with inputted corpus
    
    Args:
        corpus (str): The corpus to train the tri-gram model
        print_output (bool): Print the trigram or no
        
    Returns:
        dict: A dictionary containing the trigrams and its count
    """
    
    tokens = preprocess_text(corpus)
    
    trigram_counts = dict()
    
    # Create trigram
    for i in range(len(tokens) - 2):
        trigram = (tokens[i], tokens[i+1], tokens[i+2])
        
        if trigram in trigram_counts:
            trigram_counts[trigram] += 1
            
        else:
            trigram_counts[trigram] = 1
            
    if print_output:
        for trigram, count in trigram_counts.items():
            print(f'{trigram}: {count}')
            
    return trigram_counts

DL_BIGRAM = train_bigram(corpus=deep_learning)
DL_TRIGRAM = train_trigram(deep_learning)

# 3. Using a test sentence “The quick brown fox jumps over the lazy dog near the bank of the river.”
# OR generate your own test sentence, create a function that will determine the perplexity score
# for each trained model.

def calculate_bigram_perplexity(bigram_counts: dict, test_corpus: str) -> float:
    """
    Calculates the perplexity score of a bi-gram model given a test corpus.
    
    Args:
        bigram_counts (dict): The bigram model to calculate the perplexity score.
        test_corpus (str): The test corpus.
    
    Returns:
        float: The perplexity score.
    """
    
    tokens = preprocess_text(test_corpus)
    
    # Calculate unigram counts for normalization
    unigram_counts = {}
    for token in tokens:
        unigram_counts[token] = unigram_counts.get(token, 0) + 1
        
    # Calculate total number of words (N)
    N = len(tokens)
    
    # Initialize log probability sum
    log_prob_sum = 0.0
    
    # Calculate bi-gram probabilities and accumulate log-prob
    for i in range(len(tokens) - 1):
        bigram = (tokens[i], tokens[i + 1])
        bigram_count = bigram_counts.get(bigram, 0)
        unigram_count = unigram_counts.get(tokens[i], 0)
        
        # Calculate conditional probability P(w2 | w1)
        if unigram_count > 0:
            prob = bigram_count / unigram_count
        else:
            prob = 1e-10  # Small value to handle unseen bi-grams
        
        # Add log probability (log base 2)
        log_prob_sum += math.log2(prob) if prob > 0 else math.log2(1e-10)
    
    # Calculate perplexity
    perplexity = 2 ** (-log_prob_sum / N)
    
    return perplexity

def calculate_trigram_perplexity(trigram_counts: dict, test_corpus: str) -> float:
    """
    Calculates the perplexity score of a tri-gram model given a test corpus.
    
    Args:
        trigram_counts (dict): The trigram model to calculate the perplexity score.
        test_corpus (str): The test corpus.
    
    Returns:
        float: The perplexity score.
    """
    tokens = preprocess_text(test_corpus)
    
    # Calculate bigram counts for normalization
    bigram_counts = {}
    for i in range(len(tokens) - 1):
        bigram = (tokens[i], tokens[i + 1])
        bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1
    
    # Calculate total number of words (N)
    N = len(tokens)
    
    # Initialize log probability sum
    log_prob_sum = 0.0
    
    # Calculate tri-gram probabilities and accumulate log-prob
    for i in range(len(tokens) - 2):
        trigram = (tokens[i], tokens[i + 1], tokens[i + 2])
        trigram_count = trigram_counts.get(trigram, 0)
        bigram = (tokens[i], tokens[i + 1])
        bigram_count = bigram_counts.get(bigram, 0)
        
        # Calculate conditional probability P(w3 | w1, w2)
        if bigram_count > 0:
            prob = trigram_count / bigram_count
        else:
            prob = 1e-10  # Small value to handle unseen tri-grams
        
        # Add log probability (log base 2)
        log_prob_sum += math.log2(prob) if prob > 0 else math.log2(1e-10)
    
    # Calculate perplexity
    perplexity = 2 ** (-log_prob_sum / N)
    
    return perplexity

DL_TEST_CORPUS = 'Deep learning is a subset of machine learning that focuses on neural networks to perform tasks such as classification, regression, and representation learning. Some common architectures include convolutional neural networks and transformers.'

bigram_perplexity_result = calculate_bigram_perplexity(DL_BIGRAM, DL_TEST_CORPUS)
print(f'Bigram Perplexity -> Test Sentence "{DL_TEST_CORPUS}" -> Score: {bigram_perplexity_result:.4f}')

trigram_perplexity_result = calculate_trigram_perplexity(DL_TRIGRAM, DL_TEST_CORPUS)
print(f'Trigram Perplexity -> Test Sentence "{DL_TEST_CORPUS}" -> Score: {trigram_perplexity_result:.4f}')