import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import collections
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
# Custom stopwords list (manually created to avoid NLTK dependency)
class SpamClassifier:
    def __init__(self, words_file='words_alpha.txt'):
        self.spam_words = None
        self.ham_words = None
        self.all_words = None
        self.stop_words = stopwords.words('english')  # Use the custom stopwords list
        self.wordlist = self._load_wordlist(words_file)
        self.spam_messages = []
        self.ham_messages = []

    def _load_wordlist(self, words_file):
        """Load a list of valid English words"""
        words = pd.read_csv(words_file, encoding='UTF-8', header=None, names=['Word'])
        return set(words['Word'])

    def _process_message(self, message):
        """Preprocess a single message (lowercase, tokenize, filter, stemming)"""
        words = message.lower()  # Lowercase
        words = word_tokenize(words)  # Tokenization
        words = [word for word in words if len(word) > 1]  # Filter short words
        words = [word for word in words if word not in self.stop_words]  # Remove stopwords
        words = [word for word in words if word in self.wordlist]  # Keep only valid words
        words = [PorterStemmer().stem(word) for word in words]  # Stemming
        return words

    def _count_words(self, data):
        """Count words in a list of messages"""
        counter = collections.OrderedDict()
        for message in data:
            for word in set(self._process_message(message)):
                counter[word] = counter.get(word, 0) + 1
        return counter

    def fit(self, train_data_file):
        """Train the spam classifier using a training dataset"""
        print("Training the Spam Classifier...")
        mails = pd.read_csv(train_data_file, encoding='latin-1')
        self.spam_messages = set(mails[mails['Category'] == 1]['Message'])
        self.ham_messages = set(mails[mails['Category'] == 0]['Message'])
        
        # Process messages and count words
        self.spam_words = self._count_words(self.spam_messages)
        self.ham_words = self._count_words(self.ham_messages)
        all_messages = self.spam_messages.union(self.ham_messages)
        self.all_words = self._count_words(all_messages)
        print("Training Complete.")

    def predict(self, message, s=1, p=0.5, percentage=False):
        """Predict whether a message is spam or ham"""
        n = 0
        for word in self._process_message(message):
            spam_freq = self.spam_words.get(word, 0) / self.all_words.get(word, 1)
            ham_freq = self.ham_words.get(word, 0) / self.all_words.get(word, 1)
            if spam_freq + ham_freq > 0:
                spaminess_of_word = spam_freq / (spam_freq + ham_freq)
                corr_spaminess = (s * p + self.all_words[word] * spaminess_of_word) / (s + self.all_words[word])
                n += np.log(1 - corr_spaminess) - np.log(corr_spaminess)

        spam_result = 1 / (1 + np.e**n)

        if percentage:
            print(f"Spam probability: {spam_result * 100:.2f}%")
        return spam_result > 0.5

    def evaluate(self, test_data_file):
        """Evaluate the model on a test dataset"""
        print("Evaluating the Spam Classifier...")
        test_data = pd.read_csv(test_data_file, encoding='latin-1')
        test_messages = [i for i in test_data['Message'] if len(self._process_message(i)) >= 1]
        count_spam = 0
        count_ham = 0
        for message in test_messages:
            if self.predict(message):
                count_spam += 1
            else:
                count_ham += 1
        print(f"Spam: {count_spam}")
        print(f"Ham: {count_ham}")
        return {'spam': count_spam, 'ham': count_ham}

# Example usage
if __name__ == "__main__":
    classifier = SpamClassifier(words_file='words_alpha.txt')
    classifier.fit(train_data_file='train_data.csv')
    classifier.evaluate(test_data_file='test_data.csv')

    # Test single prediction
    message = "Congratulations! You've won a free gift card. Call now!"
    is_spam = classifier.predict(message, percentage=True)
    print("Spam" if is_spam else "Ham")
