

import os
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
########################################################################

# Directory where you want to download and save the data-set.
# Set this before you start calling any of the functions below.
data_dir = "data/europarl/"

# Base-URL for the data-sets on the internet.
data_url = "http://www.statmt.org/europarl/v7/"


########################################################################
# Public functions that you may call to download the data-set from
# the internet and load the data into memory.



def load_data(english=True, language_code="da", start="", end=""):
    """
    Load the data-file for either the English-language texts or
    for the other language (e.g. "da" for Danish).

    All lines of the data-file are returned as a list of strings.

    :param english:
      Boolean whether to load the data-file for
      English (True) or the other language (False).

    :param language_code:
      Two-char code for the other language e.g. "da" for Danish.
      See list of available codes above.

    :param start:
      Prepend each line with this text e.g. "ssss " to indicate start of line.

    :param end:
      Append each line with this text e.g. " eeee" to indicate end of line.

    :return:
      List of strings with all the lines of the data-file.
    """

    if english:
        # Load the English data.
        filename = "europarl-v7.{0}-en.en".format(language_code)
    else:
        # Load the other language.
        filename = "europarl-v7.{0}-en.{0}".format(language_code)

    # Full path for the data-file.
    path = os.path.join(data_dir, filename)

    # Open and read all the contents of the data-file.
    with open(path, encoding="utf-8") as file:
        # Read the line from file, strip leading and trailing whitespace,
        # prepend the start-text and append the end-text.
        texts = [start + line.strip() + end for line in file]

    return texts


########################################################################
class TokenizerWrap(Tokenizer):
    """Wrap the Tokenizer-class from Keras with more functionality."""

    def __init__(self, texts, padding,
                 reverse=False, num_words=None,oov_token='//'):
        """
        :param texts: List of strings. This is the data-set.
        :param padding: Either 'post' or 'pre' padding.
        :param reverse: Boolean whether to reverse token-lists.
        :param num_words: Max number of words to use.
        """

        Tokenizer.__init__(self, num_words=num_words,oov_token=oov_token)

        # Create the vocabulary from the texts.
        self.fit_on_texts(texts)

        # Create inverse lookup from integer-tokens to words.
        self.index_to_word = dict(zip(self.word_index.values(),
                                      self.word_index.keys()))

        # Convert all texts to lists of integer-tokens.
        # Note that the sequences may have different lengths.
        self.tokens = self.texts_to_sequences(texts)

        if reverse:
            # Reverse the token-sequences.
            self.tokens = [list(reversed(x)) for x in self.tokens]

            # Sequences that are too long should now be truncated
            # at the beginning, which corresponds to the end of
            # the original sequences.
            truncating = 'pre'
        else:
            # Sequences that are too long should be truncated
            # at the end.
            truncating = 'post'

        # The number of integer-tokens in each sequence.
        self.num_tokens = [len(x) for x in self.tokens]

        # Max number of tokens to use in all sequences.
        # We will pad / truncate all sequences to this length.
        # This is a compromise so we save a lot of memory and
        # only have to truncate maybe 5% of all the sequences.
        self.max_tokens =30
        self.tokens = [x for x in self.tokens if len(x)<=10 and len(x)>=4]
        self.tokens = [x for x in self.tokens if 1 not in x]

        # Pad / truncate all token-sequences to the given length.
        # This creates a 2-dim numpy matrix that is easier to use.
        self.tokens_padded = pad_sequences(self.tokens,
                                           maxlen=self.max_tokens,
                                           padding=padding,
                                           truncating=truncating)

    def token_to_word(self, token):
        """Lookup a single word from an integer-token."""

        word = " " if token == 0 else self.index_to_word[token]
        return word

    def tokens_to_string(self, tokens):
        """Convert a list of integer-tokens to a string."""

        # Create a list of the individual words.
        words = [self.index_to_word[token]
                 for token in tokens
                 if token != 0]

        # Concatenate the words to a single string
        # with space between all the words.
        text = " ".join(words)

        return text

    def text_to_tokens(self, text, reverse=False, padding=False):
        """
        Convert a single text-string to tokens with optional
        reversal and padding.
        """

        # Convert to tokens. Note that we assume there is only
        # a single text-string so we wrap it in a list.
        tokens = self.texts_to_sequences([text])
        tokens = np.array(tokens)

        if reverse:
            # Reverse the tokens.
            tokens = np.flip(tokens, axis=1)

            # Sequences that are too long should now be truncated
            # at the beginning, which corresponds to the end of
            # the original sequences.
            truncating = 'pre'
        else:
            # Sequences that are too long should be truncated
            # at the end.
            truncating = 'post'

        if padding:
            # Pad and truncate sequences to the given length.
            tokens = pad_sequences(tokens,
                                   maxlen=self.max_tokens,
                                   padding='pre',
                                   truncating=truncating)

        return tokens
if __name__ == '__main__':
    num_words = 30000


    data_dest = load_data(english=True,
                          language_code='da'
)

    tokenizer = TokenizerWrap(texts=data_dest,
                              padding='post',
                              reverse=False,
                              num_words=num_words)
    print(len(tokenizer.tokens_padded))

    # saving
    # with open('tokenizer.pickle', 'rb') as handle:
    #     tokenizer = pickle.load(handle)


