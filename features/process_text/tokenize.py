from nltk import sent_tokenize, word_tokenize
from nltk.tokenize import PunktSentenceTokenizer, RegexpTokenizer, TreebankWordTokenizer, WordPunctTokenizer, \
    WhitespaceTokenizer
from features.process_text.patterns import get_sentence_token_pattern, get_word_token_pattern
from sklearn.feature_extraction.text import CountVectorizer


_sentence_tokenizer_default = sent_tokenize

_sentence_tokenizer_punkt = PunktSentenceTokenizer.tokenize

_sentence_tokenizer_regex = RegexpTokenizer(pattern=get_sentence_token_pattern(), gaps=True).tokenize

_SENTENCE_TOKENIZER_DICT = {
    'default': _sentence_tokenizer_default,
    'punkt': _sentence_tokenizer_punkt,
    'regex': _sentence_tokenizer_regex
}


def sentence_tokenize(text, sentence_tokenizer_id='default'):
    """
    Tokenizes a sentence, based on a given tokenizer.
    Args:
        text: A string, describing a sentence.
        sentence_tokenizer_id: A tokenizer key, taken from SENTENCE_TOKENIZER_DICT.
    Returns:
        A tokenized sentence.
    """
    # If text is empty, return None.
    if not text: return None
    sentence_tokenizer = _SENTENCE_TOKENIZER_DICT.get(sentence_tokenizer_id)
    return sentence_tokenizer(text)


_word_tokenizer_default = word_tokenize

_word_tokenizer_treebank = TreebankWordTokenizer().tokenize

_word_tokenizer_regex = RegexpTokenizer(pattern=get_word_token_pattern(), gaps=False).tokenize

_word_tokenizer_punkt = WordPunctTokenizer().tokenize

_word_tokenizer_whitespace = WhitespaceTokenizer().tokenize

_WORD_TOKENIZER_DICT = {
    'default': _word_tokenizer_default,
    'treebank': _word_tokenizer_treebank,
    'regex': _word_tokenizer_regex,
    'punkt': _word_tokenizer_punkt,
    'whitespace': _word_tokenizer_whitespace
}


def word_tokenize(text, word_tokenizer_id='default'):
    """
    Word-tokenizes a given sentence, based on a defined tokenizer.
    Args:
        sentence: A string, corresponding to a sentence.
        word_tokenizer_id: A key from WORD_TOKENIZER_DICT
    Returns:
        A list of words, corresponding to the tokenized sentence.
    """
    # If text is empty, return None.
    if not text: return None
    word_tokenizer = _WORD_TOKENIZER_DICT.get(word_tokenizer_id)
    tokens = None
    try:
        tokens = word_tokenizer(text)
    except TypeError:
        print("ERROR:")
        print(text)
        return None
    tokens = [token.strip() for token in tokens]
    return tokens

def is_tokenized(text):
    return type(text) == list

def merge_tokens(tokens):
    return ' '.join(tokens)

"""adicionar categoria ao dicionário de cada lista"""
def add_category_IV(list):
    list2 = []
    for dict in list:
        dict['Category'] = 'Instant_Video'
        list2.append(dict)
    return list2

def add_category_B(list):
    list2 = []
    for dict in list:
        dict['Category'] = 'Baby'
        list2.append(dict)
    return list2

def add_category_DM(list):
    list2 = []
    for dict in list:
        dict['Category'] = 'Digital_Music'
        list2.append(dict)
    return list2

def add_category_MI(list):
    list2 = []
    for dict in list:
        dict['Category'] = 'Musical_Instruments'
        list2.append(dict)
    return list2

def word_tokenize_scikit(dataset):
    """
    Previous class tokenization was done using ntlk. In this class we learn how to do it with scikit-learn
    Args:
        dataset: a collection of documents stored in a vector
    Returns:
        A list of words, corresponding to the indexed vocabulary of the dataset
    """
    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(dataset)

    return x, vectorizer


