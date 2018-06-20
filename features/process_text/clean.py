from features.process_text.tokenize import sentence_tokenize, word_tokenize
from features.process_text.normalize import expand_contractions, remove_special_characters,\
    remove_stopwords, remove_end_characters, convert_case, remove_hyperlinks, replace_whitespaces, \
    replace_apostrophes, replace_multiple_stopwords, remove_numbers, expand_abbreviations, remove_stopwords_sentiment, expand_emoticons
from features.process_text.stemming import lemmatize_text, remove_repeated_characters
from features.process_text.Merge import merge

def clean_text(text, tokenize=False):
    clean_dict = {}
    # Replace whitespaces.
    clean_dict['replace_whitespaces'] = replace_whitespaces(text)
    # Expand abbreviations.
    #clean_dict['expand_emoticons'] = expand_abbreviations(clean_dict['replace_whitespaces'])
    # Replace multiple stopwords.
    clean_dict['replace_multiple_stopwords'] = replace_multiple_stopwords(clean_dict['replace_whitespaces'])
    # Replace apostrophes.
    clean_dict['replace_apostrophes'] = replace_apostrophes(clean_dict['replace_multiple_stopwords'])
    # Expand contractions.
    clean_dict['expand_contractions'] = expand_contractions(clean_dict['replace_apostrophes'])
    # Remove hyperlinks.
    clean_dict['remove_hyperlinks'] = remove_hyperlinks(clean_dict['expand_contractions'])
    # Expand abbreviations.
    clean_dict['expand_abbreviations'] = expand_abbreviations(clean_dict['remove_hyperlinks'])
    # Remove special characters.
    clean_dict['remove_special_characters'] = remove_special_characters(clean_dict['expand_abbreviations'])
    # Remove numbers.
    clean_dict['remove_numbers'] = remove_numbers(clean_dict['remove_special_characters'])
    # Convert to lower case.
    clean_dict['convert_case'] = convert_case(clean_dict['remove_numbers'])
    # Tokenize sentences.
    clean_dict['sentence_tokenize'] = sentence_tokenize(clean_dict['convert_case'])
    # If sentence tokenize is empty, return None.
    if not clean_dict['sentence_tokenize']:
        return clean_dict, None
    else:
        # Remove end characters.
        clean_dict['remove_end_characters'] = [remove_end_characters(item) for item in clean_dict['sentence_tokenize'] if len(item) > 1]
        # Lemmatize words.
        clean_dict['lemmatize'] = [lemmatize_text(item) for item in clean_dict['remove_end_characters'] if len(item) > 1]
        #stemming words
        #clean_dict['stemming'] = [convert_word_stem(item) for item in clean_dict['lemmatize'] if len(item) > 1]
        #removing repeated characters
        # Remove stopwords.
        clean_dict['remove_stopwords'] = [remove_stopwords(item) for item  in clean_dict['lemmatize'] if len(item) > 1]
        # Remove stopwords.
        clean_dict['remove_repeated_characters'] = [remove_repeated_characters(item) for item  in clean_dict['remove_stopwords'] if len(item) > 1]
        # If tokenize, tokenize words.
        if tokenize:
            clean_dict['word_tokenize'] = [word_tokenize(item, 'whitespace') for item in clean_dict['remove_repeated_characters'] if len(item) > 1]
            clean_dict['word_merge'] = merge(clean_dict['word_tokenize'])
            #clean_dict['removing_repeated_characters'] = [remove_repeated_characters(item) for item in clean_dict['word_merge'] if len(item) > 1]




        # Return dictionary and cleaned text.
        return clean_dict, clean_dict['word_merge'] # clean_dict



def clean_text_sentiment(text, tokenize=False):
    clean_dict = {}
    # Replace whitespaces.
    clean_dict['replace_whitespaces'] = replace_whitespaces(text)
    # Expand abbreviations.
    clean_dict['expand_emoticons'] = expand_emoticons(clean_dict['replace_whitespaces'])
    # Replace multiple stopwords.
    clean_dict['replace_multiple_stopwords'] = replace_multiple_stopwords(clean_dict['expand_emoticons'])
    # Replace apostrophes.
    clean_dict['replace_apostrophes'] = replace_apostrophes(clean_dict['replace_multiple_stopwords'])
    # Expand contractions.
    clean_dict['expand_contractions'] = expand_contractions(clean_dict['replace_apostrophes'])
    # Remove hyperlinks.
    clean_dict['remove_hyperlinks'] = remove_hyperlinks(clean_dict['expand_contractions'])
    # Expand abbreviations.
    clean_dict['expand_abbreviations'] = expand_abbreviations(clean_dict['remove_hyperlinks'])
    # Remove special characters.
    clean_dict['remove_special_characters'] = remove_special_characters(clean_dict['expand_abbreviations'])
    # Remove numbers.
    clean_dict['remove_numbers'] = remove_numbers(clean_dict['remove_special_characters'])
    # Convert to lower case.
    clean_dict['convert_case'] = convert_case(clean_dict['remove_numbers'])
    # Tokenize sentences.
    clean_dict['sentence_tokenize'] = sentence_tokenize(clean_dict['convert_case'])
    # If sentence tokenize is empty, return None.
    if not clean_dict['sentence_tokenize']:
        return clean_dict, None
    else:
        # Remove end characters.
        clean_dict['remove_end_characters'] = [remove_end_characters(item) for item in clean_dict['sentence_tokenize'] if len(item) > 1]
        # Lemmatize words.
        clean_dict['lemmatize'] = [lemmatize_text(item) for item in clean_dict['remove_end_characters'] if len(item) > 1]
        #stemming words
        #clean_dict['stemming'] = [convert_word_stem(item) for item in clean_dict['lemmatize'] if len(item) > 1]
        #removing repeated characters
        # Remove stopwords.
        clean_dict['remove_stopwords'] = [remove_stopwords_sentiment(item) for item  in clean_dict['lemmatize'] if len(item) > 1]
        # Remove stopwords.
        clean_dict['remove_repeated_characters'] = [remove_repeated_characters(item) for item  in clean_dict['remove_stopwords'] if len(item) > 1]
        # If tokenize, tokenize words.
        if tokenize:
            clean_dict['word_tokenize'] = [word_tokenize(item, 'whitespace') for item in clean_dict['remove_repeated_characters'] if len(item) > 1]
            clean_dict['word_merge'] = merge(clean_dict['word_tokenize'])
            #clean_dict['removing_repeated_characters'] = [remove_repeated_characters(item) for item in clean_dict['word_merge'] if len(item) > 1]




        # Return dictionary and cleaned text.
        if tokenize:
            return clean_dict, clean_dict['word_merge'] # clean_dict
        else: return clean_dict, clean_dict['remove_repeated_characters']


