import gzip
from review.review import create_review_from_dict
from Data.Get_file_path import get_data_path
from features.process_reviews.clean_reviews import clean_review, clean_review2, clean_review3
import nltk
from nltk.tokenize import word_tokenize as wt
import numpy as np
from scipy.sparse import coo_matrix

""" - Functions toimport the full data set without restrictions"""
def load(path):
    g = gzip.open(get_data_path("01_raw\\reviews_Amazon_Instant_Video.json.gz"), 'rb')
    for l in g:
        yield eval(l)

def importIV(path):
    list = []
    for d in load(path):
        list.append(d)
    return list



def load2(path):
    g = gzip.open(get_data_path("01_raw\\reviews_Baby.json.gz"), 'rb')
    for l in g:
        yield eval(l)

def importB(path):
    list = []
    for d in load2(path):
        list.append(d)
    return list


def load3(path):
    g = gzip.open(get_data_path("01_raw\\reviews_Digital_Music.json.gz"), 'rb')
    for l in g:
        yield eval(l)

def importDM(path):
    list = []
    for d in load3(path):
        list.append(d)
    return list

def load4(path):
    g = gzip.open(get_data_path("01_raw\\reviews_Musical_Instruments.json.gz"), 'rb')
    for l in g:
        yield eval(l)

def importMI(path):
    list = []
    for d in load4(path):
        list.append(d)
    return list


#filtering by overall functions

def class1(Dataset):
    list = []
    for d in Dataset:
        if d.get('overall') == 1.0:
            list.append(d)
    return list

def class2(Dataset):
    list = []
    for d in Dataset:
        if d.get('overall') == 2.0:
            list.append(d)
    return list

def class3(Dataset):
    list = []
    for d in Dataset:
        if d.get('overall') == 3.0:
            list.append(d)
    return list

def class4(Dataset):
    list = []
    for d in Dataset:
        if d.get('overall') == 4.0:
            list.append(d)
    return list

def class5(Dataset):
    list = []
    for d in Dataset:
        if d.get('overall') == 5.0:
            list.append(d)
    return list


def catIV(Dataset):
    list = []
    for d in Dataset:
        if d.category == 'Instant_Video':
            list.append(d)
    return list

def catDM(Dataset):
    list = []
    for d in Dataset:
        if d.category == 'Digital_Music':
            list.append(d)
    return list

def catMI(Dataset):
    list = []
    for d in Dataset:
        if d.category == 'Musical_Instruments':
            list.append(d)
    return list

def catB(Dataset):
    list = []
    for d in Dataset:
        if d.category == 'Baby':
            list.append(d)
    return list


#Making reviews function

def add_review(review_dict_list):
    review_set = []
    for review_dict in review_dict_list:
        review = create_review_from_dict(review_dict)
        review_set.append(review)
    return review_set

def clean_reviews(reviews):
    cleaned_reviews = list()
    i = 0
    for review in reviews:
        if i % 100 == 0:
            print(i / len(reviews))
        review_cleaned = clean_review(review)
        cleaned_reviews.append(review_cleaned)
        i += 1
    return cleaned_reviews


def clean_reviews3(reviews):
    cleaned_reviews3 = list()
    i = 0
    for review in reviews:
        if i % 100 == 0:
            print(i / len(reviews))
        review_cleaned = clean_review3(review)
        cleaned_reviews3.append(review_cleaned)
        i += 1
    return cleaned_reviews3



def check_nouns_adverbs_adjectives(sentence):
    tokenized = wt(sentence)
    tags = nltk.pos_tag(tokenized)
    return " ".join([i[0] for i in tags if (i[1][0] == 'N') or (i[1][0] == 'R') or (i[1][0] == 'J')])

def remove_nouns_adverbs_adjectives(list):
    list2 = []
    for string in list:
        list2.append(check_nouns_adverbs_adjectives(string))
    return list2



def pair(sentence):
    u = []
    tokenized = wt(sentence)
    tags = nltk.pos_tag(tokenized)
    size = len(tags)
    for i in range(0,len(tags)):
        word = tags[i]
        if word[1][0] == 'J':
            u.append(word[0])
            if  i+1 < size and tags[i+1][1][0] == 'N':
                u.append(" ".join([word[0] + "_" + tags[i+1][0]]))
            if  i+2 < size and tags[i+2][1][0] == 'N':
                u.append(" ".join([word[0] + "_" + tags[i+2][0]]))
    return " ".join(u)

def remove_pair(list):
    list2 = []
    for string in list:
            list2.append(pair(string))
    return list2

def loadpos():
    g = open(get_data_path("positive-words.txt"), 'r').read()
    return g


def importpos():
    dict={}
    for d in loadpos().split():
        dict[d]=1
    return dict


def loadneg():
    g = open(get_data_path("negative-words.txt"), 'r').read()
    return g

def importneg():
    dict={}
    for d in loadneg().split():
        dict[d]=-1
    return dict


def loadnegwords():
    g = open(get_data_path("negation-words.txt"), 'r').read()
    return g

def importnegwords():
    dict={}
    for d in loadnegwords().split():
        dict[d]=-1
    return dict

def loadneut():
    g = open(get_data_path("neutral-words.txt"), 'r').read()
    return g

def importneut():
    dict={}
    for d in loadneut().split():
        dict[d]=0
    return dict


def treat2(sentence,sent_lex, neg_word, neut_word):
    u = []
    tokenized = sentence.split()
    size = len(tokenized)
    for i in range(0,size):
        word = tokenized[i]
        if word in sent_lex:
            if  i+1 < size and tokenized[i+1] in neg_word:
                u.append([" ".join(["not_" + word]),-1 * sent_lex[tokenized[i]]])
            if  i+2 < size and tokenized[i+2] in neg_word:
                u.append([" ".join(["not_" + word]),-1 * sent_lex[tokenized[i]]])
            if  i+1 < size and tokenized[i+1] in neut_word:
                u.append([" ".join(["neutral_" + word]),0 * sent_lex[tokenized[i]]])
            if  i+2 < size and tokenized[i+2] in neut_word:
                u.append([" ".join(["neutral_" + word]),0 * sent_lex[tokenized[i]]])
            if  i-1 > 0 and tokenized[i-1] in neg_word:
                u.append([" ".join(["not_" + word]),-1 * sent_lex[tokenized[i]]])
            if  i-2 > 0 and tokenized[i-2] in neg_word:
                u.append([" ".join(["not_" + word]),-1 * sent_lex[tokenized[i]]])
            if  i-1 > 0 and tokenized[i-1] in neut_word:
                u.append([" ".join(["neutral_" + word]),0 * sent_lex[tokenized[i]]])
            if  i-2 > 0 and tokenized[i-2] in neut_word:
                u.append([" ".join(["neutral_" + word]),0 * sent_lex[tokenized[i]]])
            else: u.append([word, sent_lex[tokenized[i]]])
    return u



def neut_neg2(list,sent_lex, neg_word, neut_word):
    list5 = []
    for list2 in list:
        list4 = []
        for string in list2:
            list4.append(treat2(string,sent_lex, neg_word, neut_word))
        list5.append(list4)
    return list5

def merge_sentences(list):
    list5 = []
    for review in list:
        list6 = []
        for sentence in review:
            for word in sentence:
                list6.append(word)
        list5.append(list6)
    return list5



def remove_repeat(list):
    t = []
    for review in list:
        words=[]
        numbers=[]
        for word in review:
            if word[0] not in words:
                numbers.append(word[1])
                words.append(word[0])
        t.append(numbers)
    return t


def convert_dict_to_list(dictionary):
    return [(key, value) for key, value in dictionary.items()]


def extract_unique_words(list_of_tuples):
    unique_words = set()
    for review in list_of_tuples:
        for pair in review:
            unique_words.add(pair[0])
    return sorted(list(unique_words))


def create_sparse_matrix(unique_words, dataset):
    columns = []
    rows = []
    data = []
    dataset_size = len(dataset)
    unique_words_size = len(unique_words)
    for i in range(0, dataset_size):
        for pair in dataset[i]:
            index = unique_words.index(pair[0])
            columns.append(index)
            rows.append(i)
            data.append(pair[1])
    data = np.asarray(data)
    rows = np.asarray(rows)
    columns = np.asarray(columns)
    matrix = coo_matrix((data, (rows, columns)),shape=(dataset_size, unique_words_size))

    return matrix


def sum_repeated(dataset):
    final = dict()
    list_without_repeat = list()
    for review in dataset:
        final.clear()
        for pair in review:
            if pair[0] in final:
                 final[pair[0]] = final[pair[0]] + pair[1]
            else:
                final[pair[0]] = pair[1]

        list_without_repeat.append(convert_dict_to_list(final))

    return list_without_repeat