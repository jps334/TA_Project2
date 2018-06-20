from Data.Load_data import importIV, importB, importDM, importMI, add_review, clean_reviews, clean_reviews3, catB, catDM, catIV, catMI
import random
from features.process_text.tokenize import word_tokenize_scikit, add_category_B, add_category_DM, add_category_IV, add_category_MI
from Data.Load_data import class1, class2, class3, class4, class5,  extract_unique_words, create_sparse_matrix
from Data.Load_data import remove_nouns_adverbs_adjectives, importneg, importpos,  remove_pair, neut_neg2, importnegwords, importneut, merge_sentences,  sum_repeated
from sklearn.decomposition import TruncatedSVD
from features.process_reviews.feature_weighting import tfidf_bow
from models.supervised_classification import Classification, prediction
from sklearn.preprocessing import minmax_scale
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from evaluation.metrics import f1_score, accuracy, precision, recall
import pandas as pd
from features.process_reviews.webpropagate import cooccurrence_matrix, get_sorted_vocab, cosine_similarity_matrix, get_vectors, cosim, graph_propagation, propagate, format_matrix
import dill as pickle
from operator import itemgetter
from sklearn.linear_model import SGDClassifier


#Importing each categories dataset

Dataset_instant_video = importIV('reviews_Amazon_Instant_Video.json.gz')

Dataset_baby = importB('reviews_Baby.json.gz')

Dataset_music_instruments = importMI('reviews_Music_Instruments.json.gz')

Dataset_digital_music = importDM('reviews_Digital_Music.json.gz')

# Building necessary sample from each category


#Instant Video

data_instant_video_1 = random.choices(class1(Dataset_instant_video), k = 20000)
data_instant_video_2 = random.choices(class2(Dataset_instant_video), k = 20000)
data_instant_video_3 = random.choices(class3(Dataset_instant_video), k = 20000)
data_instant_video_4 = random.choices(class4(Dataset_instant_video), k = 20000)
data_instant_video_5 = random.choices(class5(Dataset_instant_video), k = 20000)

data_instant_video = []
data_instant_video.extend(data_instant_video_1)
data_instant_video.extend(data_instant_video_2)
data_instant_video.extend(data_instant_video_3)
data_instant_video.extend(data_instant_video_4)
data_instant_video.extend(data_instant_video_5)


#Baby

data_baby_1 = random.choices(class1(Dataset_baby), k = 20000)
data_baby_2 = random.choices(class2(Dataset_baby), k = 20000)
data_baby_3 = random.choices(class3(Dataset_baby), k = 20000)
data_baby_4 = random.choices(class4(Dataset_baby), k = 20000)
data_baby_5 = random.choices(class5(Dataset_baby), k = 20000)

data_baby = []
data_baby.extend(data_baby_1)
data_baby.extend(data_baby_2)
data_baby.extend(data_baby_3)
data_baby.extend(data_baby_4)
data_baby.extend(data_baby_5)


#Music Instruments


data_music_instruments_1 = random.choices(class1(Dataset_music_instruments), k = 20000)
data_music_instruments_2 = random.choices(class2(Dataset_music_instruments), k = 20000)
data_music_instruments_3 = random.choices(class3(Dataset_music_instruments), k = 20000)
data_music_instruments_4 = random.choices(class4(Dataset_music_instruments), k = 20000)
data_music_instruments_5 = random.choices(class5(Dataset_music_instruments), k = 20000)

data_music_instruments = []
data_music_instruments.extend(data_music_instruments_1)
data_music_instruments.extend(data_music_instruments_2)
data_music_instruments.extend(data_music_instruments_3)
data_music_instruments.extend(data_music_instruments_4)
data_music_instruments.extend(data_music_instruments_5)

#Digital Music

data_digital_music_1 = random.choices(class1(Dataset_digital_music), k = 20000)
data_digital_music_2 = random.choices(class2(Dataset_digital_music), k = 20000)
data_digital_music_3 = random.choices(class3(Dataset_digital_music), k = 20000)
data_digital_music_4 = random.choices(class4(Dataset_digital_music), k = 20000)
data_digital_music_5 = random.choices(class5(Dataset_digital_music), k = 20000)

data_digital_music = []
data_digital_music.extend(data_digital_music_1)
data_digital_music.extend(data_digital_music_2)
data_digital_music.extend(data_digital_music_3)
data_digital_music.extend(data_digital_music_4)
data_digital_music.extend(data_digital_music_5)

#Adding the category column
data_instant_video = add_category_IV(data_instant_video)
data_baby = add_category_B(data_baby)
data_digital_music = add_category_DM(data_digital_music)
data_music_instruments = add_category_MI(data_music_instruments)


# Building Complete dataset

Data = []

Data.extend(data_instant_video)

Data.extend(data_baby)

Data.extend(data_digital_music)

Data.extend(data_music_instruments)

#Creating and cleaning test, training and sample data


train_1 = random.choices(class1(Data), k = 56000)
train_2 = random.choices(class2(Data), k = 56000)
train_3 = random.choices(class3(Data), k = 56000)
train_4 = random.choices(class4(Data), k = 56000)
train_5 = random.choices(class5(Data), k = 56000)

train_data = []

train_data.extend(train_1)

train_data.extend(train_2)

train_data.extend(train_3)

train_data.extend(train_4)

train_data.extend(train_5)


test_1 = random.choices(class1(Data), k = 24000)
test_2 = random.choices(class2(Data), k = 24000)
test_3 = random.choices(class3(Data), k = 24000)
test_4 = random.choices(class4(Data), k = 24000)
test_5 = random.choices(class5(Data), k = 24000)

test_data = []

test_data.extend(test_1)

test_data.extend(test_2)

test_data.extend(test_3)

test_data.extend(test_4)

test_data.extend(test_5)



test_data = add_review(test_data)
train_data_cleaned = clean_reviews(train_data)
train_data2_cleaned = clean_reviews2(train_data)


train_data3_cleaned = clean_reviews3(train_data)

test_data = add_review(test_data)
test_data3_cleaned = clean_reviews3(test_data)



test_data = add_review(test_data)
test_data_cleaned = clean_reviews(test_data)
test_data2_cleaned = clean_reviews2(test_data)
test_data3_cleaned = clean_reviews3(test_data)


train_data_sample = random.choices(train_data, k = 1000)
train_data_cleaned_sample = clean_reviews(train_data_sample)
train_data2_cleaned_sample = clean_reviews2(train_data_sample)
train_data3_cleaned_sample = clean_reviews3(train_data_sample)


"""Import Train Data with Pickle"""
with open('Data\\Pickle\\train_data_cleaned.pickle', 'rb') as handle:
    train_data_cleaned = pickle.load(handle)

"""Import Test Data with Pickle"""
with open('Data\\Pickle\\test_data_cleaned.pickle', 'rb') as handle:
    test_data_cleaned = pickle.load(handle)


"""Import Train Data with Pickle"""
with open('Data\\Pickle\\train_data3_cleaned.pickle', 'rb') as handle:
    train_data3_cleaned = pickle.load(handle)

"""Import Test Data with Pickle"""
with open('Data\\Pickle\\test_data3_cleaned.pickle', 'rb') as handle:
    test_data3_cleaned = pickle.load(handle)


train_data_IV = random.choices(catIV(train_data3_cleaned), k = 70000)
train_data_DM = random.choices(catDM(train_data3_cleaned), k = 70000)
train_data_MI = random.choices(catMI(train_data3_cleaned), k = 70000)
train_data_B = random.choices(catB(train_data3_cleaned), k = 70000)



train_data_IV_1 = random.choices(class1(train_data_IV), k = 75)
train_data_IV_2 = random.choices(class2(train_data_IV), k = 75)
train_data_IV_3 = random.choices(class3(train_data_IV), k = 75)
train_data_IV_4 = random.choices(class4(train_data_IV), k = 75)
train_data_IV_5 = random.choices(class5(train_data_IV), k = 75)


train_data_DM_1 = random.choices(class1(train_data_DM), k = 75)
train_data_DM_2 = random.choices(class2(train_data_DM), k = 75)
train_data_DM_3 = random.choices(class3(train_data_DM), k = 75)
train_data_DM_4 = random.choices(class4(train_data_DM), k = 75)
train_data_DM_5 = random.choices(class5(train_data_DM), k = 75)


train_data_MI_1 = random.choices(class1(train_data_MI), k = 75)
train_data_MI_2 = random.choices(class2(train_data_MI), k = 75)
train_data_MI_3 = random.choices(class3(train_data_MI), k = 75)
train_data_MI_4 = random.choices(class4(train_data_MI), k = 75)
train_data_MI_5 = random.choices(class5(train_data_MI), k = 75)


train_data_B_1 = random.choices(class1(train_data_B), k = 75)
train_data_B_2 = random.choices(class2(train_data_B), k = 75)
train_data_B_3 = random.choices(class3(train_data_B), k = 75)
train_data_B_4 = random.choices(class4(train_data_B), k = 75)
train_data_B_5 = random.choices(class5(train_data_B), k = 75)

train_data_sample = []
train_data_sample.extend(train_data_B_1)
train_data_sample.extend(train_data_B_2)
train_data_sample.extend(train_data_B_3)
train_data_sample.extend(train_data_B_4)
train_data_sample.extend(train_data_B_5)
train_data_sample.extend(train_data_IV_1)
train_data_sample.extend(train_data_IV_2)
train_data_sample.extend(train_data_IV_3)
train_data_sample.extend(train_data_IV_4)
train_data_sample.extend(train_data_IV_5)
train_data_sample.extend(train_data_DM_1)
train_data_sample.extend(train_data_DM_2)
train_data_sample.extend(train_data_DM_3)
train_data_sample.extend(train_data_DM_4)
train_data_sample.extend(train_data_DM_5)
train_data_sample.extend(train_data_MI_1)
train_data_sample.extend(train_data_MI_2)
train_data_sample.extend(train_data_MI_3)
train_data_sample.extend(train_data_MI_4)
train_data_sample.extend(train_data_MI_5)



#Creating review and overall lists, and removing empty reviews


review_text_list3_sample = [review.reviewtext_cleaned for review in train_data_sample]
review_text_list2_sample = [' ' if v is None else v for v in review_text_list2_sample]
review_text_list3_sample = [' ' if v is None else v for v in review_text_list3_sample]


for i in review_text_list3_sample:
    print(i)


review_text_list_test = [review.reviewtext_cleaned for review in test_data_cleaned]
review_text_list_test = [' ' if v is None else v for v in review_text_list_test]

review_text_list = [review.reviewtext_cleaned for review in train_data_cleaned]
review_text_list = [' ' if v is None else v for v in review_text_list]



review_text_list3_test = [review.reviewtext_cleaned for review in test_data3_cleaned]
review_text_list3_test = [' ' if v is None else v for v in review_text_list3_test]

review_text_list3 = [review.reviewtext_cleaned for review in train_data3_cleaned]
review_text_list3 = [' ' if v is None else v for v in review_text_list3]

review_overall_list = [review.overall for review in train_data_cleaned]
review_overall_list_test = [review.overall for review in test_data_cleaned]


review_overall_list3 = [review.overall for review in train_data3_cleaned]
review_overall_list3_test = [review.overall for review in test_data3_cleaned]



review_text_list_nouns_adverbs_adjectives = remove_nouns_adverbs_adjectives(review_text_list)
review_text_list_nouns_adverbs_adjectives_test = remove_nouns_adverbs_adjectives(review_text_list_test)


review_text_list_nouns_adjectives = remove_pair(review_text_list)
review_text_list_nouns_adjectives_test = remove_pair(review_text_list_test)





#Seeing cleaned text

for i in review_text_list:
       print(i)

# Build the co-occurrence matrix.

train_words = merge_sentences(review_text_list3_sample)


traindata_words = []
for line in review_text_list3_sample:
    list5 = []
    for sentence in line:
        list5.append(sentence.split())
    traindata_words.append(list5)


traindata_words = merge_sentences(traindata_words)




traindata_words = tuple(tuple(x) for x in traindata_words)


d1 = cooccurrence_matrix(traindata_words)

"""Export Reviews with Pickle"""
with open('Data\\Pickle\\d1_protocol1.pickle', 'wb') as handle:
    pickle.dump(d1, handle, protocol=1)


with open('Data\\Pickle\\d1_protocol1.pickle', 'rb') as handle:
    d1 = pickle.load(handle)

pickle.dump( d1, open( "d1.pickle", "wb" ) )
# Get the vocab.
vocab1 = get_sorted_vocab(d1)



positive_words = importpos()
negative_words = importneg()

overall_1 = [review.reviewtext_cleaned for review in train_data3_cleaned if review.overall == 1]
overall_5 = [review.reviewtext_cleaned for review in train_data3_cleaned if review.overall == 5]
overall_1 = [' ' if v is None else v for v in overall_1]
overall_5 = [' ' if v is None else v for v in overall_5]
count = dict()
ov_total = overall_5
for review in ov_total:
    review = ' '.join(review)
    for word in review.split():

        if word in count:
            count[word] = count[word] + 1
        else:
            if word in positive_words:
                count[word] = 1

import operator

sorted_d = sorted(count.items(), key=operator.itemgetter(1))

new_positive = []

for word in sorted_d[::-1]:

    if word[1] > 100 and len(new_positive) < 100:
        new_positive.append(word[0])



# Build the cosine matrix.
cm1 = cosine_similarity_matrix(vocab1, d1)
prop1 = graph_propagation(cm1, vocab1, new_positive, new_negative, 2)

sent_lex1500 = []
for key, val in sorted(prop1.items(), key=itemgetter(1), reverse=True):
    sent_lex1500.append([key, val])


hu_liu = {**positive_words,**negative_words}



negation_words = importnegwords()
neutral_words = importneut()

#Qualitative analysis
hu_liu_analysis=[]
for (key,val) in hu_liu.items():
        hu_liu_=[]
        hu_liu_.append(key)
        hu_liu_.append(val)
        hu_liu_analysis.append(test)


hu_liu_analysis = tuple(tuple(x) for x in hu_liu_analysis)


sent_lex1500 = tuple(tuple(x) for x in sent_lex1500)
hlsentiment=[]
for i in sent_lex1500:
    for u in hu_liu_analysis:
        if i[0]==u[0]:
            sentimentx=[]
            sentimentx.append(i[0])
            sentimentx.append(i[1])
            sentimentx.append(u[1])
            hlsentiment.append(sentimentx)
for i in hlsentiment:
    print(i)

review_text_list3 = tuple(tuple(x) for x in review_text_list3)

review_text_list3_test = tuple(tuple(x) for x in review_text_list3_test)


sent_lex1500 = tuple(tuple(x) for x in sent_lex1500)
len(sent_lex1500)
listt=[]
for test in sent_lex1500:
    if test[1] > 0 or test[1]<0:
        listt.append(test)
len(listt)

listt = tuple(tuple(x) for x in listt)

sent_lex2 = {}
for i in listt:
    sent_lex2[i[0]] = i[1]

reviews1500 = neut_neg2(review_text_list3,sent_lex2,negation_words,neutral_words)
reviews1500_test = neut_neg2(review_text_list3_test,sent_lex2,negation_words,neutral_words)


reviews1500hl = neut_neg2(review_text_list3,hu_liu,negation_words,neutral_words)
reviews1500hl_test = neut_neg2(review_text_list3_test,hu_liu,negation_words,neutral_words)


reviews1500 = merge_sentences(reviews1500)

reviews1500 = tuple(tuple(tuple(x) for x in y) for y in reviews1500)

reviews1500 = sum_repeated(reviews1500)

reviews1500_test = merge_sentences(reviews1500_test)

reviews1500_test = tuple(tuple(tuple(x) for x in y) for y in reviews1500_test)

reviews1500_test = sum_repeated(reviews1500_test)

reviews1500hl = merge_sentences(reviews1500hl)

reviews1500hl = tuple(tuple(tuple(x) for x in y) for y in reviews1500hl)

reviews1500hl = sum_repeated(reviews1500hl)

reviews1500hl_test = merge_sentences(reviews1500hl_test)

reviews1500hl_test = tuple(tuple(tuple(x) for x in y) for y in reviews1500hl_test)

reviews1500hl_test = sum_repeated(reviews1500hl_test)



reviews_lex = reviews1500 + reviews1500_test
review_hl = reviews1500hl + reviews1500hl_test

uniquewords = extract_unique_words(reviews_lex)
uniquewordshl = extract_unique_words(review_hl)


reviews_matrix = create_sparse_matrix(uniquewords,reviews1500)

reviews_matrix_test = create_sparse_matrix(uniquewords,reviews1500_test)

reviews_matrixhl = create_sparse_matrix(uniquewordshl,reviews1500hl)

reviews_matrixhl_test = create_sparse_matrix(uniquewordshl,reviews1500hl_test)



svd_model = Classification(SGDClassifier(),reviews_matrix, review_overall_list3)
svd_modelhl = Classification(SGDClassifier(),reviews_matrixhl, review_overall_list3)
predicted_svd = prediction(svd_model, reviews_matrix_test)
predicted_svdhl = prediction(svd_modelhl, reviews_matrixhl_test)


f1_score_svd = f1_score(review_overall_list3_test, predicted_svd, average='macro')
#
print('F1-SCORE RESULT: '+str(f1_score_svd))


accuracy_svd = accuracy(review_overall_list3_test, predicted_svd)
#
print('ACCURACY RESULT: '+str(accuracy_svd))

precision_svd = precision(review_overall_list3_test, predicted_svd, average='macro')
#
print('PRECISION RESULT: '+str(precision_svd))


recall_svd = recall(review_overall_list3_test, predicted_svd, average='macro')
#
print('RECALL RESULT: '+str(recall_svd))


f1_score_svdhl = f1_score(review_overall_list3_test, predicted_svdhl, average='macro')
#
print('F1-SCORE RESULT: '+str(f1_score_svdhl))


accuracy_svdhl = accuracy(review_overall_list3_test, predicted_svdhl)
#
print('ACCURACY RESULT: '+str(accuracy_svdhl))


precision_svdhl = precision(review_overall_list3_test, predicted_svdhl, average='macro')
#
print('PRECISION RESULT: '+str(precision_svdhl))


recall_svdhl = recall(review_overall_list3_test, predicted_svdhl, average='macro')
#
print('RECALL RESULT: '+str(recall_svdhl))





# # Tokenization (BOW)

bow_index, bow_vectorizer = word_tokenize_scikit(review_text_list)


bigram_index_sample, bigram_vectorizer_sample = word_tokenize_scikit(review_text_list_nouns_adjectives)
bigram_index, bigram_vectorizer = word_tokenize_scikit(review_text_list_nouns_adjectives)

bow_index_nouns, bow_vectorizer_nouns = word_tokenize_scikit(review_text_list_nouns_adverbs_adjectives)




# # TFIDF



tfidf_index, tfidf_vectorizer = tfidf_bow(review_text_list)


bigramtfidf_index, bigramtfidf_vectorizer = tfidf_bow(review_text_list_nouns_adjectives)


tfidf_index_naa, tfidf_vectorizer_naa = tfidf_bow(review_text_list_nouns_adverbs_adjectives)




tfidf_index_test = tfidf_vectorizer.transform(review_text_list_test)

bigramtfidf_index_test = bigramtfidf_vectorizer.transform(review_text_list_nouns_adjectives_test)

tfidf_index_naa_test = tfidf_vectorizer_naa.transform(review_text_list_nouns_adverbs_adjectives_test)






# LSA

#Regular BOW

lsa = TruncatedSVD(n_components=100)
lsa_matrix = lsa.fit_transform(tfidf_index)

lsa2 = lsa.fit(tfidf_index)
lsa_matrix = lsa_matrix[:,[0,1,2,4,6]]
lsa_matrix_test = lsa2.transform(tfidf_index_test)
lsa_matrix_test = lsa_matrix_test[:,[0,1,2,4,6]]


names=tfidf_vectorizer.get_feature_names()

#Top ten words in each concept
for i, comp in enumerate(lsa.components_):
    termsInComp = zip (names,comp)
    sortedTerms =  sorted(termsInComp, key=lambda x: x[1], reverse=True) [:10]
    print("Concept %d:" % i )
    for term in sortedTerms:
        print(term[0])
    print (" ")


for i, comp in enumerate(lsa.components_):
    termsInComp = zip (names,comp)
    sortedTerms =  sorted(termsInComp, key=lambda x: x[1], reverse=False) [:10]
    print("Concept %d:" % i )
    for term in sortedTerms:
        print(term[0])
    print (" ")



#Singular values Representaion of categories per concept

lsa2.singular_values_
df = lsa_matrix
df = pd.DataFrame(df)
df['overall'] = review_overall_list
df1=df[[0, 1, 2, 3, 4, 5, 6, 'overall']]
df1.groupby(['overall',]).mean()


#BOW with only naa

lsa_naa = TruncatedSVD(n_components=100)
lsa_naa_matrix = lsa_naa.fit_transform(tfidf_index_naa)
lsa_naa2 = lsa_naa.fit(tfidf_index_naa)
lsa_naa_matrix= lsa_naa_matrix[:,[0,1,2,3,4]]
lsa_naa_matrix_test = lsa_naa2.transform(tfidf_index_naa_test)
lsa_naa_matrix_test= lsa_naa_matrix_test[:,[0,1,2,3,4]]
names2=tfidf_vectorizer_naa.get_feature_names()


#Top ten words for each concept

for i, comp in enumerate(lsa_naa.components_):
    termsInComp = zip (names2,comp)
    sortedTerms =  sorted(termsInComp, key=lambda x: x[1], reverse=True) [:10]
    print("Concept %d:" % i )
    for term in sortedTerms:
        print(term[0])
    print (" ")

for i, comp in enumerate(lsa_naa.components_):
    termsInComp = zip (names2,comp)
    sortedTerms =  sorted(termsInComp, key=lambda x: x[1], reverse=False) [:10]
    print("Concept %d:" % i )
    for term in sortedTerms:
        print(term[0])
    print (" ")


#Representation of each overall in each concept
lsa_naa.singular_values_
df2 = lsa_naa_matrix
df2 = pd.DataFrame(df2)
df2['overall'] = review_overall_list
df3=df2[[0, 1, 2, 3, 4, 5, 6, 7, 'overall']]
df3.groupby(['overall',]).mean()



#Words with Bigrams

lsa_bigram = TruncatedSVD(n_components=50)
lsa_bigram2 = lsa_bigram.fit(bigramtfidf_index)
lsa_bigram_matrix = lsa_bigram.fit_transform(bigramtfidf_index)
lsa_bigram_matrix = lsa_bigram_matrix[:,[0,1,2,3,5]]
lsa_bigram_matrix_test = lsa_bigram2.transform(bigramtfidf_index_test)
lsa_bigram_matrix_test = lsa_bigram_matrix_test[:,[0,1,2,3,5]]
names3=bigramtfidf_vectorizer.get_feature_names()


#Top ten words for each concept
for i, comp in enumerate(lsa_bigram2.components_):
    termsInComp = zip (names3,comp)
    sortedTerms =  sorted(termsInComp, key=lambda x: x[1], reverse=True) [:10]
    print("Concept %d:" % i )
    for term in sortedTerms:
        print(term[0])
    print (" ")

    print (" ")

for i, comp in enumerate(lsa_bigram.components_):
    termsInComp = zip (names3,comp)
    sortedTerms =  sorted(termsInComp, key=lambda x: x[1], reverse=False) [:10]
    print("Concept %d:" % i )
    for term in sortedTerms:
        print(term[0])
    print(" ")

#Singular values and representation of each overall i each concept

lsa_bigram2.singular_values_
df3 = lsa_bigram_matrix
df3 = pd.DataFrame(df3)
df3['overall'] = review_overall_list
df4=df3[[0, 1, 2, 3, 4, 5, 'overall']]
df4.groupby(['overall',]).mean()




# Regular BOW

mnb_matrix = minmax_scale(lsa_matrix)

mnb_matrix_test = minmax_scale(lsa_matrix_test)


mnb_model = Classification(MultinomialNB(),mnb_matrix, review_overall_list)

svc_model =  Classification(SVC(),lsa_matrix, review_overall_list)

svc_model_class =  Classification(SVC(C=1.0, kernel='linear'),lsa_matrix, review_overall_list)


predicted_mnb = prediction(mnb_model, mnb_matrix_test)

predicted_svc = prediction(svc_model, lsa_matrix_test)

predicted_svc_class = prediction(svc_model_class, lsa_matrix_test)

    #bow with naa

mnb_matrix_naa = minmax_scale(lsa_naa_matrix)

mnb_matrix_naa_test = minmax_scale(lsa_naa_matrix_test)

mnb_model_naa = Classification(MultinomialNB(),mnb_matrix_naa, review_overall_list)

svc_model_naa =  Classification(SVC(),lsa_naa_matrix, review_overall_list)

svc_model_naa_class =  Classification(SVC(C=1.0, kernel='linear'),lsa_naa_matrix, review_overall_list)



predicted_mnb_naa = prediction(mnb_model_naa, mnb_matrix_naa_test)

predicted_svc_naa = prediction(svc_model_naa, lsa_naa_matrix_test)

predicted_svc_naa_class = prediction(svc_model_naa_class, lsa_naa_matrix_test)



    #Words with Bigrams

mnb_matrix_bigrams = minmax_scale(lsa_bigram_matrix)

mnb_matrix_bigrams_test = minmax_scale(lsa_bigram_matrix_test)

mnb_model_bigrams = Classification(MultinomialNB(),mnb_matrix_bigrams, review_overall_list)

svc_model_bigrams_class =  Classification(SVC(C=1.0, kernel='linear'),lsa_bigram_matrix, review_overall_list)

svc_model_bigrams =  Classification(SVC(),lsa_bigram_matrix, review_overall_list)



predicted_mnb_bigrams = prediction(mnb_model_bigrams, mnb_matrix_bigrams_test)

predicted_svc_bigrams_class = prediction(svc_model_bigrams_class, lsa_bigram_matrix_test)

predicted_svc_bigrams = prediction(svc_model_bigrams, lsa_bigram_matrix_test)



"""
    Evaluation metrics
"""

#BOW

f1_score_mnb = f1_score(review_overall_list3_test, predicted_mnb, average='macro')
#
print('F1-SCORE RESULT: '+str(f1_score_mnb))


accuracy_mnb = accuracy(review_overall_list3_test, predicted_mnb)
#
print('ACCURACY RESULT: '+str(accuracy_mnb))


precision_mnb = precision(review_overall_list3_test, predicted_mnb, average='macro')
#
print('PRECISION RESULT: '+str(precision_mnb))


recall_mnb = recall(review_overall_list3_test, predicted_mnb, average='macro')
#
print('RECALL RESULT: '+str(recall_mnb))






f1_score_svc = f1_score(review_overall_list3_test, predicted_svc, average='macro')
#
print('F1-SCORE RESULT: '+str(f1_score_svc))


accuracy_svc = accuracy(review_overall_list3_test, predicted_svc)
#
print('ACCURACY RESULT: '+str(accuracy_svc))


precision_svc = precision(review_overall_list3_test, predicted_svc, average='macro')
#
print('PRECISION RESULT: '+str(precision_svc))


recall_svc = recall(review_overall_list3_test, predicted_svc, average='macro')
#
print('RECALL RESULT: '+str(recall_svc))




f1_score_svc_class = f1_score(review_overall_list3_test, predicted_svc_class, average='macro')
#
print('F1-SCORE RESULT: '+str(f1_score_svc_class))


accuracy_svc_class = accuracy(review_overall_list3_test, predicted_svc_class)
#
print('ACCURACY RESULT: '+str(accuracy_svc_class))


precision_svc_class = precision(review_overall_list3_test, predicted_svc_class, average='macro')
#
print('PRECISION RESULT: '+str(precision_svc_class))


recall_svc_class = recall(review_overall_list3_test, predicted_svc_class, average='macro')
#
print('RECALL RESULT: '+str(recall_svc_class))


#BOW (naa)



f1_score_mnb_naa = f1_score(review_overall_list3_test, predicted_mnb_naa, average='macro')
#
print('F1-SCORE RESULT: '+str(f1_score_mnb_naa))


accuracy_mnb_naa = accuracy(review_overall_list3_test, predicted_mnb_naa)
#
print('ACCURACY RESULT: '+str(accuracy_mnb_naa))


precision_mnb_naa = precision(review_overall_list3_test, predicted_mnb_naa, average='macro')
#
print('PRECISION RESULT: '+str(precision_mnb_naa))


recall_mnb_naa = recall(review_overall_list3_test, predicted_mnb_naa, average='macro')
#
print('RECALL RESULT: '+str(recall_mnb_naa))





f1_score_svc_naa = f1_score(review_overall_list3_test, predicted_svc_naa, average='macro')
#
print('F1-SCORE RESULT: '+str(f1_score_svc_naa))


accuracy_svc_naa = accuracy(review_overall_list3_test, predicted_svc_naa)
#
print('ACCURACY RESULT: '+str(accuracy_svc_naa))


precision_svc_naa = precision(review_overall_list3_test, predicted_svc_naa, average='macro')
#
print('PRECISION RESULT: '+str(precision_svc_naa))


recall_svc_naa = recall(review_overall_list3_test, predicted_svc_naa, average='macro')
#
print('RECALL RESULT: '+str(recall_svc_naa))




f1_score_svc_naa_class = f1_score(review_overall_list3_test, predicted_svc_naa_class, average='macro')
#
print('F1-SCORE RESULT: '+str(f1_score_svc_naa_class))


accuracy_svc_naa_class = accuracy(review_overall_list3_test, predicted_svc_naa_class)
#
print('ACCURACY RESULT: '+str(accuracy_svc_naa_class))


precision_svc_naa_class = precision(review_overall_list3_test, predicted_svc_naa_class, average='macro')
#
print('PRECISION RESULT: '+str(precision_svc_naa_class))


recall_svc_naa_class = recall(review_overall_list3_test, predicted_svc_naa_class, average='macro')
#
print('RECALL RESULT: '+str(recall_svc_naa_class))



#Words with Bigrams



f1_score_mnb_bigrams = f1_score(review_overall_list3_test, predicted_mnb_bigrams, average='macro')
#
print('F1-SCORE RESULT: '+str(f1_score_mnb_bigrams))


accuracy_mnb_bigrams = accuracy(review_overall_list3_test, predicted_mnb_bigrams)
#
print('ACCURACY RESULT: '+str(accuracy_mnb_bigrams))


precision_mnb_bigrams = precision(review_overall_list3_test, predicted_mnb_bigrams, average='macro')
#
print('PRECISION RESULT: '+str(precision_mnb_bigrams))


recall_mnb_bigrams = recall(review_overall_list3_test, predicted_mnb_bigrams, average='macro')
#
print('RECALL RESULT: '+str(recall_mnb_bigrams))





f1_score_svc_bigrams = f1_score(review_overall_list3_test, predicted_svc_bigrams, average='macro')
#
print('F1-SCORE RESULT: '+str(f1_score_svc_bigrams))


accuracy_svc_bigrams = accuracy(review_overall_list3_test, predicted_svc_bigrams)
#
print('ACCURACY RESULT: '+str(accuracy_svc_bigrams))


precision_svc_bigrams = precision(review_overall_list3_test, predicted_svc_bigrams, average='macro')
#
print('PRECISION RESULT: '+str(precision_svc_bigrams))


recall_svc_bigrams = recall(review_overall_list3_test, predicted_svc_bigrams, average='macro')
#
print('RECALL RESULT: '+str(recall_svc_bigrams))




f1_score_svc_bigrams_class = f1_score(review_overall_list3_test, predicted_svc_bigrams_class, average='macro')
#
print('F1-SCORE RESULT: '+str(f1_score_svc_bigrams_class))


accuracy_svc_bigrams_class = accuracy(review_overall_list3_test, predicted_svc_bigrams_class)
#
print('ACCURACY RESULT: '+str(accuracy_svc_bigrams_class))


precision_svc_bigrams_class = precision(review_overall_list3_test, predicted_svc_bigrams_class, average='macro')
#
print('PRECISION RESULT: '+str(precision_svc_bigrams_class))


recall_svc_bigrams_class = recall(review_overall_list3_test, predicted_svc_bigrams_class, average='macro')
#
print('RECALL RESULT: '+str(recall_svc_bigrams_class))
