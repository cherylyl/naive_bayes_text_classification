#!/usr/bin/python

import sys
import os
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
import math
import string
import nltk

from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer




###############################################################################


def find_voc(Path, K):
    lem = WordNetLemmatizer()
    ps = PorterStemmer()
    doc = ""
    neg_train_files = os.listdir(os.path.join(Path, 'training_set', 'neg'))
    for neg_train in neg_train_files:
        path = os.path.join(Path, 'training_set', 'neg', neg_train)
        with open(path,encoding="utf8") as f:
            temp = f.read()
            doc = doc+temp

    pos_train_files = os.listdir(os.path.join(Path, 'training_set', 'pos'))
    for pos_train in pos_train_files:
        path = os.path.join(Path, 'training_set', 'pos', pos_train)
        with open(path,encoding="utf8") as f:
            temp = f.read()
            doc = doc + temp

    doc = doc.replace("loved", "love")
    doc = doc.replace("loves", "love")
    doc = doc.replace("loving", "love")


    tokens = doc.strip().split()

    sw = stopwords.words('english')
    punc = list(string.punctuation)
    punc.remove('!')
    punc.remove('?')
    sw += punc

    others = ["movie", "film", "one", 'two', 'wa', 'ha', '--', "get", "character", "plot", "that's",'see', 'make',
              'scene','something', 'story', "series", "doe"]
    for other in others:
        sw.append(other)

    cleaned = []
    for token in tokens:
        token = lem.lemmatize(token)
        if token not in sw:
            # print(token)
            cleaned.append(token)

    fdist = FreqDist(cleaned)

    freq_K = fdist.most_common(K)
    vocab = []
    for i in range(K):
        vocab.append(freq_K[i][0])
    return vocab




def transfer(fileDj, vocabulary):
    lem = WordNetLemmatizer()

    with open(fileDj, 'r',encoding="utf8") as f:
        doc = f.read()
        doc = doc.replace("loved","love")
        doc = doc.replace("loves","love")
        doc = doc.replace("loving","love")

    BOWDj = [0 for i in range(len(vocabulary))]
    words = doc.split()



    for word in words:
        word = lem.lemmatize(word)
        if (word in vocabulary):
            index = vocabulary.index(word)
            BOWDj[index] += 1
        else:
            BOWDj[-1] += 1

    return BOWDj


def loadData(Path, i):
    Xtrain = []
    Xtest = []
    ytrain = []
    ytest = []



    if i == 0:
        words = ['love', 'wonderful', 'best', 'great', 'superb', 'still', 'beautiful', 'bad', 'worst', 'stupid',
                 'waste', 'boring', '?', '!', 'UNK']
    else:
        words = find_voc(Path,1400)


    neg_train_files = os.listdir(os.path.join(Path, 'training_set', 'neg'))
    for neg_train in neg_train_files:
        path = os.path.join(Path, 'training_set', 'neg', neg_train)
        Xtrain.append(transfer(path,words))
        ytrain.append(0)

    pos_train_files = os.listdir(os.path.join(Path, 'training_set', 'pos'))
    for pos_train in pos_train_files:
        path = os.path.join(Path, 'training_set', 'pos', pos_train)
        Xtrain.append(transfer(path, words))
        ytrain.append(1)


    neg_test_files = os.listdir(os.path.join(Path, 'test_set', 'neg'))
    for neg_test in neg_test_files:
        path = os.path.join(Path, 'test_set', 'neg', neg_test)
        Xtest.append(transfer(path, words))
        ytest.append(0)



    pos_test_files = os.listdir(os.path.join(Path, 'test_set', 'pos'))
    for pos_test in pos_test_files:
        path = os.path.join(Path, 'test_set', 'pos', pos_test)
        Xtest.append(transfer(path, words))
        ytest.append(1)

    Xtrain = np.reshape(Xtrain, (1400,len(words)))
    ytrain = np.reshape(ytrain, (1400,1))

    Xtest = np.reshape(Xtest, (600,len(words)))
    ytest = np.reshape(ytest, (600,1))

    return Xtrain, Xtest, ytrain, ytest, words




def naiveBayesMulFeature_train(Xtrain, ytrain, voc):
    alpha = 1

    thetaPos = [0 for i in range(voc)]
    thetaNeg = [0 for i in range(voc)]
    num_words_neg = np.sum(Xtrain[0:700,:])
    num_words_pos = np.sum(Xtrain[700:1400,:])
    for i in range(voc):
        thetaNeg[i] = (np.sum(Xtrain[0:700,i:i+1])+alpha)/(num_words_neg+voc)
        thetaPos[i] = (np.sum(Xtrain[700:1400,i:i+1])+alpha)/(num_words_pos+voc)


    return thetaPos,thetaNeg


def naiveBayesMulFeature_test(Xtest, ytest,thetaPos, thetaNeg, voc):
    yPredict = []
    thetaNeg = np.reshape(thetaNeg, (voc, 1))
    thetaPos = np.reshape(thetaPos, (voc, 1))

    neg_p = Xtest.dot(np.log(thetaNeg))
    pos_p = Xtest.dot(np.log(thetaPos))

    for i in range(np.size(Xtest,0)):
        if (neg_p[i][0] < pos_p[i][0]):
            yPredict.append(1)
        else:
            yPredict.append(0)
    ytest = np.reshape(ytest, 600)

    Accuracy = np.sum(ytest==yPredict)/600


    return yPredict, Accuracy


def naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest,voc):
    ytrain = np.reshape(ytrain,1400)
    model = MultinomialNB()
    model.fit(Xtrain,ytrain)

    yPredict = model.predict(Xtest)

    ytest = np.reshape(ytest, 600)
    Accuracy = np.sum(ytest == yPredict) / 600


    return Accuracy

def naiveBayesMulFeature_sk_Bern(Xtrain, ytrain, Xtest, ytest,voc):
    ytrain = np.reshape(ytrain,1400)
    model =BernoulliNB()
    model.fit(Xtrain,ytrain)

    yPredict = model.predict(Xtest)

    ytest = np.reshape(ytest, 600)
    Accuracy = np.sum(ytest == yPredict) / 600


    return Accuracy



def naiveBayesBernFeature_train(Xtrain, ytrain, voc):
    thetaPosTrue = [0 for i in range(voc)]
    thetaNegTrue = [0 for i in range(voc)]
    neg_X = Xtrain[0:700,:]
    pos_X = Xtrain[700:1400,:]


    for i in range(voc):
        col = neg_X[:,i:i+1]

        count = 1
        for j in range(700):
            if (col[j][0])>0:
                count += 1
            thetaNegTrue[i] = float(count)/(700+2)

    for i in range(voc):
        col = pos_X[:,i:i+1]

        count = 1
        for j in range(700):
            if (col[j][0])>0:
                count += 1
            thetaPosTrue[i] = float(count)/(700+2)

    return thetaPosTrue, thetaNegTrue

    
def naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue, voc):

    yPredict = []

    neg_p = [math.log(0.5) for i in range(600)]
    pos_p = [math.log(0.5) for i in range(600)]

    for i in range(600):
        doc = Xtest[i:i+1,:]
        doc = np.reshape(doc,voc)
        for j in range(voc):
            if doc[j]>0:
                neg_p[i] += math.log(thetaNegTrue[j])
                pos_p[i] += math.log(thetaPosTrue[j])
            else:
                neg_p[i] += math.log((1-thetaNegTrue[j]))
                pos_p[i] += math.log((1 - thetaPosTrue[j]))


    for i in range(np.size(Xtest, 0)):
        if (neg_p[i] < pos_p[i]):
            yPredict.append(1)
        else:
            yPredict.append(0)
    ytest = np.reshape(ytest, 600)


    Accuracy = float(np.sum(ytest == yPredict) / 600)

    return yPredict, Accuracy


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python naiveBayes.py dataSetPath testSetPath")
        sys.exit()

    textDataSetsDirectoryFullPath = sys.argv[1]
    testFileDirectoryFullPath = sys.argv[2]

    words = ['love', 'wonderful', 'best', 'great', 'superb', 'still', 'beautiful', 'bad', 'worst', 'stupid',
             'waste', 'boring', '?', '!', 'UNK']


    # textDataSetsDirectoryFullPath = ''
    # voc = find_voc(textDataSetsDirectoryFullPath)
    # print(voc)
    # voc = transfer(textDataSetsDirectoryFullPath1,words)
    # print(voc)

    trials = ['First trial with predefined 15 words in vocabulary', 'Second trial with different preprocessing of new vocabulary']

    for i in range(2):
        print("--------------------")
        print(trials[i])
        print("--------------------")

        Xtrain, Xtest, ytrain, ytest, vocab = loadData(textDataSetsDirectoryFullPath, i)
        v = len(vocab)
        thetaPos, thetaNeg = naiveBayesMulFeature_train(Xtrain, ytrain, v)
        # print("thetaPos =", thetaPos)
        # print("thetaNeg =", thetaNeg)
        print("--------------------")

        yPredict, Accuracy = naiveBayesMulFeature_test(Xtest, ytest, thetaPos, thetaNeg,v)
        print("MNBC classification accuracy =", Accuracy)

        Accuracy_sk = naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest, v)
        print("Sklearn MultinomialNB accuracy =", Accuracy_sk)

        Accuracy_ber = naiveBayesMulFeature_sk_Bern(Xtrain, ytrain, Xtest, ytest, v)
        print("Sklearn Multivariate Bernoulli accuracy =", Accuracy_ber)

        thetaPosTrue, thetaNegTrue = naiveBayesBernFeature_train(Xtrain, ytrain, v)
        # print("thetaPosTrue =", thetaPosTrue)
        # print("thetaNegTrue =", thetaNegTrue)
        print("--------------------")

        yPredict, Accuracy = naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue, v)
        print("BNBC classification accuracy =", Accuracy)
        print("--------------------")



     



