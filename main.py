import csv
import math
import nltk
import scipy.sparse
from nltk.stem import PorterStemmer
import re
import operator
import string
import gensim
from pprint import pprint
from gensim.corpora import Dictionary
import sklearn
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#IR evaluation
qrelsfile = open('qrels.csv', 'r')
qrelsreader = csv.reader(qrelsfile)
qrelsheader = next(qrelsreader)
qrelsdocs = {}
for row in qrelsreader:
    if row[0] not in qrelsdocs:
        qrelsdocs[row[0]]={}
        qrelsdocs[row[0]][row[1]] = row[2]
    else:
        qrelsdocs[row[0]][row[1]] = row[2]

systemresultsfile = open('system_results.csv')
systemresultsreader = csv.reader(systemresultsfile)
systemresultsheader = next(systemresultsreader)
systemresultsdocs={}
for row in systemresultsreader:
    systemnumber=row[0]
    querynumber=row[1]
    docnumber=row[2]
    rankofdoc=row[3]
    score=row[4]
    a_list = [docnumber, rankofdoc, score]
    if systemnumber not in systemresultsdocs:
        systemresultsdocs[systemnumber]={}
        systemresultsdocs[systemnumber][querynumber] = []
        systemresultsdocs[systemnumber][querynumber].append(a_list)
    else:
        if querynumber in systemresultsdocs[systemnumber]:
            systemresultsdocs[systemnumber][querynumber].append(a_list)
        else:
            systemresultsdocs[systemnumber][querynumber] = []
            systemresultsdocs[systemnumber][querynumber].append(a_list)

def calculatepatr(listforpatr, query, limit):
    patr = 0
    for row in listforpatr:
        docnumber = row[0]
        if docnumber in qrelsdocs.get(query):
            patr+=1
    patr/=limit
    return round(patr, 3)

def calculater50(listforr50, query):
    r50 = 0
    for row in listforr50:
        docnumber = row[0]
        if docnumber in qrelsdocs.get(query):
            r50+=1
    r50/=len(qrelsdocs.get(query))
    return round(r50, 3)

def averageprecision(listforavgprecision, query):
    avgprec = 0
    count = 0
    for row in listforavgprecision:
        docnumber = row[0]
        rank = row[1]
        if docnumber in qrelsdocs.get(query):
            count+=1
            avgprec+= count/int(rank)
    avgprec/=len(qrelsdocs.get(query))
    return round(avgprec, 3)

def ndcgatlimit(listforndcglimit):
    dg = []
    dcg = []
    relevances=[]
    for row in listforndcglimit:
        docnumber = row[0]
        rank = row[1]
        if docnumber in qrelsdocs.get(query):
            relevance = qrelsdocs.get(query).get(docnumber)
            relevances.append(int(relevance))
            if int(rank)==1:
                dg.append(int(relevance))
            else:
                dg.append(int(relevance)/math.log2(int(rank)))
        else:
            dg.append(0)
    dcg.append(dg[0])
    for i in range(1,len(listforndcglimit)):
        dcg.append(dcg[i-1]+dg[i])
    relevances.sort(reverse=True)
    relevances.extend([0]*(len(listforndcglimit)-len(relevances)))
    idg=[]
    idcg=[]
    for index, rel in enumerate(relevances):
        if index==0:
            idg.append(relevances[index])
        else:
            idg.append(rel / math.log2(index+1))
    idcg.append(idg[0])
    for i in range(1, len(listforndcglimit)):
        idcg.append(idcg[i - 1] + idg[i])
    ndcg = []
    for index, element in enumerate(dcg):
        if dcg[index]!=0 and idcg[index]!=0:
            ndcg.append(dcg[index]/idcg[index])
        else:
            ndcg.append(0)
    return round(ndcg[len(listforndcglimit)-1], 3)

#for every system, for each query caluclate P@10, R@50, r-precision, AP, nDCG@10 and nDCG@20
irevaluationfile = open('ir_eval.csv','w')
csvwriter = csv.writer(irevaluationfile, lineterminator='\n')
csvwriter.writerow(['system_number','query_number','P@10','R@50','r-precision','AP','nDCG@10','nDCG@20'])
for system in systemresultsdocs:
    avgp10=0
    avgr50=0
    avgrprecision=0
    avgap=0
    avgndcg10=0
    avgndcg20=0
    for query in systemresultsdocs.get(system):
        #p10
        listforp10 = systemresultsdocs[system].get(query)[:10]
        p10 = calculatepatr(listforp10, query, 10)
        #r50
        listforr50 = systemresultsdocs[system].get(query)[:50]
        r50 = calculater50(listforr50, query)
        #r-precision
        r = len(qrelsdocs.get(query))
        listforrprecision = systemresultsdocs[system].get(query)[:r]
        rprecision = calculatepatr(listforrprecision, query, r)
        #AP
        avgprecision = averageprecision(systemresultsdocs[system].get(query), query)
        #nDCG@10
        listforndcg10 = systemresultsdocs[system].get(query)[:10]
        ndcg10 = ndcgatlimit(listforndcg10)
        #nDCG@20
        listforndcg20 = systemresultsdocs[system].get(query)[:20]
        ndcg20 = ndcgatlimit(listforndcg20)
        avgp10+=p10
        avgr50+=r50
        avgrprecision+=rprecision
        avgap+=avgprecision
        avgndcg10+=ndcg10
        avgndcg20+=ndcg20
        csvwriter.writerow([system,query,p10,r50,rprecision,avgprecision,ndcg10,ndcg20])
    csvwriter.writerow([system,'mean',round(avgp10/10,3),round(avgr50/10,3),round(avgrprecision/10,3),round(avgap/10,3),round(avgndcg10/10,3),round(avgndcg20/10,3)])

#text analysis

textanalysisfile = open('train_and_dev.tsv','r')
textanalysiscontent = textanalysisfile.readlines()
versedict={}
for line in textanalysiscontent:
    corpus=line.split('\t')[0].strip()
    verse =  line.split('\t')[1].strip()
    if corpus not in versedict:
        versedict[corpus] = []
        versedict[corpus].append(verse)
    else:
        versedict[corpus].append(verse)
numberofOTverses = len(versedict.get('OT'))
numberofNTverses = len(versedict.get('NT'))
numberofQuranverses = len(versedict.get('Quran'))
totalnumberofverses = numberofOTverses+numberofNTverses+numberofQuranverses

OTtokendict={}
NTtokendict={}
Qurantokendict={}

porter = PorterStemmer()
stopwordsfile = open('EnglishStopWords.txt', 'r')
stopwords = stopwordsfile.read().split()
for index, stopword in enumerate(stopwords):
    stopwords[index] = porter.stem(stopword)

def preprocessing(verse, corpora):
    preprocessedtokens = []
    tokens = re.split(r"[ ,.:;></|+-@#!$%^&*_=\n\'\"\[\]\{\}\(\)]", verse)  # tokenisation
    tokens = set(tokens)
    for token in tokens:
        if token.lower() not in stopwords and token != '':  # stopping
            token.lower()  # case-folding
            stemmedtoken = porter.stem(token)  # porter-stemming
            if stemmedtoken.lower() not in stopwords:  # stopping
                if corpora=='OT':
                    if stemmedtoken.lower() not in OTtokendict:
                        OTtokendict[stemmedtoken.lower()]=1
                    else:
                        OTtokendict[stemmedtoken.lower()]+=1
                elif corpora=='NT':
                    if stemmedtoken.lower() not in NTtokendict:
                        NTtokendict[stemmedtoken.lower()] = 1
                    else:
                        NTtokendict[stemmedtoken.lower()] += 1
                elif corpora=='Quran':
                    if stemmedtoken.lower() not in Qurantokendict:
                        Qurantokendict[stemmedtoken.lower()] = 1
                    else:
                        Qurantokendict[stemmedtoken.lower()] += 1

for corpora in versedict:
    for verse in versedict.get(corpora):
        preprocessing(verse, corpora)
#sort tokens by highest document frequency
OTtokendict = dict(sorted(OTtokendict.items(), key=operator.itemgetter(1), reverse=True))
NTtokendict = dict(sorted(NTtokendict.items(), key=operator.itemgetter(1), reverse=True))
Qurantokendict = dict(sorted(Qurantokendict.items(), key=operator.itemgetter(1), reverse=True))

#remove tokens with frequency < 10
OTtokendict = {k:v for k,v in OTtokendict.items() if v>=10}
NTtokendict = {k:v for k,v in NTtokendict.items() if v>=10}
Qurantokendict = {k:v for k,v in Qurantokendict.items() if v>=10}

#mutual information
mutualinformationdict={}
mutualinformationdict['OT']={}
mutualinformationdict['NT']={}
mutualinformationdict['Quran']={}

#chi-squared
chisquareddict={}
chisquareddict['OT']={}
chisquareddict['NT']={}
chisquareddict['Quran']={}

def findmutualinformationandchisquared(tokendict, corpora):
    for token in tokendict:
        if corpora=='OT':
            if token not in mutualinformationdict.get('OT'):
                ntfreq=0
                quranfreq=0
                if token in NTtokendict:
                    ntfreq=NTtokendict.get(token)
                if token in Qurantokendict:
                    quranfreq = Qurantokendict.get(token)
                N11 = OTtokendict.get(token)
                N10 = ntfreq+quranfreq
                N01 = numberofOTverses-OTtokendict.get(token)
                N00 = numberofNTverses+numberofQuranverses-N10
        elif corpora == 'NT':
            if token not in mutualinformationdict.get('NT'):
                otfreq = 0
                quranfreq = 0
                if token in OTtokendict:
                    otfreq = OTtokendict.get(token)
                if token in Qurantokendict:
                    quranfreq = Qurantokendict.get(token)
                N11 = NTtokendict.get(token)
                N10 = otfreq + quranfreq
                N01 = numberofNTverses - NTtokendict.get(token)
                N00 = numberofOTverses + numberofQuranverses - N10
        elif corpora == 'Quran':
            if token not in mutualinformationdict.get('Quran'):
                ntfreq = 0
                otfreq = 0
                if token in NTtokendict:
                    ntfreq = NTtokendict.get(token)
                if token in OTtokendict:
                    otfreq = OTtokendict.get(token)
                N11 = Qurantokendict.get(token)
                N10 = ntfreq + otfreq
                N01 = numberofQuranverses - Qurantokendict.get(token)
                N00 = numberofNTverses + numberofOTverses - N10
        mutualinformationdict[corpora][token] = (((N11/totalnumberofverses)*math.log2((totalnumberofverses*N11)/((N10+N11)*(N01+N11))))
                                              +((N01/totalnumberofverses)*math.log2((totalnumberofverses*N01)/((N00+N01)*(N01+N11))))
                                              +((N00/totalnumberofverses)*math.log2((totalnumberofverses*N00)/((N00+N01)*(N00+N10)))))
        if N10!=0:
            mutualinformationdict[corpora][token]+=((N10/totalnumberofverses)*math.log2((totalnumberofverses*N10)/((N10+N11)*(N00+N10))))
        chisquareddict[corpora][token] = ((N11+N10+N01+N00)*(((N11*N00)-(N10*N01))**2))/((N11+N01)*(N11+N10)*(N10+N00)*(N01+N00))

findmutualinformationandchisquared(OTtokendict, 'OT')
findmutualinformationandchisquared(NTtokendict, 'NT')
findmutualinformationandchisquared(Qurantokendict, 'Quran')

mutualinformationdict['OT'] = dict(sorted( mutualinformationdict['OT'].items(), key=operator.itemgetter(1), reverse=True))
mutualinformationdict['NT'] = dict(sorted( mutualinformationdict['NT'].items(), key=operator.itemgetter(1), reverse=True))
mutualinformationdict['Quran'] = dict(sorted( mutualinformationdict['Quran'].items(), key=operator.itemgetter(1), reverse=True))

chisquareddict['OT'] = dict(sorted( chisquareddict['OT'].items(), key=operator.itemgetter(1), reverse=True))
chisquareddict['NT'] = dict(sorted( chisquareddict['NT'].items(), key=operator.itemgetter(1), reverse=True))
chisquareddict['Quran'] = dict(sorted( chisquareddict['Quran'].items(), key=operator.itemgetter(1), reverse=True))

print(mutualinformationdict)
print(chisquareddict)

#LDA
dictofpreprocessedverses={}
dictofpreprocessedverses['OT']=[]
dictofpreprocessedverses['NT']=[]
dictofpreprocessedverses['Quran']=[]

listofverses = []
for verse in versedict.get('OT'):
    listofverses.append(verse)
for verse in versedict.get('NT'):
    listofverses.append(verse)
for verse in versedict.get('Quran'):
    listofverses.append(verse)
listofpreprocessedverses=[]
for verse in listofverses:
    preprocessedverse=[]
    tokens = re.split(r"[ ,.:;></|+-@#!$%^&*_=\n\'\"\[\]\{\}\(\)]", verse)  # tokenisation
    for token in tokens:
        if token.lower() not in stopwords and token != '':  # stopping
            token.lower()  # case-folding
            stemmedtoken = porter.stem(token)  # porter-stemming
            preprocessedverse.append(stemmedtoken)
    listofpreprocessedverses.append(preprocessedverse)

id2word=Dictionary(listofpreprocessedverses)
texts = listofpreprocessedverses
# Term Document Frequency
ldacorpus = [id2word.doc2bow(text) for text in texts]

lda_model = gensim.models.LdaModel(corpus=ldacorpus,
                                   id2word=id2word,
                                   num_topics=20)

#printing the top topics identified by the ldamodel
pprint(lda_model.print_topics())

top_topics=lda_model.top_topics(ldacorpus)
averagescoreot={}
averagescorent = {}
averagescorequran = {}
for j in range(0, 20):
    averagescoreot[j] = 0
    averagescorent[j] = 0
    averagescorequran[j] = 0

for i in range(0,33494):
    doc_topic_score=lda_model.get_document_topics(ldacorpus[i])
    if i>=0 and i<=20765:
        #update ot avg score
        for list in doc_topic_score:
            averagescoreot[list[0]]=averagescoreot[list[0]]+list[1]
    elif i>=20766 and i<=27877:
        #update nt avg score
        for list in doc_topic_score:
            averagescorent[list[0]] = averagescorent[list[0]] + list[1]
    elif i>=27878 and i<=33493:
        #update quran avg score
        for list in doc_topic_score:
            averagescorequran[list[0]] = averagescorequran[list[0]] + list[1]

for j in range(0, 20):
    averagescoreot[j] =averagescoreot[j]/numberofOTverses
    averagescorent[j] = averagescorent[j]/numberofNTverses
    averagescorequran[j] = averagescorequran[j]/numberofQuranverses

averagescoreot=dict(sorted(averagescoreot.items(), key=operator.itemgetter(1), reverse=True))
averagescorent = dict(sorted(averagescorent.items(), key=operator.itemgetter(1), reverse=True))
averagescorequran = dict(sorted(averagescorequran.items(), key=operator.itemgetter(1), reverse=True))

i=0
for key in averagescoreot:
    print("#",i," Topic top terms of Old Testament. Topic No#",key)
    print(lda_model.get_topic_terms(key))
    i+=1
    if i==3:
        break
i=0
for key in averagescorent:
    print("#", i , " Topic top terms of New Testament. Topic No#",key)
    print(lda_model.get_topic_terms(key))
    i += 1
    if i == 3:
        break
i=0
for key in averagescorequran:
    print("#",i ," Topic top terms of Quran. Topic No#",key)
    print(lda_model.get_topic_terms(key))
    i+=1
    if i==3:
        break

#text classification
classificationfile = open('classification.csv','w')
classificationfilewriter = csv.writer(classificationfile, lineterminator='\n')
classificationfilewriter.writerow(['system','split','p-quran','r-quran','f-quran','p-ot','r-ot','f-ot','p-nt','r-nt','f-nt','p-macro','r-macro','f-macro'])

categories=[]
for num in range(0,20765):
    categories.append('OT')
for num in range(0, 7113):
    categories.append('NT')
for num in range(0,5616):
    categories.append('Quran')

def generateword2id(Inputverses):
    #preprocess training data and form the vocab
    vocab = set([])
    for preprocessedverse in Inputverses:
        for word in preprocessedverse:
            vocab.add(word)

    word2id = {}
    for index, word in enumerate(vocab):
        word2id[word] = index
    return word2id

def generatecat2id(categories):
    cat2id = {}
    for index, category in enumerate(set(categories)):
        cat2id[category] = index
    return cat2id

def generatebowmatrix(listofpreprocessedverses, word2id):
    #generate sparse matrix
    sparsematrixsize=(len(listofpreprocessedverses), len(word2id)+1)
    oovindex=len(word2id)
    TrainingMatrix = scipy.sparse.dok_matrix(sparsematrixsize)

    for verseid, verse in enumerate(listofpreprocessedverses):
        for word in verse:
            TrainingMatrix[verseid, word2id.get(word, oovindex)]+=1
    return TrainingMatrix

def createlistforbothmodels(listofverses, baselinemodel):
    listforbaselinemodel=[]
    for verse in listofverses:
        preprocessedverse=[]
        tokens = re.split(r"[ ,.:;></|+-@#!$%^&*_=\n\'\"\[\]\{\}\(\)]", verse)  # tokenisation
        for token in tokens:
            if token.lower() not in stopwords and token != '':  # stopping
                token.lower()  # case-folding
                if baselinemodel==False:
                    stemmedtoken = porter.stem(token)
                    preprocessedverse.append(stemmedtoken)
                else:
                    preprocessedverse.append(token)
        listforbaselinemodel.append(preprocessedverse)
    return listforbaselinemodel

def findincorrectpredictions(predictions, dev_y_train, baselinedevdata):
    incorrectcount = 0
    for index,predicted in enumerate(predictions):
        if predicted!=dev_y_train[index]:
            incorrectcount += 1
            if incorrectcount<4:
                print("Verse : ",baselinedevdata[index],"Predicted as: ",predicted," Actual is: ",dev_y_train[index])

#test data
testfile = open('test.tsv','r')
listoftestverses=[]
listoftestcategories=[]
for line in testfile.readlines():
    listoftestcategories.append(line.split('\t')[0].strip())
    listoftestverses.append(line.split('\t')[1].strip())
listoftestversesforbaselinemodel = createlistforbothmodels(listoftestverses, True)
listoftestversesforimprovedmodel = createlistforbothmodels(listoftestverses, False)

#baseline model
listforbaselinemodel = createlistforbothmodels(listofverses, True)
baselinetrainingdata, baselinedevdata, baselinetrainingcategories, baselinedevcategories = model_selection.train_test_split(listforbaselinemodel, categories, test_size=0.30)

word2id = generateword2id(baselinetrainingdata)
cat2id = generatecat2id(categories)
cat_names = set(categories)

baselinetrainingMatrix = generatebowmatrix(baselinetrainingdata, word2id)
y_train = [cat2id[category] for category in baselinetrainingcategories]
baselinemodel = sklearn.svm.LinearSVC(C=1000, max_iter=5000, dual=False)
baselinemodel.fit(baselinetrainingMatrix, y_train)
#prediction for training data
baselinetrainingprediction = baselinemodel.predict(baselinetrainingMatrix)
print("SVM accuracy score for baseline training data->",accuracy_score(baselinetrainingprediction, y_train)*100)
print("Classification report for baseline model training data:\n")
btcr = classification_report(y_train, baselinetrainingprediction, target_names=cat_names, output_dict=True)
classificationfilewriter.writerow(['baseline','train',btcr.get('Quran').get('precision'), btcr.get('Quran').get('recall'), btcr.get('Quran').get('f1-score'),
                                   btcr.get('OT').get('precision'), btcr.get('OT').get('recall'), btcr.get('OT').get('f1-score'),
                                   btcr.get('NT').get('precision'), btcr.get('NT').get('recall'), btcr.get('NT').get('f1-score'),
                                   btcr.get('macro avg').get('precision'), btcr.get('macro avg').get('recall'), btcr.get('macro avg').get('f1-score')])
baselinedevelopmentMatrix = generatebowmatrix(baselinedevdata, word2id)
devy_train = [cat2id[cat] for cat in baselinedevcategories]
#make a prediction
devprediction = baselinemodel.predict(baselinedevelopmentMatrix)

print("SVM accuracy score for baseline dev data->",accuracy_score(devprediction, devy_train)*100)
print("Classification report for baseline model dev data:\n")
bdcr = classification_report(devy_train, devprediction, target_names=cat_names, output_dict=True)
classificationfilewriter.writerow(['baseline','dev',bdcr.get('Quran').get('precision'), bdcr.get('Quran').get('recall'), bdcr.get('Quran').get('f1-score'),
                                   bdcr.get('OT').get('precision'), bdcr.get('OT').get('recall'), bdcr.get('OT').get('f1-score'),
                                   bdcr.get('NT').get('precision'), bdcr.get('NT').get('recall'), bdcr.get('NT').get('f1-score'),
                                   bdcr.get('macro avg').get('precision'), bdcr.get('macro avg').get('recall'), bdcr.get('macro avg').get('f1-score')])

TestMatrix = generatebowmatrix(listoftestversesforbaselinemodel, word2id)
test_y_train = [cat2id[cat] for cat in listoftestcategories]
#make a prediction
testprediction = baselinemodel.predict(TestMatrix)

print("SVM accuracy score for baseline test data->", accuracy_score(testprediction, test_y_train)*100)
print("Classification report for baseline model test data:\n")
btestcr = classification_report(test_y_train, testprediction, target_names=cat_names, output_dict=True)
classificationfilewriter.writerow(['baseline','test',btestcr.get('Quran').get('precision'), btestcr.get('Quran').get('recall'), btestcr.get('Quran').get('f1-score'),
                                   btestcr.get('OT').get('precision'), btestcr.get('OT').get('recall'), btestcr.get('OT').get('f1-score'),
                                   btestcr.get('NT').get('precision'), btestcr.get('NT').get('recall'), btestcr.get('NT').get('f1-score'),
                                   btestcr.get('macro avg').get('precision'), btestcr.get('macro avg').get('recall'), btestcr.get('macro avg').get('f1-score')])
#print the incorrect predictions
findincorrectpredictions(devprediction, devy_train, baselinedevdata)

#improved model
improvedtrainingdata, improveddevdata, improvedtrainingcategories, improveddevcategories = model_selection.train_test_split(listofpreprocessedverses, categories, test_size=0.15)

improvedword2id = generateword2id(improvedtrainingdata)
improvedtrainingMatrix = generatebowmatrix(improvedtrainingdata, improvedword2id)
improved_y_train = [cat2id[category] for category in improvedtrainingcategories]
improvedmodel = sklearn.svm.LinearSVC(C=1000, max_iter=5000, dual=False)
improvedmodel.fit(improvedtrainingMatrix, improved_y_train)

#prediction for training data
improvedtrainingprediction = improvedmodel.predict(improvedtrainingMatrix)
print("SVM accuracy score for improved training data->",accuracy_score(improvedtrainingprediction, improved_y_train)*100)
print("Classification report for improved model training data:\n")
itcr = classification_report(improved_y_train, improvedtrainingprediction, target_names=cat_names, output_dict=True)
classificationfilewriter.writerow(['improved','train',itcr.get('Quran').get('precision'), itcr.get('Quran').get('recall'), itcr.get('Quran').get('f1-score'),
                                   itcr.get('OT').get('precision'), itcr.get('OT').get('recall'), itcr.get('OT').get('f1-score'),
                                   itcr.get('NT').get('precision'), itcr.get('NT').get('recall'), itcr.get('NT').get('f1-score'),
                                   itcr.get('macro avg').get('precision'), itcr.get('macro avg').get('recall'), itcr.get('macro avg').get('f1-score')])

improveddevelopmentMatrix = generatebowmatrix(improveddevdata, improvedword2id)
improved_dev_y_train = [cat2id[cat] for cat in improveddevcategories]
#make a prediction
improveddevprediction = improvedmodel.predict(improveddevelopmentMatrix)

print("SVM accuracy score for improved dev data->",accuracy_score(improveddevprediction, improved_dev_y_train)*100)
print("Classification report for improved model dev data:\n")
idcr = classification_report(improved_dev_y_train, improveddevprediction, target_names=cat_names, output_dict=True)
classificationfilewriter.writerow(['improved','dev',idcr.get('Quran').get('precision'), idcr.get('Quran').get('recall'), idcr.get('Quran').get('f1-score'),
                                   idcr.get('OT').get('precision'), idcr.get('OT').get('recall'), idcr.get('OT').get('f1-score'),
                                   idcr.get('NT').get('precision'), idcr.get('NT').get('recall'), idcr.get('NT').get('f1-score'),
                                   idcr.get('macro avg').get('precision'), idcr.get('macro avg').get('recall'), idcr.get('macro avg').get('f1-score')])

#make a prediction
improvedtestmatrix = generatebowmatrix(listoftestversesforimprovedmodel, improvedword2id)
improvedtestprediction = improvedmodel.predict(improvedtestmatrix)

print("SVM accuracy score for improved test data->", accuracy_score(improvedtestprediction, test_y_train)*100)
print("Classification report for improved model test data:\n")
itestcr = classification_report(test_y_train, improvedtestprediction, target_names=cat_names, output_dict=True)
classificationfilewriter.writerow(['improved','test',itestcr.get('Quran').get('precision'), itestcr.get('Quran').get('recall'), itestcr.get('Quran').get('f1-score'),
                                   itestcr.get('OT').get('precision'), itestcr.get('OT').get('recall'), itestcr.get('OT').get('f1-score'),
                                   itestcr.get('NT').get('precision'), itestcr.get('NT').get('recall'), itestcr.get('NT').get('f1-score'),
                                   itestcr.get('macro avg').get('precision'), itestcr.get('macro avg').get('recall'), itestcr.get('macro avg').get('f1-score')])
