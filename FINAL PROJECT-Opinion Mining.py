import nltk
import xlrd
import pandas as pd
import numpy as np
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.corpus import sentiwordnet
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from operator import itemgetter
import re
from pandas_ml import ConfusionMatrix
#import pyfim
from sys import argv, stderr, maxsize
from math import ceil
from time import time
# from fim import eclat, fim, apriori, fpgrowth
import sys
from itertools import chain, combinations
import ast

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"at&t", "att", phrase)
    phrase = re.sub(r"b&h", "bh", phrase)
    phrase = re.sub(r"AT&T", "att", phrase)
    phrase = re.sub(r"B&H", "bh", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

 
#=======================================================================
#=======================================================================
#=======================================================================
#=======================================================================
#=======================================================================
#=======================================================================
#=======================================================================
#=======================================================================
#=======PREPROCESSING===================================================
temp = list()
stop_words = set(stopwords.words('english')) 
book = xlrd.open_workbook("EXPERT REVIEWS LG.xlsx")
sh = book.sheet_by_index(0)
ps = PorterStemmer()
lm = WordNetLemmatizer()
sw = pd.read_csv('stopword.txt')
stopword = sw.iloc[:, 0].values


for i in range(sh.nrows):
    if (i > 0):
        temp.append(decontracted(sh.cell_value(i,2)))
         

allDataSet = ''.join(temp)
 
datas = list()
for i in range(len(temp)): 
    datas.append(nltk.word_tokenize(temp[i].lower()))
    
datafixed = list()
for j in range(len(temp)):
    prefix = list()
    for k in range(len(datas[j])):
        prefix.append(decontracted(datas[j][k]))
    datafixed.append(prefix)
appendFile = open('datafixed.csv','w', encoding = 'utf-8')
for l in range(0,len(datafixed)):
    appendFile.write(str(datafixed[l])+","+"\n") 
appendFile.close()

tagged_token = list()
for i in range(0,len(datafixed)):
    tagged_token.append(nltk.pos_tag(datafixed[i]))
tagged_easyk = list()
for k in range(0,len(tagged_token)):
    tagged_e = list()
    for l in range(0,len(tagged_token[k])):
        tagged_e.append(list(tagged_token[k][l]))
    tagged_easyk.append(tagged_e)
appendFile = open('taggedtoken.txt','w' ,encoding = 'utf-8')
for j in range(0,len(tagged_easyk)):
    appendFile.write(str(tagged_easyk[j])+","+"\n") 
appendFile.close()
   
pre_sw = list()
for j in range(len(temp)):
    filteredtext = list()
    for k in range(len(datafixed[j])):
        if not datafixed[j][k] in stopword:
            filteredtext.append(datafixed[j][k])
    pre_sw.append(filteredtext)
    
filteredwords = list()        
for i in range(0,len(pre_sw)):
    filteredalm = list()
    for j in range(len(pre_sw[i])):
        if pre_sw[i][j] != ',':
            filteredalm.append(pre_sw[i][j])
    filteredwords.append(filteredalm)
appendFile = open('filteredwords.csv','w' ,encoding = 'utf-8')
for k in range(0,len(filteredwords)):
    appendFile.write(str(filteredwords[k])+","+"\n") 
appendFile.close()       
    
lemmatizedwords = list()
for i in range(0,len(filteredwords)):
    prelem = list()
    for j in range (len(filteredwords[i])):
        prelem.append(lm.lemmatize(filteredwords[i][j]))
    lemmatizedwords.append(prelem)
appendFile = open('lemmatizedwords.csv','w' ,encoding = 'utf-8')
for k in range(0,len(lemmatizedwords)):
    appendFile.write(str(lemmatizedwords[k])+","+"\n") 
appendFile.close()
    
tagged = list()    
for i in range(0,len(lemmatizedwords)):
    tagged.append(nltk.pos_tag(lemmatizedwords[i]))
appendFile = open('tagged.txt','w' ,encoding = 'utf-8')
for j in range(0,len(tagged)):
    appendFile.write(str(tagged[j])+","+"\n") 
appendFile.close()

tagged_clean = list()
for i in range (0,len(tagged)):
    tagged_clean.append([x for (x,y) in tagged[i] if y in ('NN')])
appendFile = open('tagged_clean.csv','w' ,encoding = 'utf-8')
for j in range(0,len(tagged_clean)):
    appendFile.write(str(tagged_clean[j])+","+"\n") 
appendFile.close()

tid = list()#VER2----------------- enter a transcation list. this step provide an input for association mining
for i in range (0,len(tagged_clean)):
    tid.append(",".join(tagged_clean[i]))
appendFile = open('transactionlist2.txt','w', encoding ='utf-8')
for k in range(0,len(tid)):
    appendFile.write(str(tid[k])+","+"\n") 
appendFile.close() 
    
#=======================================================================
#=======================================================================
#=======================================================================
#=======================================================================
#=======================================================================
#=======================================================================
#=======================================================================
#=======================================================================
#=======FEATURE EXTRACTION==============================================
kot = 0
FreqItems = dict()
support = dict()

def eclat(prefix, items, dict_id):
    global isupp
    while items:
        i,itids = items.pop()
        isupp = len(itids)
        if isupp >= minsup:
            FreqItems[frozenset(prefix + [i])] = isupp
            suffix = []
            for j, ojtids in items:
                jtids = itids & ojtids
                if len(jtids) >= minsup:
                    suffix.append((j,jtids))
            dict_id += 1
            eclat(prefix+[i], sorted(suffix, key=lambda item: len(item[1]), reverse=True), dict_id)

def rules(FreqItems, confidence):
    Rules = []
    cnt = 0

    for items, support in FreqItems.items():
        # print(items)
        # print(support)
        if(len(items) > 1):
            lst = list(items)
            lst.sort()
            antecedent = lst[:len(lst) - 1]
            consequent = lst[-1:]
            
            conf = float(FreqItems[frozenset(items)]/FreqItems[frozenset(antecedent)]*100)
            if (conf >= confidence):
                cnt += 1
                lift = float(conf/FreqItems[frozenset(consequent)])
                if lift >= 1:
                    Rules.append((antecedent, consequent, support, conf, lift))

    print('Found %d Rules ' % (cnt))
    return Rules

def getantecendent(FreqItems, confidence):
    ant = []
    cnt = 0

    for items, support in FreqItems.items():
        if(len(items) > 1):
            lst = list(items)
            lst.sort()
            antecedent = lst[:len(lst) - 1]
            consequent = lst[-1:]
            
            conf = float(FreqItems[frozenset(items)]/FreqItems[frozenset(antecedent)]*100)
            if (conf >= confidence):
                cnt += 1
                lift = float(conf/FreqItems[frozenset(consequent)])
                if lift >= 1:
                    ant.append((antecedent))

    print('Print %d attributes' % (cnt))
    return ant

def print_Frequent_Itemsets(output_FreqItems, FreqItems):
    file = open(output_FreqItems, 'w+')
    for item, support in FreqItems.items():
        file.write(" {} : {} \n".format(list(item), round(support,4)))

def print_Rules(output_Rules, Rules):
    file = open(output_Rules, 'w+')
    for a, b,supp, conf, lift in sorted(Rules):
        file.write("{} ==> {} support: {} confidence: {} \n".format((a), (b), round(supp, 4),round(conf, 4),round(lift, 4)))
    file.close()
    
def print_Antecendent(ant):
    file = open('output_antecendent.csv', 'w+')
    for a in sorted(ant):
        file.write("[] \n".format((a)))
    file.close()
    
def Read_Data(filename, delimiter=','):
    data = {}
    trans = 0
    f = open('transactionlist2.txt', 'r', encoding = "utf-8")
    for row in f:
        trans += 1
        for item in row.split(delimiter):
            if item not in data:
                data[item] = set()
            data[item].add(trans)
    f.close()
    return data

if __name__ == "__main__":
    minsup   = 8
    confidence = 75
    output_FreqItems = 'output_freqitems.csv'
    output_Rules = 'output_rule.csv'
    dict_id = 0
    dataeclat = Read_Data('transactionlist2.txt', ',') #change the delimiter based on your input file
    dataeclat.pop("\n",None)
    dataeclat.pop("",None)

    eclat([], sorted(dataeclat.items(), key=lambda item: len(item[1]), reverse=True), dict_id)
    Rules = rules(FreqItems, confidence)
    Antecendent = getantecendent(FreqItems,confidence)
    print('found %d Frequent items'%len(FreqItems))
    print('Writing polarity .....')
    print_Frequent_Itemsets(output_FreqItems, FreqItems)
    print_Rules(output_Rules, Rules)
    print_Antecendent(Antecendent)
    
    Ant1d = np.hstack(Antecendent)
    
    combined = np.array(Ant1d)
    key, val = np.unique(combined, return_counts=True)
    dict(zip(key, val))
    countedval = np.stack((key, val), axis=1)
    
    df = pd.DataFrame(countedval, columns=['feature','count'])
    df["count"] = pd.to_numeric(df["count"])
    sortfeatures = df.sort_values(["count"], axis=0, ascending=[False]) 
    features = sortfeatures.drop(sortfeatures[sortfeatures['count']<2].index)
    
    listfeatures = list(features.iloc[:, 0].values)
    appendFile = open('feature.txt','w')
    appendFile.write(str(listfeatures)+","+"\n") 
    appendFile.close()
    

#=======================================================================
#=======================================================================
#=======================================================================
#=======================================================================
#=======================================================================
#=======================================================================
#=======================================================================
#=======================================================================
#=======OPINION WORDS IDENTIFY==========================================
    

def identifyOpinionWords(a, b, outputAspectOpinionListStr,printResult):       
    inputReviewList = a
    inputAspectList = b
    outputAspectOpinionList=open (outputAspectOpinionListStr,"w")
    # inputReviewsTuples=ast.literal_eval(inputReviewList)
    # inputAspectTuples=ast.literal_eval(inputAspectList)
    # global feat, tag, outputAspectOpinionTuples, negativeWordSet, orientationCache, value, testtemp, wordbefore, tagbefore, wordafter, tagafter, word
    global outputAspectOpinionTuples, predfeatcount, sysfeatcount, listvalue, aspect
    outputAspectOpinionTuples={}
    sysfeatcount={}
    orientationCache={}
    negativeWordSet = {"don't","never", "nothing", "nowhere", "noone", "none", "not",
                  "hasn't","hadn't","can't","couldn't","shouldn't","won't",
                  "wouldn't","don't","doesn't","didn't","isn't","aren't","ain't"}
    for aspect in inputAspectList:
        aspectTokens= nltk.word_tokenize(aspect)
        count=0
        for listvalue in inputReviewList:
            # print(value)
            condition=int(True)
            isNegativeSen=int(False)
            for subWord in aspectTokens:
                if(subWord in str(listvalue)):
                    condition = condition and int(True)
                else: 
                    condition = condition and int(False)
            if(condition):
                for negWord in negativeWordSet:
                    if(not isNegativeSen):#once senetence is negative no need to check this condition again and again
                        if negWord in str(listvalue):
                            isNegativeSen=isNegativeSen or int(True)
                outputAspectOpinionTuples.setdefault(aspect,[0,0,0])
                sysfeatcount.setdefault(aspect,[])
                for i in range(0, len(listvalue)-1):
                    word = listvalue[i][0]
                    tag = listvalue[i][1]
                    if(word in aspectTokens and (tag == 'NN' or tag == "NNS")):
                        count += 1
                        wordbefore = listvalue[i-1][0]
                        wordbefore2 = listvalue[i-2][0]
                        tagbefore = listvalue[i-1][1]
                        tagbefore2 = listvalue[i-2][1]
                        wordafter = listvalue[i+1][0]
                        tagafter = listvalue[i+1][1]
                        if(tagbefore == 'JJ' or tagbefore =='JJS' or tagbefore == 'VBN' or tagbefore =='VBD' or tagbefore =='RB' or tagbefore == 'RBR' or tagbefore == 'RBS' or tagbefore == 'VBG' or tagbefore == 'IN' or
                           tagbefore2 == 'JJ' or tagbefore2 =='JJS' or tagbefore2 == 'VBN' or tagbefore2 =='VBD' or tagbefore2 =='RB' or tagbefore2 == 'RBR' or tagbefore2 == 'RBS' or tagbefore2 == 'VBG' or tagbefore2 == 'IN' or
                           tagafter == 'JJ' or tagafter =='JJS' or tagafter == 'VBN' or tagafter =='VBD' or tagafter =='RB' or tagafter == 'RBR' or tagafter == 'RBS' or tagafter == 'VBG'):
                                orien = orientation(wordbefore)
                                orien2 = orientation(wordbefore2)
                                orien3 = orientation(wordafter)
                                if(isNegativeSen and orien != 2):
                                      orien= int(not orien)
                                if(isNegativeSen and orien2 != 2):
                                      orien2= int(not orien2)
                                if(isNegativeSen and orien3 != 2):
                                      orien3= int(not orien3)
                                if(orien==1 or orien2 == 1 or orien3 == 1):
                                      outputAspectOpinionTuples[aspect][1]+=1
                                      sysfeatcount[aspect].append(1)
                                elif(orien == 0 or orien2 == 0  or orien3 == 0):
                                      outputAspectOpinionTuples[aspect][0]+=1
                                      sysfeatcount[aspect].append(0)
                                elif(orien == 2 or orien2 == 2  or orien3 == 2):
                                      outputAspectOpinionTuples[aspect][2]+=1 
                                      sysfeatcount[aspect].append(2)
                        else:
                                outputAspectOpinionTuples[aspect][2]+=1 
                                sysfeatcount[aspect].append(2)
        if(count>0):
            #print(aspect,' ', outputAspectOpinionTuples[aspect][0], ' ',outputAspectOpinionTuples[aspect][1], ' ',outputAspectOpinionTuples[aspect][2])
            outputAspectOpinionTuples[aspect][1]=round((outputAspectOpinionTuples[aspect][1])/count,2)
            outputAspectOpinionTuples[aspect][0]=round((outputAspectOpinionTuples[aspect][0])/count,2)
            outputAspectOpinionTuples[aspect][2]=round((outputAspectOpinionTuples[aspect][2])/count,2)
            print(aspect,':\t\tPos => ', outputAspectOpinionTuples[aspect][1], '\tNeg => ',outputAspectOpinionTuples[aspect][0], '\tNeut => ',outputAspectOpinionTuples[aspect][2])
    if(printResult):
        print(outputAspectOpinionList)
    outputAspectOpinionList.write(str(outputAspectOpinionTuples))
    outputAspectOpinionList.close();
    
#-----------------------------------------------------------------------------------
def orientation(inputWord): 
    wordSynset=wordnet.synsets(inputWord)
    if(len(wordSynset) != 0): 
        word=wordSynset[0].name()
        orientation=sentiwordnet.senti_synset(word)
        if(orientation.pos_score()>orientation.neg_score()):
            return int(True)
        elif(orientation.pos_score()<orientation.neg_score()):
            return int(False)
        else:
            return 2
           

inputReviewListStr = tagged
inputAspectListStr = listfeatures
outputAspectOpinionListStr = 'opinion.txt'
printResult = True

identifyOpinionWords(inputReviewListStr, inputAspectListStr, outputAspectOpinionListStr,printResult)
#=======================================================================
#=======================================================================
#=======================================================================
#=======================================================================
#=======================================================================
#=======================================================================
#=======================================================================
#=======================================================================
#=======EVALUATION======================================================
dataexpert = pd.read_csv('re-EXPERT REVIEWS LG.csv', sep=";")

featexp = []
for list_feature in dataexpert['Expected Feature']:
    data_split = list_feature.split(',')
    featexp.append(data_split)
    
polexp = []
for list_polarity in dataexpert['Polarity']:
    data_split = list_polarity.split(',')
    polexp.append(data_split)
    
feature_only = []
for feature in featexp:
    for text in feature:
        if text not in feature_only:
            feature_only.append(text)

feature_only.remove("-")

feature_score = []
for feature in feature_only:
    feature_polarity = []
    for idx_i in range(len(featexp)):
        scores = []
        for idx_j in range(len(featexp[idx_i])):
            if feature == featexp[idx_i][idx_j]:
                score = polexp[idx_i][idx_j]
                scores.append(int(score))
        if len(scores) == 0:
            scores.append(2)
        feature_polarity.append(scores)
    feature_score.append(feature_polarity)
for i in range(0,len(feature_score)):
    for j in range(0,len(feature_score[i])):
        count0 = 0
        count1 = 0
        count2 = 0
        for k in range(0,len(feature_score[i][j])):
            if feature_score[i][j][k] == 0:
                count0+=1
            elif feature_score[i][j][k] == 1:
                count1+=1
            elif feature_score[i][j][k] == 2:
                count2+=1
        feature_score[i][j].clear()
        if (count0>=count1 and count0>=count2):
            feature_score[i][j].append(0)
        elif (count1>=count0 and count1>=count2):
            feature_score[i][j].append(1)
        elif (count2>=count0 and count2>=count1):
            feature_score[i][j].append(2)

feature_scoreit = list()
for i in range(0, len(feature_score)):
    feature_scoreit.append(list(chain.from_iterable(feature_score[i])))

for i in range(0, len(feature_scoreit)):
    count0 = 0
    count1 = 0
    count2 = 0
    for k in range(0,len(feature_scoreit[i])):
        if feature_scoreit[i][k] == 0:
            count0+=1
        elif feature_scoreit[i][k] == 1:
            count1+=1
        elif feature_scoreit[i][k] == 2:
            count2+=1
    feature_scoreit[i].clear()
    if (count0>=count1 and count0>=count2):
        feature_scoreit[i].append(0)
    elif (count1>=count0 and count1>=count2):
        feature_scoreit[i].append(1)
    elif (count2>=count0 and count2>=count1):
        feature_scoreit[i].append(2)
        
featuresexp = dict(zip(feature_only,feature_scoreit))   

for i in sysfeatcount:
    count3 = 0
    count4 = 0
    count5 = 0
    for k in range(0, len(sysfeatcount[i])):
        if sysfeatcount[i][k] == 0:
            count3+=1
        elif sysfeatcount[i][k] == 1:
            count4+=1
        elif sysfeatcount[i][k] == 2:
            count5+=1
    sysfeatcount[i].clear()
    if (count3>=count4 and count3>=count5):
        sysfeatcount[i].append(0)
    elif (count4>=count3 and count4>=count5):
        sysfeatcount[i].append(1)
    elif (count5>=count3 and count5>=count4):
        sysfeatcount[i].append(2)

featurespred ={}
for key in sysfeatcount:
    if key in featuresexp:
        featurespred[key] = sysfeatcount[key]

featurestest = {}
for key in featuresexp:
    if key in featurespred:
        featurestest[key] = featuresexp[key]        
    
poltest = list()
poltest.append(list(chain.from_iterable(featurestest.values())))

polpred = list()
polpred.append(list(chain.from_iterable(featurespred.values())))

tp = set(listfeatures) & set(feature_only)
fp = [i for i in listfeatures if i not in feature_only and i not in tp] 
fn = [i for i in feature_only if i not in listfeatures and i not in tp] 
tn = fp + fn

ctp = len(tp)
cfp = len(fp)
ctn = len(tn)
cfn = len(fn)

acc1 = (ctp+ctn)/(ctp+ctn+cfp+cfn)
p1 = ctp/(ctp+cfp)
r1 = ctp/(ctp+cfn)
f1score = 2*ctp/((2*ctp)+cfp+cfn) 
print("accuracy = ",acc1)
print("precission = ",p1)
print("recall = ",r1)
print("f1-score = ",f1score)
print("")

count = 0
for i in range (0,len(polpred)):
    if (polpred[i] == poltest[i]):
        count+=1
if (count == len(polpred)) :
    print("accuracy = 1")
    print("precission = 1")
    print("recall = 1")
    print("f1-score = 1")
else:
    confmatrix = ConfusionMatrix(polpred[0], poltest[0])
    confmatrix.print_stats()
# for key in featuresexp:
#     if key in sysfeatcount:
#         for i in range(len(featuresexp[key])-len(sysfeatcount[key])):
#             sysfeatcount[key].append(2)
#             i+=0
            
# for i in listfeatures:
#     if(len(sysfeatcount[i]) > len(featuresexp[i])):
#         for j in range(0,abs(len(sysfeatcount[i])-len(featuresexp[i]))):
#             featuresexp[i].append(2)
#             j+=0
#     elif(len(sysfeatcount[i]) < len(featuresexp[i])):
#         for k in range(0,abs(len(sysfeatcount[i])-len(featuresexp[i]))):
#             sysfeatcount[i].append(2)
#             k+=0
#     print("")
#     print("EVALUATION of features " + i +":")
#     confmatrix = ConfusionMatrix(sysfeatcount[i], featuresexp[i])
#     # print(confmatrix)
#     confmatrix.print_stats()
#     print("")
#     print("")

# from sklearn.metrics import accuracy_score
# y_pred = [0, 0, 0, 0]
# y_true = [0, 0, 0, 0]
# accuracy_score(y_true, y_pred)
# accuracy_score(y_true, y_pred, normalize=False)
# confmatrix = ConfusionMatrix(y_true, y_pred)
# print(confmatrix)

# confmatrix.print_stats()