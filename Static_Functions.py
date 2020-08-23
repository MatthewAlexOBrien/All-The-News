import nltk
import re
import  pandas
import numpy
from tensorflow.keras.preprocessing.text import Tokenizer
from sner import Ner
from nltk.tokenize.treebank import TreebankWordDetokenizer



################ Extracting Names ####################

# This method takes a text and extracts the names within in.
# It keys off of PERSON tags and if we see multiple PERSON
# tags in a row, then it knows that the words are part of the same name.
# We will have to run this before cleaning the text because removing
# stopwords and punctuation could result in different names with
# PERSON tags being adjacent to each other, and thus appended
# as the same name.

def extract_names(text):
    st = Ner(host='localhost', port=9199)
    text = st.get_entities(text)
    current_chunk = []
    continuous_chunk = []
    for word, tag in text:
        if tag =="PERSON":
            current_chunk.append(word)
        else:
            if current_chunk:
                current_chunk = ' '.join(current_chunk)
                continuous_chunk.append(current_chunk)
                current_chunk = []
            else:
                pass

    if current_chunk:
        current_chunk = ' '.join(current_chunk)
        continuous_chunk.append(current_chunk)

    return continuous_chunk


############## Split Text #################

# split the text into individual tuples of sentences, where
# each sentence tuple is an array of split up words


def split(text):
    sentences = nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    return sentences


######## Remove Punctuation and Numbers #######

# numbers, punctuation and other special character are too
# ambiguous to be valuable in sentiment analysis, Here we are going
# to also remove urls and html tags. Somewhat annoyingly, the split function
# above splits things on : as well, which generally we want to have
# however, this means that urls get split up on : before they go
# through this method, so essentially I have 2 seperate regex lines
# that deal with the different halves of the url.


def remove_special_characters(text):
    txt = []
    for sentences in text:
        res = []
        for word in sentences:
            #first remove urls and html tags
            newtext = re.sub(r'http\S+', '', word)
            newtext = re.sub(r'//\S+', '', newtext)
            newtext = re.sub('<.*?>', '', newtext)

            #Next remove puntuation, numbers, and anything thats not a letter
            newtext = re.sub('[^A-Za-z]+', '', newtext)

            if newtext != '':
                res.append(newtext)

        txt.append(res)
    return txt


######## Lemmitize words into roots #######

# here the idea is to map all words to their respective root words.
# we don't necessarily care about the part of speech the word is used
# as, but rather the sentiment of the root word itself.

def lemmatize(text):
    lemma = nltk.WordNetLemmatizer()
    txt = []
    for sentences in text:
        res = []
        sentences = nltk.pos_tag(sentences)
        for tuple in sentences:
            tag = tuple[1][0].lower()
            word = tuple[0]
            if tag in ['n', 'v', 'r', 'j']:
                if tag == 'j':
                    tag = 'a'
                word = lemma.lemmatize(word, pos=tag)
                res.append(word)
            else:
                word = lemma.lemmatize(word)
                res.append(word)
        txt.append(res)
    return txt


############ Remove Stopwords ##############

# typically we always want to remove stop words before conducting
# any analysis. Since pronouns are going to be be key indicators for
# identifying gender, Im going to keep them. Potentially we could remove
# pronouns in the first portion where we try to identify specific articles
# of interest - this method is easy enough to adjust accordingly.


def clean_non_pronoun_stopwords(text):
    stopwords = set(nltk.corpus.stopwords.words("english"))
    txt = []
    for sentences in text:
        res = []
        for word in sentences:
            exclude_words = set(["her", "him", "hers", "his", "himself", "herself", "he's"])
            if word not in stopwords.difference(exclude_words):
                res.append(word)
        txt.append(res)
    return txt



############## Make lowercase #################

# Now, we want to make all word lowercase unless they are a
# proper noun, because we are going to need to identify names
# further down the road. For this portion, we are going to use
# the Stanford NER (Named Entry Recognition) tagger to and only
# make words lowercase if they they have an O tag (meaning that
# that they are of type 'other' and aren't a proper noun. For this step, you
# will need to download the Stanford NLP core library here to your
# computer https://stanfordnlp.github.io/CoreNLP/download.html
# and set the environment variables CLASSPATH and STANFORD_MODELS
# to the downloaded locations of stanford-ner.jar and english.all.3class.distsim.crf.ser.gz
# on your computer, respectivley.

# This method is also VERY slow if you dont connect to a Java virtual
# machine. Before running code we should do the following in our terminal to
# initiate a java virtual machine.

# 1) Set directory to stanford ner folder 'cd /Users/matthewobrien/stanford-ner-2015-04-20/'
# 2) Run the ner server 'java -classpath stanford-ner.jar edu.stanford.nlp.ie.NERServer -port 9199 -loadClassifier ./classifiers/english.all.3class.distsim.crf.ser.gz'

def non_proper_to_lower(text):
    st = Ner(host='localhost',port=9199)
    tree = TreebankWordDetokenizer()
    txt = []
    for sentences in text:
        res = []
        sentences = tree.detokenize(sentences)
        sentences = st.get_entities(sentences)
        for word in sentences:
            if word[1] == "O":
                word = word[0].lower()
            else:
                word = word[0]
            res.append(word)
        txt.append(res)
    return txt



################ Concatenate Back ####################

# I thought it was important to conduct the lemminzation
# in the context of each sentence, as opposed to the context
# of the entire article. But to get the text in the format we need
# to tokenize, we need to bring it back to a single array of words
# for each article, instead of an array of an array of sentences.

def concat(text):
    txt = []
    for sentences in text:
        for word in sentences:
            txt.append(word)
    return txt



################ Tokenizing Text ####################

# This is the process of representing each word as a number.
# Thus, instead of having a tuple of words for each article,
# we will have a tuple of numbers. Eventually, we will use
# these tuples as the input layer to a more sophisticated
# neural network which uses the word2vec vectors as the weight matrix

def tokenize(oldcolumn):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(oldcolumn)
    newcolumn = tokenizer.texts_to_sequences(oldcolumn)
    return newcolumn


