import pandas
import numpy
import nltk
from Master import base
from Static_Functions import split, clean_non_pronoun_stopwords, remove_special_characters, lemmatize
from Static_Functions import concat, tokenize, non_proper_to_lower, extract_names

######################################################
#################### Read Data #######################
######################################################

news = pandas.read_csv(base + '2 - Data/all_the_news50k.csv',
                       usecols=['date', 'year', 'month', 'day', 'title', 'article', 'section', 'url', 'publication'],
                       engine='c')

######################################################
#################### Clean Data ######################
######################################################

# Set the basic exclusion criteria for aricles we are going
# to allow into our sample

#Drop missing values in any columns we need
news = news.dropna(subset=['date', 'year', 'month', 'day', 'title', 'article', 'publication', 'section'])

#Set Variable classes
news[['date']] = news[['date']].apply(pandas.to_datetime)
news[['year', 'month', 'day']] = news[['year', 'month', 'day']].apply(pandas.to_numeric)
news[['title', 'article', 'section', 'publication']] = news[['title', 'article', 'section', 'publication']].astype(str)

#Remove consecutive spaces
news['title'] = news['title'].replace('\s+', ' ', regex=True)
news['article'] = news['article'].replace('\s+', ' ', regex=True)

#Only take articles that have between 100 and 2500 words and title with more then 3 words
news = news[(news['article'].str.count(' ') >= 100) & (news['article'].str.count(' ') <= 2500)]
news = news[news['title'].str.count(' ') >= 3]


######################################################
############ Identify Names / Gender / Race ##########
######################################################

news = news.iloc[0:10]

# Create a new column combing title and text
news['text'] = news['title'] + " " + news['article']

news['names'] = news['text'].apply(lambda x: extract_names(x))
print('names done')

######################################################
#################### Parse Text ######################
######################################################

# Most of the heavy lifting here is done in Static_Functions,
# where we have slightly modified versions of typical text cleaning
# where we conduct the cleaning a way that is specifically tailored to
# our research question. If you are curious about how each step is conducted,
# Static_Functions has more information.

# split the text column up
news['text'] = news['text'].apply(lambda x: split(x))
print('split done')

# remove punctuation, numbers, urls, and other special characters
news['text'] = news['text'].apply(lambda x: remove_special_characters(x))
print('sc done')

# Lemmatize words to their root word
news['text'] = news['text'].apply(lambda x: lemmatize(x))
print('lem done')

# remove all non-pronoun stopwords
news['text'] = news['text'].apply(lambda x: clean_non_pronoun_stopwords(x))
print('stop done')

# convert all non proper nouns to lowercases
news['text'] = news['text'].apply(lambda x: non_proper_to_lower(x))
print('lower done')

# Concatenating arrays of sentences back to a single array of the full text
news['text'] = news['text'].apply(lambda x: concat(x))
print('concat done')


######################################################
################## Vectorize Text ####################
######################################################

# Now we need to represent the parsed text as a vectors
# so we can start to perform some mathematical operations on
# them

# Tokenizing - represent each word with a unique number
news['tokenized'] = tokenize(news['text'])
print('token done')

news.to_csv(base + '2 - Data/50kCleaned.csv', index=False, line_terminator='\n')