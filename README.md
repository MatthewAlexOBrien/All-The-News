# ****Project Backround****

The intention for this project is to examine within-article differences in slant towards towards individuals with different ethnic and gender backrounds for major  news puplications between 2000 and 2020. The dataset is a combination of the Components dataset found here: https://components.one/datasets/all-the-news-2-news-articles-dataset/ with other major news publications scrapped independently. The full list of publications includes: New York Times, Breitbart, CNN, Business Insider, the Atlantic, Fox News, Talking Points Memo, Buzzfeed News, National Review, New York Post, the Guardian, NPR, Reuters, Vox, and the Washington Post.



# ****Identifying Named Subjects****

See: ***subjects.ipnyb*** 

We start by extracting the 'subject' of each sentence for all articles using Stanford NLP Group's dependency parser pipeline, availble through their python NLP package 'Stanza'. https://stanfordnlp.github.io/stanza/depparse.html. Next, we use Stanza's Named Entity Regognition pipline to determine which subjects are named entities (people) https://stanfordnlp.github.io/stanza/ner.html. Sentences for which the nominal subject is not named-entity are removed from the dataset. We repeat this exercise for article headlines.

# ****Identifying Race and Gender****

See: ***ethnicity.ipnyb*** 

Race of named subjects is identified with two methods. 
* i. DEEP FACE We use google images and DeepFace https://pypi.org/project/deepface/ to measure ethnicity and gender, which has an overall accuracy of 97% for predicting gender and 72% for predicting race. We obtain images of named subjects by 'googling' them and dowloading the first 3 images. 

* ii. ETHNICOLR We use the ethnicolr package https://ethnicolr.readthedocs.io/ethnicolr.html to predict race based on the letter sequences in the subjects' name. The authors of this package have a corresponding paper https://arxiv.org/pdf/1805.02109.pdf which outlines their procedure for predicting race. They use voter registration data in the US to train deep neural models that identify letter sequences that most correspond to specific ethnic origins.  The measure privides 85% accuracy when both the first and last name are identified. 
 
For robustness, we exclude subjects whos predicted ethnicity is inconsistent across the two methods.


# ****Sentiment Anlysis****

See: ***sentiment.ipnyb*** 

We use three well established measures of sentiment. All measures of sentiment are standardized for comparabity.
*   i. VADER (Valence Aware Dictionary and sEntiment Reasoner https://ojs.aaai.org/index.php/ICWSM/article/view/14550)
*   ii. AFINN (AFINN lexicon https://arxiv.org/abs/1103.2903). 
*   iii. STANZA We also include the sentiment alayzer from the STANZA pipeline (https://stanfordnlp.github.io/stanza/sentiment.html)

