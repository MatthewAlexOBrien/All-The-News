# ****Project Backround****

The intention for this project is to examine within-article differences in slant towards towards individuals with different ethnic and gender backrounds, for major US news puplications between 2000 and 2020. The dataset is a combination of the Components dataset found here: https://components.one/datasets/all-the-news-2-news-articles-dataset/ with other major news publications scrapped independently. The full list of publications includes: New York Times, Breitbart, CNN, Business Insider, the Atlantic, Fox News, Talking Points Memo, Buzzfeed News, National Review, New York Post, the Guardian, NPR, Reuters, Vox, and the Washington Post.



# ****Part 1: Data Pre-Processing****
The exercise of data pre-processing is describled below. (1A) We start by identifying sentences and headlines for which the nominal subject is a named entity. (1B) Next, we predict race and gender of each named entity. (1C) Finally, we perform sentiment analysis on headlines and sentences. The exact details for each step are shown in their corresponding sections below. For your conveinience, the pre-processed dataset can be cloned directly into this notbook in Part 1B.

***1A. Identifying Named Nominal Subjects***

We start by examining the nominal subject of each sentence within in each article, for which we use the Stanford NLP Group's dependency parser pipeline, availble through their python NLP package 'Stanza'. https://stanfordnlp.github.io/stanza/depparse.html. Next, we use Stanza's Named Entity Regognition pipline to determine which nominal subjects are named entities (people) https://stanfordnlp.github.io/stanza/ner.html. Sentences for which the nominal subject is not named-entity are removed from the dataset. We repeat this exercise for article headlines.

***1B. Identifying Race and Gender of Nominal Subjects***

Race of all named nominal subjects is identified with two methods. First, we use the ethnicolr package https://ethnicolr.readthedocs.io/ethnicolr.html to predict race based on the letter sequences in the subjects' name. The authors of this package have a corresponding paper here https://arxiv.org/pdf/1805.02109.pdf which outlines their procedure for predicting race. Essentially, they use voter registration data in the US to train deep neural models that identify letter sequences that most correspond to specific ethnic origins.  The measure is somewhat noisy, with ~ 85% accuracy when both the first and last name are identified.  Second, we use google images and DeepFace https://pypi.org/project/deepface/ to measure ethnicity, which has an overall accuracy of 97% for predicting race and gender. We obtain images of named subjects by effectively 'googling' them and dowloading the first 2 images. For robustness, we exclude subjects whos predicted ethnicity is inconsistent across the two methods.

***1C. Performing Sentiment Analysis***

Sentiment Analysis is well estblished in the world of NLP. We start by looking at two of the most common measures of sentiment, VADER (Valence Aware Dictionary and sEntiment Reasoner https://ojs.aaai.org/index.php/ICWSM/article/view/14550) and AFINN (AFINN lexicon https://arxiv.org/abs/1103.2903). We also include the sentiment alayzer from the STANZA pipeline (https://stanfordnlp.github.io/stanza/sentiment.html). All measures of sentiment are standardized for comparability. 
