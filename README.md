# ****Project Backround****

The intention for this project is to examine within-article differences in slant towards towards individuals with different ethnic backrounds, for major US news puplications between 2000 and 2020. The dataset is a combination of the Components dataset found here: https://components.one/datasets/all-the-news-2-news-articles-dataset/ with other major news publications scrapped independently. The full list of publications includes: New York Times, Breitbart, CNN, Business Insider, the Atlantic, Fox News, Talking Points Memo, Buzzfeed News, National Review, New York Post, the Guardian, NPR, Reuters, Vox, and the Washington Post.



# ****Part 1: Data Pre-Processing****
The exercise of data pre-processing is describled below. (1A) We start by identifying sentences and headlines for which the nominal subject is a named entity. (1B) Next, we predict race and gender of each named entity. (1C) Finally, we perform sentiment analysis on headlines and sentences. The exact details for each step are shown in their corresponding sections below. For your conveinience, the pre-processed dataset can be cloned directly into this notbook in Part 1B.
