import re
import time
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
from spellchecker import SpellChecker

# Import data
data = pd.read_csv('../../data/twitter data/Labelled/raw/labelled_twitter_data.csv',encoding='latin-1')
#data = data.sample(frac=1).reset_index(drop=True)
data = data.head(20)

# import emoticon dictionary
emoticon_dictionary = pd.read_csv('../../data/tools/emoticon_dictionary.csv')

# import acronym dictionary
acronym_dictionary = pd.read_csv('../../data/tools/acronym_dictionary.csv')

# create variable of stopwords
stop = stopwords.words('english') 

# set spell checker & lemmatizer variables
spell = SpellChecker()
lemmatizer = WordNetLemmatizer()


def extract_emoticon_sentiment(tweet, emoticon_dictionary):
    """
    Extracts the number of positive and negative sentiment emoticons in a tweet
    
    Parameters
    ----------
    tweet: str
        the tweet containing emoticons to be extracted
    
    Returns
    -------
    pos: int
        the number of positive emoticons contained in the tweet
    neg: int
        the number of negative emoticons contained in the tweet
    """   
    
    # initialise variables
    sentiment = ''
    pos = 0
    neg = 0
    
    # create list of emoticons
    emoticon_list = emoticon_dictionary['Emoticon']
    
    # search for emoticons in tweets
    for emoticon in emoticon_list:
        if emoticon in tweet:
            sentiment = emoticon_dictionary['Sentiment'][emoticon_dictionary['Emoticon'] == emoticon].values[0]
            count = tweet.count(emoticon)
        else:
            sentiment = ''
            count = 0
        if sentiment == 'Positive':
            pos += count
        if sentiment == 'Negative':
            neg += count
            
    return pos, neg


def remove_emoticons(tweet, emoticon_dictionary):
    """
    Removes any emoticons contained in a tweet
    
    Parameters
    ----------
    tweet: str
        the tweet containing emoticons to be removed
    
    Returns
    -------
    tweet: str
        the tweet with emoticons removed
    """     
    
    # create list of emoticons
    emoticon_list = emoticon_dictionary['Emoticon']
    
    # search for emoticons in tweets
    for emoticon in emoticon_list:
        if emoticon in tweet:
            # replace emoticon with ""
            tweet = tweet.replace(emoticon, "")
    
    return tweet


def remove_specific_language(tweet):
    """
    Removes any twitter specific language (hashtags, targets, urls and RT) 
    contained in a tweet
    
    Parameters
    ----------
    tweet: str
        the tweet containing twitter specific language to be removed
    
    Returns
    -------
    tweet: str
        the tweet with twitter specific language removed
    """    
    
    # remove RT
    tweet = tweet.replace('RT','')
    
    # remove hashtags, targets and urls
    tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)","",tweet).split())
    
    return tweet



def lowercase(tweet):
    """
    Converts all characters to lowercase
    
    Parameters
    ----------
    tweet: str
        the tweet containing characters to be converted to lower case
    
    Returns
    -------
    tweet: str
        the tweet with all characters converted to lowercase
    """ 
    
    # convert text to lowercase
    tweet = tweet.lower()
    
    return tweet


def expand_acronym(tweet, acronym_dictionary):
    """
    Expand any acronyms contained in a tweet
    
    Parameters
    ----------
    tweet: str
        the tweet containing acronyms to be expanded
    
    Returns
    -------
    tweet: str
        the tweet with all acronyms expanded
    """ 
    
    # convert to strings
    acronym_dictionary['Acronym'] = acronym_dictionary['Acronym'].astype(str)
    
    # create list of words
    words = tweet.split()
    words_new = []
    
    for word in words:
        for i in range(len(acronym_dictionary)):
            if word == acronym_dictionary['Acronym'][i]:
                word = acronym_dictionary['Meaning'][i]
        words_new.append(word)
        
    # concatenate list of words
    tweet = ' '.join(words_new)
        
    return tweet


def remove_punctuation(tweet):
    """
    Remove all punctuation contained in a tweet
    
    Parameters
    ----------
    tweet: str
        the tweet containing punctuation to be removed
    
    Returns
    -------
    tweet: str
        the tweet with punctuation removed
    """ 
    
    # remove punctuation from tweet
    tweet = re.sub(r'[^\w\s]','',tweet)
    
    return tweet


def remove_stopwords(tweet, stop):
    """
    Remove all stopwords in a tweet
    
    Parameters
    ----------
    tweet: str
        the tweet containing stopwords to be removed
    
    Returns
    -------
    tweet: str
        the tweet with stopwords removed
    """ 
    
    # create list of words
    words = tweet.split()
    words_new = []
    
    # search for emoticons in tweets
    for word in words:
        for stopword in stop:
            if word == stopword:
                # set each stopword to be an empty string
                word = ''
        words_new.append(word)
                
    # concatenate list of words
    tweet = ' '.join(words_new)
        
    return tweet


def spelling_correction(tweet, spell, lemmatizer):
    """
    Correct all spelling mistakes and lemmatize all words in a tweet
    
    Parameters
    ----------
    tweet: str
        the tweet containing words to be corrected a lemmatized
    
    Returns
    -------
    tweet: str
        the tweet with all spelling mistakes corrected and all words lemmatized
    """ 
    
    # remove extra alphabets
    pattern = re.compile(r"(.)\1{2,}")
    tweet = pattern.sub(r"\1\1", tweet)
    
    # split string into list of words
    words = tweet.split()
    words_new = []
    
    # correct & lemmatize each word in the list
    for word in words:
        # correct spelling of word
        word = spell.correction(word)
        # lemmatize word
        word = lemmatizer.lemmatize(word)
        # append to list
        words_new.append(word)
        
    # concatenate list of words
    tweet = ' '.join(words_new)
    
    return tweet


start = time.time()
# apply functions to tweets
data['emoticon_sentiment'] = data['tweet'].apply(lambda x: extract_emoticon_sentiment(x, emoticon_dictionary))
data[['pos_emoticon', 'neg_emoticon']] = pd.DataFrame(data['emoticon_sentiment'].tolist(), index=data.index)   
data = data.drop(columns = ['emoticon_sentiment'])
data['tweet'] = data['tweet'].apply(lambda x: remove_emoticons(x, emoticon_dictionary))
data['tweet'] = data['tweet'].apply(remove_specific_language)
data['tweet'] = data['tweet'].apply(lowercase)
data['tweet'] = data['tweet'].apply(lambda x: expand_acronym(x, acronym_dictionary))
data['tweet'] = data['tweet'].apply(remove_punctuation)
data['tweet'] = data['tweet'].apply(lambda x: remove_stopwords(x, stop))
#data['tweet'] = data['tweet'].apply(lambda x: spelling_correction(x, spell, lemmatizer))
data['sentiment'] = data['sentiment'].replace(4,1)
end = time.time()

print(end-start)

# save to csv
#data.to_csv('../../data/twitter data/Labelled/clean/labelled_twitter_data.csv', index=False)


