import re
from spellchecker import SpellChecker
spell = SpellChecker()


def extract_emoticon_sentiment(str tweet, emoticon_dictionary):
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
    cdef str sentiment = ''
    cdef int pos = 0
    cdef int neg = 0
    cdef int count = 0
    
    # search for emoticons in tweets
#    for i in range(len(emoticon_list)):
#        if emoticon_list[i] in tweet:
#            sentiment = emoticon_sentiment[i]
#            count = tweet.count(emoticon_list[i])
#        else:
#            sentiment = ''
#            count = 0
#        if sentiment == 'Positive':
#            pos += count
#        if sentiment == 'Negative':
#            neg += count
            
    
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
    emoticon_list = emoticon_dictionary['Emoticon'].tolist()
    
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


def expand_acronym(tweet, acronym_dictionary, acronyms, meanings):
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
    #acronym_dictionary['Acronym'] = acronym_dictionary['Acronym'].astype(str)
    
    # create acronym variables
    #cdef list acronyms = acronym_dictionary['Acronym'].tolist()
    #cdef list meanings = acronym_dictionary['Meaning'].tolist()
    
    # create list of words
    cdef list words = tweet.split()
    cdef list words_new = []
    #cdef str full_word = ''
    
    for word in words:
        for i in range(len(acronym_dictionary)):
            if word == acronyms[i]:
                word = meanings[i]
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


def remove_stopwords(tweet, list stop):
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
    cdef list words = tweet.split()
    cdef list words_new = []
    
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


def spelling_correction(tweet, spell, lemmatizer, word_dictionary):
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
    
    # create list of words
    cdef list word_list = word_dictionary['Word'].tolist()
    
    # remove extra alphabets
    pattern = re.compile(r"(.)\1{2,}")
    tweet = pattern.sub(r"\1\1", tweet)
    
    # split string into list of words
    cdef list words = tweet.split()
    cdef list words_new = []
    
    # correct & lemmatize each word in the list
    for word in words:
        if word not in word_list:
            # correct spelling of word
            word = spell.correction(word)
        # lemmatize word
        word = lemmatizer.lemmatize(word)
        # append to list
        words_new.append(word)
        
    # concatenate list of words
    tweet = ' '.join(words_new)
    
    return tweet

