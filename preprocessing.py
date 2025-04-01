# preprocessing.py
#The the data files as input, and output the edible data for the model
import re
import string

# this patters checks if string only contains digists or punctuation, copies from
# https://stackoverflow.com/questions/21696649/filtering-out-strings-that-only-contains-digits-and-or-punctuation-python
pattern = re.compile("[\d{}]+$".format(re.escape(string.punctuation)))
  

def normalize(data):
    # not yet emojis implemented, because the list of emojis is very long, this would slow down the program terribly
    for list_of_tokens in data:
        for token in list_of_tokens:
            # replace links by filler

            pass
            # replace usernames by filler
            #re.sub("^@", "CODE_username")

def tokens_to_sentences(data):
    tweets = []
    # make list of tweets, first initiate empty tweet: 
    tweet = ""
    # loop over the data
    for line in data.readlines():
        # skip if empty line
        if len(line) == 0 or line.isspace():
            continue
        
        token = line.split()[0]

        # normalization for hyperlinks, #tags and @usernames
        # regular expression taken from: https://stackoverflow.com/questions/3809401/what-is-a-good-regular-expression-to-match-a-url
        regex_url = "https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,}"
        token = re.sub(regex_url, "URLREPLACED", token)

        
        # check if new tweet is starting
        if "# sent_enum = " in line:
            # This check is nessesary because I dont want the initialized empty tweet to be appended to the dataset
            if not tweet == "":
                #append previous tweet to list of tweets
                tweets.append(tweet)
                # empty tweet variable for next tweet
                tweet = ""
        # Append word to tweet
        else:
            tweet = tweet + " " + token
    # add last tweet
    tweets.append(tweet)

    return tweets 

def concaternate_tokens(predictions, data):
    # The bert tokenizer has split some words into subwords, and has encoded the spaces as an underscore. 
    # this function will concaternate the split tokens, and change the labels accodingly
    # so the data will be suitable for SpaCy.
    all_tokens = []
    all_labels = []
    
    for n, tweet in enumerate(data):
        list_of_words = []
        word = ""
        # keep track of all labels per word. 
        labels_per_word = []
        labels_per_sentence = []

        for i, token in enumerate(tweet):
            # concaternate tokens to words (units separated by space in original text)
            # The ▁ symbol encodes a space in the original tweet, thus new word is starting
            if token[0] == "▁" and word:
                # add completed word to list of words
                list_of_words.append(word)
                # find correct label and add to list of labels per sentence
                labels_per_sentence.append(labeler(word, labels_per_word))
                labels_per_word = []
                
                # remove first character and start new word, add label to list of labels
                word = token[1:]
                labels_per_word.append(predictions[n][i])

            # the first word is not taken into account by the precious if cause. So this is done now. 
            elif token[0] == "▁":
                word = token[1:]
                labels_per_word.append(predictions[n][i])

            # take into account that punctuation has to be split from the words
            # only look at the fist character to avoid separating @users and #'s (because they always begin with the thick underscore)
            elif token[0] in string.punctuation:
                # add previous word to list of words
                if word:
                    list_of_words.append(word)
                    labels_per_sentence.append(labeler(word, labels_per_word))
                    labels_per_word = []

                    # add punctuation to list of word
                    list_of_words.append(token)
                    labels_per_sentence.append(labeler(word, labels_per_word))
                    labels_per_word = []
                    # reset word. 
                    word = ''
            else:
                # In this case, token is a subwords in the middle or end of a word
                # so append to current word.
                word = word + token
                labels_per_word.append(predictions[n][i])

        # append last word, but first check if it is not empty. 
        if word:
            list_of_words.append(word)
            labels_per_sentence.append(labeler(word, labels_per_word))
            labels_per_word = []
        
        # append to final dataset
        all_tokens.append(list_of_words)
        all_labels.append(labels_per_sentence)

    return all_labels, all_tokens


def labeler(word, labels):
    # input: word and all labels attributed to the word by the model
    # output: predicted label

    # label as 'other' if string only contains punctuation. 
    if pattern.match(word):
        return 'other'
    # All tokens that start with @ are labeled as other
    if word[0]== '@':
        return 'other'
    

    # map model labels to task labels
    set_labels = set(labels)
    if set_labels == {'I-ES'}:
        return 'lang2'
    elif set_labels == {'I-EN'}:
        return 'lang1'
    elif len(set_labels) == 1:
        return 'fw'
    else:
        return 'mixed'



  # postprocessing errors:
  # gold:       this program:
  # "'s"        "'", "s"     
  # "@m_boy"    "@m", "-", "boy"
  # ">>>>"       '>', '>>>'
