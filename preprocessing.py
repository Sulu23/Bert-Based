#The the data files as input, and output the edible data for the model
import re
import string

def normalize(data):
    # not yet emojis implemented, because the list of emojis is very long, this would slow down the program terribly
    for list_of_tokens in data:
        for token in list_of_tokens:
            # replace links by filler
            # regular expression taken from: https://stackoverflow.com/questions/3809401/what-is-a-good-regular-expression-to-match-a-url
            regex_url = "(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})"
            #re.sub(regex_url, "CODE_URL")

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
        # normalization for hyperlinks, #tags and @usernames
        token = line.split()[0]
        normalize(token)
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

def concaternate_tokens(data, predictions):
    # The bert tokenizer has split some words into subwords, and has encoded the spaces as an underscore. 
    # this function will concaternate the split tokens, and change the labels accodingly
    # so the data will be suitable for SpaCy. 
    for tweet in data:
        #first remove beginning/end of line mark
        tweet.pop()
        tweet.pop(0)
        list_of_words = []
        word = ""
        for token in tweet:
            # concaternate tokens to words (units separated by space in original text)
            # The ▁ symbol encodes a space in the original tweet, thus new word is starting
            if token[0] == "▁" and word:
                # add completed word to list of words
                list_of_words.append(word)
                # remove first character and start new word
                word = token[1:]

            # the first word is not taken into account by the precious if. So this is done now. 
            elif token[0] == "▁":
                word = token[1:]

            # take into account that punctuation has to be split from the words
            # only look at the fist character to avoid separating @users and #'s (because they always begin with the thick underscore)
            elif token[0] in string.punctuation:
                # add previous word to list of words
                if word:
                    list_of_words.append(word)
                # add punctuation to list of word.
                list_of_words.append(token)
                # reset word. 
                word = ''

            else:
                # In this case, token is a subwords in the middle or end of a word
                # so append to current word.
                word = word + token
        # append last word, but first check if it is not empty. 
        if word:
            list_of_words.append(word)

        print(list_of_words, '\n')



    # silly way of printing predictions and tokens
    #i = 0
    #for label in predictions:
    #    print("\nTokens: " , tokens[i])
    #    print("Labels: " ,label)
    #    print("len tokens: ", len(tokens[i]), "len labels: ", len(label))
    #    i += 1
  