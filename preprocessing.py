# preprocessing.py
#The the data files as input, and output the edible data for the model
import re
import string

# This set contains all unk classified tokens from the training data
unks = {'umm', 'ew', 'u*', '#jdm', 'mhmmm', '6hrs', 'ma', 'hmm', 'bamb', '#r', 'bora',
       'hammmmmmm', 'wya', 'nahhh', 'naw', 'goga', 'mhm', 'h52d', 'laɪvz', 'ily', 'uhh',
        'yayy', 'ayyye', 'eh', 'sh#+', 'qhfkfieodhwnppvl897jd', 'uu', 'nqv', 'yales',
        'dawwwwwhhh', 'cu', '#pinsheswercaslesbis', 'iraaa', 'xoxo', 'foo', 'ajaaaaijaaai',
        '6-7pm', 'nɪ', 'xxx', 'foh', 'laɪf', 'rr', 'uff', 'meow', '10am', 'pfftt', 'ringo',
        'dksishslbehelbeksbdjjdbksbs', 'ricesichuu', 'ia', 'ijuuuuuu', 'nd', 'akdiciwjcoajf',
        'ohhh', 'sesh', 'oh', 'eww', 'emm', 'simpa', 'ayyee', 'mm', 'uffff', 'awww', '90s',
        'beeeeaaappp', '#s.e.m', 'p2', 'marciiiii', 'wiggy', 'p', 'napearía', 'ehh', 'x',
        'frfr', 'imy', 'psh', 'pe', 'pmo', '#allmaton', 'aw', 'ey', 'nismo', 'ughh',
        'nawh', 'wow', 'apush', 'zzzaaaaayuuuummmm', 't', 'wjfk', 'em', 'pawsome', 'k',
        'p1', 'tl', 'mayneee', 'yay', 'otc', 'l', 'ayyyjuuuesuu', 'kjooo', 'ummm', 'na',
        'auuuuuuuuush', 'eeeaahhh', 'awwww', 'ouch', 'bujaja', '#poetuit', 'eaaa', 'hmmm',
        'awws', 'jajajajajajajajajajajajaja', 'yahoos', 'pffftt', 'boo', 'jab', 'uh',
        'silverado', 'ughhhhhh', 'muah', 'aii', 'tacuachooonnn', 'tssss', 'mlp', '38k', 'ig',
        'ghvsdfhjjds', 'ff', 'natin', 'bubu', 'ugh', 'bb', '-spm', 'tsk', 'leɪt', 'kəˈmyu', 'ijuuu',
        'svvfsdbhf', 'ouchh', 'j°na', 'bev', 'mmm', 'fənt', 'ggggruto', 'aww',
        '#freefabo', 'c', 'djfer&djroy', 'og', 'tss', 'iuuuuuu', 'mv', 'awito', 'beffo', 'pip',
        'ti', '#mochomista', 'cc', 'awh', 'chi', 'f150', 'b', 'woojuuu', 'n', 'bmo'}


def tokens_to_sentences(data):
    """ seperates tweets and removes urls from data """
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


def concatenate_tokens(predictions, data):
    """ The bert tokenizer has split some words into subwords, and has encoded the spaces as an underscore.
    this function will concaternate the split tokens, and change the labels accodingly
    so the data will be suitable for SpaCy. """
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
            # chr(9601) encodes a space in the original tweet, thus new word is starting
            if token[0] != chr(9601):
                word += tweet[i]
                labels_per_word.append(predictions[n][i])


            elif token[0] == chr(9601):
                if word:
                    list_of_words.append(word)
                    labels_per_sentence.append(labeler(word, labels_per_word))

                word = token[1:]
                labels_per_word = [predictions[n][i]]

        # appends last word
        if word:
            list_of_words.append(word)
            labels_per_sentence.append(labeler(word, labels_per_word))


        # append to final dataset
        all_tokens.append(list_of_words)
        all_labels.append(labels_per_sentence)

    return all_labels, all_tokens


def labeler(word, labels):
    """ input: word and all labels attributed to the word by the model
        output: predicted label """

    # label as 'other' if string contains punctuation, but not hashtags
    if word.isalpha() == False and word[0] != '#':
        return 'other'

    # All tokens that start with @ are labeled as other
    if word.startswith('@'):
        return 'other'

    # check hardcoded list of unk labeled tokens from training data, lowecased, because the set is lowercased as well
    if word.lower() in unks:
        return 'unk'

    # When a character occurs 3 or more times consecutively in a word, we assume it is an unk.
    # The regular expression is from:
    # https://stackoverflow.com/questions/6518154/matching-3-or-more-of-the-same-character-in-python/6518192
    if re.search(r'(\w)\1\1', word):
        return 'unk'

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
