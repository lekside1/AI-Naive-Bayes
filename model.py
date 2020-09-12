# imported packages
import string
import os

# this method returns the raw title year data
def get_raw_title(data, year):
    # get raw title from data + convert to lowercase
    raw_title_year = [d['Title'].lower() for d in data if d['year'] == year]
    return raw_title_year


# this method get the count of each class
def get_post_type_count(data):
    # post type counts
    story_count = ask_count = show_count = poll_count = 0
    for d in data:
        if d['Post Type'] == 'story':
            story_count += 1
        elif d['Post Type'] == 'ask_hn':
            ask_count += 1
        elif d['Post Type'] == 'show_hn':
            show_count += 1
        elif d['Post Type'] == 'poll':
            poll_count += 1
    # total count of all classes
    total_class_count = story_count + ask_count + show_count + poll_count
    return story_count, ask_count, show_count, poll_count, total_class_count


# this method get the raw title data for each class type
def get_raw_class_title(data):
    data_2018 = [d for d in data if d['year'] == '2018']
    story_list = [w['Title'].lower() for w in data_2018 if w['Post Type'] == 'story']
    ask_list = [w['Title'].lower() for w in data_2018 if w['Post Type'] == 'ask_hn']
    show_list = [w['Title'].lower() for w in data_2018 if w['Post Type'] == 'show_hn']
    poll_list = [w['Title'].lower() for w in data_2018 if w['Post Type'] == 'poll']
    # poll_list = [w['Title'].lower() for w in data_2018 if w['Post Type'] != 'story'
    #              and w['Post Type'] != 'ask_hn' and w['Post Type'] != 'show_hn']
    return story_list, ask_list, show_list, poll_list


# tokenize title data by removing white lines
def tokenize_data(raw_data):
    tokens = []
    for title in raw_data:
        tokens += title.split()
    return tokens


# this method remove punction from a token
def remove_punctuation_from_token(token):
    removed_punct = []
    for c in token:
        if c in string.punctuation:
            token = token.replace(c, "")
            removed_punct.append(c)
    return token, removed_punct


# this method removes punctuation from tokens
def remove_punctuation(tokens):
    new_tokens = []
    removed_punct = []
    for t in tokens:
        token, punct = remove_punctuation_from_token(t)
        new_tokens.append(token)
        removed_punct += punct
    return new_tokens, removed_punct


# this method removes non alphabetic tokens
def remove_non_alpha(no_punct_tokens, removed_tokens):
    alpha_tokens = []
    for word in no_punct_tokens:
        if word.isalpha():
            alpha_tokens.append(word)
        else:
            removed_tokens.append(word)
    return alpha_tokens, removed_tokens


# this method cleans the tokenized data
def clean_raw_data(raw_data):
    # tokenize by removing white lines
    tokens = tokenize_data(raw_data)
    # remove punctuation from tokens
    no_punctuation_tokens, removed_punct = remove_punctuation(tokens)
    # remove non alphabetic tokens
    clean_words, removed_tokens = remove_non_alpha(no_punctuation_tokens, removed_punct)
    # sort alphabetically
    clean_words.sort()
    removed_tokens = sorted(set(removed_tokens))
    return clean_words, removed_tokens


# this method return a dictionary of word frequency pairs
def get_word_freq_dict(words):
    return dict(zip(words, [words.count(w) for w in words]))


# this method calculates the smooth probability of each word, returns
def get_word_prob(freq_dict, class_type, prob_name, class_words_count, vocab_size, delta=0.5):
    class_prob = []
    for s_freq in freq_dict[class_type].values:
        # prob = round(s_freq / (class_words_count + (delta * vocab_size)), 6)
        prob = s_freq / (class_words_count + (delta * vocab_size))
        class_prob.append(prob)
    freq_dict[prob_name] = class_prob
    return freq_dict, class_prob


# this method create the model file
def create_model_file(model_file, df, story_prob, ask_prob, show_prob, poll_prob):
    os.makedirs(os.path.dirname('./output/'), exist_ok=True) # create output directory if doesn't exist
    model_file_path = './output/'+model_file
    all_words = df.index.tolist()
    story_freq = df['story'].values
    ask_freq = df['ask_hn'].values
    show_freq = df['show_hn'].values
    poll_freq = df['poll'].values
    with open(model_file_path, 'w') as f:
        for i in range(len(df)):
            model = (str(i + 1) + '  ' + all_words[i]
                     + '  ' + str(story_freq[i]) + '  ' + str(story_prob[i])
                     + '  ' + str(ask_freq[i]) + '  ' + str(ask_prob[i])
                     + '  ' + str(show_freq[i]) + '  ' + str(show_prob[i])
                     + '  ' + str(poll_freq[i]) + '  ' + str(poll_prob[i])
                     + "\r")
            f.write(model)
        f.close()
    print('Done writing to', model_file)


