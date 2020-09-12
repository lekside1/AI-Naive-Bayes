# imported packages
from model import *
from numpy import log, inf
import os


# this method gets the testing data
def get_testing_title(data, year='2019'):
    testing_year = []
    testing_titles = []
    # get raw testing title from data + convert to lowercase
    for d in data:
        if d['year'] == year:
            testing_year.append(([d.get('Title').lower()], d.get('Post Type')))
            testing_titles.append(d.get('Title').lower())
    return testing_year, testing_titles


# this method cleans the tokenized data
def clean_testing_data(raw_data):
    # tokenize by removing white lines
    tokens = tokenize_data(raw_data)
    # remove punctuation from tokens
    no_punctuation_tokens, _ = remove_punctuation(tokens)
    # remove non alphabetic tokens
    alpha_tokens = [word for word in no_punctuation_tokens if word.isalpha()]
    return alpha_tokens


# this method smooth (0.5) words not in vocabulary
def get_unknown_smooth_prob(class_words_count, vocab_size):
    # prob = round(0.5 / (class_words_count + (0.5 * vocab_size)), 6)
    prob = 0.5 / (class_words_count + (0.5 * vocab_size))
    return prob


# this method compute the score for each title
def get_score(test_titles, p_class, class_word_pob, class_words_count, vocab_size, class_type):
    # handle divide by zero runtime warning
    # score(class) = log(p(class)) + log(p(first word | class)) + ... + log(p(last word | class))
    class_score_list = []
    for title in test_titles:
        class_score = log(p_class) if p_class > 0 else -inf
        for i in range(len(title)):
            if class_word_pob.get(title[i]):
                class_score += log(class_word_pob.get(title[i]))
            else:
                class_score += log(get_unknown_smooth_prob(class_words_count, vocab_size))
        class_score_list.append(class_score)
    # print(class_type, 'score:', class_score_list)
    return class_score_list


# this method returns the max class score
def get_max_score(story, ask, show, poll):
    return max([story, ask, show, poll])


# this method computes max score
def compute_max_scores(test_score_df):
    max_scores = []
    for i in range(len(test_score_df)):
        # score for each class
        story_score = test_score_df.get('story')[i]
        ask_score = test_score_df.get('ask_hn')[i]
        show_score = test_score_df.get('show_hn')[i]
        poll_score = test_score_df.get('poll')[i]
        # store max score in list
        max_scores.append(get_max_score(story_score, ask_score, show_score, poll_score))
    test_score_df['max_score'] = max_scores  # add max_score column to df
    return max_scores


# this method set classified label
def set_classify_label(test_score_df, max_scores):
    classify_label = []
    for i in range(len(test_score_df)):
        if max_scores[i] == test_score_df.get('story')[i]:
            classify_label.append('story')
        elif max_scores[i] == test_score_df.get('ask_hn')[i]:
            classify_label.append('ask_hn')
        elif max_scores[i] == test_score_df.get('show_hn')[i]:
            classify_label.append('show_hn')
        else:
            classify_label.append('poll')
    test_score_df['classify_label'] = classify_label     # add classify_label column to df
    return classify_label


# this method check if classification is right of wrong
def check_classification(test_score_df):
    classify_result = []
    for i in range(len(test_score_df)):
        # compare actual label and classified label
        classify_result.append('right') \
            if test_score_df.get('actual_label')[i] == test_score_df.get('classify_label')[i] \
            else classify_result.append('wrong')
    test_score_df['classify_result'] = classify_result       # add classify_result column to df
    return classify_result


# this method computes the number of right and wrong
def get_number_right_wrong(testing_score_df):
    right_count = wrong_count = 0
    for r in testing_score_df['classify_result'].values:
        if r == 'right':
            right_count += 1
        else:
            wrong_count += 1
    print('right: ', right_count, 'wrong: ', wrong_count)
    return right_count, wrong_count


# this method creates baseline-result.txt file
def create_baseline(baseline_file, test_score_df, test_titles, classify_label,
                    story_scores, ask_scores, show_scores, poll_scores,
                    actual_labels, classify_result):
    os.makedirs(os.path.dirname('./output/'), exist_ok=True) # create output directory if doesn't exist
    baseline_file_path = './output/'+baseline_file
    with open(baseline_file_path, 'w') as f:
        for i in range(len(test_score_df)):
            baseline = (str(i + 1) + '  ' + test_titles[i] + '  ' + classify_label[i]
                        + '  ' + str(story_scores[i]) + '  ' + str(ask_scores[i])
                        + '  ' + str(show_scores[i]) + '  ' + str(poll_scores[i])
                        + '  ' + str(actual_labels[i]) + '  ' + str(classify_result[i]) + "\r")
            f.write(baseline)
        f.close()
    print('Done writing to', baseline_file)
