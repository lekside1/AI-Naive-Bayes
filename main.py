# imported packages
import pandas as pd
from inoutput import *
from naiveBayes import *
from freq import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcol


# main method
def main():
    sep = '='*100+"\n"   # separator

    # get user input of filename
    # read the file using csv
    data = read_csv_file(get_user_input())

    # ================================================================================================================ #

    # Task 1: Extract data and build model
    print('==========\nTask 1\n==========')
    # extract raw title from data
    raw_title_2018 = get_raw_title(data, '2018')

    # clean the raw title data
    clean_words_2018, removed_words_2018 = clean_raw_data(raw_title_2018)

    # get the vocabulary 2018 words and size
    vocabulary_2018 = sorted(set(clean_words_2018))
    vocabulary_size = len(vocabulary_2018)

    # extract class raw title data
    story_title_raw, ask_title_raw, show_title_raw, poll_title_raw = get_raw_class_title(data)

    # clean raw class title data
    clean_story_words, _ = clean_raw_data(story_title_raw)
    clean_ask_words, _ = clean_raw_data(ask_title_raw)
    clean_show_words, _ = clean_raw_data(show_title_raw)
    clean_poll_words, _ = clean_raw_data(poll_title_raw)

    # write the vocabulary to vocabulary.txt
    write_to_file('vocabulary.txt', vocabulary_2018)

    # write the removed words to removed_word.txt
    write_to_file('removed_word.txt', removed_words_2018)
    print('Done cleaning data...')
    print(sep)

    # words frequency
    print('Computing word frequencies...')
    story_freq_dict = get_word_freq_dict(clean_story_words)
    ask_freq_dict = get_word_freq_dict(clean_ask_words)
    show_freq_dict = get_word_freq_dict(clean_show_words)
    poll_freq_dict = get_word_freq_dict(clean_poll_words)
    print('Done computing word frequencies')
    print(sep)

    # Probability smoothing (delta = 0.5)
    # get count of classes
    story_count, ask_count, show_count, poll_count, total_count = get_post_type_count(data)

    # probability of each class
    p_story = float(story_count / total_count)
    p_ask = float(ask_count / total_count)
    p_show = float(show_count / total_count)
    p_poll = float(poll_count / total_count)

    # total number of words in each class
    story_words_count = len(story_freq_dict)
    ask_words_count = len(ask_freq_dict)
    show_words_count = len(show_freq_dict)
    poll_words_count = len(poll_freq_dict)

    delta = 0.5  # smooth word frequency from vocabulary_2018
    # create a dictionary of words and frequency with smoothing
    words_freq_dict = {w: {'story': story_freq_dict.get(w, 0) + delta, 'ask_hn': ask_freq_dict.get(w, 0) + delta,
                           'show_hn': show_freq_dict.get(w, 0) + delta, 'poll': poll_freq_dict.get(w, 0) + delta}
                       for w in vocabulary_2018}

    # create a data frame from words freq dictionary
    pd.set_option("display.max_rows", None)  # print all dataframe rows
    words_freq_prob_df = pd.DataFrame.from_dict(words_freq_dict, orient='index',
                                                columns=['story', 'ask_hn', 'show_hn', 'poll'])

    # compute the smoothed probability (delta = 0.5)
    words_freq_prob_df, story_prob = get_word_prob(words_freq_prob_df, 'story', 'story_prob', story_words_count,
                                                   vocabulary_size, delta=delta)
    words_freq_prob_df, ask_prob = get_word_prob(words_freq_prob_df, 'ask_hn', 'ask_prob', ask_words_count,
                                                 vocabulary_size, delta=delta)
    words_freq_prob_df, show_prob = get_word_prob(words_freq_prob_df, 'show_hn', 'show_prob', show_words_count,
                                                  vocabulary_size, delta=delta)
    words_freq_prob_df, poll_prob = get_word_prob(words_freq_prob_df, 'poll', 'poll_prob', poll_words_count,
                                                  vocabulary_size, delta=delta)
    print('Done computing smoothed probabilities')
    print(sep)
    print(words_freq_prob_df)
    print(sep)

    # create dictionary of words: prob for each class
    story_word_prob = dict(list(zip(words_freq_prob_df.index, story_prob)))
    ask_word_prob = dict(list(zip(words_freq_prob_df.index, ask_prob)))
    show_word_prob = dict(list(zip(words_freq_prob_df.index, show_prob)))
    poll_word_prob = dict(list(zip(words_freq_prob_df.index, poll_prob)))

    # create the model-2018.txt file
    create_model_file('model-2018.txt', words_freq_prob_df, story_prob, ask_prob, show_prob, poll_prob)
    print(sep)

    # ================================================================================================================ #

    # Task 2: Use ML Classifier to test data set
    print('==========\nTask 2\n==========')
    # get the testing data titles
    testing_2019, testing_titles = get_testing_title(data)

    clean_testing_titles = []   # titles only
    actual_labels = []         # actual labels only
    for t, l in testing_2019:
        clean_testing_titles.append(clean_testing_data(t))  # clean titles
        actual_labels.append(l)

    # get scores for the each title and each class type
    story_scores = get_score(clean_testing_titles, p_story, story_word_prob, story_words_count, vocabulary_size, 'story')
    ask_scores = get_score(clean_testing_titles, p_ask, ask_word_prob, ask_words_count, vocabulary_size, 'ask_hn')
    show_scores = get_score(clean_testing_titles, p_show, show_word_prob, show_words_count, vocabulary_size, 'show_hn')
    poll_scores = get_score(clean_testing_titles, p_poll, poll_word_prob, poll_words_count, vocabulary_size, 'poll')

    # create dictionary of title score
    title_score_dict = {i: {'title': clean_testing_titles[i], 'story': story_scores[i], 'ask_hn': ask_scores[i],
                            'show_hn': show_scores[i], 'poll': poll_scores[i], 'actual_label': actual_labels[i]}
                        for i in range(len(clean_testing_titles))}

    # create a data frame from title score dictionary
    testing_score_df = pd.DataFrame.from_dict(title_score_dict, orient='index',
                                              columns=['title', 'story', 'ask_hn', 'show_hn', 'poll', 'actual_label'])

    # compute max scores
    max_scores = compute_max_scores(testing_score_df)

    # set classified label
    classify_label = set_classify_label(testing_score_df, max_scores)

    # check if classification is right of wrong
    classify_result = check_classification(testing_score_df)
    print(testing_score_df)
    print(sep)

    # compute the number of right and wrong
    right_count, wrong_count = get_number_right_wrong(testing_score_df)
    print(sep)

    # create baseline-result.txt file
    create_baseline('baseline-result.txt', testing_score_df, testing_titles, classify_label,
                    story_scores, ask_scores, show_scores, poll_scores,
                    actual_labels, classify_result)
    print(sep)

    # ================================================================================================================ #

    # Task 3: Experiments with the classifier
    print('==========\nTask 3\n==========')

    # Experiment 1: Stop-word Filtering
    print('----------\nExperiment 1\n----------')
    stopwords_file = './resource/stopwords.txt'
    stopword_model = 'stopword-model.txt'
    stopword_result = 'stopword-result.txt'

    # Experiment 1 Task 1
    # read stopwords file and store words in list
    stopwords_list = read_file(stopwords_file)

    # filter stopwords from vocabulary
    e1_vocabulary = [w for w in vocabulary_2018 if w not in stopwords_list]
    e1_vocabulary_size = len(e1_vocabulary)

    # create a dictionary of words and frequency with smoothing from experiment1_vocabulary
    e1_words_freq_dict = {w: {'story': story_freq_dict.get(w, 0) + delta, 'ask_hn': ask_freq_dict.get(w, 0) + delta,
                              'show_hn': show_freq_dict.get(w, 0) + delta, 'poll': poll_freq_dict.get(w, 0) + delta}
                          for w in e1_vocabulary}

    # create a data frame from experiment1 words freq dictionary
    e1_words_freq_dict_df = pd.DataFrame.from_dict(e1_words_freq_dict, orient='index',
                                                   columns=['story', 'ask_hn', 'show_hn', 'poll'])

    # compute the smoothed probability (delta = 0.5)
    e1_words_freq_dict_df, e1_story_prob = get_word_prob(e1_words_freq_dict_df, 'story', 'story_prob',
                                                         story_words_count, e1_vocabulary_size, delta=delta)
    e1_words_freq_dict_df, e1_ask_prob = get_word_prob(e1_words_freq_dict_df, 'ask_hn', 'ask_prob',
                                                       ask_words_count, e1_vocabulary_size, delta=delta)
    e1_words_freq_dict_df, e1_show_prob = get_word_prob(e1_words_freq_dict_df, 'show_hn', 'show_prob',
                                                        show_words_count, e1_vocabulary_size, delta=delta)
    e1_words_freq_dict_df, e1_poll_prob = get_word_prob(e1_words_freq_dict_df, 'poll', 'poll_prob',
                                                        poll_words_count, e1_vocabulary_size, delta=delta)

    # create dictionary of words: prob for each class
    e1_story_word_prob = dict(list(zip(e1_words_freq_dict_df.index, e1_story_prob)))
    e1_ask_word_prob = dict(list(zip(e1_words_freq_dict_df.index, e1_ask_prob)))
    e1_show_word_prob = dict(list(zip(e1_words_freq_dict_df.index, e1_show_prob)))
    e1_poll_word_prob = dict(list(zip(e1_words_freq_dict_df.index, e1_poll_prob)))

    # create the stopword model
    create_model_file(stopword_model, e1_words_freq_dict_df, e1_story_prob, e1_ask_prob,
                      e1_show_prob, e1_poll_prob)
    print(sep)

    # Experiment 1 Task 2
    # get scores for the each title and each class type
    e1_story_scores = get_score(clean_testing_titles, p_story, e1_story_word_prob,
                                story_words_count, e1_vocabulary_size, 'story')
    e1_ask_scores = get_score(clean_testing_titles, p_ask, e1_ask_word_prob,
                              ask_words_count, e1_vocabulary_size, 'ask_hn')
    e1_show_scores = get_score(clean_testing_titles, p_show, e1_show_word_prob,
                               show_words_count, e1_vocabulary_size, 'show_hn')
    e1_poll_scores = get_score(clean_testing_titles, p_poll, e1_poll_word_prob,
                               poll_words_count, e1_vocabulary_size, 'poll')

    # create dictionary of title score
    e1_title_score_dict = {i: {'title': clean_testing_titles[i], 'story': e1_story_scores[i],
                               'ask_hn': e1_ask_scores[i], 'show_hn': e1_show_scores[i], 'poll': e1_poll_scores[i],
                               'actual_label': actual_labels[i]} for i in range(len(clean_testing_titles))}

    # create a data frame from title score dictionary
    e1_testing_score_df = pd.DataFrame.from_dict(e1_title_score_dict, orient='index',
                                                 columns=['title', 'story', 'ask_hn', 'show_hn', 'poll', 'actual_label'])

    # compute e1 max scores
    e1_max_scores = compute_max_scores(e1_testing_score_df)

    # set e1 classified label
    e1_classify_label = set_classify_label(e1_testing_score_df, e1_max_scores)

    # check if e1 classification is right of wrong
    e1_classify_result = check_classification(e1_testing_score_df)

    # compute the number of right and wrong
    e1_right_count, wrong_count = get_number_right_wrong(e1_testing_score_df)
    print(sep)

    # create stopword-result file
    create_baseline(stopword_result, e1_testing_score_df, testing_titles, e1_classify_label,
                    e1_story_scores, e1_ask_scores, e1_show_scores, e1_poll_scores,
                    actual_labels, e1_classify_result)
    print(sep)

    # ---------------------------------------------------------------------------------------------------------------- #
    # Experiment 2: Word Length Filtering
    print('----------\nExperiment 2\n----------')

    wordlength_model = 'wordlength-model.txt'
    wordlength_result = 'wordlength-result.txt'

    # Experiment 2 Task 1
    # remove words with length <= 2 and >=9 from vocabulary
    e2_vocabulary = [w for w in vocabulary_2018 if 2 < len(w) < 9]
    e2_vocabulary_size = len(e2_vocabulary)

    # create a dictionary of words and frequency with smoothing from experiment1_vocabulary
    e2_words_freq_dict = {w: {'story': story_freq_dict.get(w, 0) + delta, 'ask_hn': ask_freq_dict.get(w, 0) + delta,
                              'show_hn': show_freq_dict.get(w, 0) + delta, 'poll': poll_freq_dict.get(w, 0) + delta}
                          for w in e2_vocabulary}

    # create a data frame from experiment1 words freq dictionary
    e2_words_freq_dict_df = pd.DataFrame.from_dict(e2_words_freq_dict, orient='index',
                                                   columns=['story', 'ask_hn', 'show_hn', 'poll'])

    # compute the smoothed probability (delta = 0.5)
    e2_words_freq_dict_df, e2_story_prob = get_word_prob(e2_words_freq_dict_df, 'story', 'story_prob',
                                                         story_words_count, e2_vocabulary_size, delta=delta)
    e2_words_freq_dict_df, e2_ask_prob = get_word_prob(e2_words_freq_dict_df, 'ask_hn', 'ask_prob',
                                                       ask_words_count, e2_vocabulary_size, delta=delta)
    e2_words_freq_dict_df, e2_show_prob = get_word_prob(e2_words_freq_dict_df, 'show_hn', 'show_prob',
                                                        show_words_count, e2_vocabulary_size, delta=delta)
    e2_words_freq_dict_df, e2_poll_prob = get_word_prob(e2_words_freq_dict_df, 'poll', 'poll_prob',
                                                        poll_words_count, e2_vocabulary_size, delta=delta)

    # create dictionary of words: prob for each class
    e2_story_word_prob = dict(list(zip(e2_words_freq_dict_df.index, e2_story_prob)))
    e2_ask_word_prob = dict(list(zip(e2_words_freq_dict_df.index, e2_ask_prob)))
    e2_show_word_prob = dict(list(zip(e2_words_freq_dict_df.index, e2_show_prob)))
    e2_poll_word_prob = dict(list(zip(e2_words_freq_dict_df.index, e2_poll_prob)))

    # create the wordlength_model
    create_model_file(wordlength_model, e2_words_freq_dict_df, e2_story_prob, e2_ask_prob,
                      e2_show_prob, e2_poll_prob)
    print(sep)

    # Experiment 2 Task 2
    # get scores for the each title and each class type
    e2_story_scores = get_score(clean_testing_titles, p_story, e2_story_word_prob,
                                story_words_count, e2_vocabulary_size, 'story')
    e2_ask_scores = get_score(clean_testing_titles, p_ask, e2_ask_word_prob,
                              ask_words_count, e2_vocabulary_size, 'ask_hn')
    e2_show_scores = get_score(clean_testing_titles, p_show, e2_show_word_prob,
                               show_words_count, e2_vocabulary_size, 'show_hn')
    e2_poll_scores = get_score(clean_testing_titles, p_poll, e2_poll_word_prob,
                               poll_words_count, e2_vocabulary_size, 'poll')

    # create dictionary of title score
    e2_title_score_dict = {i: {'title': clean_testing_titles[i], 'story': e2_story_scores[i],
                               'ask_hn': e2_ask_scores[i], 'show_hn': e2_show_scores[i], 'poll': e2_poll_scores[i],
                               'actual_label': actual_labels[i]} for i in range(len(clean_testing_titles))}

    # create a data frame from title score dictionary
    e2_testing_score_df = pd.DataFrame.from_dict(e2_title_score_dict, orient='index',
                                                 columns=['title', 'story', 'ask_hn', 'show_hn', 'poll', 'actual_label'])

    # compute e2 max scores
    e2_max_scores = compute_max_scores(e2_testing_score_df)

    # set e2 classified label
    e2_classify_label = set_classify_label(e2_testing_score_df, e2_max_scores)

    # check if e2 classification is right of wrong
    e2_classify_result = check_classification(e2_testing_score_df)

    # compute the number of right and wrong
    e2_right_count, e2_wrong_count = get_number_right_wrong(e2_testing_score_df)
    print(sep)

    # create wordlength_result file
    create_baseline(wordlength_result, e2_testing_score_df, testing_titles, e2_classify_label,
                    e2_story_scores, e2_ask_scores, e2_show_scores, e2_poll_scores,
                    actual_labels, e2_classify_result)
    print(sep)

    # ---------------------------------------------------------------------------------------------------------------- #
    # Experiment 3: Infrequent Word Filtering (unfinished)
    print('----------\nExperiment 3\n----------')

    # remove freq = 1
    story_no1_freq_dict, ask_no1_freq_dict, show_no1_freq_dict, poll_no1_freq_dict \
        = remove_freq_1(story_freq_dict, ask_freq_dict, show_freq_dict, poll_freq_dict)

    # create a dictionary of words and frequency with smoothing
    words_no1_freq_dict = {w: {'story': story_no1_freq_dict.get(w, 0) + delta,
                               'ask_hn': ask_no1_freq_dict.get(w, 0) + delta,
                               'show_hn': show_no1_freq_dict.get(w, 0) + delta,
                               'poll': poll_no1_freq_dict.get(w, 0) + delta} for w in vocabulary_2018}
    words_no1_freq_dict_df = pd.DataFrame.from_dict(words_no1_freq_dict, orient='index',
                                                    columns=['story', 'ask_hn', 'show_hn', 'poll'])
    print("Removed freq = 1")
    print(words_no1_freq_dict_df)
    print(sep)

    # remove freq <= 5
    story_no5_freq_dict, ask_no5_freq_dict, show_no5_freq_dict, poll_no5_freq_dict \
        = remove_freq_lte_5(story_freq_dict, ask_freq_dict, show_freq_dict, poll_freq_dict)

    # create a dictionary of words and frequency with smoothing
    words_no5_freq_dict = {w: {'story': story_no5_freq_dict.get(w, 0) + delta,
                               'ask_hn': ask_no5_freq_dict.get(w, 0) + delta,
                               'show_hn': show_no5_freq_dict.get(w, 0) + delta,
                               'poll': poll_no5_freq_dict.get(w, 0) + delta} for w in vocabulary_2018}
    words_no5_freq_dict_df = pd.DataFrame.from_dict(words_no5_freq_dict, orient='index',
                                                    columns=['story', 'ask_hn', 'show_hn', 'poll'])
    print("Removed freq <=5")
    print(words_no5_freq_dict_df)
    print(sep)

    # remove freq <= 10
    story_no10_freq_dict, ask_no10_freq_dict, show_no10_freq_dict, poll_no10_freq_dict \
        = remove_freq_lte_10(story_freq_dict, ask_freq_dict, show_freq_dict, poll_freq_dict)

    # create a dictionary of words and frequency with smoothing
    words_no10_freq_dict = {w: {'story': story_no10_freq_dict.get(w, 0) + delta,
                               'ask_hn': ask_no10_freq_dict.get(w, 0) + delta,
                               'show_hn': show_no10_freq_dict.get(w, 0) + delta,
                               'poll': poll_no10_freq_dict.get(w, 0) + delta} for w in vocabulary_2018}
    words_no10_freq_dict_df = pd.DataFrame.from_dict(words_no10_freq_dict, orient='index',
                                                     columns=['story', 'ask_hn', 'show_hn', 'poll'])
    print("Removed freq <= 10")
    print(words_no10_freq_dict_df)
    print(sep)

    # remove freq <= 15
    story_no15_freq_dict, ask_no15_freq_dict, show_no15_freq_dict, poll_no15_freq_dict \
        = remove_freq_lte_15(story_freq_dict, ask_freq_dict, show_freq_dict, poll_freq_dict)

    # create a dictionary of words and frequency with smoothing
    words_no15_freq_dict = {w: {'story': story_no15_freq_dict.get(w, 0) + delta,
                               'ask_hn': ask_no15_freq_dict.get(w, 0) + delta,
                               'show_hn': show_no15_freq_dict.get(w, 0) + delta,
                               'poll': poll_no5_freq_dict.get(w, 0) + delta} for w in vocabulary_2018}
    words_no15_freq_dict_df = pd.DataFrame.from_dict(words_no15_freq_dict, orient='index',
                                                     columns=['story', 'ask_hn', 'show_hn', 'poll'])
    print("Removed freq <= 15")
    print(words_no15_freq_dict_df)
    print(sep)

    # remove freq <= 20
    story_no20_freq_dict, ask_no20_freq_dict, show_no20_freq_dict, poll_no20_freq_dict \
        = remove_freq_lte_20(story_freq_dict, ask_freq_dict, show_freq_dict, poll_freq_dict)

    # create a dictionary of words and frequency with smoothing
    words_no20_freq_dict = {w: {'story': story_no20_freq_dict.get(w, 0) + delta,
                               'ask_hn': ask_no20_freq_dict.get(w, 0) + delta,
                               'show_hn': show_no20_freq_dict.get(w, 0) + delta,
                               'poll': poll_no20_freq_dict.get(w, 0) + delta} for w in vocabulary_2018}
    words_no20_freq_dict_df = pd.DataFrame.from_dict(words_no20_freq_dict, orient='index',
                                                     columns=['story', 'ask_hn', 'show_hn', 'poll'])
    print("Removed freq <= 20")
    print(words_no20_freq_dict_df)
    print(sep)


# end of main
if __name__ == '__main__':
    main()
