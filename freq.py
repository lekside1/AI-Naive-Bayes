
# this method return dict of class without freq = 1
def remove_freq_1(story_freq_dict, ask_freq_dict, show_freq_dict, poll_freq_dict):
    story_freq_dict = {w: f for w, f in story_freq_dict.items() if f != 1}
    ask_freq_dict = {w: f for w, f in ask_freq_dict.items() if f != 1}
    show_freq_dict = {w: f for w, f in show_freq_dict.items() if f != 1}
    poll_freq_dict = {w: f for w, f in poll_freq_dict.items() if f != 1}
    return story_freq_dict, ask_freq_dict, show_freq_dict, poll_freq_dict


# this method return dict of class without freq <= 5
def remove_freq_lte_5(story_freq_dict, ask_freq_dict, show_freq_dict, poll_freq_dict):
    story_freq_dict = {w: f for w, f in story_freq_dict.items() if f > 5}
    ask_freq_dict = {w: f for w, f in ask_freq_dict.items() if f > 5}
    show_freq_dict = {w: f for w, f in show_freq_dict.items() if f > 5}
    poll_freq_dict = {w: f for w, f in poll_freq_dict.items() if f > 5}
    return story_freq_dict, ask_freq_dict, show_freq_dict, poll_freq_dict


# this method return dict of class without freq <= 10
def remove_freq_lte_10(story_freq_dict, ask_freq_dict, show_freq_dict, poll_freq_dict):
    story_freq_dict = {w: f for w, f in story_freq_dict.items() if f > 10}
    ask_freq_dict = {w: f for w, f in ask_freq_dict.items() if f > 10}
    show_freq_dict = {w: f for w, f in show_freq_dict.items() if f > 10}
    poll_freq_dict = {w: f for w, f in poll_freq_dict.items() if f > 10}
    return story_freq_dict, ask_freq_dict, show_freq_dict, poll_freq_dict


# this method return dict of class without freq <= 15
def remove_freq_lte_15(story_freq_dict, ask_freq_dict, show_freq_dict, poll_freq_dict):
    story_freq_dict = {w: f for w, f in story_freq_dict.items() if f > 15}
    ask_freq_dict = {w: f for w, f in ask_freq_dict.items() if f > 15}
    show_freq_dict = {w: f for w, f in show_freq_dict.items() if f > 15}
    poll_freq_dict = {w: f for w, f in poll_freq_dict.items() if f > 15}
    return story_freq_dict, ask_freq_dict, show_freq_dict, poll_freq_dict


# this method return dict of class without freq <= 20
def remove_freq_lte_20(story_freq_dict, ask_freq_dict, show_freq_dict, poll_freq_dict):
    story_freq_dict = {w: f for w, f in story_freq_dict.items() if f > 20}
    ask_freq_dict = {w: f for w, f in ask_freq_dict.items() if f > 20}
    show_freq_dict = {w: f for w, f in show_freq_dict.items() if f > 20}
    poll_freq_dict = {w: f for w, f in poll_freq_dict.items() if f > 20}
    return story_freq_dict, ask_freq_dict, show_freq_dict, poll_freq_dict