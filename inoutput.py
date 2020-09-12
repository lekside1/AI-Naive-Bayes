# imported packages
import csv
import os

# this method gets the user input
def get_user_input():
    filename = input('Enter filename: ')
    if not filename:
        filename = 'hns_2018_2019.csv'
    return './data/'+filename


# this method read the csv file and returns the data as a list
def read_csv_file(filename):
    data = []
    with open(filename) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            data.append(row)
    csv_file.close()
    print('Done reading', filename)
    return data


# this method writes to a list to a file
def write_to_file(filename, str_list):
    os.makedirs(os.path.dirname('./output/'), exist_ok=True) # create output directory if doesn't exist
    filepath = './output/'+filename
    with open(filepath, 'w') as f:
        for word in str_list:
            f.write('%s\n' % word)
    f.close()
    print('Done writing to', filename)


# this method read a txt file
def read_file(filename):
    with open(filename) as f:
        file_data_list = f.read().splitlines()
    f.close()
    return file_data_list
