# This code performs the pre-processing of the raw data

import re
import csv


def remove_punctuation(str):
    punctuation = [".", ",", "'", "?", "!", "\\", "/", ":", '"', "-", "#", "@", "(", ")", "^"]
    for punc in punctuation:
        if punc in str:
            str = str.replace(punc, "")

    if "_" in str:
        str = str.replace("_", " ")
    return str


# regex expression sources from
# # https://stackoverflow.com/questions/5020906/python-convert-camel-case-to-space-delimited-using-regex-and-taking-acronyms-in
def split_camel_case(str):
    str = re.sub("([a-z])([A-Z])","\g<1> \g<2>",str)
    return str


def remove_urls(str):
    if "http" in str:
        i = str.index("http")
        # check that the url is at the end of the string before removing it
        if " " not in str[i:]:
            str = str[:i]
    return str


def remove_emojis(str):
    str = re.sub("\\\\u....", " ", str)
    return str


def remove_whitespace(str):
    str = re.sub(' +', ' ', str)
    return str

# process file
dev = open("dev-raw.tsv")
training = open("train-raw.tsv")
testing = open("test-raw.tsv")


with open('processed_train_raw.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t', lineterminator='\n')
    for line in training:
        line = line.split("\t")
        tweet = line[2].rstrip()
        tweet = remove_urls(tweet)
        tweet = remove_emojis(tweet)
        tweet = remove_punctuation(tweet)
        tweet = split_camel_case(tweet)
        tweet = tweet.lower()
        tweet = remove_whitespace(tweet)
        tsv_writer.writerow([line[0], line[1], tweet])

with open('processed_dev_raw.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t', lineterminator='\n')
    for line in dev:
        line = line.split("\t")
        tweet = line[2].rstrip()
        tweet = remove_urls(tweet)
        tweet = remove_emojis(tweet)
        tweet = remove_punctuation(tweet)
        tweet = split_camel_case(tweet)
        tweet = tweet.lower()
        tweet = remove_whitespace(tweet)
        tsv_writer.writerow([line[0], line[1], tweet])

with open('processed_test_raw.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t', lineterminator='\n')
    for line in testing:
        line = line.split("\t")
        tweet = line[2].rstrip()
        tweet = remove_urls(tweet)
        tweet = remove_emojis(tweet)
        tweet = remove_punctuation(tweet)
        tweet = split_camel_case(tweet)
        tweet = tweet.lower()
        tweet = remove_whitespace(tweet)
        tsv_writer.writerow([line[0], line[1], tweet])