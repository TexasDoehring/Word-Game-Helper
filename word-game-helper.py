import polars as pl
import numpy as np
from string import ascii_lowercase as letters
letters = letters+"_"

#word_dictionary_file_path stolen from:
# https://github.com/dwyl/english-words/blob/master/words_alpha.txt
word_dictionary_file_path = r"C:\Users\brett.doehring\Downloads\words_alpha.txt".replace("\\","/")

def word_to_matrix(word):
    numbers = list(map(int, range(1, len(letters)+2)))
    let_to_num = dict(zip(letters, numbers))
    num_to_let = dict(zip(numbers, letters))

    word = list(word)
    word = np.array(word)
    
    get_val = np.vectorize(lambda word: let_to_num.get(word, 'Unknown'))
    return np.array(get_val(word))

def word_finder(known, length=-1, starts_with="", ends_with="", contains="", does_not_contain="",lnixp={},pprint=True):

    contains = contains.replace(" ","")
    contains = ''.join(set(contains) | set(''.join(lnixp.values())))
    contains = "".join(set(contains))

    if "_" in known:
        length = len(known)
    
    does_not_contain = does_not_contain.replace(" ","")
    does_not_contain = ''.join(set(does_not_contain) - set(contains))
    does_not_contain = ''.join(set(does_not_contain) - set(known.replace('_','')))
    does_not_contain = ''.join(set(does_not_contain) - set(starts_with))
    does_not_contain = ''.join(set(does_not_contain) - set(ends_with))
    does_not_contain = ''.join(set(does_not_contain) - set(''.join(lnixp.values())))
    
    
    with open(word_dictionary_file_path,'r') as file:
        words = file.read()
        words = words.split("\n")
    ids = range(len(words))
    df = pl.DataFrame({"index": ids,"words": words})
    word_lengths = [ len(word) for word in df["words"] ]
    df = df.with_columns(word_length=pl.Series(word_lengths))
    
    #length check
    df = df.filter(pl.col("word_length") != 0)
    if length != -1:
        df = df.filter(pl.col("word_length") == length)
    elif length == -1:
        df = df.filter(pl.col("word_length") >= len(known))
        df = df.filter(pl.col("word_length") >= len(starts_with))
        df = df.filter(pl.col("word_length") >= len(ends_with))

    num_matricies = [word_to_matrix(word) for word in df['words']] #converts strings to matricies

    df = df.with_columns(num_matrix=pl.Series(num_matricies)) #puts the matricies in the df
    if len(known) > 0:
        known_matrix = np.array(word_to_matrix(known)) #creates a code for our known word
    
    #filter for the starts_with variable
    if len(starts_with) != 0:
        starts_with_matrix = np.array(word_to_matrix(starts_with))
        right_end = starts_with_matrix.shape[0]
        df = df.with_columns(truth_matrix=pl.Series([starts_with_matrix==np.array(num_matrix)[0:right_end] for num_matrix in df['num_matrix']])) #returns a matrix with where our matrix matches any other word's matrix and adds that column to the df
        df = df.with_columns(truth_sums=pl.Series([sum(truth_matrix) for truth_matrix in df['truth_matrix']])) #returns the sums of truths for all truth matricies
        df = df.filter(pl.col("truth_sums") == df["truth_sums"].max()) #filters the df to only contain the most matching matricies

    if len(ends_with) != 0:
        ends_with_matrix = np.array(word_to_matrix(ends_with))
        left_end = ends_with_matrix.shape[0]
        df = df.with_columns(truth_matrix=pl.Series([ends_with_matrix==np.array(num_matrix)[-left_end:] for num_matrix in df['num_matrix']])) #returns a matrix with where our matrix matches any other word's matrix and adds that column to the df
        df = df.with_columns(truth_sums=pl.Series([sum(truth_matrix) for truth_matrix in df['truth_matrix']])) #returns the sums of truths for all truth matricies
        df = df.filter(pl.col("truth_sums") == df["truth_sums"].max()) #filters the df to only contain the most matching matricies

    if len(known) != 0:
        df = df.with_columns(truth_matrix=pl.Series([known_matrix==np.array(num_matrix) for num_matrix in df['num_matrix']])) #returns a matrix with where our matrix matches any other word's matrix and adds that column to the df
        df = df.with_columns(truth_sums=pl.Series([sum(truth_matrix) for truth_matrix in df['truth_matrix']])) #returns the sums of truths for all truth matricies
        df = df.filter(pl.col("truth_sums") == df["truth_sums"].max()) #filters the df to only contain the most matching matricies

    #filter for the does_not_contain variable
    if len(does_not_contain) != 0:
        dnc_matrix = np.array(word_to_matrix(does_not_contain))
        df = df.with_columns(dnc_overlap=pl.Series([set(dnc_matrix)&set(num_matrix) for num_matrix in df['num_matrix']]))
        df = df.with_columns(len_of_dnc_overlap=pl.Series([len(dnc_overlap) for dnc_overlap in df['dnc_overlap']]))
        df = df.filter(pl.col("len_of_dnc_overlap") == 0) #filters the df to only contain the most matching matricies

    #filter for the contains variable
    if len(contains) != 0:
        contains_matrix = np.array(word_to_matrix(contains))
        df = df.with_columns(contains_overlap=pl.Series([set(contains_matrix)&set(num_matrix) for num_matrix in df['num_matrix']]))
        df = df.with_columns(len_of_contains_overlap=pl.Series([len(contains_overlap) for contains_overlap in df['contains_overlap']]))
        df = df.filter(pl.col("len_of_contains_overlap") == len(contains)) #filters the df to only contain the most matching matricies

    #filter for the "lnixp" variable (letter_not_in_x_position)
    if lnixp.values():
        for key in lnixp.keys():
            if len(lnixp[key]) != 0:
                lnixp_matrix = np.array(word_to_matrix(lnixp[key]))
                
                df = df.with_columns(lnixp_overlap=pl.Series([set(lnixp_matrix)&set(num_matrix[key-1:key]) for num_matrix in df['num_matrix']]))
                df = df.with_columns(len_of_lnixp_overlap=pl.Series([len(lnixp_overlap) for lnixp_overlap in df['lnixp_overlap']]))
                df = df.filter(pl.col("len_of_lnixp_overlap") == 0) #filters the df to remove any words that contian letters in wrong positions

    #pretty printing vs. normal printing
    if df.height==0:
        print(f'\nNo possible words recorded\n')
    elif pprint == False:
        for i,word in enumerate(df["words"]):
            print(f"{i:4}\t{word}")
    else:
        fixed_width = df['word_length'].max()
        dashes = ('-'*fixed_width)+'-----'
        for i,word in enumerate(df["words"]):
            print(f"+-----+{dashes}+")
            print(f"|{i:4} |  {word.ljust(fixed_width)}   |")
        print(f"+-----+{dashes}+")


not_in_puzzle = "fbake tngwin lve"

lnixp = {1:"",
         2:"",
         3:"",
         4:"",
         5:""}

word_finder("", length=8,
            contains="gamnoial",
            starts_with="",
            ends_with="",
            does_not_contain = "rom"+not_in_puzzle,
            lnixp=lnixp,
            pprint=False)
"""
lnixp2 = {1:"",
          2:"",
          3:"",
          4:"",
          5:""}
word_finder("___r", length=4,
            contains="o",
            starts_with="",
            ends_with="",
            does_not_contain = "d"+not_in_puzzle,
            lnixp=lnixp2,
            pprint=False)

lnixp3 = {1:"",
          2:"",
          3:"",
          4:"",
          5:""}
word_finder("mind", length=4,
            contains="",
            starts_with="",
            ends_with="",
            does_not_contain = "ps"+not_in_puzzle,
            lnixp=lnixp3,
            pprint=False)

lnixp4 = {1:"",
          2:"t",
          3:"",
          4:"",
          5:""}
word_finder("thick", length=5,
            contains="it",
            starts_with="",
            ends_with="",
            does_not_contain = not_in_puzzle+"",
            lnixp=lnixp4,
            pprint=False)

lnixp5 = {1:"",
          2:"",
          3:"",
          4:"",
          5:"r"}
word_finder("erred", length=5,
            contains="er",
            starts_with="",
            ends_with="",
            does_not_contain = not_in_puzzle+"",
            lnixp=lnixp5,
            pprint=False)
"""