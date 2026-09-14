import polars as pl
import numpy as np
from string import ascii_lowercase as letters
letters = letters+"_"

#word_dictionary_file_path stolen from:
# https://github.com/dwyl/english-words/blob/master/words_alpha.txt
word_dictionary_file_path = r"C:\Users\brett.doehring\Downloads\words_alpha.txt".replace("\\","/")

def word_to_matrix(word: str):
    numbers = list(map(int, range(1, len(letters)+2)))
    let_to_num = dict(zip(letters, numbers))
    num_to_let = dict(zip(numbers, letters))

    word = list(word)
    word = np.array(word)
    
    get_val = np.vectorize(lambda word: let_to_num.get(word, 'Unknown'))
    return np.array(get_val(word))

def smush_score_calculator(word: str, score_dict: dict):
    score: int = 0
    spicy_uses: int = 0
    last_life_uses: int = 0
    for letter in set(list(word)):
        letter_count: int = word.count(letter)
        if score_dict[letter]['spicy'] == True:
            spicy_uses: int = letter_count
        if score_dict[letter]['uses'] == letter_count:
            last_life_uses: int = last_life_uses+1
        score: int = score+(letter_count*score_dict[letter]['score'])
    spice_and_llu: int = spicy_uses+last_life_uses
    score: int = score*(spice_and_llu+1)
    return score

def word_finder(known: str, length: int = -1, 
                starts_with: str = "", ends_with: str = "", 
                lis: list = [], lio: list = [], lcc:list=[], #lis = letters in sequence, lio = letters in order, lcc = letter count constraint
                contains: str = "", does_not_contain: str = "", 
                letter_pool: str = "", 
                lnixp: dict[int, str] = {}, 
                pprint: bool = True, 
                smush_dict: dict = {}):

    if smush_dict:
        letter_pool:str = ''.join(smush_dict.keys())

    if len(letter_pool) != 0:
            letter_pool:str = letter_pool.replace(" ","")
            does_not_contain:str = does_not_contain+''.join(set(letters)^set(letter_pool))
            does_not_contain:str = does_not_contain.replace("_",'')
            print(type(letter_pool),letter_pool)
            print(type(does_not_contain),does_not_contain)

    contains:str = contains.replace(" ","")
    contains:str = ''.join(set(contains) | set(''.join(lnixp.values())))
    contains:str = ''.join(set(contains))

    if "_" in known:
        length:int = len(known)
    
    does_not_contain:str = does_not_contain.replace(" ","")
    does_not_contain:str = ''.join(set(does_not_contain) - set(contains))
    does_not_contain:str = ''.join(set(does_not_contain) - set(known.replace('_','')))
    does_not_contain:str = ''.join(set(does_not_contain) - set(starts_with))
    does_not_contain:str = ''.join(set(does_not_contain) - set(ends_with))
    does_not_contain:str = ''.join(set(does_not_contain) - set(''.join(lnixp.values())))
    
    with open(word_dictionary_file_path,'r') as file:
        words:str = file.read()
        words:list[str] = words.split("\n")
    ids:list[int] = range(len(words))
    df = pl.DataFrame({"index": ids,"words": words})
    word_lengths = [ len(word) for word in df["words"] ]
    df = df.with_columns(word_length=pl.Series(word_lengths))
    
    #length check
    df = df.filter(pl.col("word_length") != 0)
    if length != -1:
        df = df.filter(pl.col("word_length") == length)
    else:
        df = df.filter(pl.col("word_length") >= len(known))
        df = df.filter(pl.col("word_length") >= len(starts_with))
        df = df.filter(pl.col("word_length") >= len(ends_with))

    num_matricies:list[np.ndarray] = [word_to_matrix(word) for word in df['words']] #converts strings to matricies

    df = df.with_columns(num_matrix=pl.Series(num_matricies)) #puts the matricies in the df
    if len(known) > 0:
        known_matrix = np.array(word_to_matrix(known)) #creates a code for our known word

    #filter for the known variable
    if len(known) != 0:
        df = df.with_columns(truth_matrix=pl.Series([known_matrix==np.array(num_matrix) for num_matrix in df['num_matrix']])) #returns a matrix with where our matrix matches any other word's matrix and adds that column to the df
        df = df.with_columns(truth_sums=pl.Series([sum(truth_matrix) for truth_matrix in df['truth_matrix']])) #returns the sums of truths for all truth matricies
        df = df.filter(pl.col("truth_sums") == df["truth_sums"].max()) #filters the df to only contain the most matching matricies
    
    #filter for the starts_with variable
    if len(starts_with) != 0:
        starts_with_matrix = np.array(word_to_matrix(starts_with))
        right_end = starts_with_matrix.shape[0]
        df = df.with_columns(truth_matrix=pl.Series([starts_with_matrix==np.array(num_matrix)[0:right_end] for num_matrix in df['num_matrix']])) #returns a matrix with where our matrix matches any other word's matrix and adds that column to the df
        df = df.with_columns(truth_sums=pl.Series([sum(truth_matrix) for truth_matrix in df['truth_matrix']])) #returns the sums of truths for all truth matricies
        df = df.filter(pl.col("truth_sums") == df["truth_sums"].max()) #filters the df to only contain the most matching matricies

    #filter for the ends_with variable
    if len(ends_with) != 0:
        ends_with_matrix = np.array(word_to_matrix(ends_with))
        left_end = ends_with_matrix.shape[0]
        df = df.with_columns(truth_matrix=pl.Series([ends_with_matrix==np.array(num_matrix)[-left_end:] for num_matrix in df['num_matrix']])) #returns a matrix with where our matrix matches any other word's matrix and adds that column to the df
        df = df.with_columns(truth_sums=pl.Series([sum(truth_matrix) for truth_matrix in df['truth_matrix']])) #returns the sums of truths for all truth matricies
        df = df.filter(pl.col("truth_sums") == df["truth_sums"].max()) #filters the df to only contain the most matching matricies

    #filter for the lis variable (lis meaning "letters in sequence")
    # for example, if lis = "ave" then we would get items like "heaven" "cave" or "avenue"
    if len(lis) != 0:
        if lis[0] != "":
            for sequence in lis:
                df = df.filter(sequence in word for word in df['words'])

    #filter for lio variable (lio meaning "letters in order")
    # so if lio = ["so"] we would only get words where so comes before o, but letters can be in between them, like "shore"
    if len(lio) != 0:
        if lio[0] != "":
            for lio_item in lio:
                for lio_letter_index in range(len(lio_item)-1):
                    df = df.filter(pl.col('words').str.contains(lio_item[lio_letter_index]) & pl.col('words').str.contains(lio_item[lio_letter_index+1]) & (pl.col('words').str.find(lio_item[lio_letter_index]) < pl.col('words').str.find(lio_item[lio_letter_index+1])))

    #filer for the lcc variable (lcc meaning "letter count constraint")
    if len(lcc) != 0:
        if lcc[0] != "":
            for lcc_item in lcc:
                lcc_item = lcc_item.replace(" ","")
                lcc_item = lcc_item.lower()
                lcc_item = [lcc_item[0],lcc_item[1:-1],lcc_item[-1]]
                operators = set(['>=','<=','==','>','<'])
                operator = ''.join(set(lcc_item) & operators)
                letter = ''.join(set(lcc_item) & set(letters))
                num = ''.join(set(lcc_item) & set('0123456789'))
                if operator == '==':
                    df = df.filter(pl.col('words').str.count_matches(letter) == int(num))
                elif operator == '>=':
                    df = df.filter(pl.col('words').str.count_matches(letter) >= int(num))
                elif operator == '<=':
                    df = df.filter(pl.col('words').str.count_matches(letter) <= int(num))
                elif operator == '>':
                    df = df.filter(pl.col('words').str.count_matches(letter) > int(num))
                elif operator == '<':
                    df = df.filter(pl.col('words').str.count_matches(letter) < int(num))

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

    if smush_dict:
        for letter in smush_dict.keys():
            if smush_dict[letter]['uses'] != -1:
                df = df.filter(pl.col("words").apply(lambda word: word.count(letter) <= smush_dict[letter]['uses']))
        df = df.with_columns(smush_score=pl.Series([smush_score_calculator(word, smush_dict) for word in df['words']]))
        df = df.sort("smush_score")
        for i,word in enumerate(df["words"]):
            print(f"{i:4}\t{word.ljust(10)}\t{df['smush_score'][i]}")

    else:
        #pretty printing vs. normal printing
        fixed_width = df['word_length'].max()
        if df.height==0:
            print(f'\nNo possible words recorded\n')
        elif pprint == False:
            for i,word in enumerate(df["words"]):
                print(f"{i:4}\t{word.ljust(fixed_width)}")
        else:
            dashes = ('-'*fixed_width)+'-----'
            for i,word in enumerate(df["words"]):
                print(f"+-----+{dashes}+")
                print(f"|{i:4} |  {word.ljust(fixed_width)}   |")
            print(f"+-----+{dashes}+")

not_in_puzzle:str = ""
not_in_puzzle:str = ""+not_in_puzzle
not_in_puzzle:str = ""+not_in_puzzle
not_in_puzzle:str = ""+not_in_puzzle

lnixp:dict[int, str] = {1:"",
                        2:"",
                        3:"",
                        4:"",
                        5:""}

word_finder("", #length=7,
            contains="",
            starts_with="",
            ends_with="",
            lis=[""],
            lio=[""],
            lcc=[""],
            does_not_contain = ""+not_in_puzzle,
            letter_pool = "",
            lnixp=lnixp,
            pprint=True)
"""
            smush_dict={'i':{'score':1, 'uses':2, 'spicy':False},
                        'r':{'score':1, 'uses':0, 'spicy':False},
                        'a':{'score':1, 'uses':1, 'spicy':False},
                        'l':{'score':1, 'uses':0, 'spicy':False},
                        'e':{'score':1, 'uses':-1,'spicy':False},
                        'o':{'score':1, 'uses':2, 'spicy':False},
                        'z':{'score':10,'uses':0, 'spicy':False},
                        'g':{'score':2, 'uses':0, 'spicy':False},
                        'm':{'score':3, 'uses':2, 'spicy':True}})
#"""

"""
lnixp2 = {1:"",
          2:"",
          3:"",
          4:"",
          5:""}
word_finder("the", length=3,
            contains="he",
            starts_with="",
            ends_with="",
            does_not_contain = ""+not_in_puzzle,
            lnixp=lnixp2,
            pprint=False)

lnixp3 = {1:"",
          2:"",
          3:"",
          4:"",
          5:""}
word_finder("__c_p_t", length=7,
            contains="cpat",
            starts_with="",
            ends_with="",
            does_not_contain = ""+not_in_puzzle,
            lnixp=lnixp3,
            pprint=False)

lnixp4 = {1:"",
          2:"",
          3:"",
          4:"",
          5:""}
word_finder("greet", length=5,
            contains="greet",
            starts_with="",
            ends_with="",
            does_not_contain = ""+not_in_puzzle,
            lnixp=lnixp4,
            pprint=False)

lnixp5 = {1:"",
          2:"",
          3:"",
          4:"",
          5:""}
word_finder("sense", length=5,
            contains="sense",
            starts_with="",
            ends_with="",
            does_not_contain = ""+not_in_puzzle,
            lnixp=lnixp5,
            pprint=False)
#"""