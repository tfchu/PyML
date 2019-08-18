import pandas as pd
import numpy as np

def hello_pd():
    # note: python dictionary: d = {'index0': 'a', 'index1': 'b', 'index2': 'c'}
    my_dict_list = [{'a': 1, 'b': 2, 'c': 3}, {'a': 10, 'b': 20, 'c': 30}, {'a': 100, 'b': 200, 'c': 300}]
    df = pd.DataFrame(my_dict_list)
    print(df)

# https://stackoverflow.com/questions/17071871/select-rows-from-a-dataframe-based-on-values-in-a-column-in-pandas
def filter_test():
    df = pd.DataFrame({'A': 'foo bar foo bar foo bar foo foo'.split(),
                   'B': 'one one two three two two one three'.split(),
                   'C': np.arange(8), 'D': np.arange(8) * 2})
    print(df)
    print(df[df['A'] == 'foo'])
def main():
    filter_test()

if __name__ == '__main__':
    main()