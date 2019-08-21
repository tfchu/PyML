import random
from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd

def web_scraping():
    url = 'http://localhost/pokemon_stat.html'
    #url = 'https://pokemondb.net/pokedex/all'
    html = urlopen(url)                         # http.client.HTTPResponse object
    soup = BeautifulSoup(html, 'lxml')          # soup is the html code

    list_rows = []
    rows = soup.find_all('tr')
    for row in rows:
        td_list = row.find_all('td')
        is_data = len(td_list)
        if is_data:
            str_cell = BeautifulSoup(str(td_list), 'lxml').get_text()
        else:
            str_cell = BeautifulSoup(str(row.find_all('th')), 'lxml').get_text()
        list_rows.append(str_cell)

    df = pd.DataFrame(list_rows)
    df = df[0].str.split(',', expand=True)  # expand 1 column to multiple columns with splitting ','
    #df[0] = df[0].str.strip('[')
    #df[df.columns[-1]] = df[df.columns[-1]].str.strip(']')
    df = df.rename(columns=df.iloc[0])      # set column name to name of 1st row
    df = df.drop(df.index[0])               # remove first row
    # remove extra space in header
    df = df.rename(columns={'[#' : '#', ' Name': 'Name', ' Type': 'Type', ' Total': 'Total', ' HP': 'HP', ' Attack': 'Attack', ' Defense': 'Defense', ' Sp. Atk': 'Sp_Atk', ' Sp. Def': 'Sp_Def', ' Speed]': 'Speed'})
    # strip unwanted characters
    df['#'] = df['#'].str.strip('[')            # strip '['
    df['Type'] = df['Type'].str.strip()         # strip whitespace
    df['Speed'] = df['Speed'].str.strip(']')    # strip ']'
    # type conversion
    df['#'] = df['#'].astype(int)                           # '#' column to int (instead of string)
    df['Total'] = df['Total'].astype(int)
    df['HP'] = df['HP'].astype(int)
    df['Attack'] = df['Attack'].astype(int)
    df['Defense'] = df['Defense'].astype(int)
    df['Sp_Atk'] = df['Sp_Atk'].astype(int)
    df['Sp_Def'] = df['Sp_Def'].astype(int)
    df['Speed'] = df['Speed'].astype(int)
    #print(df.head(10))
    print('total df row: {}, columns: {}'.format(len(df.index), len(df.columns)))
    return df

# print a list of Pokemon, each represented by features vector [attack, sp_atk]
# use the print-out to set training and validation set
# e.g. 
# water: [[48, 50], [63, 65], ...]
# normal: [[45, 35], [60, 50], ...]
def get_features():
    def get_features_of_a_type(type):
        type_features = list()
        df_type = df[df['Type'].str.contains(type)]
        #df_type = df[df['Type'] == type]
        print('{} df row: {}, columns: {}'.format(type, len(df_type.index), len(df_type.columns)))
        atk_list = df_type['Attack'].tolist()
        sp_atk_list = df_type['Sp_Atk'].tolist()
        # more features
        # def_list = df_type['Defense'].tolist()
        # sp_def_list = df_type['Sp_Def'].tolist()
        # spd_list = df_type['Speed'].tolist()

        for i in range(len(df_type.index)): 
            type_features.append([atk_list[i], sp_atk_list[i]])
            # type_features.append([atk_list[i], sp_atk_list[i], def_list[i], sp_def_list[i], spd_list[i]])
        print(type_features)
        return atk_list, sp_atk_list

    df = web_scraping()
    df = df[df['#'] <= 400]             # <= 400 as training set, > 400 as validation set
    x1, y1 = get_features_of_a_type('Water')
    x2, y2 = get_features_of_a_type('Normal')

    # fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    # axs[0].plot([x1], [y1], 'o', color='blue')
    # axs[1].plot([x2], [y2], 'o', color='red')
    # axs[0].set(xlabel='Attack', ylabel='Sp_Atk', title='WATER')
    # axs[1].set(xlabel='Attack', ylabel='Sp_Atk', title='NORMAL')
    # axs[0].grid(True)
    # axs[1].grid(True)

    # plt.plot([x1], [y1], 'o', color='blue')
    # plt.plot([x2], [y2], 'o', color='red')
    # plt.xlabel('Attack')
    # plt.ylabel('Sp_Atk')
    # plt.xlim(0, 200)
    # plt.ylim(0, 200)
    # plt.show()    

def fake_features():
    feature_list = list()
    for i in range(50):
        feature_list.append([float(random.randint(120, 181)), float(random.randint(120, 181))])
    print(feature_list)
    feature_list.clear()
    for i in range(50):
        feature_list.append([float(random.randint(20, 81)), float(random.randint(20, 81))])
    print(feature_list)    

def main():
    get_features()

if __name__ == '__main__':
    main()