'''
* tutorial
https://www.datacamp.com/community/tutorials/web-scraping-using-python
(TBD) https://towardsdatascience.com/how-to-web-scrape-with-python-in-4-minutes-bc49186a8460 (for batch download)
(TBD) https://docs.python-guide.org/scenarios/scrape/

* BeautifulSoup
https://www.crummy.com/software/BeautifulSoup/bs4/doc/

* pandas
https://pandas.pydata.org/pandas-docs/stable/index.html

* seaborn
https://seaborn.pydata.org/

* numpy+MKL (note. use platform.architecture() to see required package)
https://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns               # require numpy+MKL (intel math kernel library), not numpy
from urllib.request import urlopen
from bs4 import BeautifulSoup

def main():
    #url = "http://www.hubertiming.com/results/2017GPTR10K"
    url = 'http://localhost/dataset.html'
    html = urlopen(url)                         # http.client.HTTPResponse object
    soup = BeautifulSoup(html, 'lxml')          # soup is the html code
    #print(soup.title)                          # title, e.g. <title>...</title>
    #text = soup.get_text()                     # text inside webpage (no html tags)
    # for link in soup.find_all('a'):
    #     print(link.get('href'))

    # Header  
    list_header = []
    header = soup.find_all('th')                # a list of bs4.element.tag objects: [<th>text1</th>, <th>text2</th>, ...]
    str_cells = BeautifulSoup(str(header), 'lxml').get_text()   # BS(): html code (<html>) with <th> elements inside (rest are removed)
                                                                # a string of a list: '[text1, text2, ...]'
    list_header.append(str_cells)               # list_header: ['[text1, text2, ...]']

    df_header = pd.DataFrame(list_header)       # pandas.core.frame.DataFrame obj
    df_header = df_header[0].str.split(',', expand=True)        # df_header[0]: column 0, a pandas.core.series.Series obj 
                                                                # .str.split(): split with ',' and expand to multiple columns
    
    # Data
    list_rows = []
    rows = soup.find_all('tr')                  # a list of bs4.element.tag objects: [<tr>text1</tr>, <tr>text2</tr>, ...]
                                                # text can be <th>head1</th><th>head2</th>..., or <td>data1</td><td>data2</td>...
    for row in rows: 
        str_cells = BeautifulSoup(str(row.find_all('td')), 'lxml').get_text()   # a string of a list: '[data1, data2, ...]'
        list_rows.append(str_cells)             # ['[data, data, ...]', '[data, data, ...]', ...]

    df_data = pd.DataFrame(list_rows)           # n rows x 1 column data set

    df_data = df_data[0].str.split(',', expand=True)            # split column 0 with ',' and expand
    df_data[0] = df_data[0].str.strip('[')                      # remove '[' in column 0 (labeled 0) and replace it
    df_data[df_data.columns[-1]] = df_data[df_data.columns[-1]].str.strip(']')  # df.columns: a range between 0 and n (n-1 columns)

    # Concat header and data
    df = pd.concat([df_header, df_data])        # concatenate

    # Cleaning
    df = df.rename(columns=df.iloc[0])          # rename label (header) with first row (df.iloc[0])
    df = df.dropna(axis=0, how='any')           # remove the row as long as any column value is None
    df = df.drop(df.index[0])                   # drop a row (axis = 0 , default) or a column (axis = 1)
                                                # df.index[0] is the index (row label) of the DataFrame
                                                # row is called 'index', column is called 'columns'
    df = df.rename(columns={'[Place': 'Place', ' Team]': 'Team'})   # rename 2 labels (header)
    #df_data['Team'] = df_data['Team'].str.strip(']')               # use label to call the column (After label is changed)
    #print(df.head(10))

    # df.info()
    # df.shape

    # visualization: check if chip_time_mins follows normal distribution
    chip_time_list = df[' Chip Time'].tolist()  # hh:mm:ss

    chip_time_mins = []
    for i in chip_time_list:
        hms = i.split(':')
        if len(hms) == 2:
            m, s = hms
            chip_time_mins.append((int(m)*60 + int(s))/60)
        else:
            h, m, s = hms
            chip_time_mins.append((int(h)*3600 + int(m)*60 + int(s))/60)
    df['chip_time_mins'] = chip_time_mins

    x = df['chip_time_mins']
    sns.distplot(x, hist=True, kde=True, rug=False, color='m', bins=25, hist_kws={'edgecolor':'black'})
    plt.show()

    # visualization: compare female and male differences
    f_fuko = df.loc[df[' Gender']==' F']['chip_time_mins']
    m_fuko = df.loc[df[' Gender']==' M']['chip_time_mins']
    sns.distplot(f_fuko, hist=True, kde=True, rug=False, hist_kws={'edgecolor':'black'}, label='Female')
    sns.distplot(m_fuko, hist=False, kde=True, rug=False, hist_kws={'edgecolor':'black'}, label='Male')
    plt.legend()
    plt.show()
    

if __name__ == '__main__':
    main()
