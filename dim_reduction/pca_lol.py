'''
https://leemeng.tw/essence-of-principal-component-analysis.html
'''
import pandas as pd
from IPython.display import display

# 將事先預處理過的英雄數據讀取為 pandas 的 DataFrame
# 你可以從同樣的 url 獲得本文 demo 的數據
df = pd.read_csv("https://bit.ly/2FkIaTv", sep="\t", index_col="名稱")

print("df.shape:", df.shape)

# 展示前 5 rows
print("前五名英雄數據：")
display(df.head(5))

# 顯示各特徵的平均與標準差
print("各特徵平均與標準差：")
df_stats = df.describe().loc[['mean', 'std']]
df_stats.style.format("{:.2f}")
display(df_stats)