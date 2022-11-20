import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor


data = pd.read_csv("zomato.csv")
# print(data.shape)
# print(data.columns)

'''Data Cleaning'''
# print(data.duplicated().sum()) # 0 Duplicated
# print(data.isnull().sum()) # Null Values present at [approx_cost, cuisines, dish_liked, rest_type, location, phone, rate]

# print(data.info())
# print(data.describe())
# print(data.nunique())

'''remove url and phone columns, it's not helpful for analysis'''
df = data.drop(["url", "phone"], axis=1)

# print(df.duplicated().sum())
df.drop_duplicates(inplace=True)
# print(df.duplicated().sum())

# print(df.isnull().sum())
df.dropna(how="any", inplace=True)
# print(df.isnull().sum())

# print(df.columns)
'''Here some column name is too long like approx_cost(for two people)','listed_in(type)', 'listed_in(city)
so fix it'''''

df = df.rename(columns={"approx_cost(for two people)": "cost", "listed_in(type)": "type", "listed_in(city)": "city"})
# print(df.columns)

'''Main Character is cost so, observ it'''
# print(df["cost"].unique())

'''in cost some elements that will throw error, some elements like "2,100"  in "," so, replace it to like 2100 '''
df["cost"] = df["cost"].apply(lambda x: x.replace(",", ""))
# print(df["cost"].head())

# print(df["cost"].dtype)
'''change the type of cost values object to flot'''
df["cost"] = df["cost"].astype(float)
# print(df["cost"].dtype)

# print(df["rate"].unique())
df = df.loc[df.rate != 'NEW']
'''Remove "/5" from rate values'''
df["rate"] = df["rate"].apply(lambda x: x.replace("/5", ""))
# print(df["rate"].head())

'''Now, Data Set is looking good'''

'''Make Bar-plot for famous restaurant chains'''
# plt.figure(figsize=(15, 10))
# chain = df["name"].value_counts()[:20]
# sns.barplot(x=chain, y=chain.index, palette="Paired")
# plt.title("Most famous restaurents chains")
# plt.xlabel("Number of outlets")
# plt.show()

'''Pie_chart - Table booking(Y/N)'''
# x = df["book_table"].value_counts()
# plt.pie(x, labels=x, autopct="%2.1f%%")
# plt.title("Table Booking")
# plt.legend(x.index)
# plt.show()

'''Restaurants Food Delivery'''
# sns.countplot(x=df["online_order"], data=df)
# plt.title("Whether Restaurants deliver online or Not")
# plt.show()

'''Rate Distribution Plot'''
# plt.figure(figsize=(15, 10))
# sns.histplot(df["rate"], bins=20, kde=True)
# plt.xticks(rotation="vertical")
# plt.show()

'''Cost Vs rating Scatter'''
# plt.figure(figsize=(15,7))
# sns.scatterplot(x="rate", y='cost', hue='online_order', data=df)
# plt.xticks(rotation=90)
# plt.show()

# print("Min. Rating", df["rate"].min())
# print("Max. Rating", df["rate"].max())
'''change the type of Rating'''
df["rate"] = df["rate"].astype(float)

'''rating counts between 1 and 2.'''
# print("Rating Range: 1 to 2---", ((df["rate"] >= 1) & (df["rate"] <= 2)).sum())
# print("Rating Range: 2 to 3---", ((df["rate"] >= 2) & (df["rate"] <= 3)).sum())
# print("Rating Range: 3 to 4---", ((df["rate"] >= 3) & (df["rate"] <= 4)).sum())
# print("Rating Range: 4 to 5---", (df["rate"] >= 4).sum())

# ranges = ["Rating Range: 1 to 2", "Rating Range: 2 to 3", "Rating Range: 3 to 4", "Rating Range: 4 to 5"]
'''Make pie-chart for rating count'''
# part = [((df["rate"] >= 1) & (df["rate"] <= 2)).sum(),
#         ((df["rate"] >= 2) & (df["rate"] <= 3)).sum(),
#         ((df["rate"] >= 3) & (df["rate"] <= 4)).sum(),
#         (df["rate"] >= 4).sum()]
# label = ['1', '2', '3', '>4']
# colors = ['red', 'blue', 'green', 'yellow']
# plt.figure(figsize=(15, 10))
# plt.pie(part, colors=colors, labels=label, autopct="%2.1f%%", shadow=True)
# plt.title("Percentage of Restaurants Rating")
# plt.legend(ranges)
# plt.show()

'''Count-plot Service Type'''
# plt.figure(figsize=(15, 7))
# sns.countplot(x=df["type"], data=data)
# plt.title('Type of Service')
# plt.show()

'''most famous restaurants'''
# plt.figure(figsize=(15, 7))
# chains = df['name'].value_counts()[:20]
# sns.barplot(x=chains, y=chains.index, palette='Set1')
# plt.title("Most famous restaurant chains", size=20, pad=20)
# plt.xlabel("Number of outlets", size=15)
# plt.show()

'''online_order data replace 1 & 0'''
print(df.online_order.value_counts())
df.online_order[df.online_order == "Yes"] = 1
df.online_order[df.online_order == "No"] = 0

print(df.online_order.value_counts())
df.online_order = pd.to_numeric(df.online_order) #convert argument to a numeric type.
print(df.online_order.value_counts())

df.book_table[df.book_table == 'Yes'] = 1
df.book_table[df.book_table == 'No'] = 0

df.book_table = pd.to_numeric(df.online_order)
print(df.book_table.value_counts())

'''Object transform in number Use LableEncoder'''
le = LabelEncoder()

df.location = le.fit_transform(df.location)
df.rest_type = le.fit_transform(df.rest_type)
df.cuisines = le.fit_transform(df.cuisines)
df.menu_item = le.fit_transform(df.menu_item)

# print(df.columns)

'''Make new data set file'''
new_dataset = df.loc[:, ['online_order', 'book_table', 'votes',
                         'location', 'rest_type', 'cuisines',
                         'cost', 'menu_item']]
new_dataset.to_csv("Zomato_df.csv")

x = new_dataset
# print(x.head())

y = df['rate']
# print(y)

'''Model:- LinearRegression()'''
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)
lr_model = LinearRegression()
lr_model.fit(x_train, y_train)

"Prediction with r2_score"
y_pred = lr_model.predict(x_test)
print("LinearRegression Model", r2_score(y_test, y_pred))# 0.20289143117748099

'''Model:- RandomForestRegressor'''
rfr_model = RandomForestRegressor()
rfr_model.fit(x_train, y_train)
y_rfr_predict = rfr_model.predict(x_test)
print("Random Forest Regressor model", r2_score(y_test, y_rfr_predict))# 0.905538781723273

'''Model:- Extra Trees Regressor'''
etr_model = ExtraTreesRegressor(n_estimators=120)
etr_model.fit(x_train, y_train)
y_etr_predict = etr_model.predict(x_test)
print("Extra Trees Regression Model", r2_score(y_test, y_etr_predict))# 0.9323699765726636

'''Here, we get high accuracy in Extra Trees Regressor Model then Random Forest Model'''

