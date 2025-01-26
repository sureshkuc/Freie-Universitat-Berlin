import numpy as np
import pandas as pd

table = pd.read_csv('/home/sofya/Documents/UNI/ML/ml2020/project3/actor_err.csv')

table # first let's have a look. first thing we see is line 24 uses some other separator

table.at[24] = table.loc[24].Actor.split(':')

table # we also see that there are two similar columns actor and Firstname + Lastname. We see some of lastnames are missing. maybe we can compare them to the actor names, if noone uses psedonim left out the 'actor' column

table[table.Actor != table.Firstname+ ' '+ table.Lastname] # so wee see there are no pseudonames, just misspellings or caps; correct that
names_index = table[table.Actor != table.Firstname+ ' '+ table.Lastname].index

table.loc[names_index]

table.at[names_index, ['Firstname']] = table.loc[names_index].Actor.str.split(' ').str[0]
table.at[names_index, ['Lastname']] = table.loc[names_index].Actor.str.split(' ').str[1]

table



# we also see that in raw 18 values are enclaused, correct that
table.columns

table.at[18] = table.loc[18].str.replace("'", '')

#Price column has some different notations and nan, since it binary variable we change it for true and false
table['Price'] = ['no' if pd.isnull(i) or i[-1]!= 's' else('yes') for i in table.Price]


# finally resort column to start wit name and surname, remove column actor since it's redundant and check for dups
table = table[['Firstname','Lastname', 'Total Gross', 'Number of Movies', 'Average per Movie','#1 Movie', 'Gross', 'Price']]
set(table.duplicated().to_list()) # no duplications
table # I can't say what gross is

table.dtypes

table['Total Gross'] = table['Total Gross'].astype('float')
table['Number of Movies'] = table['Number of Movies'].astype('float')
table['Average per Movie'] = table['Average per Movie'].astype('float')
table['Gross'] = table['Gross'].astype('float')

table.sum()# from the available hint I understand that the Gross is always positiv and price nan equals no

table['Gross'] = abs(table['Gross'])

table

print(table.sum())
table.Price.value_counts()

