# Sport theory what do they have in common?
# In this test I will analyse from the website https://www.just-fly-sports.com to determine if the four groups share a common structure. 
# The website contains over 

# import packages 
import pandas as pd
import numpy as np 
import requests
import urllib.request
import time
from bs4 import BeautifulSoup
import re


# loop over each page there are 55 pages
page_count = 1
max_page_count = 55

# initialise article variable
articles = []
while page_count <= max_page_count:

	# read page request
	url = "https://www.just-fly-sports.com/sports-performance-articles/page/{}/".format(page_count)

	# Connect to the URL
	response = requests.get(url)

	# Parse HTML and save to BeautifulSoup object¶
	soup = BeautifulSoup(response.text, "html.parser")

	# find each article on the page
	name_box = soup.find_all('a',class_='more-link button')
	
	# collect href for new urls
	article_urls = []
	for tags in name_box:
		article_urls.append(tags['href'])

	# loop through each article in the page
	for url in article_urls:
		# initialise total article variable
		article = ""

		# connect to url
		response = requests.get(url)

		# parse HTML to BS4
		soup = BeautifulSoup(response.text, "html.parser")

		# find article text only on page
		article_only = soup.find('div',class_="entry-content entry clearfix")
		text = article_only.find_all('p')
		
		# append to one big article 
		for paragraphs in text[:-4]:
			article += paragraphs.text + " "

		articles.append(article)

	print("PAGE {} COMPLETED".format(page_count))
	page_count += 1

print(len(articles))
# analyse using TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

vector = TfidfVectorizer(stop_words='english')
data = vector.fit_transform(articles)
words = vector.get_feature_names()

# Import NMF
from sklearn.decomposition import NMF

# Create an NMF instance: model, let's use 5 groupings.
model = NMF(n_components=5)

# Fit the model to articles
model.fit(data)

# Transform the articles: nmf_features
nmf_features = model.transform(data)

# Print the NMF features
print(nmf_features)

# Create a DataFrame: components_df
components_df = pd.DataFrame(model.components_,columns=words)

# Print the shape of the DataFrame
print(components_df.shape)

for i in range(5):
	print("MOST IMPORTANT WORDS COMPONENT {}:\n{}\n\n".format(i,components_df.iloc[i,:].nlargest()),file=open("5_sport_families.txt","a"))

# Save file for future use 
data_save = pd.DataFrame(data)
data_save.to_csv("just_fly_sports_tfidf.csv",columns=words)

