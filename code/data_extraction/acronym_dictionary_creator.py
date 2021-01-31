import requests
import pandas as pd
from bs4 import BeautifulSoup

# create empty lists
acronyms = []
meanings = []

# create soup from website html
main_link = 'https://www.noslang.com/dictionary/'
data = requests.get(main_link)
soup = BeautifulSoup(data.content, 'html.parser')

# find all links on the page
pages = soup.find("ul", class_ = "list-inline text-center").find_all("li")

# loop through the links
for page in pages:
    # create soup for each sub link
    link = "https://www.noslang.com" + str(page.a).split('"')[1]
    page_data = requests.get(link)
    soup = BeautifulSoup(page_data.content, 'html.parser')
    # find all rows on each page
    items = soup.find_all("div", class_="dictionary-word")
    # loop though items on each page
    for item in items:
        # extract the acronym and meaning
        acronym = item.a.text[:-2]
        meaning = item.find("span", class_="dictonary-replacement").text
        # append the attributes to the list variables
        acronyms.append(acronym)
        meanings.append(meaning)
        
# create dataframe and save to csv        
acronym_dictionary = pd.DataFrame({'Acronym':acronyms, 'Meaning':meanings})
acronym_dictionary.to_csv('../../data/Tools/acronym_dictionary.csv')