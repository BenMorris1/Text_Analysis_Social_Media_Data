import re
import requests
import pandas as pd

# get data from API
request = requests.get('https://projects.fivethirtyeight.com/2020-general-data/presidential_poll_averages_2020.csv')
content = request.text
data = pd.DataFrame([x.split(',') for x in content.split('\n')[1:]], columns=[x for x in content.split('\n')[0].split(',')])

#filter data & concatenate
california_data = data[data['state'] == 'California']
florida_data = data[data['state'] == 'Florida']
illinois_data = data[data['state'] == 'Illinois']
new_york_data = data[data['state'] == 'New York']
texas_data = data[data['state'] == 'Texas']
data_final = pd.concat([california_data, florida_data, illinois_data, new_york_data, texas_data])

#clean data
# convert dates to datetime
data_final['date'] = pd.to_datetime(data_final['modeldate'], format = '%m/%d/%Y')
# drop columns not needed
data_final = data_final.drop(columns = ['cycle','modeldate'])
#
data_final = data_final.replace("Joseph R. Biden Jr.", "Biden")
data_final = data_final.replace("Donald Trump", "Trump")
data_final = data_final.replace("Convention Bounce for Joseph R. Biden Jr.", "Biden")
data_final = data_final.replace("Convention Bounce for Donald Trump", "Trump")


# output
output = data_final.to_string(index=False)
output = re.sub("\n", ";", output.strip())
output = re.sub("\s\s+", ",", output.strip())
output = re.sub(" ", ",", output.strip())
output = output.replace(";,","\n")
print(output)
#print(output[:500]))