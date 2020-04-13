#!/usr/bin/env python
# coding: utf-8

# In[69]:


from bs4 import BeautifulSoup
import requests
import time, os
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from random_user_agent.user_agent import UserAgent
from random_user_agent.params import SoftwareName, OperatingSystem
import re
import random
import string
from datetime import datetime
from progressbar import ProgressBar
import pandas as pd
pbar = ProgressBar()
import collections
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[70]:


print('start ' + str(datetime.now()))


# In[71]:


# you can also import SoftwareEngine, HardwareType, SoftwareType, Popularity from random_user_agent.params
# you can also set number of user agents required by providing `limit` as parameter
software_names = [SoftwareName.CHROME.value]
operating_systems = [OperatingSystem.WINDOWS.value, OperatingSystem.LINUX.value]   
user_agent_rotator = UserAgent(software_names=software_names, operating_systems=operating_systems, limit=100)
# Get list of user agents.
user_agents = user_agent_rotator.get_user_agents()
user_agent = user_agent_rotator.get_random_user_agent()

# Get Random User Agent String.
def new_agent():
    return user_agent_rotator.get_random_user_agent()


# In[72]:


chromedriver = "/Applications/chromedriver" # path to the chromedriver executable
os.environ["webdriver.chrome.driver"] = chromedriver
reviews = "https://apps.apple.com/us/app/tinder/id547702041#see-all/reviews"
#user_agent = {'User-agent': user_agent}
driver = webdriver.Chrome(chromedriver)
driver.get(reviews)
time.sleep(1)


# In[73]:


elem = driver.find_element_by_tag_name("body")

no_of_pagedowns = (5000)

while no_of_pagedowns:
    elem.send_keys(Keys.PAGE_DOWN)
    if no_of_pagedowns <= 2000:
        time.sleep(2*random.random())
    else:
        time.sleep(random.random())
    print(no_of_pagedowns)
    no_of_pagedowns-=1


# In[74]:


html = driver.page_source
soup = BeautifulSoup(html, 'html5lib')


# In[96]:


review_data = []
headers = ['app','time','stars','title','review']


# In[97]:


reviews = soup.findAll(attrs={"class": "ember-view l-column--grid l-column small-12 medium-6 large-4 small-valign-top l-column--equal-height"})
for review in reviews:
    try:
        time = review.find(class_="we-customer-review__date").text
        full_stars = review.find(class_ = "we-star-rating ember-view we-customer-review__rating we-star-rating--large").get('aria-label')
        stars = re.search(r"([0-5]) out of ([0-5])",full_stars).group(1)
        title = review.find(class_ = "we-truncate we-truncate--single-line ember-view we-customer-review__title").text.replace("\n", " ")   
        if review.find('blockquote',attrs={"class": "we-truncate we-truncate--multi-line we-truncate--interactive we-truncate--truncated ember-view we-customer-review__body"}):
            review_txt = review.find('blockquote',attrs={"class": "we-truncate we-truncate--multi-line we-truncate--interactive we-truncate--truncated ember-view we-customer-review__body"})
        else: 
            review_txt = review.find('blockquote',attrs={"class": "we-truncate we-truncate--multi-line we-truncate--interactive ember-view we-customer-review__body"})      
        text = ' '.join([result.getText() for result in review_txt.findAll('p')])

         #APPEND TO DICT
        review_dict = dict(zip(headers, [  
                                    'tinder',
                                    time,
                                    stars,
                                    title,
                                    text
                                 ]))

        review_data.append(review_dict)
        
    except StopIteration:
        raise
    except Exception as e:
        print(e) # or whatever kind of logging you want
        print(review)
        pass


# In[98]:


df_review_data = pd.DataFrame.from_dict(review_data)


# In[99]:


df_review_data.to_csv('tinder.csv')

print('end ' + str(datetime.now()))


# In[ ]:





# In[ ]:




