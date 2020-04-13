from bs4 import BeautifulSoup
import requests
import time, os
from datetime import datetime
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
import numpy as np
pbar = ProgressBar()
#%config InlineBackend.figure_formats = ['retina']
import collections

from multiprocessing import Pool

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

def scrape_yelp(user):
    #for user in user_list:
    try:
        #NEW PAGE
        user_agent=new_agent()
        link = 'https://www.yelp.com/user_details?userid='+str(user)
        page = requests.get(link, user_agent).text
        soup = BeautifulSoup(page, 'html5lib')

        #COLLECT VARIABLES
        if soup.find(class_ = 'badge-bar u-space-r1'):
            elite = soup.find(class_ = 'badge-bar u-space-r1').text
        else:
            elite = None

        return user, elite
        #GO TO SLEEP
        time.sleep(random.random()*3)

        print(datetime.now(),user)

    except StopIteration:
        raise
    except Exception as e:
        print(e) # or whatever kind of logging you want
        pass

def write_csv(records,start,end):
    file_name = str(start)+"_"+str(end)+".csv"
    pd.DataFrame(records).to_csv('../project3/data/yelp_dataset_2020/'+file_name, index=None, header=None)

df_user_18 = pd.read_csv('../project3/need_to_scrape.csv', #skiprows = range(1, 1000000), nrows=1000000
                        )
yelp_users = df_user_18['user_id'].unique()
yelp_users = list(yelp_users)
random.seed(4)
yelp_users = random.sample(yelp_users, k=200001)

chromedriver = "/Applications/chromedriver" # path to the chromedriver executable
os.environ["webdriver.chrome.driver"] = chromedriver
users_all = "https://www.yelp.com/"
#user_agent = {'User-agent': user_agent}
#driver = webdriver.Chrome(chromedriver)
#driver.get(user_all)
time.sleep(1)

start = 0
#for end in range(10000,len(yelp_users), 10000):
for end in range(1000,200001, 1000):
    print(datetime.now(), time.time, start, end)
    selected_yelp = yelp_users[start:end]
    p = Pool(10)  # Pool tells how many at a time
    records = p.map(scrape_yelp, selected_yelp)
    p.terminate()
    p.join()
    write_csv(records,start,end)
    records = ''
    start = end
    time.sleep(random.random()*6)
