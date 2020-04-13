#scrapy crawl get_userlogin -a user=grongrong101 -a password=26172617!
#https://stackoverflow.com/questions/29179519/submit-form-that-renders-dynamically-with-scrapy

import json

import scrapy
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time, os

class Okcupid_Main(scrapy.Spider):
    name = "get_userlogin"
    allowed_domains = ["okcupid.com"]
    start_urls = ["https://www.okcupid.com/login"]

    def parse(self, response):
        profile = response.xpath('//a[text()='View Profile']/text()').extract()

    yield {'profile':profile}
