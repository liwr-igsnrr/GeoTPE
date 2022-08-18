import re
import pandas as pd
import itertools
from collections.abc import Iterable
import urllib.request
import requests
from lxml import etree
from fake_useragent import UserAgent
import urllib3
import time
from time import sleep
import json
import itertools
import re
import ast
urllib3.disable_warnings()
import copy
import itertools

def phrases_filtered():
    geo_phrases=[]

    headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.132 Safari/537.36"}

    read_file = pd.read_excel("Wikipedia_article.xlsx")
    phrases = list(read_file['Hyperlinked Phrases'])

    for phrase_sem in phrases:
        phrases_split=phrase_sem.split(" ; ")
        for phrase in phrases_split:
            phrase_process="_".join(phrase.split())
            link_full="https://en.jinzhao.wiki/wiki/"+phrase_process
            Crawling= requests.get(link_full, headers=headers, verify=False, timeout=30)
            html_article = etree.HTML(Crawling.text)

            Infobox_label = "".join(html_article.xpath("//table[@class='infobox']/tbody/tr/th/text()"))
            if len(Infobox_label)==0:
               geo_phrases.append(phrase)

    wp = pd.DataFrame({"geo_phrases": geo_phrases})
    wp.to_excel("Wiki_geo_words.xlsx", index=False)

if __name__ == '__main__':
    phrases_filtered()