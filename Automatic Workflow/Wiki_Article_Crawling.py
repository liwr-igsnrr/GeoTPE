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

def crawler_wiki():
    Titles = []
    Lead_Sections = []
    Hyperlinked_Phrases = []
    Hyperlinks = []

    headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.132 Safari/537.36"}

    read_file = pd.read_excel("geo_phrases.xlsx")
    geo_phrases = list(read_file['geo_phrases'])

    for geo_phrase in geo_phrases:
        phrase_process="_".join(geo_phrase.split())
        link_full="https://en.jinzhao.wiki/wiki/"+phrase_process
        Crawling= requests.get(link_full, headers=headers, verify=False, timeout=30)
        html_article = etree.HTML(Crawling.text)

        # 标题
        First_Heading = "".join(html_article.xpath("//h1[@id='firstHeading']/text()"))
        Titles.append(First_Heading)

        # 摘要
        lead_section_test = html_article.xpath("//p")[0].xpath("string(.)")

        if len(lead_section_test) <= 5:
            lead_section = html_article.xpath("//p")[1].xpath("string(.)")
            remove_brackets = re.sub(r'\[(.*?)\]', "", lead_section).replace("\n", " ")
            remove_comma = re.sub(r'\((.*?)\)', "", remove_brackets)
            Lead_Sections.append(remove_comma)

            linked_phrase = " ; ".join(html_article.xpath("//p[2]/a/text()"))
            phrase_link = " ; ".join(html_article.xpath("//p[2]/a/@href"))

            Hyperlinked_Phrases.append(linked_phrase)
            Hyperlinks.append(phrase_link)

        else:
            lead_section = lead_section_test
            remove_brackets = re.sub(r'\[(.*?)\]', "", lead_section).replace("\n", " ")
            remove_comma = re.sub(r'\((.*?)\)', "", remove_brackets)
            Lead_Sections.append(remove_comma)


            linked_phrase = " ; ".join(html_article.xpath("//p[1]/a/text()"))
            phrase_link = " ; ".join(html_article.xpath("//p[2]/a/@href"))

            Hyperlinked_Phrases.append(linked_phrase)
            Hyperlinks.append(phrase_link)

    wp = pd.DataFrame({"title": Titles, "lead sections": Lead_Sections, "Hyperlinked Phrases": Hyperlinked_Phrases,"Hyperlinks":Hyperlinks})
    wp.to_excel("E:\\Wikipedia_article.xlsx", index=False)

if __name__ == '__main__':
    crawler_wiki()