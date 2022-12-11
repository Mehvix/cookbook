import csv

import requests
from bs4 import BeautifulSoup as bs

url = "https://universityresearchpark.org/resident-companies/"
soup = bs(requests.get(url).content, "html.parser")

f = open("file.csv", "w")
writer = csv.writer(f)
soup = soup.find("div", attrs={"class": "cn-list-body"})
soup = soup.find_all("div", attrs={"class": "cn-entry"})

for i in soup:
    try:
        store = i.find("div", attrs={"cn-left"}).find("a").text
        phone = (
            i.find("span", attrs={"cn-phone-number"})
            .find("spamv n", attrs={"value"})
            .text
        )
        email = i.find("span", attrs={"cn-email-address"}).find("a").text
        writer.writerow([store, phone, email])
    except AttributeError:
        pass
