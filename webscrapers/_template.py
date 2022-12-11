import requests
from bs4 import BeautifulSoup as bs

URL = ""

with requests.get(URL, stream=True) as resp:
    raw = resp.content

html = bs(raw, "html.parser")

print(html.select("head"))
