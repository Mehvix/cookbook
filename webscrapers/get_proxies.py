import json

import requests
from bs4 import BeautifulSoup as bs

url = "https://www.us-proxy.org/"
soup = bs(requests.get(url).content, "html.parser")
proxies = []
for row in soup.find("table", attrs={"id": "proxylisttable"}).find_all("tr")[1:]:
    tds = row.find_all("td")
    try:
        ip = tds[0].text.strip()
        port = tds[1].text.strip()
        proxies.append(f"{ip}:{port}")
    except IndexError:
        continue

    with open("proxies.json", "w") as f:
        json.dump(proxies, f)
