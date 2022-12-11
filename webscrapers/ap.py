import os
import urllib.request
from contextlib import closing

import requests
import wget
from bs4 import BeautifulSoup
from requests import get
from requests.exceptions import RequestException

url = input(
    'Enter the "Past Exam Questions" URL, e.g.\n'
    "https://apcentral.collegeboard.org/courses/ap-computer-science-a/exam/past-exam-questions\n"
)
# url = "https://apcentral.collegeboard.org/courses/ap-computer-science-a/exam/past-exam-questions"
base = url.split("/")[:-4]
base = "/".join(base)
name = url.strip().split("/")[4].split("?")[0]
print("Downloading", name)

try:
    with closing(get(url, stream=True)) as resp:
        raw = resp.content

except RequestException as e:
    print("Error during requests to {0} : {1}".format(url, str(e)))

html = BeautifulSoup(raw, "html.parser")

for div in html.select('div[id="accordion"]'):
    for table in div.select("table"):
        # print(table, "\n\n\n===\n\n\n")
        cap = table.select("caption")
        year = cap[0].text.strip().split(":")[0]

        directory = name + r"\\" + year
        # print(directory)
        if not os.path.exists(directory):
            os.makedirs(directory)

        files = table.select("a")
        for i in files:
            file_name, file_url = i.text, i["href"]
            if "Form" in cap or "form" in cap:
                file_dir = directory + "\\Form B " + file_name + ".pdf"
            else:
                file_dir = directory + "\\" + file_name + ".pdf"

            if "http" not in file_url:
                file_url = base + file_url
            print("Downloading " + file_url + " to " + file_dir)
            if not os.path.isfile(file_dir):
                try:
                    wget.download(file_url, out=file_dir)  # faster
                except urllib.error.HTTPError as e:
                    print("wget 404!", file_url, file_dir, str(e))

                    print("Trying with URLLib...")
                    try:
                        urllib.request.urlretrieve(file_url, file_dir)
                    except:
                        print(
                            "URLLib 404!",
                            file_url,
                            file_dir,
                        )
            else:
                print("File exists already!")
