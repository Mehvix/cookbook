import csv

from selenium import webdriver
from selenium.common.exceptions import *
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

URL = "https://www.compass.com/agents/locations/new-york-ny/21429/?page="
CLASSES = ["agentCard-name", "agentCard-email", "agentCard-phone"]
TIMEOUT = 5

driver = webdriver.Chrome(
    executable_path="/usr/bin/chromedriver",
    service_args=["--verbose", "--log-path=./chromedriver.log"],
)


driver.get(URL)

with open("out.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["name", "email", "phone"])
    writer.writeheader()

    next_button = True
    while next_button:
        try:
            WebDriverWait(driver, TIMEOUT).until(
                EC.presence_of_element_located((By.ID, "agent-card"))
            )
        except TimeoutException:
            print("Timed out waiting for page to load")

        for x in driver.find_elements(by=By.CLASS_NAME, value="agentCard"):
            phone = [""]
            email = [""]
            name = [""]
            vars = [name, email, phone]
            try:
                for var, classname in zip(vars, CLASSES):
                    var[0] = x.find_element(by=By.CLASS_NAME, value=classname).text
            except NoSuchElementException:
                continue

            print(name[0], email[0], phone[0])
            writer.writerow({"name": name[0], "email": email[0], "phone": phone[0]})

        next_button = driver.find_element(
            by=By.CLASS_NAME, value="cx-react-pagination-next"
        )
        next_button.click()
