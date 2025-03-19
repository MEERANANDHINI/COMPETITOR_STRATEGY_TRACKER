import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# Set up Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run Chrome in headless mode
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# Set correct paths
chrome_options.binary_location = "/usr/bin/google-chrome-stable"
service = Service("/usr/bin/chromedriver")

# Launch Chrome
driver = webdriver.Chrome(service=service, options=chrome_options)

# Store names and their URLs
stores = {
    "samsung.com": "https://www.samsung.com/in/smartphones/galaxy-z-flip6/",
    "Sathya Retails": "https://www.sathya.in/samsung-galaxy-z-flip6",
    "Fliptwirls": "https://www.fliptwirls.com/products/samsung-galaxy-z-flip6",
    "Croma": "https://www.croma.com/search/?q=samsung+galaxy+z+flip6"
}

# Store extracted data
data = []

for store, url in stores.items():
    driver.get(url)
    time.sleep(5)  # Wait for page to load

    try:
        price_element = driver.find_element(By.XPATH, "//span[contains(text(),'₹')]")  # Adjust XPath if needed
        price = price_element.text.strip()
    except:
        price = "Not Found"

    data.append({"Store": store, "Price": price})

driver.quit()

# Save to CSV
df = pd.DataFrame(data)
df.to_csv("samsung_zflip6_prices.csv", index=False)
print("✅ CSV file saved successfully: samsung_zflip6_prices.csv")
