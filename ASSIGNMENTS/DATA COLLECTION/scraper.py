import time
import csv
import random
from datetime import datetime
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Setup Selenium with undetected Chrome driver
options = uc.ChromeOptions()
options.add_argument("--headless")  # Run in headless mode
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

driver = uc.Chrome(options=options)

# Define product URLs
PRODUCTS = {
  # Existing Products
    "Samsung Galaxy Z Flip 6": {
        "Amazon": "https://www.amazon.in/Samsung-Galaxy-Smartphone-Storage-Offers/dp/B0DH6ZPHR1?th=1",
        "Flipkart": "https://www.flipkart.com/samsung-galaxy-z-flip6-5g-mint-256-gb/p/itmef6672178d8d7?pid=MOBH2HG9E4NG2BXN",
        "JioMart": "https://www.jiomart.com/p/electronics/samsung-galaxy-flip6-256-gb-12-gb-ram-light-green-mobile-phone/609495704"
    },
    "Samsung Galaxy Z Fold 6": {
        "Amazon": "https://www.amazon.in/Samsung-Galaxy-Fold6-Smartphone-Storage/dp/B0D83ZN7TR?th=1",
        "Flipkart": "https://www.flipkart.com/samsung-galaxy-z-fold6-5g-pink-256-gb/p/itmc5b0d65ae951b?pid=MOBH2HG9BAR2XV3Z"
    },
    "Apple iPhone 16 Pro Max": {
        "Amazon": "https://www.amazon.in/iPhone-16-Pro-Max-256/dp/B0DGHYDZR9/",
        "Flipkart": "https://www.flipkart.com/apple-iphone-16-pro-max-desert-titanium-256-gb/p/itm5a8453e89cbd4?pid=MOBH4DQFTQHZAKAF"
    },
    "OnePlus 13": {
        "Amazon": "https://www.amazon.in/OnePlus-Smarter-512GB-Storage-Arctic/dp/B0DQ8S77V6/",
        "Flipkart": "https://www.flipkart.com/oneplus-13-arctic-dawn-256-gb/p/itm7e57559a9aa18?pid=MOBH8CHPY6Y8PYEQ"
    },
    "Apple Watch Series 9": {
        "Amazon": "https://www.amazon.in/Apple-Cellular-Smartwatch-Stainless-Resistant/dp/B0CHY27FW9?th=1",
        "Flipkart": "https://www.flipkart.com/apple-watch-series-9-gps-41mm-midnight-aluminium-case-sport-loop/p/itm5d18d8f02eff7?pid=SMWGTC2YHN84CW4Y"
    },

    # New Products
    "Apple MacBook Air Laptop": {
        "Amazon": "https://www.amazon.in/2022-Apple-MacBook-Laptop-chip/dp/B0DLHGQ17K",
        "Flipkart": "https://www.flipkart.com/apple-macbook-air-m2-8-gb-256-gb-ssd-mac-os-monterey-mly33hn-a/p/itmdc5308fa78421"
    },
    "Apple AirPods Pro (2nd Generation)": {
        "Amazon": "https://www.amazon.in/Apple-AirPods-Pro-2nd-Generation/dp/B0BDKD8DVD",
        "Flipkart": "https://www.flipkart.com/apple-airpods-pro-2nd-generation-magsafe-case-usb-c-bluetooth/p/itm60c8f5a308352"
    },
    "NIKE Mens Jordan Stay Loyal 3 Running Shoes": {
        "Amazon": "https://www.amazon.in/Jordan-Stay-Loyal-3-White-RED-BLACK-FB1396-101-9UK/dp/B0D6Z478Y1",
        "Flipkart": "https://www.flipkart.com/nike-jordan-stay-loyal-3-sneakers-men/p/itm6d20571705871"
    },
    "Premium Aquatic Eau De Cologne": {
        "Amazon": "https://www.amazon.in/Premium-Eau-de-cologne-200ml/dp/B01MQVWUAN",
        "Flipkart": "https://www.flipkart.com/premium-eau-de-cologne-100-ml/p/itmf3wgvsjxz3eyh"
    },
    "Samsung S24 Ultra": {
        "Amazon": "https://www.amazon.in/Samsung-Galaxy-Smartphone-Titanium-Storage/dp/B0CS5XW6TN",
        "Flipkart": "https://www.flipkart.com/samsung-galaxy-s24-ultra-5g-titanium-gray-512-gb/p/itm463827d6eb2be"
    },
    "Redmi Note 13 Pro": {
        "Amazon": "https://www.amazon.in/Redmi-Fusion-Purple-Storage-Without/dp/B0CXXB7LYC",
        "Flipkart": "https://www.flipkart.com/redmi-note-13-pro-5g-coral-purple-128-gb/p/itm810ee84cdaac6"
    },
    "L'Oreal Paris Hyaluron Moisture 72HR Moisture Filling Shampoo": {
        "Amazon": "https://www.amazon.in/Paris-Moisture-Hyaluronic-Dehydrated-Hyaluron/dp/B0B6XQMCLM",
        "Flipkart": "https://www.flipkart.com/l-oral-paris-hyaluron-moisture-72h-filling-shampoo-dull-dehydrated-hair/p/itme7ac87dd6cbd5"
    },
    "Lakmé 9 to 5 Kajal Twin Pack": {
        "Amazon": "https://www.amazon.in/LAKM%C3%89-Pencil-Eyeconic-Finish-2-Deep/dp/B09L7QWYC3",
        "Flipkart": "https://www.flipkart.com/lakm-9-5-kajal-twin-pack-lasts-upto-24hrs-pack-2/p/itm3904ee14ce26f"
    },
    "Puma Women's Pacific Maze Sneaker": {
        "Amazon": "https://www.amazon.in/Puma-Pacific-Black-Lime-Squeeze-Sneakers/dp/B0BLNTN44H",
        "Flipkart": "https://www.flipkart.com/puma-pacific-maze-wn-s-running-shoes-women/p/itm54d515a0306aa"
    }
}

# Selectors dictionary
selectors = {
    "price": {
        "Amazon": ["span.a-price-whole"],
        "Flipkart": ["div.Nx9bqj.CxhGGd", "div.yRaY8j.A6+E6v"],
    },
    "discount": {
        "Amazon": "span.savingsPercentage",
        "Flipkart": "div.UkUFwK.WW8yVX span",
    },
    "reviews": {
        "Amazon": "div[data-hook='review-collapsed'] span",
        "Flipkart": "div.ZmyHeo div",
    }
}

def clean_price(price):
    """Clean price by removing unwanted characters."""
    if price == "N/A" or not price:
        return "N/A"
    price = price.replace("₹", "").replace(",", "").strip()
    return price if price.isdigit() else "N/A"

def clean_discount(discount):
    """Clean discount by removing unwanted characters."""
    if discount == "N/A" or not discount:
        return "N/A"
    discount = discount.replace("%", "").replace("off", "").strip()
    return discount if discount.isdigit() else "N/A"

def extract_data(url, selector):
    """Extracts text data from a given site using the provided selector."""
    driver.get(url)
    time.sleep(random.uniform(5, 10))  # Random delay to mimic human behavior

    # Scroll to load dynamic content
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(random.uniform(2, 5))  # Random delay after scrolling

    try:
        if isinstance(selector, list):  # Handle multiple selectors (fallback)
            for sel in selector:
                try:
                    elements = WebDriverWait(driver, 10).until(
                        EC.presence_of_all_elements_located((By.CSS_SELECTOR, sel))
                    )
                    if elements:
                        # Filter out empty strings and unwanted elements
                        filtered_elements = [element.text.strip() for element in elements if element.text.strip()]
                        return filtered_elements if filtered_elements else ["N/A"]
                except Exception as e:
                    print(f"Selector {sel} failed: {e}")
            return ["N/A"]
        else:
            elements = WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, selector))
            )
            filtered_elements = [element.text.strip() for element in elements if element.text.strip()]
            return filtered_elements if filtered_elements else ["N/A"]
    except Exception as e:
        print(f"❌ Error extracting data from {url}: {e}")
        # Save page source for debugging
        with open("debug_page.html", "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        print("Saved page source to debug_page.html")
        return ["N/A"]

# CSV file setup
price_csv = "prices_dataset.csv"
review_csv = "reviews_dataset.csv"

with open(price_csv, "w", newline="", encoding="utf-8") as p_file, open(review_csv, "w", newline="", encoding="utf-8") as r_file:
    price_writer = csv.writer(p_file)
    review_writer = csv.writer(r_file)
    
    # Write headers
    price_writer.writerow(["product_name", "price", "discount", "date", "source"])
    review_writer.writerow(["product_name", "reviews", "source"])
    
    for product, sites in PRODUCTS.items():
        for site, url in sites.items():
            print(f"🔍 Scraping {product} from {site}...")

            # Extract price and discount
            price = extract_data(url, selectors["price"][site])
            discount = extract_data(url, selectors["discount"][site])
            date = datetime.now().strftime("%Y-%m-%d")
            
            # Log extracted values before cleaning
            print(f"Extracted price: {price}, discount: {discount}")

            # Clean price and discount
            price = clean_price(price[0]) if price and price != ["N/A"] else "N/A"
            discount = clean_discount(discount[0]) if discount and discount != ["N/A"] else "N/A"
            
            # Format discount values
            if site == "Amazon" and discount != "N/A":
                discount = f"-{discount}%"
            elif site == "JioMart" and discount != "N/A":
                discount = f"{discount} off"
            elif site == "Flipkart" and discount != "N/A":
                discount = f"{discount} off"

            # Save price data to CSV
            price_writer.writerow([product, price, discount, date, site])
            
            # Extract all reviews
            reviews = extract_data(url, selectors["reviews"][site])
            for review in reviews:
                if review and review != "N/A":  # Skip blank reviews
                    review_writer.writerow([product, review, site])
            
            print(f"✅ Done: {product} from {site}")

driver.quit()
print("🎉 Scraping Completed!")