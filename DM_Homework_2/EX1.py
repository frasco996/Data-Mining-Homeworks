import requests
from bs4 import BeautifulSoup
import csv
import time
import sys
import pandas as pd
import matplotlib.pyplot as plt
from fake_useragent import UserAgent
import re
import math
import numpy as np
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from scipy.sparse import save_npz
import json
# Function to fetch and parse Amazon product data based on keyword and page number
def scrape_amazon_products(keyword, page):
    url = f"https://www.amazon.it/s?k={keyword}&page={page}"
    #url=f"https://www.amazon.it/s?k={keyword}&page={page}/gp/yourstore/home?ref_=nav_cs_ys"
    ua = UserAgent()

    random_user_agent = ua.random
    print("Random User-Agent:", random_user_agent)
    headers = {
        "User-Agent": random_user_agent
    }       

    response = requests.get(url, headers=headers)
    products = []
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        
        for product in soup.find_all(attrs={"data-asin": True}):
            if("sg-col-4-of-24" in product.get("class","N/A")) : # Extract the data-asin value
                
                description_element = product.find('span', class_='a-text-normal')
                description = description_element.text.strip() if description_element else 'N/A'
                
                price_element = product.find('span', class_='a-price-whole')
                price = price_element.text.strip() if price_element else 'N/A'

                prime_status=product.find(attrs={"aria-label": "Amazon Prime"})
                if prime_status:
                    prime_status = "Prime"  
                else:
                    prime_status = "Non-Prime"

                product_url_element = product.find('a', class_='a-link-normal')
                product_url = "https://www.amazon.it" + product_url_element['href'] if product_url_element else 'N/A'
                
                stars_element = product.find('span', class_='a-icon-alt')
                stars = stars_element.text.strip() if stars_element else 'N/A'
                
                reviews_element = product.find('span', class_='a-size-base')
                reviews = reviews_element.text.strip() if reviews_element else 'N/A'

                products.append([description, price, prime_status, product_url, stars, reviews])
            
    else:
        print("Error:",response)
    return products

'''
change the price list from string to float
'''
def parse_elem(l):
    if str(l) != 'nan':
        if '.' in l:
            l = l.replace(".", "")
        if ',' in l:
            l = float(l.replace(",", "."))
    return l
'''
Price Ranges: Determine the price range of products within different categories
'''
def Eda1(df):
    support=[]
    cable=[]
    gpu=[]
    adapter=[]
    termic=[]
    price=[]
    j=0
    
    for i in df['Product Description']:
        p = parse_elem(df['Price'][j])
        if ('supporto' in i or 'Supporto' in i) and  'supporto 5K' not in i:
            support.append(p)
        elif 'cavo' in i or 'Cavo' in i:
            cable.append(p)
        elif 'adattatore' in i or 'Adattatore' in i:
            adapter.append(p)
        elif 'termica' in i or 'termico' in i or  'Termica' in i or 'Termico' in i:
            termic.append(p)
        elif 'gpu' or 'scheda grafica' in i or 'GPU' or 'Scheda Grafica' in i:
            gpu.append(p)
        j=j+1
        price.append(p)
    df['Price'] = price

    print('\n##### Exercise 1 ######')
    print("The ranges are:")
    print(f"Range Gpu: MIN {min(gpu)}, MAX {max(gpu)}")
    print(f"Range Supports for Gpu: MIN {min(support)}, MAX {max(support)}")
    print(f"Range Cable: MIN {min(cable)}, MAX {max(cable)}")
    print(f"Range Thermal paste: MIN {min(termic)}, MAX {max(termic)}")
    print(f"Range Adapter for Gpu: MIN {min(adapter)}, MAX {max(adapter)}")

'''
Customer Reviews: Analyze customer reviews to find the products with the highest ratings
'''
def Eda2(df):
    rev=[]
    stars=[]
    
    for i in df['Number of Reviews']:

        if type(i)!=float:
            x = re.search('[a-zA-Z ]+', i)
        if x:
            rev.append(np.nan)
        else:
            if '.' in str(i):
                i = str(i).replace(".", "")
            rev.append(float(i))
    filtered_ratings = [rating if not np.isnan(rating) else 0 for rating in rev]
    df['Number of Reviews'] = filtered_ratings

    min_rating = min(filtered_ratings)
    max_rating = max(filtered_ratings)
    new_min = 1
    new_max = 5

    # Normalize ratings to the new range with this formula
    normalized_ratings = [(rating - min_rating) / (max_rating - min_rating) * (new_max - new_min) + new_min for rating in filtered_ratings]
   
    stars_list = df['Stars'].str.extract(r'(\d+,\d+) su 5 stelle')
    stars = stars_list[0].str.replace(',', '.').astype(float)
    
    # Replace NaN values with 0
    stars = stars.fillna(0)
    stars = stars.tolist()
    df['Stars'] = stars
    sum=[]
    for i in range(len(df['Number of Reviews'])):
        sum.append((normalized_ratings[i]+stars[i])/2)
    
    df['Real rating'] = sum
    top_10_ratings = df.drop_duplicates(subset='Real rating', keep='first').nlargest(10, 'Real rating')
    print('\n##### Exercise 2 ######')
    print("The 10 top ratings products are:")
    print(top_10_ratings)
    
    
'''
Primeness: is there any relation with the product being ‘Prime’ with its price/rating?

'''
def Eda3(df):
    prime_df = df[df['Prime Status'] == 'Prime']
    notprime_df = df[df['Prime Status'] == 'Non-Prime']
    PrimeLen=len(prime_df)
    notPrimeLen=len(notprime_df)
    PricePrime_df = prime_df['Price'].sum()
    PriceNotPrime_df = notprime_df['Price'].sum()
    StarsPrime_df = prime_df['Stars'].sum()
    StarsNotPrime_df = notprime_df['Stars'].sum()

    avgPricePrime = PricePrime_df/PrimeLen
    avgPriceNotPrime = PriceNotPrime_df/notPrimeLen
    avgStarsPrime = StarsPrime_df/PrimeLen
    avgStarsNotPrime = StarsNotPrime_df/notPrimeLen
    print('\n##### Exercise 3 ######')
    print(f"The number of products with Prime are: {PrimeLen} with an average Price of: {avgPricePrime} and an average of Stars of: {avgStarsPrime}")
    print(f"The number of products with Non-Prime are: {notPrimeLen} with an average Price of: {avgPriceNotPrime} and an average of Stars of: {avgStarsNotPrime}")


'''
Plot the top 10 products in term of ratings
Plot the top 10 products in term of price

'''
def Eda4(df):
    print('\n##### Exercise 4 ######')
    print("Loading Plots of best rating products and higher price products")
    
    # Make a copy of the dataframe
    df_copy = df.copy()

    # Convert 'Real rating' and 'Price' columns to numeric
    df_copy['Real rating'] = pd.to_numeric(df_copy['Real rating'], errors='coerce')
    df_copy['Price'] = pd.to_numeric(df_copy['Price'], errors='coerce')

    # Filter the dataframe to remove duplicates based on 'Real rating' column and keep the first occurrence
    top_10_ratings = df_copy.drop_duplicates(subset='Real rating', keep='first').nlargest(10, 'Real rating')

    # Modify the 'Product Description' column to contain only 5 words
    top_10_ratings['Product Description'] = top_10_ratings['Product Description'].apply(lambda x: ' '.join(re.findall(r'\w+', x)[:5]))

    top_10_price = df_copy.drop_duplicates(subset='Price', keep='first').nlargest(10, 'Price')

    # Modify the 'Product Description' column in the filtered dataframe to contain only 8 words
    top_10_price['Product Description'] = top_10_price['Product Description'].apply(lambda x: ' '.join(re.findall(r'\w+', x)[:8]))

    # Create a bar plot for top 10 products by rating and price
    fig = sp.make_subplots(rows=1, cols=2)
    fig.add_trace(go.Bar(x=top_10_ratings['Product Description'], y=top_10_ratings['Real rating'], name='Real Rating'), row=1, col=1)
    fig.update_xaxes(title_text='Product', row=1, col=1)
    fig.update_yaxes(title_text='Ratings', row=1, col=1)
    
    fig.add_trace(go.Bar(x=top_10_price['Product Description'], y=top_10_price['Price'], name='Price'), row=1, col=2)
    fig.update_xaxes(title_text='Product', row=1, col=2)
    fig.update_yaxes(title_text='Price', row=1, col=2)
    
    fig.update_layout(title_text='Top 10 Products', showlegend=True)
    fig.show()


def main():
    keyword = "gpu"
    num_pages = 7
    print("Taking the amazon products...")
    #IF YOU WANT TO TRY HOW I MAKE THE REQUESTS TO AMAZON, UNCOMMENT THIS PART OF CODE, THE CODE WILL START ANYWAY BECAUSE I ADDED THE FILE THAT I PREVIOUSLY DOWNLOADED CALLED amazon_products.tsv
    
    with open('amazon_products.tsv', 'w', newline='', encoding='utf-8') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')
        writer.writerow(['Product Description', 'Price', 'Prime Status', 'Product URL', 'Stars', 'Number of Reviews'])

        for page in range(1, num_pages + 1):
            product = scrape_amazon_products(keyword, page)
            for product in product:
                writer.writerow(product)
            time.sleep(4) 
    
    # Read the TSV file into a pandas DataFrame
    # Change this path to ./amazon_products.tsv if you want to analyze the current page.
    df = pd.read_csv('./amazon_productsRaw.tsv', sep='\t')

    Eda1(df)

    Eda2(df)

    Eda3(df)

    Eda4(df)
    df.to_csv('./amazon_productsClean.tsv', sep='\t', index=False)
    
    
if __name__ == "__main__":
    main()
