# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| code-fold: false
#| code-summary: "Show the code" 
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
iris = fetch_ucirepo(id=53) 
  
# data (as pandas dataframes) 
X = iris.data.features 
y = iris.data.targets 
  
# metadata 
print(iris.metadata) 
  
# variable information 
print(iris.variables) 
#
#
#
#
#
#
#
#
#| code-fold: false
X.head()
#
#
#
#
#
#
#
#| code-fold: false
#| warning: false
import bs4                      # library for BeautifulSoup
from bs4 import BeautifulSoup   # import the BeautifulSoup object
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from seaborn import set_style
set_style("whitegrid")
#
#
#
#
#| code-fold: false
html_doc="""
<!DOCTYPE html>
<html lang="en">
<head>
    <title>My Dummy HTML Document</title>
</head>
<body>
    <h1>Welcome to My Dummy HTML Document</h1>
    <p>This is a paragraph in my dummy HTML document.</p>
    <a href="https://mrislambd.github.io/blog" class="blog" id="blog"> Blog </a>
    <a href="htpps://mrislambd.github.io/research" class="research" id="research"> Research </a>
</body>
</html>
"""
#
#
#
#
#
#| code-fold: false
soup=BeautifulSoup(html_doc, features='html.parser')
```  
#
#
#| code-fold: false 
soup.html.head.title
#
#
#
#
#
#| code-fold: false 
soup.title 
```    
#
#
#
#| code-fold: false 
soup.title.text
#
#
#
#
#
#| code-fold: false  
soup.title.parent
#
#
#
#
#
#| code-fold: false  
soup.a
#
#
#
#
#| code-fold: false  
soup.a['class']
#
#
#
#
#| code-fold: false  
soup.findAll('a')
#
#
#
#
#| code-fold: false  
soup.findAll('a')[0]['id']
#
#
#
#
#| code-fold: false  
soup.p.text 
#
#
#
#
#| code-fold: false  
[h['href'] for h in soup.findAll('a')]
#
#
#
#
#
#
#
#| code-fold: false
import requests
response = requests.get(url="https://research.nvidia.com/graduate-fellowships/archive")
response.status_code
#
#
#
#
#
#| code-fold: false
soup = BeautifulSoup(response.text, 'html.parser')
#
#
#
#
#
#| code-fold: false
pf = soup.find_all("div", class_="archive-group")
#
#
#
#
#
#| code-fold: false
pf[0]
#
#
#
#
#
#| code-fold: false
#| code-overflow: scroll
data=[]

for group in pf:
    year = group.find(
        "h4",class_="archive-group__title"
        ).get_text(strip=True).split()[0]

    fellows = group.find_all("div", class_="views-row")
    for fellow in fellows:
        name = fellow.find(
            "div", class_="views-field-title"
            ).get_text(strip=True) 
        institute = fellow.find(
            "div", class_="views-field-field-grad-fellow-institution"
            ).get_text(strip=True)

        data.append({"Name": name, "Year": year, "Institute": institute})

data=pd.DataFrame(data)
data.head()
#
#
#
#
#
#| code-fold: false
# Count the number of fellows each year
year_counts = data['Year'].value_counts().sort_values(ascending=False)
# Create a DataFrame where years are columns and counts are values in the next row
year_data = {
    'Year': year_counts.index,
    'Count': year_counts.values
}
# Create the DataFrame
year_data_counts = pd.DataFrame(year_data)

# Transpose the DataFrame and reset index to get years as columns
year_data_counts = year_data_counts.set_index('Year').T

# Display the DataFrame
print(year_data_counts)
#
#
#
#
#

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
