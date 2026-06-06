import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "https://www.howstat.com/Cricket/Statistics/IPL/MatchList.asp"

response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

tables = soup.find_all("table")

rows = tables[2].find_all("tr")

match_data = []

for row in rows:

    cols = row.find_all("td")

    data = [col.get_text(strip=True) for col in cols]

    if len(data) >= 7:

        team1 = data[3]
        team2 = data[4]

        if (
            ("Royal Challengers Bengaluru" in team1 and "Chennai Super Kings" in team2)
            or
            ("Chennai Super Kings" in team1 and "Royal Challengers Bengaluru" in team2)
        ):

            print(data)

            match_data.append({
                "Date": data[2],
                "Team1": data[3],
                "Team2": data[4],
                "Venue": data[5],
                "Result": data[6]
            })

# Save CSV
df = pd.DataFrame(match_data)

df.to_csv("match_data.csv", index=False)

print("\nCSV Saved Successfully!")
print(df.head())