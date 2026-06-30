import os
import requests
from urllib.parse import quote
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("RIOT_API_KEY")

if not API_KEY:
    raise ValueError("Missing RIOT_API_KEY. Run: export RIOT_API_KEY='your_key_here'")

# Change these
GAME_NAME = "MunchyPunchyLOL"
TAG_LINE = "TTV1"

region = "americas"

game_name_encoded = quote(GAME_NAME)
tag_line_encoded = quote(TAG_LINE)

url = f"https://{region}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{game_name_encoded}/{tag_line_encoded}"

headers = {
    "X-Riot-Token": API_KEY
}

response = requests.get(url, headers=headers)

print("Status code:", response.status_code)
print(response.text)

if response.status_code == 200:
    data = response.json()
    print("\nYour PUUID is:")
    print(data["puuid"])