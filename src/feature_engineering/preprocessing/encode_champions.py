import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def get_all_champions():
    # Fetch the latest live patch version dynamically from Riot
    version_url = "https://ddragon.leagueoflegends.com/api/versions.json"
    latest_version = requests.get(version_url).json()[0]

    data_url = f"https://ddragon.leagueoflegends.com/cdn/{latest_version}/data/en_US/champion.json"
    response = requests.get(data_url).json()

    # Extract names
    champion_data = response["data"]
    champion_names = [info["name"] for info in champion_data.values()]

    # Sort alphabetically
    return sorted(champion_names)


all_champions = get_all_champions()

# --- 2. MOCK MATCH DATA SETUP ---
# (Replace this section with your actual raw dataframe)
np.random.seed(42)
num_samples = 100
mock_data = {
    # Even if our data only randomly hits a few champions...
    "champion": np.random.choice(["Ahri", "Jinx", "Thresh"], num_samples),
    "role": np.random.choice(
        ["Top", "Jungle", "Mid", "ADC", "Support"], num_samples
    ),
    "queue_type": np.random.choice(["Ranked Solo", "Flex", "Normal"], num_samples),
    "rank": np.random.choice(
        ["Iron", "Bronze", "Silver", "Gold", "Platinum", "Diamond", "Challenger"],
        num_samples,
    ),
    "win": np.random.choice([0, 1], num_samples),
}
df = pd.DataFrame(mock_data)


# --- 3. THE ENCODING PIPELINE ---

# Step A: Lock in the full universe of champions using your API list
df["champion"] = pd.Categorical(df["champion"], categories=all_champions)

# Step B: One-hot encode Champion, Role, and Queue Type
# Because 'champion' is an explicit Category, pandas builds vectors for ALL API champions automatically.
df_encoded = pd.get_dummies(
    df, columns=["champion", "role", "queue_type"], dtype=int
)

# Step C: Ordinal map the Ranks
rank_mapping = {
    "Iron": 0,
    "Bronze": 1,
    "Silver": 2,
    "Gold": 3,
    "Platinum": 4,
    "Diamond": 5,
    "Master": 6,
    "Grandmaster": 7,
    "Challenger": 8,
}
df_encoded["rank"] = df_encoded["rank"].map(rank_mapping)


# --- 4. CREATE FEATURE MATRIX AND SPLIT ---

X = df_encoded.drop(columns=["win"])
y = df_encoded["win"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# --- 5. VERIFICATION ---
print("\n--- FINAL MATRIX VERIFICATION ---")
print(f"X_train shape: {X_train.shape}")
print(
    f"Number of feature columns: {X_train.shape[1]}"
)  # Will be 160+ columns for champions + roles + queues + rank

# Verify a specific champion vector column exists even if nobody played them in the mock data
print(f"Does 'champion_Aatrox' column exist? {'champion_Aatrox' in X_train.columns}")