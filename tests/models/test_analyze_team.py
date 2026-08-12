from src.models.analyze_team import generate_explanation

# test: python -m pytest tests/models/test_analyze_team.py

# Isolated mock data rather than real champions.json / feature_matrices.json.
# generate_explanation needs champion_winrates and feature_matrices to
# produce meaningful avg_ap_ratio / avg_role_freq values - without them,
# both default to values that always trigger "Heavy Physical Damage" and
# "off-role" weaknesses regardless of team composition, making a
# zero-weaknesses assertion impossible to satisfy no matter which real
# champions are picked.

CHAMPION_DATA = {
    "TankA": {
        "tags": ["Tank"],
        "info": {"attack": 4, "defense": 8, "magic": 2, "difficulty": 3},
        "stats": {"attackrange": 175, "hp": 650, "armor": 45, "attackdamage": 55},
    },
    "TankB": {
        "tags": ["Tank"],
        "info": {"attack": 4, "defense": 8, "magic": 2, "difficulty": 3},
        "stats": {"attackrange": 175, "hp": 650, "armor": 45, "attackdamage": 55},
    },
    "MageA": {
        "tags": ["Mage"],
        "info": {"attack": 2, "defense": 3, "magic": 9, "difficulty": 6},
        "stats": {"attackrange": 550, "hp": 480, "armor": 20, "attackdamage": 45},
    },
    "MarksmanA": {
        "tags": ["Marksman"],
        "info": {"attack": 8, "defense": 3, "magic": 1, "difficulty": 5},
        "stats": {"attackrange": 600, "hp": 500, "armor": 22, "attackdamage": 58},
    },
    "SupportA": {
        "tags": ["Support"],
        "info": {"attack": 2, "defense": 5, "magic": 5, "difficulty": 4},
        "stats": {"attackrange": 450, "hp": 520, "armor": 30, "attackdamage": 40},
    },
}

# Fixed draft order: Top, Jungle, Mid, ADC, Support
TEAM = ["TankA", "TankB", "MageA", "MarksmanA", "SupportA"]

# Empty dict (not None) -> every champion falls back to a neutral 0.5 win
# rate, which sits between the "strong"/"weak" thresholds and adds neither
# a reason nor a weakness.
CHAMPION_WINRATES = {}

FEATURE_MATRICES = {
    "champ_ap_ratios": {
        "TankA": 0.3,
        "TankB": 0.3,
        "MageA": 0.8,
        "MarksmanA": 0.2,
        "SupportA": 0.4,
    },
    "champ_ap_variances": {},
    # Each champion strongly favors the role it's assigned to here, keeping
    # avg_role_freq well above the 0.5 off-role threshold.
    "role_freqs": {
        "TankA": {"top": 0.8},
        "TankB": {"jungle": 0.8},
        "MageA": {"mid": 0.8},
        "MarksmanA": {"bot": 0.8},
        "SupportA": {"support": 0.8},
    },
    "counter_matrix": {},
}


def test_generate_explaination():
    result = generate_explanation(
        TEAM,
        CHAMPION_DATA,
        CHAMPION_WINRATES,
        FEATURE_MATRICES,
    )

    assert "Strong frontline" in result["reasons"]
    assert "Balanced mixed damage (AD/AP)" in result["reasons"]
    assert len(result["weaknesses"]) == 0