from src.models.analyze_team import generate_explanation

#test: python -m pytest unit_tests/test_predict.py -v

def test_generate_explaination():
    features = {
        "num_tanks": 2,
        "avg_attack": 5.0,
        "avg_magic": 5.0,
    }

    result = generate_explanation(features)

    assert "Strong frontline" in result["reasons"]
    assert "Balanced mixed damage (AD/AP)" in result["reasons"]
    assert len(result["weaknesses"]) == 0