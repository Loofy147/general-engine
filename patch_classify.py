with open("engine/core.py", "r") as f:
    code = f.read()

target = """def classify_model(hybrid_score, theta_promote=0.85):
    \"\"\"
    Classify model as PROMOTED, INHIBITED, or GAP.
    \"\"\"
    if hybrid_score > theta_promote:
        return "PROMOTED"
    elif hybrid_score < 0.3:
        return "INHIBITED"
    else:
        return "GAP\"\"\""""

# Let's search and replace with regex or clean string manipulation.
# Let's view the exact content of classify_model first.
