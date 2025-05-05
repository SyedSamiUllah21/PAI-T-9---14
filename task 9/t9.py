import re


text = "example of : tokenization, it's  v fun."

tokens = re.findall(r"\b\w+\b", text)

print("Tokens:", tokens)
