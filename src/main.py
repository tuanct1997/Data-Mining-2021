import json
import tensorflow as tf
import pandas as pd

PATH = "../data/"
f = open(PATH+"patients.json")

check = json.loads(f.read())

df = pd.DataFrame(check)

print(df.info())
print(df.head(5))