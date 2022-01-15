import json
import tensorflow as tf
import pandas as pd
from json import JSONDecoder, JSONDecodeError
import timeit

PATH = "../data/"



def create_necessary_set(path,file):
	f = open(path)
	data = json.loads(f.read())
	print(data.keys())
	print("!!!!!!!!!!!!")
	conditions_map = pd.DataFrame(data["Conditions"])
	print((conditions_map.info()))
	therapies_map = pd.DataFrame(data["Therapies"])
	print(therapies_map.info())
	patients_df = pd.DataFrame(data["Patients"])
	print(patients_df.info())
	start = timeit.default_timer()
	x = list([a for b in patients_df["trials"].tolist() for a in b])
	y = list([a for b in patients_df["conditions"].tolist() for a in b])
	stop = timeit.default_timer()
	print("@@@@@@@@@@@@@@@@@@@")
	print("TIME IS {}".format(stop - start))
	print(len(x))
	df = pd.DataFrame(x)
	df1 = pd.DataFrame(y)
	print(df1.head(1))
	print(df1.info())
	print(df.head(1))
	print(df.info())
	new = conditions_map.merge(therapies_map, how = "cross")
	new1 = new.merge(df, how = "outer", left_on = ["id_x","id_y"], right_on = ["condition","therapy"])
	print(new1.info())

create_necessary_set(PATH+"datasetB.json", 2)