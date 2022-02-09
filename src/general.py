import json
import pandas as pd
from json import JSONDecoder, JSONDecodeError
import timeit
import numpy as np
import copy

PATH = "../data/"


##function to create necessary component such as matrix
def create_necessary_set(path,file):
	f = open(path)
	data = json.loads(f.read())

	conditions_map = pd.DataFrame(data["Conditions"])
	therapies_map = pd.DataFrame(data["Therapies"])
	patients_df = pd.DataFrame(data["Patients"])
	start = timeit.default_timer()
	# Create df for conditions and trials from patients record
	x = list([a for b in patients_df["trials"].tolist() for a in b])
	y = list([a for b in patients_df["conditions"].tolist() for a in b])
	stop = timeit.default_timer()
	patient_trials = pd.DataFrame(x)
	patient_condition = pd.DataFrame(y)
	patients_info = patient_trials.merge(patient_condition[["kind","id"]], how = "inner", left_on = ["condition"], right_on = ["id"])
	# Create test_set
	test_set = copy.deepcopy(patients_info)
	test_set = test_set[test_set["successful"] == 100]
	check = test_set["therapy"].tolist()
	test_set = test_set.iloc[1:20000]
	#create ultility matrix base on conditions-therapy pairs
	# patient_condition.to_csv("a.csv",index = False)
	patients_info.drop(["condition","end","id_x","start","id_y"], axis = 1,inplace = True)
	patients_info = patients_info.groupby(["kind","therapy"])["successful"].mean().reset_index(name = "score")
	return conditions_map, therapies_map, patients_info, test_set, patient_condition

# Perform evaluation - precision and recall at k
def ultility_evaluation(df,mat):
	df["suggestion"] = ""
	df["label"] = ""
	labels_frame = df.groupby(["kind","therapy"])["successful"].count().reset_index(name = "frequent")
	# Only consider the therapy which is frequent than 1 to reduce the size of relevant therapy
	labels_frame = labels_frame[labels_frame["frequent"] > 1]
	precision = []
	recall = []

	for idx,row in df.iterrows():
		temps = mat[mat["kind"] == row["kind"]].nlargest(5,"score")
		df.at[idx,"suggestion"] = temps["therapy"].tolist()
		temps = labels_frame[labels_frame["kind"] == row["kind"]]
		df.at[idx,"label"] = temps["therapy"].tolist()
		del temps
		intersection = list(set(df.at[idx,"label"]) & set(df.at[idx,"suggestion"]))
		precision.append(len(intersection)/5)
		recall.append(len(intersection)/len(df.at[idx,"label"]))

	print("PRECISION IS = {}".format(sum(precision)/len(precision)))
	print("RECALL IS = {}".format(sum(recall)/len(recall)))

	return df

# Prediction function for the test case
def predict_general(df,mat):
	df["suggestion"] = ""
	for idx,row in df.iterrows():
		temps = mat[mat["kind"] == row["kind"]].nlargest(5,"score")
		df.at[idx,"suggestion"] = temps["therapy"].tolist()
		del temps
	
	return df

# Read testcase file to create input
def read_testcase(path):
	conds = []
	pat = []
	with open(path) as f:
		lines = [line.rstrip() for line in f]
	for line in lines[1:]:
		for word in line.split():
			if "pc" in word:
				conds.append(word)
			else:
				pat.append(word)
	df_dict = {"patients":pat, "patient_condition":conds}
	df = pd.DataFrame(df_dict)
	return df

test_case = read_testcase(PATH+"datasetB_cases.txt")
cd_map , th_map , matrix , testset, patient_condition = create_necessary_set(PATH+"datasetB.json", 2)
test_case = test_case.merge(patient_condition[["id","kind"]], how = "left",left_on = ["patient_condition"], right_on = ["id"])

test_case = predict_general(test_case,matrix)

test_case.to_csv("../result/result_general.csv", index = False)
