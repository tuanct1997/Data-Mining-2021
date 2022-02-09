import json
import pandas as pd
from json import JSONDecoder, JSONDecodeError
import timeit
import scipy.sparse as sparse
import numpy as np
import copy
from lightfm import LightFM
from sklearn.preprocessing import MinMaxScaler
from pandas.api.types import CategoricalDtype
from lightfm.datasets import fetch_stackexchange
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k
from lightfm.cross_validation import random_train_test_split
import pickle
import os.path

PATH = "../data/"

#function to create necessary component such as matrix
def create_necessary_set(path):
	f = open(path)
	data = json.loads(f.read())
	print(data.keys())

	conditions_map = pd.DataFrame(data["Conditions"])

	therapies_map = pd.DataFrame(data["Therapies"])

	patients_df = pd.DataFrame(data["Patients"])

	# Create df for conditions and trials from patients record
	x = list([a for b in patients_df["trials"].tolist() for a in b])
	y = list([a for b in patients_df["conditions"].tolist() for a in b])
	
	patient_trials = pd.DataFrame(x)
	patient_condition = pd.DataFrame(y)
	patients_info = patient_trials.merge(patient_condition, how = "inner", left_on = ["condition"], right_on = ["id"])
	#create ultility matrix base on conditions-therapy pairs
	patients_info.drop(["end","id_x","start","id_y","cured","diagnosed","isCured","isTreated","kind"], axis = 1,inplace = True)
	#Scale back rating
	scaler = MinMaxScaler()
	patients_info["successful"] = scaler.fit_transform(patients_info["successful"].values.reshape(-1,1))


	# Making sparse matrix
	cond_ids = list(np.sort(patients_info.condition.unique()))
	the_ids =   list(np.sort(patients_info.therapy.unique()))
	rating = list(patients_info.successful)
	#Create dummy in int so lightFM can input that
	patients_info['cond_ids'] = patients_info['condition'].astype(CategoricalDtype(categories=cond_ids)).cat.codes
	patients_info['the_ids'] = patients_info['therapy'].astype(CategoricalDtype(categories=the_ids)).cat.codes

	#convert to csr_matrix
	sparse_matrix = sparse.csr_matrix((rating, (patients_info["cond_ids"],patients_info["the_ids"])), shape=(len(cond_ids), len(the_ids)))

	#Create conditions features with necessary fields only
	conditions_features = patient_condition.merge(conditions_map, how = "inner", left_on = "kind", right_on = "id")
	conditions_features = pd.get_dummies(conditions_features, columns = ['isCured', 'isTreated','type'])
	conditions_features.drop(["cured","diagnosed","id_y","kind","name"], axis = 1,inplace = True)

	conditions_features = conditions_features.sort_values('id_x').reset_index().drop('index', axis=1)
	conditions_features_scr = sparse.csr_matrix(conditions_features.drop(['id_x','isCured_False','isCured_True','isTreated_False','isTreated_True'], axis=1).values)
	return conditions_map, therapies_map, patients_info, sparse_matrix, conditions_features,conditions_features_scr

# Cosine_similarity function for condition-condition recommendation
def cosine_similarity(model,uid, conditions_map, features):
	ls = conditions_map.index[conditions_map["id_x"] == uid].tolist()
	ids = ls[0]
	item_representations = features.dot(model.item_embeddings)
	scores = item_representations.dot(item_representations[ids, :])
	item_norms = np.linalg.norm(item_representations, axis=1)
	scores /= item_norms
	best = np.argpartition(scores, -1000)[-1000:]
	for val in best:
		if conditions_map.loc[val]["isTreated_True"] == 1:
			return conditions_map.loc[val]["id_x"]

# Train model
def hybrid_system(rating, features):
	#hyper param
	NUM_THREADS = 1
	NUM_COMPONENTS = 10
	NUM_EPOCHS = 10
	ITEM_ALPHA = 1e-6
	train,test = random_train_test_split(rating,test_percentage = 0.2)

	model = LightFM(loss='warp',item_alpha=ITEM_ALPHA, no_components=NUM_COMPONENTS)

	model = model.fit(train, epochs=NUM_EPOCHS, num_threads=NUM_THREADS, item_features=features)
	train_precision = precision_at_k(model, train, k = 10, item_features = features).mean()
	print('Collaborative filtering train precision_at_k: %s' % train_precision)
	test_precision = precision_at_k(model, test, k = 10, item_features = features).mean()
	print('Collaborative filtering test precision_at_k: %s' % test_precision)
	train_recall = recall_at_k(model, train, k = 10, item_features = features).mean()
	print('Collaborative filtering train recall_at_k: %s' % train_recall)
	test_recall = recall_at_k(model, test, k = 10, item_features = features).mean()
	print('Collaborative filtering test recall_at_k: %s' % test_recall)
	train_auc = auc_score(model, train, num_threads=NUM_THREADS, item_features=features).mean()
	print('Collaborative filtering train AUC: %s' % train_auc)
	test_auc = auc_score(model, test, train_interactions=train, num_threads=NUM_THREADS, item_features=features).mean()
	print('Collaborative filtering test AUC: %s' % test_auc)

	return model


# making predictions with collaborative
def prediction(model, patients_map, conditions_map, uid, features):
	temps = patients_map[patients_map["condition"] == uid]
	therapies =  np.asarray(list(set(patients_map["the_ids"].tolist())))
	# if cold-start -> perform cosine similarity and regression
	if len(temps) == 0:
		uid = cosine_similarity(model,uid, conditions_map, features)
		ls = prediction(model, patients_map, conditions_map, uid, features)
		return ls
	else:
		scores = model.predict(int(temps.iloc[0]["cond_ids"]), therapies, item_features = features)
		list_therapy = np.argsort(-scores)
		list_therapy = list_therapy[:5]
		top_therapy = patients_map["therapy"][list_therapy]
		return top_therapy

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

start = timeit.default_timer()
test_case = read_testcase(PATH+"datasetB_cases.txt")
cd_map , th_map , patients_label, rating_matrix,cb_label ,item_features = create_necessary_set(PATH+"datasetB.json")

# check if already have model
if os.path.isfile('./model/model.pickle'):
	with open('./model/model.pickle','rb') as f:
            model = pickle.load(f)
else:
	model = hybrid_system(rating_matrix, item_features)
	with open('./model/model.pickle', 'wb') as file:
		pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)

test_case["suggestion"] = ""
for idx,val in test_case.iterrows():
	condtions_id = val["patient_condition"]
	ls = prediction(model, patients_label, cb_label, condtions_id, item_features)
	test_case.at[idx,"suggestion"] = ls.tolist()
stop = timeit.default_timer()
print("TIME IS {}".format(stop - start))
print(test_case.info())
test_case.to_csv("../result/result_hybrid.csv", index = False)
