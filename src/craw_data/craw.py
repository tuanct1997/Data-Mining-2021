import pandas as pd
import scrapy
import re
import random
import json
import time

# first_name = pd.read_csv("first_names.all.csv", sep = ",", index_col = 0, header = None, names = ["first_name"])
# last_name = pd.read_csv("last_names.all.csv", index_col = 0, header = None, names = ["last_name"])

# print(first_name.info())
# print(last_name.info())


# class CondtionCraw(scrapy.Spider):
# 	name = "craw"
# 	start_urls = ["https://www.nhsinform.scot/illnesses-and-conditions/a-to-z"]

# 	def parse(self, response):
# 		conditions = []
# 		SET_SELECTOR = ".nhs-uk__az-link"
# 		for i in response.css(SET_SELECTOR):
# 			NAME_SELECTOR = 'a::text'
# 			conditions.append(re.sub(r'[^A-Za-z0-9]+', ' ', i.css(NAME_SELECTOR).extract_first()).strip())

# 		df = pd.DataFrame(conditions, columns = ["conditions"])
# 		df.to_csv("conditions.csv", index = False)

def str_time_prop(start, end, time_format, prop):
	stime = time.mktime(time.strptime(start, time_format))
	etime = time.mktime(time.strptime(end, time_format))

	ptime = stime + prop * (etime - stime)

	return time.strftime(time_format, time.localtime(ptime))


def random_date(start, end, prop):
	return str_time_prop(start, end, '%Y%m%d', prop)

print(random_date("20201210","20211210", random.random()))

first_name = pd.read_csv("first_names.all.csv", sep = ",", index_col = 1, header = None, names = ["first_name"])
last_name = pd.read_csv("last_names.all.csv", index_col = 1, header = None, names = ["last_name"])
conditions = pd.read_csv("conditions.csv")
therapy = pd.read_csv("therapy.csv",encoding='latin-1')
print(therapy.info())
print(first_name.info())
print(last_name.info())
conditions = conditions.sample(frac=1).reset_index(drop=True)
therapy = therapy.sample(frac=1).reset_index(drop=True)
conditions_json = []
therapies_json = []
patiens_json = []
# patiens_json = {"id":[], "name":[],"condtions":{"id":[], "diagnosed":[],"cured":[],"kind":[]}, "trial":{"id":[],"start":[],"end":[],"condtion":[],"therapy":[],"successful":[]}}
print(conditions.info())

far_date = '20200112'
end_date = '20210112'

for idx,val in conditions.iterrows():
	d = {"id":"Cond"+str(idx+1)}
	d["name"] = val.item()
	d["type"] = val.item()
	conditions_json.append(d)

for idx,val in therapy.iterrows():
	d = {"id":"Th"+str(idx+1)}
	d["name"] = val.item()
	d["type"] = val.item()
	therapies_json.append(d)

for x in range(25000):
	d = {"id":x+1}
	ft_name = first_name.sample().first_name.item()
	lt_name = last_name.sample().last_name.item()
	name = ft_name + " " + lt_name
	d["name"] = name
	d["conditions"] = []
	d["trials"] = []
	rand1 = random.randint(1,3)
	rand = random.randint(1,5)
	total_trials = rand1*rand
	check = 0
	for i in range(rand):
		temp = 0
		sub_d = {"id":"pc"+str(i+1)}
		sub_d["diagnosed"] = random_date(far_date, end_date,random.random())
		sub_d["cured"] = random_date(far_date, end_date,random.random())
		while int(sub_d.get("cured")) < int(sub_d.get("diagnosed")):
			if sub_d.get("cured") == 0:
				break
			sub_d["cured"] = random_date(far_date, end_date,random.random())
		sub_d["kind"] = "Cond"+str(random.randint(1,len(conditions_json)))
		d["conditions"].append(sub_d)	
		for j in range(rand1):
			check += 1
			if check > total_trials:
				break
			sub_trials = {"id":"tr"+str(check), "start":0, "end":0,"therapy":0, "successful":0}
			if temp == 0:
				sub_trials["start"] = sub_d.get("diagnosed")
				sub_trials["end"] = random_date(sub_d.get("diagnosed"),sub_d.get("cured"),random.random())
				start = sub_trials.get("start")
				end = sub_trials.get("end")
			else:
				while int(sub_trials.get("start")) < int(end):
					sub_trials["start"] = random_date(end, sub_d.get("cured"),random.random())
					sub_trials["end"] = random_date(sub_trials.get("start"), sub_trials.get("cured"),random.random())
					start = sub_trials.get("start")
					end = sub_trials.get("end")
			sub_trials["conditions"] = sub_d.get("id")
			sub_trials["therapy"] = "Th"+str(random.randint(1,len(therapies_json)))
			sub_trials["successful"] = random.randint(0,100)
			d["trials"].append(sub_trials)

	print("!!!!!!!!!!!!!!!!!!!!!!!!")
	print(x)
	patiens_json.append(d)
	
	# therapies_json["id"].append("Th"+str(x))
with open('conditions.json', 'w', encoding='utf-8') as f:
	json.dump(conditions_json, f, ensure_ascii=False)

with open('therapies.json', 'w', encoding = 'utf-8') as f:
	json.dump(therapies_json, f, ensure_ascii=False)

with open('patiens.json', 'w', encoding = 'utf-8') as f:
	json.dump(patiens_json, f, ensure_ascii=False)


