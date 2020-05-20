import pymongo

try:
	client=pymongo.MongoClient("mongodb://localhost:27017")
	print("Connected successfully!!!")
except:
	print("Could not connect to MongoDB")

db=client["Project"]
col=db["Users"]
col1=db["Ads"]



def insertUser(u):
	usr={'username':u.username,'email':u.email,'password':u.password}
	result=col.insert(usr)
	print("Employee inserted with Id:",result)



def searchLoginUser(email,password):
	doc=col.find({"email":email,"password":password})
	if doc.count() == 0:
		return False
	else:
		return True

def searchRegisterEmail(email):
	doc=col.find({"email":email})
	if doc.count() >= 1:
		return True
	else:
		return False

def getAds():
	doc=col1.find()
	lst = list(doc)
	print(lst)
	return lst
