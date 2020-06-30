import pymongo


class MDatabase:
	def __init__(self):
		try:
			self.client=pymongo.MongoClient("mongodb+srv://suraj13mj:surajmj@8thsemproject-oqp0h.mongodb.net/test?retryWrites=true&w=majority")
			print("Connected successfully!!!")
		except:
			print("Could not connect to MongoDB")

		self.db=self.client["Project"]
		self.col_user=self.db["Users"]
		self.col_ads=self.db["Ads"]



	def insertUser(self,u):
		usr={'userid':u.userid,'username':u.username,'email':u.email,'password':u.password,'date':u.date,'time':u.time}
		result=self.col_user.insert(usr)
		print("Employee inserted with Id:",result)

	def deleteUser(self,userid):
		result = self.col_user.remove({'userid':userid})


	def searchLoginUser(self,email,password):
		doc=self.col_user.find({"email":email,"password":password})
		if doc.count() == 0:
			return False
		else:
			return True

	def searchRegisterEmail(self,email):
		doc=self.col_user.find({"email":email})
		if doc.count() >= 1:
			return True
		else:
			return False

	def searchUserId(self,userid):
		doc=self.col_user.find({"userid":userid})
		if doc.count() >= 1:
			return True
		else:
			return False

	def getUsername(self,email,password):
		doc=self.col_user.find({"email":email,"password":password},{"username":1,"_id":0})
		return list(doc)

	def getUserInfo(self):
		doc=self.col_user.find({},{"userid":1,"username":1,"email":1,"date":1,"time":1,"_id":0})
		return list(doc)








	def insertAd(self,a):
		ad = {'adid':a.adid,'adname':a.adname,'adpath':a.adpath,'adage':a.adage,'adgender':a.adgender,'adcategory':a.adcategory,'adviews':0,'adimpressions':[],'male':0,'female':0}
		result=self.col_ads.insert(ad)
		print("Ad inserted with Id:",result)


	def deleteAd(self,adid):
		result = self.col_ads.remove({'adid':adid})


	def searchAdPath(self,adid):
		doc = self.col_ads.find({"adid":adid},{"adpath":1,"_id":0})
		lst = list(doc)
		return lst


	def getAds(self):
		doc=self.col_ads.find()
		lst = list(doc)
		return lst

	def getPredictedAds(self,gender,age_group,age):
		doc=self.col_ads.find({"adgender":gender,"adage":age_group},{"adpath":1,"adid":1,"adviews":1,"_id":0})
		lst = list(doc)
		lst_adviews = [ad['adviews'] for ad in lst]
		minimum = min(lst_adviews)
		min_adviews = [{"adid":ad['adid'],"adpath":ad['adpath']} for ad in lst if ad['adviews'] == minimum]

		if min_adviews:
			p_id = min_adviews[0]['adid']
			if gender == "Male":
				doc=self.col_ads.update({'adid':p_id},{'$inc':{"adviews":1,"male":1}})
			else:
				doc=self.col_ads.update({'adid':p_id},{'$inc':{"adviews":1,"female":1}})
			doc=self.col_ads.update({'adid':p_id},{'$push':{"adimpressions":{"age":age,"gender":gender}}})
		print(min_adviews[0])
		return min_adviews[0]


	def getAdViews(self):
		doc=self.col_ads.find({},{"adid":1,"adname":1,"adviews":1,"_id":0})
		lst = list(doc)
		return lst

	def getAdImpressions(self):
		doc = self.col_ads.find({},{"adid":1,"adname":1,"adimpressions":1,"_id":0})
		lst = list(doc)
		return lst

	def getMaleViews(self):
		doc=self.col_ads.find({},{"adid":1,"adname":1,"male":1,"_id":0})
		lst = list(doc)
		return lst

	def getFemaleViews(self):
		doc=self.col_ads.find({},{"adid":1,"adname":1,"female":1,"_id":0})
		lst = list(doc)
		return lst

	def getAdNames(self):
		doc=self.col_ads.find({},{"adid":1,"adname":1,"_id":0})
		lst = list(doc)
		return lst