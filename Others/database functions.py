
#Functions defined for the performing C.R.U.D operations to the MongoDB Database

class MDatabase:
	def __init__(self):
		pass
		#Initialises and connects to MongoDB Database

	def insertUser(self,u):
		pass
		#Function to add User Registration details into the Database
		
	def deleteUser(self,userid):
		pass
		#Function to delete User Details from the Database
		
	def searchLoginUser(self,email,password):
		pass
		#Function to validate User Login Details
		
	def searchRegisterEmail(self,email):
		pass
		#Function to validate duplicate Registered Email during User Registration
		
	def searchUserId(self,userid):
		pass
		#Function to search Registered User-ID of the User
		
	def getUsername(self,email,password):
		pass
		#Function to fetch Registered Username of the User
		
	def getUserInfo(self):
		pass
		#Function to fetch Other Details of the Registered User
		
	def insertAd(self,a):
		pass
		#Function to insert new Advertise into the Database
		
	def deleteAd(self,adid):
		pass
		#Function to delete existing Advertise from the Database
		
	def searchAdPath(self,adid):
		pass
		#Function to fetch the pathname of the Advertise stored on the Local machine/Server
		
	def getAds(self):
		pass
		#Function to fetch all the Advertise details present in the Database
		
	def getPredictedAds(self,gender,age_group,age):
		pass
		#Function to fetch Ads based on age and gender predicted by the AI model
		
	def getAdViews(self):
		pass
		#Function to fetch the No of Advertise views
		
	def getAdImpressions(self):
		pass
		#Function to fetch the Advertise Impression w.r.t to each Advertise

	def getMaleViews(self):
		pass
		#Function to fetch the No of Male Adviews
		
	def getFemaleViews(self):
		pass
		#Function to fetch the No of Female Adviews

		