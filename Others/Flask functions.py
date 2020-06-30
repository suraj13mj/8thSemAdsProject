#Route Functions for the HTML pages to be rendered using Flask Web Framework


@app.route("/")
@app.route("/home/")
@app.route("/index/")
def index():
	pass
	#Function to render the Home Page 

@app.route("/about/")
def about():
	pass
	#Function to render the About Page
	
@app.route("/admin/",methods=["GET","POST"])
def admin():
	pass
	#Function to render the Admin Page
	
@app.route("/adtable/",methods=["GET","POST"])
def adtable():
	pass
	#Function to render the Dashboard Page

@app.route("/register",methods=["GET","POST"])
def register():
	pass
	#Function to render the User Registration Page
	
@app.route("/login/",methods=["GET","POST"])
def login():
	pass
	#Function to render the User Login Page
	
@app.route("/logout/")
def logout():
	pass
	#Function to remove the Logged in user from the Session i.e Logout the user
	
@app.route("/analytics/")
def analytics():
	pass
	#Function to render the Analytics Page
	
@app.route("/detect/")
def detection():
	pass
	#Function to render the Detection Page

@app.route("/video_feed/")
def video_feed():
	pass
	#Function to start the Video feed through Browser inorder to capture Frames for Age and Gender detection
	