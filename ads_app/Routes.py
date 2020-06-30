from flask import render_template, url_for, flash, redirect, request, session, json
from flask import Response
from werkzeug.utils import secure_filename

from ads_app import app
from ads_app.Forms import RegistrationForm, LoginForm
from ads_app.User import User
from ads_app.Ad import Ad
from ads_app.Reg_MongoDB import MDatabase
import threading


import os
import signal
import datetime
import pytz

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}


@app.route("/")
@app.route("/home/")
@app.route("/index/")
def index():
	try:
		m.end_pgm()
	except:
		print("not a variable")
	return render_template("index.html",title="Home")




@app.route("/about/")
def about():
	try:
		m.end_pgm()
	except:
		print("not a variable")
	return render_template("about.html",title="About")


@app.route("/admin/",methods=["GET","POST"])
def admin():
	try:
		m.end_pgm()
	except:
		print("not a variable")
	try:
		adsdb = MDatabase()
		if request.method == 'POST':
			if request.form['submit'] == "Delete User":
				delete_id = int(request.form["user_id"])
				bool_id = adsdb.searchUserId(delete_id)
				if not bool_id:
					flash(f"Invalid User ID","danger")
				else:
					adsdb.deleteUser(delete_id)
					flash(f"User deleted successfully","success")
			else:
				username = request.form["username"]
				email = request.form["email"]
				password = request.form["password"]

				tz_india = pytz.timezone('Asia/Kolkata') 
				x = datetime.datetime.now(tz_india)
				date = x.strftime("%d/%b/%Y")
				time = x.strftime("%I:%M %p")

				fh=open(".\\ads_app\\static\\User_counter.txt","r")
				userid=int(fh.read())
				fh.close()

				userid+=1

				fh=open(".\\ads_app\\static\\User_counter.txt","w")
				fh.write(str(userid))
				fh.close()

				usr = User(userid,username,email,password,date,time)
				if adsdb.searchRegisterEmail(email):
					flash(f"Already an account exists with this email","danger")
				adsdb.insertUser(usr)
				flash(f'Account created for {username}!','success')

		user = adsdb.getUserInfo()
		lst_userid = [li['userid'] for li in user]
		lst_username = [li['username'] for li in user]
		lst_email = [li['email'] for li in user]
		lst_date = [li['date'] for li in user]
		lst_time = [li['time'] for li in user]
		return render_template("admin.html",title="Admin",userid=lst_userid,username=lst_username,email=lst_email,date=lst_date,time=lst_time,length=len(user))
	except:
		return render_template("404.html")
	



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/adtable/",methods=["GET","POST"])
def adtable():
	try:
		m.end_pgm()
	except:
		print("not a variable")
	try:
		adsdb = MDatabase()
		if request.method == 'POST':
			if request.form['submit'] == "Upload Advertise":
				if 'file' not in request.files:
					flash(f"No file part available","danger")
					return redirect(request.url)
				file = request.files['file']
				if file.filename == '':
					flash(f"No file selected to upload","danger")
					return redirect(request.url)
				if file and allowed_file(file.filename):
					filename = secure_filename(file.filename)
					file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
					print(filename)

					fh=open(".\\ads_app\\static\\Ad_counter.txt","r")
					adid=int(fh.read())
					fh.close()

					adid+=1

					fh=open(".\\ads_app\\static\\Ad_counter.txt","w")
					fh.write(str(adid))
					fh.close()

					aname = request.form["adname"]
					agroup = request.form.getlist("agegroup")
					agender = request.form.getlist("gender")
					acategory = request.form["category"]
					apath = "C:\\Users\\Geralt\\Desktop\\AdsApp v5\\ads_app\\static\\Uploads\\"+filename
					new_ad = Ad(adid,aname,apath,agroup,agender,acategory)
					adsdb.insertAd(new_ad) 
					flash(f"Advertise uploaded successfully","success")

			else:
				delete_id = int(request.form["ad_id"])
				img_path = adsdb.searchAdPath(delete_id)
				if not img_path:
					flash(f"Invalid Advertise ID","danger")
				else:
					adsdb.deleteAd(delete_id)
					img_path = img_path[0]["adpath"]
					img_path = img_path.replace("\\","/")
					os.remove(img_path)
					flash(f"Advertise deleted successfully","success")


		ads=adsdb.getAds()
		lst_adid = [li['adid'] for li in ads]
		lst_adname = [li['adname'] for li in ads ]
		lst_adpath = [li['adpath'] for li in ads ]
		for i in range(len(lst_adpath)):
			lst_adpath[i] = lst_adpath[i].split("\\")[-1]
			lst_adpath[i] = "Uploads/"+lst_adpath[i]
		lst_adage = [li['adage'] for li in ads ]
		lst_adgender = [li['adgender'] for li in ads ]
		lst_adcategory = [li['adcategory'] for li in ads ]
		lst_adviews = [li['adviews'] for li in ads]
		length = len(lst_adid)
		return render_template("adtable.html",adid=lst_adid,adname=lst_adname,adpath=lst_adpath,adcategory=lst_adcategory,adage=lst_adage,adgender=lst_adgender,adviews=lst_adviews,length=length,title="Dashboard")
	except:
		return render_template("404.html",title="Error")



@app.route("/register",methods=["GET","POST"])
def register():
	try:
		m.end_pgm()
	except:
		print("not a variable")
	form = RegistrationForm()
	if form.validate_on_submit():
		username = form.username.data
		email = form.email.data
		password = form.password.data

		tz_india = pytz.timezone('Asia/Kolkata') 
		x = datetime.datetime.now(tz_india)
		date = x.strftime("%d-%b-%Y")
		time = x.strftime("%I:%M %p")

		fh=open(".\\ads_app\\static\\User_counter.txt","r")
		userid=int(fh.read())
		fh.close()

		userid+=1

		fh=open(".\\ads_app\\static\\User_counter.txt","w")
		fh.write(str(userid))
		fh.close()

		userdb = MDatabase()
		usr = User(userid,username,email,password,date,time)
		if userdb.searchRegisterEmail(email):
			flash(f"Already an account exists with this email","danger")
			return redirect(url_for("register"))
		userdb.insertUser(usr)
		flash(f'Account created for {form.username.data}!...Please Login','success')
		return redirect(url_for('index'))
	return render_template('register.html',title='Register',form=form)






@app.route("/login/",methods=["GET","POST"])
def login():
	try:
		m.end_pgm()
	except:
		print("not a variable")
	form = LoginForm()
	if form.validate_on_submit():
		email = form.email.data
		password = form.password.data
		userdb = MDatabase()
		if userdb.searchLoginUser(email,password):
			session.permanent = True
			username = userdb.getUsername(email,password)
			session["user"] = username[0]["username"]
			flash(f"You have been logged in!","success")
			return redirect(url_for("index"))
		elif email == "user@admin.com" and password == "admin":
			session.permanent = True
			username = admin
			session["user"] = "admin"
			flash(f"You have been logged in!","success")
			return redirect(url_for("index"))
		else:
			flash(f"Login unsuccessful. Please check username and password","danger")
	return render_template('login.html',title='Login',form=form)




@app.route("/logout/")
def logout():
	try:
		m.end_pgm()
	except:
		print("not a variable")
	if "user" in session:
		flash(f"Logged out successfully!!","success")
		session.pop("user",None)
	return redirect(url_for("login"))


@app.route("/analytics/")
def analytics():
	try:
		m.end_pgm()
	except:
		print("not a variable")
	adsdb = MDatabase()
	adviews_data = adsdb.getAdViews()
	adimpressions_data = adsdb.getAdImpressions()
	adviews_male = adsdb.getMaleViews()
	adviews_female = adsdb.getFemaleViews()
	advertise_names = adsdb.getAdNames()
	
	inc=0
	labels=[] 
	values=[]
	label=['Male','Female'] 
	datam=[]
	dataf=[]
	for i in adviews_data:
		labels.append(adviews_data[inc]['adname'])
		values.append(adviews_data[inc]['adviews'])
		label.append(advertise_names[inc]['adname'])
		datam.append(adviews_male[inc]['male'])
		dataf.append(adviews_female[inc]['female'])

		inc=inc+1
		
	
	print(adviews_male)
	print(adviews_female) 
	count=["10","10","10"]
	color=['#f56954','#00a65a','#00c0ef','#3c8dbc',
	"#0074D9", "#FF4136", "#2ECC40", "#3D9970", "#FF851B", 
	"#7FDBFF", "#B10DC9", "#FFDC00", "#001f3f",
	"#39CCCC", "#01FF70", "#85144b", '#d2d6de',
	"#F012BE", "#111111", "#AAAAAA"]
	heighlight=['#f56954','#00a65a','#00c0ef','#3c8dbc','#d2d6de']
	return render_template("analytics.html",datam=datam,dataf=dataf,labels=labels,pieset=zip(values,labels,color),title="Analytics")





@app.route("/detect/")
def detection():
	adsdb = MDatabase()
	age=8
	gender="Male"
	if 6<=age<=12:
		agegroup="06-12"
	elif 13<=age<=18:
		agegroup="13-18"
	elif 19<=age<=24:
		agegroup="19-24"
	elif 25<=age<=30:
		agegroup="25-30"
	elif 31<=age<=40:
		agegroup="31-40"
	elif 41<=age<=50:
		agegroup="41-50"
	elif 51<=age<=60:
		agegroup="51-60"
	elif 61<=age<=70:
		agegroup="61-70"
	ad=adsdb.getPredictedAds(gender,agegroup,age)
	if not ad:
		flash(f". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . Could not find appropriate Advertise for you . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .","danger")
		predicted_ads=["/static/Uploads/Loading.gif"]
		length=1
		return render_template('detection.html',images=predicted_ads,length=length,age=age,gender=gender,title="Detection")
	else:
		predicted_ads= ["/static/Uploads/"+ad['adpath'].split("\\")[-1]]
		length=len(predicted_ads)
		return render_template('detection.html',images=predicted_ads,length=length,age=age,gender=gender,title="Detection")

