from flask import Flask, render_template, url_for, flash, redirect
from forms import RegistrationForm, LoginForm
import User
import Reg_MongoDB

app = Flask(__name__)

app.config['SECRET_KEY'] = '5d135e41154cd1265943696698d9be76'



posts = [
{
	"title":"About Miachel Scofield",
	"author":"Suraj Janmane",
	"content":"Miachel Scofield is a structural Engineer",
	"dateposted":"21,Apr 2020"
},
{
	"title":"About Morgan Grechen",
	"author":"Sagar Raymond",
	"content":"Grechen is trained Marine soldier",
	"dateposted":"13,May 2020"
}
]

@app.route("/")
@app.route("/home/")
@app.route("/index/")
def index():
	return render_template("index.html",title="Home",fposts=posts)




@app.route("/about/")
def about():
	return render_template("about.html",title="About")

@app.route("/test/")
def test():
	ads=Reg_MongoDB.getAds()
	lst_adname = [li['adname'] for li in ads ]
	lst_adcategory = [li['adcategory'] for li in ads ]
	lst_admale = [li['admale'] for li in ads ]
	lst_adfemale = [li['adfemale'] for li in ads ]
	length = len(lst_adname)
	print(lst_adname)
	return render_template("test.html",adname=lst_adname,adcategory=lst_adcategory,admale=lst_admale,adfemale=lst_adfemale,length=length)





@app.route("/register",methods=["GET","POST"])
def register():
	form = RegistrationForm()
	if form.validate_on_submit():
		username = form.username.data
		email = form.email.data
		password = form.password.data
		usr = User.User(username,email,password)
		if Reg_MongoDB.searchRegisterEmail(email):
			flash(f"Already an account exists with this email","danger")
			return redirect(url_for("register"))
		Reg_MongoDB.insertUser(usr)
		flash(f'Account created for {form.username.data}!','success')
		return redirect(url_for('index'))
	return render_template('register.html',title='Register',form=form)




@app.route("/login/",methods=["GET","POST"])
def login():
	form = LoginForm()
	if form.validate_on_submit():
		email = form.email.data
		password = form.password.data
		if Reg_MongoDB.searchLoginUser(email,password):
			flash(f"You have been logged in!","success")
			return redirect(url_for("index"))
		else:
			flash(f"Login unsuccessful. Please check username and password","danger")
	return render_template('login.html',title='Login',form=form)




if __name__ == "__main__":
	app.run(debug=True)