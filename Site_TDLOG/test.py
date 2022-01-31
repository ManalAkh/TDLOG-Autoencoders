import os

#-----Modules for creating our app and our database

from flask import Flask, url_for, redirect, render_template, request, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from flask_login import UserMixin, login_user, LoginManager, logout_user, current_user
from flask_bcrypt import Bcrypt


from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
import requests
import numpy as np
from datetime import datetime, timedelta

#----Modules that we wrote 
import potentials as pt
#import autoencoders as ae
#import dihedral_angles as rama

N=1000

app = Flask(__name__)

basedir = os.path.abspath(os.path.dirname(__file__))
print(basedir)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'app.sqlite')
app.config['SECRET_KEY'] = 'thisisasecretkey'
app.config['SQLACLHEMY_TRACK_MODIFICATIONS']=True

db= SQLAlchemy(app)
bcrypt=Bcrypt(app)

app.permanent_session_lifetime = timedelta(minutes=5)

def fetch_url(name):
    """ When the user choose a molecule to be displayed,
        create the url that leads to the sdf file in the pubchem dataset

        :param name: string, molecule's name the user had entered

        :return: the url that leads to the sdf file 
        """
    url="https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"+name+"/SDF?record_type=3d"
    return url

class User(db.Model,UserMixin):
    id=db.Column(db.Integer, primary_key=True)
    username=db.Column(db.String(20),nullable=False, unique=True)
    password=db.Column(db.String(80),nullable=False)

class RegisterForm(FlaskForm):
    username=StringField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder":'username'})
    password=PasswordField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder":'Password'})
    submit=SubmitField("Register")

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username=username.data).first()
        if existing_user_username:
            raise ValidationError("That username already exists. Please choose a different one.")


class LoginForm(FlaskForm):
    username=StringField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder":'username'})
    password=PasswordField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder":'Password'})
    submit=SubmitField("Login")


login_manager=LoginManager()
login_manager.init_app(app)
#user_manager = UserManager(app, db, User)
login_manager.login_view="login"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


#***********************************************************************************************************

@app.route('/register', methods=['GET', 'POST'])
def register():
    
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(
            form.password.data).decode('utf-8')
        new_user = User(username=form.username.data,
                        password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash("Your account has successfully been created!")
        return redirect(url_for('login')) 
    return render_template('register.html', form = form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    
    form = LoginForm()
    if form.validate_on_submit():
        user=User.query.filter_by(username=form.username.data).first()
        if user :
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                flash("Login Succesfull")
                return redirect(url_for('index'))
            else : 
                flash("Wrong password - Try Again !")
        else :
            flash("That user doesn't exist - Try Again !")
    
    return render_template('login.html', form = form)

@app.route('/logout', methods=['GET','POST'])
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/', methods=['GET','POST'])
def index():
    return render_template('index.html', title='Homepage')


@app.route('/visualisation',  methods=['GET','POST'])
def visualisation():
    if request.method == "POST":
        errors = False #tracking the errors 
        molecule = request.form['molecule']
        try:
            beta = int(request.form['beta'])
        except ValueError:
            flash("Please insert a new beta, it must be a float (floats should be with a point (exp : 0.5 not 0,5)")
            errors = True
        num = int(request.form['number'])
        bowls=[]
        
        for i in range(1,num+1):
            coord="bowl"+str(i)+"xy"
            xy=request.form[coord]
            try:
                x0,y0=xy.split(";")
                try:
                    x0=float(x0)
                except ValueError:
                    flash(f"Please insert a new x coordinate in the high intensity area number {i}, it must be a float (floats should be with a point (exp : 0.5 not 0,5)")
                    errors = True            
                try:
                    y0=float(y0)
                except ValueError:
                    flash(f"Please insert a new y coordinate in the high intensity area number {i} , it must be a float (floats should be with a point (exp : 0.5 not 0,5)")
                    errors = True
            except ValueError:
                flash(f"Please respect the coordinates format in the high intensity area number {i}, it must be at the form x;y ")
                errors = True            
            
            r_id="bowl"+str(i)+"r"
            try:
                r=float(request.form[r_id])
            except ValueError:
                flash(f"Please insert a new radius in the high intensity area number {i}, it must be a float (floats should be with a point (exp : 0.5 not 0,5)")
                errors = True

            a_id="bowl"+str(i)+"a"
            try:
                a=float(request.form[a_id])
            except ValueError:
                flash(f"Please insert a new amplitude in the high intensity area number {i}, it must be a float (floats should be with a point (exp : 0.5 not 0,5)")
                errors = True
            
            if not errors :
                if abs(x0) > 2:
                    flash(f"Please choose a value for x in the interval [-2,2] in the high intensity area number {i}")
                    errors = True
                if abs(y0) > 2:
                    flash(f"Please choose a value for x in the interval [-2,2] in the high intensity area number {i}")
                    errors = True
                bowls.append([x0,y0,r,a])

        url = fetch_url(molecule)
        response = requests.get(url)
        if response.status_code != 200: #the url doesn't exist
            flash(f"The molecule {molecule} you looked for seems as if it doesn't exist in the database. Make sure it is spelled correctly !")
            errors = True

        if errors : 
            return render_template('visualisation.html', title='Visualisation', formulaire_rempli=False)
        else:
            bowl=np.array(bowls)
            potential=pt.MultimodalPotential(bowl, beta)
            fig_pot=pt.create_plots(potential)
            trajectory, _ = pt.UnbiasedTraj(potential)
            fig_traj=pt.plot_trajectory(potential, trajectory)
            fig_loss, fig_rc = ae.plot_results(potential, trajectory)
            now = datetime.now() # current date and time to identify plots
            date_time = now.strftime("%m%d%Y%H%M%S%f")

            #creating paths
            path_plot_pot ='static\\plot\\plot3D'+date_time+'.html' #path to 3D potential plot
            path_plot_pot= os.path.join(basedir,path_plot_pot)
            path_plot_trajectory = 'static\\img\\traj'+date_time+'.png' #path to trajectory plot
            path_plot_trajectory = os.path.join(basedir, path_plot_trajectory )
            path_plot_losses = 'static\\img\\loss'+date_time+'.png' #path to loss plot
            path_plot_losses = os.path.join(basedir, path_plot_losses )
            path_plot_rc = 'static\\img\\rc'+date_time+'.png' #path to trajectory plot
            path_plot_rc = os.path.join(basedir, path_plot_rc)

            #creating the files where we store the plots
            file1 = open(path_plot_pot, 'w') 
            file1.close()
            file2 = open(path_plot_trajectory, 'w')
            file2.close()
            file3 = open(path_plot_losses, 'w')
            file3.close()
            file4 = open(path_plot_rc, 'w')
            file4.close()

            #saving the plots
            fig_pot.write_html(path_plot_pot)
            fig_traj.savefig(path_plot_trajectory)
            fig_loss.savefig(path_plot_losses)
            fig_rc.savefig(path_plot_rc)     

            return render_template('visualisation.html', title='Visualisation', molecule=molecule , url=url, pot_path='/static/plot/plot3D'+date_time+'.html', traj_path='/static/img/traj'+date_time+'.png', loss_path = '/static/img/loss'+date_time+'.png', rc_path = '/static/img/rc'+date_time+'.png', formulaire_rempli = True)
 
    return render_template('visualisation.html', title='Visualisation', formulaire_rempli=False)

@app.route('/explication/0', methods=['GET','POST'])
def explication():
    """ Vizualisation of our work during MOPSI project """
    if current_user.is_authenticated:
        session['tuto0']=True
    return render_template('explication0.html', title='Explanation')

@app.route('/profil/<string:username>/<string:status>')
def profil_visualization_history(username,status):
    return render_template('profil_visualization_history.html', username=username, status=status, title="Visualization history")

@app.route('/profil/<string:username>/<string:status>/parameters')
def profil_change_parameters(username,status):
    return render_template('profil_change_parameters.html', username=username, status=status, title="Change parameters")

@app.route('/profil/<string:username>/<string:status>/tutorial')
def profil_tutorial(username,status):
    try :
        tuto_0=session['tuto0']
        print(tuto_0)
    except:
        tuto_0=False

    try :
        tuto_1=session['tuto1']
        print(tuto_1)
    except:
        tuto_1=False

    try :
        tuto_2=session['tuto2']
        print(tuto_2)
    except:
        tuto_2=False

    try :
        tuto_3=session['tuto3']
        print(tuto_3)
    except:
        tuto_3=False

    try :
        tuto_4=session['tuto4']
        print(tuto_4)
    except:
        tuto_4=False

    return render_template('profil_tutorial.html', username=username, status=status, title="Tutorial", tuto0=tuto_0, tuto1=tuto_1, tuto2=tuto_2, tuto3=tuto_3, tuto4=tuto_4)

@app.route('/explication/1')
def explication1():
    """ Vizualisation of our work during MOPSI project """

    if current_user.is_authenticated:
        session['tuto1']=True
    return render_template('explication1.html', title='Explanation')

@app.route('/explication/2')
def explication2():
    """ Vizualisation of our work during MOPSI project """

    if current_user.is_authenticated:
        session['tuto2']=True
    return render_template('explication2.html', title='Explanation')

@app.route('/explication/3')
def explication3():
    """ Vizualisation of our work during MOPSI project """

    if current_user.is_authenticated:
        session['tuto3']=True
    return render_template('explication3.html', title='Explanation')

@app.route('/explication/4', methods=['GET', 'POST'])
def explication4():
    """ Display the last step of the tutorial 
        The user can choose the atoms and plot the correcponding Rama plot 
        Display an error message if the user didn't fill the cases correctly """

    if current_user.is_authenticated:
        session['tuto4']=True
    url=fetch_url("dialanine")

    if request.method == 'POST':
        try :
            phi_atom = [int(request.form['phi_atom1'])-1, int(request.form['phi_atom2'])-1, int(request.form['phi_atom3'])-1, int(request.form['phi_atom4'])-1]
            psi_atom = [int(request.form['psi_atom1'])-1, int(request.form['psi_atom2'])-1, int(request.form['psi_atom3'])-1, int(request.form['psi_atom4'])-1]
            fig = rama.rama_plot(phi_atom, psi_atom)
        except ValueError:
            return render_template('explication4.html', title='Explanation', url=url, error=True, completed_form=False)
        
        fig.write_html('static/plot/rama_user.html',full_html=False,include_plotlyjs='cdn')
        anim_fig = rama.rama_frame(phi_atom, psi_atom)
        anim_fig.write_html('static/plot/rama_frame.html', full_html=False,include_plotlyjs='cdn')
        return render_template('explication4.html', title='Explanation', url=url, error=False, completed_form=True)  

    return render_template('explication4.html', title='Explanation', url=url, error=False, completed_form=False)

@app.route('/explication/codeNN')
def codeAE():
    """ Show one of the notebooks we use in our MOPSI project
        This show thetraining for a 3D trajectory  """

    return render_template('TrainingAE.html')


if __name__ == "__main__":
    db.create_all()
    app.run(debug=True)