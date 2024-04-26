from flask import Flask, render_template, request, redirect, session,jsonify
import google.generativeai as genai
import pandas as pd
import logging
import sys
import os
import pickle
from sklearn import preprocessing 
import hashlib
import psycopg2
from werkzeug.utils import secure_filename
import csv
from gtts import gTTS
import tempfile
import pyglet



# Current directory
current_dir = os.path.dirname(__file__)

# Flask app
app = Flask(__name__, static_folder = 'static', template_folder = 'template')

app.secret_key = 'sandeep project'
# Logging
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

# # Load the trained model from the pickle file
with open('LR.pkl', 'rb') as f:
    model = pickle.load(f)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Home page
@app.route('/')
def home():
	return render_template('index.html')


@app.route('/prediction', methods=['POST'])
def predict():
	if request.method == 'POST':
		data = {}
		info = {}
		for field in request.form:
			if field not in ['name', 'birthdate', 'agree-term', 'signup']:
				data[field] = int(request.form[field])
			else:
				info[field] = request.form[field]

		df = pd.DataFrame([data])
		LE = preprocessing.LabelEncoder() 
		obj = (df.dtypes == 'object') 
		for col in list(obj[obj].index): 
			df[col] = LE.fit_transform(df[col])
		result = model.predict(df)

		name = info['name']
		# Determine the output
		if int(result) == 1:
			prediction = 'Dear {name}, your loan is approved!'.format(name = name)
		else:
			prediction = 'Sorry {name}, your loan is rejected!'.format(name = name)

		# Return the prediction
		return render_template('prediction.html', prediction=prediction)
	else:
		return render_template('error.html', prediction="Error occured")

def create_upload_folder():
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

create_upload_folder()

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('error.html', prediction="No file uploaded")
        file = request.files['file']
        if file.filename == '':
            return render_template('error.html', prediction="No file selected")
        try:
            df1 = pd.read_csv(file)
        except Exception as e:
            return render_template('error.html', prediction="Error occurred while reading the file")
        df = df1[['Gender', 'Married' , 'Dependents', 'Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term', 'Credit_History', 'Property_Area']]
        names = df1['Name']
        all_predictions = []
        LE = preprocessing.LabelEncoder()
        obj = (df.dtypes == 'object')
        for col in list(obj[obj].index):
            df[col] = LE.fit_transform(df[col])
        result = model.predict(df)
        for i, res in enumerate(result):
            if int(res) == 1:
                prediction = 'Approved'
            else:
                prediction = 'Rejected'
            all_predictions.append({'sequence': i+1, 'name': names[i], 'prediction': prediction})
        return render_template('prediction.html', predictions=all_predictions)
    else:
        return render_template('error.html', prediction="Error occurred")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to establish connection with PostgreSQL
def connect_to_db():
    conn = psycopg2.connect(
        dbname="flask_data",
        user="postgres",
        password="Skd6397@@",
        host="localhost"
    )
    return conn


# Route for signup and login page
@app.route('/login', methods=['GET', 'POST'])
def signup_login():
    if request.method == 'POST':
        if 'signup' in request.form:
            username = request.form['username']
            mobile = request.form['mobile']
            email = request.form['email']
            password = request.form['password']

            # Hash the password
            hashed_password = hashlib.sha256(password.encode()).hexdigest()

            conn = connect_to_db()
            cur = conn.cursor()

            cur.execute("INSERT INTO users_1 (username, mobile, email, password) VALUES (%s, %s, %s, %s)", (username, mobile, email, hashed_password))
            conn.commit()

            cur.close()
            conn.close()

            return redirect('/')
        
        elif 'login' in request.form:
            email = request.form['email']
            password = request.form['password']

            # Hash the password for comparison
            hashed_password = hashlib.sha256(password.encode()).hexdigest()

            conn = connect_to_db()
            cur = conn.cursor()

            cur.execute("SELECT * FROM users_1 WHERE email = %s AND password = %s", (email, hashed_password))
            user = cur.fetchone()

            cur.close()
            conn.close()

            if user:
                session['user'] = user[0]  # Storing user id in session
                return redirect('/')
            else:
                return "Invalid email or password. Please try again."

    return render_template('login.html')

@app.route('/interest')
def loan_form():
    return render_template('loan_form.html')

@app.route('/loan_calculator', methods=['GET', 'POST'])
def loan_calculator():
    if request.method == 'POST':
        principal = float(request.form['principal'])
        rate = 11  # Fixed interest rate at 11%
        time = int(request.form['time'])

        interest = (principal * rate * time) / 100

        return render_template('loan_form.html', interest=interest, principal=principal, rate=rate, time=time)
    else:
        return render_template('loan_form.html')


def play_audio(filename):
    sound = pyglet.media.load(filename, streaming=False)
    sound.play()
    pyglet.app.run()

@app.route('/speak', methods=['POST'])
def speak():
    data = request.get_json()
    print("Received JSON data:", data)  # Print received JSON data for debugging
    text = data.get('text')
    lang = data.get('lang')

    if text is None or lang is None:
        return jsonify({'error': 'Missing text or lang in request'}), 400

    # Choose language for gTTS
    lang_code = 'en' if lang == 'en' else 'hi'

    # Ensure text is not empty
    if not text.strip():
        return jsonify({'error': 'Text cannot be empty'}), 400

    try:
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_audio:
            tts = gTTS(text, lang=lang_code)
            tts.write_to_fp(tmp_audio)

        # Play the audio file
        play_audio(tmp_audio.name)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'success': True})


# Configure the generative AI
genai.configure(api_key="AIzaSyBTc042cwNJZNyRUQcrUtoHr-LAzLcEA6o")

# Set up the model
generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
]

gmodel = genai.GenerativeModel(model_name="gemini-pro",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

@app.route('/chatbot')
def chatbot():
    return render_template('bot.html')

@app.route('/generate', methods=['POST'])
def generate():
    user_input = request.form['user_input']
    convo = gmodel.start_chat(history=[
        {
            "role": "user",
            "parts": [user_input]
        },
        {
            "role": "model",
            "parts": [""]  # Dummy message
        },
    ])

    # Send user input and get the model's response
    convo.send_message(user_input)

    # Return the model's response
    return jsonify({'response': convo.last.text})

# Route for logout
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/')

# About route
@app.route("/about")
def about():
    return render_template("about.html")

# Contact route
@app.route("/condition")
def contact():
    return render_template("T&C.html")

@app.route("/index")
def index_home():
    return render_template("index.html")
	
if __name__ == '__main__':
    app.run(debug=True)
