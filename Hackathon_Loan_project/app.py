# import all necessary library for the app

from flask import Flask, render_template, request, redirect, session, jsonify  
import google.generativeai as genai  # Import generative AI module
import pandas as pd  
import logging  # Import logging for error handling
import sys  # Import sys for system-specific parameters and functions
import os  # Import os for interacting with the operating system
import pickle  # Import pickle for serializing and deserializing Python objects
from sklearn import preprocessing  # Import preprocessing from scikit-learn
import hashlib  # Import hashlib for secure hashing algorithms
import psycopg2  
from werkzeug.utils import secure_filename  # Import secure_filename for secure file uploads
import csv  
from gtts import gTTS  # Import gTTS for text-to-speech conversion
import tempfile  # Import tempfile for creating temporary files and directories
import pyglet  # Import pyglet for media playback

current_dir = os.path.dirname(__file__)  # Get the current directory of the file

app = Flask(__name__, static_folder='static', template_folder='template')  

app.secret_key = 'sandeep project'  # Set secret key for session management

app.logger.addHandler(logging.StreamHandler(sys.stdout))  # Add a handler to log messages to standard output
app.logger.setLevel(logging.ERROR)  # Set logging level to ERROR

with open('LR.pkl', 'rb') as f:  # Open the trained model pickle file
    model = pickle.load(f)  

UPLOAD_FOLDER = 'uploads'  # Define upload folder and allowed file extensions
ALLOWED_EXTENSIONS = {'csv'}  

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  # Configure upload folder in Flask app

@app.route('/')  # Define route for home page
def home():
    return render_template('index.html')  

@app.route('/prediction', methods=['POST'])  # Define route for prediction
def predict():
    if request.method == 'POST':  
        data = {} 
        info = {}  
        for field in request.form:  # 
            if field not in ['name', 'birthdate', 'agree-term', 'signup']:  # Exclude specific fields
                data[field] = int(request.form[field])  # Convert form field values to integers
            else:
                info[field] = request.form[field]  
        df = pd.DataFrame([data])  
        LE = preprocessing.LabelEncoder()  # Initialize LabelEncoder
        obj = (df.dtypes == 'object')  # Check for object dtype columns
        for col in list(obj[obj].index):  # Iterate through object dtype columns
            df[col] = LE.fit_transform(df[col])  # Encode object dtype columns
        result = model.predict(df)  # Predict using the trained model
        name = info['name']  # Get name from additional information
        if int(result) == 1:  # Check if result is positive
            prediction = 'Dear {name}, your loan is approved!'.format(name=name) 
        else:
            prediction = 'Sorry {name}, your loan is rejected!'.format(name=name)  
        return render_template('prediction.html', prediction=prediction)  
    else:
        return render_template('error.html', prediction="Error occurred")  

def create_upload_folder():  # Define function to create upload folder
    if not os.path.exists(UPLOAD_FOLDER):  
        os.makedirs(UPLOAD_FOLDER)  

create_upload_folder()  

@app.route('/predict_csv', methods=['POST'])  # rout for the csv prediction
def predict_csv():
    if request.method == 'POST':  
        if 'file' not in request.files:  
            return render_template('error.html', prediction="No file uploaded")  
        file = request.files['file']  # Get uploaded file
        if file.filename == '':  
            return render_template('error.html', prediction="No file selected")  
        try:
            df1 = pd.read_csv(file)  # Read CSV file into DataFrame
        except Exception as e:
            return render_template('error.html', prediction="Error occurred while reading the file")  # Render error template
        df = df1[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome',
                  'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']]  # Select relevant columns
        names = df1['Name']  # Get names from DataFrame
        all_predictions = []  
        LE = preprocessing.LabelEncoder()  # Initialize LabelEncoder
        obj = (df.dtypes == 'object')  # Check for object dtype columns
        for col in list(obj[obj].index):  
            df[col] = LE.fit_transform(df[col])  # Encode object dtype columns
        result = model.predict(df)  # Predict using the trained model
        for i, res in enumerate(result):  
            if int(res) == 1:  # Check if result is positive
                prediction = 'Approved'  
            else:
                prediction = 'Rejected'  
            all_predictions.append({'sequence': i + 1, 'name': names[i], 'prediction': prediction})  
        return render_template('prediction.html', predictions=all_predictions)  
    else:
        return render_template('error.html', prediction="Error occurred")  # Render error template

def allowed_file(filename):  # Define function to check allowed file extensions
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS  # Check if file extension is allowed

def connect_to_db():  # Define function to establish connection with PostgreSQL
    conn = psycopg2.connect(
        dbname="dhp2024",
        user="postgres",
        password="pradeep",
        host="localhost"
    )
    return conn 

@app.route('/login', methods=['GET', 'POST']) # route for the login and signup page
def signup_login():
    if request.method == 'POST':  
        if 'signup' in request.form:  
            username = request.form['username']  
            mobile = request.form['mobile']  
            email = request.form['email']  
            password = request.form['password']  
            hashed_password = hashlib.sha256(password.encode()).hexdigest()  
            conn = connect_to_db()  # Establish connection with database
            cur = conn.cursor()  
            cur.execute("INSERT INTO users_1 (username, mobile, email, password) VALUES (%s, %s, %s, %s)",
                        (username, mobile, email, hashed_password))  # Execute SQL query to insert user data
            conn.commit()  
            cur.close()  
            conn.close()  
            return redirect('/')  
        elif 'login' in request.form:  # Check if login button is clicked
            email = request.form['email']  # Get email from form
            password = request.form['password']  # Get password from form
            hashed_password = hashlib.sha256(password.encode()).hexdigest()  
            conn = connect_to_db()  
            cur = conn.cursor()  
            cur.execute("SELECT * FROM users_1 WHERE email = %s AND password = %s", (email, hashed_password))  # Execute SQL query to fetch user data
            user = cur.fetchone()  
            cur.close()  
            conn.close()  
            if user:  
                session['user'] = user[0]  
                return redirect('/')  # Redirect to home page
            else:
                return "Invalid email or password. Please try again."  
    return render_template('login.html')  

@app.route('/interest')  # Define route for loan form
def loan_form():
    return render_template('loan_form.html')  

@app.route('/loan_calculator', methods=['GET', 'POST'])  # Define route for loan calculator
def loan_calculator():
    if request.method == 'POST':  
        principal = float(request.form['principal'])  # Get principal amount from form
        rate = 11  # Fixed interest rate at 11%
        time = int(request.form['time'])  # Get time period from form
        interest = (principal * rate * time) / 100 
        return render_template('loan_form.html', interest=interest, principal=principal, rate=rate, time=time)  
    else:
        return render_template('loan_form.html')  

def play_audio(filename):  # Define function to play audio
    sound = pyglet.media.load(filename, streaming=False)  # Load audio file
    sound.play()  
    pyglet.app.run()  

@app.route('/speak', methods=['POST'])  # Define route for text-to-speech conversion
def speak():
    data = request.get_json()  # Get JSON data from request
    text = data.get('text')  # Get text from JSON data
    lang = data.get('lang')  # Get language from JSON data
    if text is None or lang is None:  # Check if text or language is missing
        return jsonify({'error': 'Missing text or lang in request'}), 400  
    lang_code = 'en' if lang == 'en' else 'hi'  
    if not text.strip():  
        return jsonify({'error': 'Text cannot be empty'}), 400  
    try:
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_audio:  
            tts = gTTS(text, lang=lang_code)  # Initialize gTTS with text and language
            tts.write_to_fp(tmp_audio)  # Write audio to temporary file
        play_audio(tmp_audio.name)  # Play audio file
    except Exception as e:
        return jsonify({'error': str(e)}), 500  
    return jsonify({'success': True})  

# Configure the generative AI
genai.configure(api_key="your_api_key_here")

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

@app.route('/chatbot')  # Define route for chatbot page
def chatbot():
    return render_template('bot.html')  # Render chatbot page

@app.route('/generate', methods=['POST'])  # Define route for generating responses
def generate():
    user_input = request.form['user_input']  
    convo = gmodel.start_chat(history=[
        {
            "role": "user",
            "parts": [user_input]
        },
        {
            "role": "model",
            "parts": [""]
        },
    ])  # Start conversation with user input
    convo.send_message(user_input)
    return jsonify({'response': convo.last.text})  

@app.route('/logout')  # Define route for logout
def logout():
    session.pop('user', None)  # Remove user from session
    return redirect('/')  

@app.route("/about")  # Define route for about page
def about():
    return render_template("about.html")  

@app.route("/condition")  # Define route for terms and conditions page
def contact():
    return render_template("T&C.html")  

@app.route("/index")  # Define route for index page
def index_home():
    return render_template("index.html")  

if __name__ == '__main__':  
    app.run(debug=True)  
