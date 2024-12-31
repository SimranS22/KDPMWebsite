from flask import Flask, request, render_template

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split

app = Flask(__name__)

df = pd.read_csv('encoded.csv')

X = df.drop(['target'],axis=1)
Y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train , X_test, Y_train, Y_test = train_test_split(X_scaled,Y,test_size=0.30,random_state=0)

gb = GradientBoostingClassifier()
gb.fit(X_train, Y_train)
# gb_pred = gb.predict(X_test)


@app.route('/')
def man():
    return render_template('index.html')

@app.route('/back', methods=['GET'])
def back():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def get_form_data():
    Name = request.form.get('Name')
    Age = request.form.get('Age')
    BloodPressure = request.form.get('BloodPressure')
    SpecificGravity = request.form.get('SpecificGravity')
    Albumin = request.form.get('Albumin')
    Sugar = request.form.get('Sugar')
    RedBloodCells = request.form.get('RedBloodCells')
    PusCell = request.form.get('PusCell')
    PusCellClumps = request.form.get('PusCellClumps')
    Bacteria = request.form.get('Bacteria')
    BloodGlucoseRandom = request.form.get('BloodGlucoseRandom')
    BloodUrea = request.form.get('BloodUrea')
    SerumCreatinine = request.form.get('SerumCreatinine')
    Sodium = request.form.get('Sodium')
    Potassium = request.form.get('Potassium')
    Haemoglobin = request.form.get('Haemoglobin')
    PackedCellVolume = request.form.get('PackedCellVolume')
    WhiteBloodCellCount = request.form.get('WhiteBloodCellCount')
    RedBloodCellCount = request.form.get('RedBloodCellCount')
    Hypertension = request.form.get('Hypertension')
    DiabetesMellitus = request.form.get('DiabetesMellitus')
    CoronaryArteryDisease = request.form.get('CoronaryArteryDisease')
    Appetite = request.form.get('Appetite')
    PedalEdema = request.form.get('PedalEdema')
    Anaemia = request.form.get('Anaemia')

    # Standardization function
    def standardize(x, mean, standard_deviation):
        return (float(x) - float(mean)) / float(standard_deviation)

    # Standardize numerical features
    age = standardize(Age, 51.852792, 16.642435)
    blood_pressure = standardize(BloodPressure, 76.649746, 12.187850)  # You need to define mean_bp and std_bp
    blood_glucose_random = standardize(BloodGlucoseRandom, 144.319797, 75.023655) 
    specific_gravity = standardize(SpecificGravity,1.017754,0.005430)
    albumin_lvl = standardize(Albumin,0.885787,1.297738)
    sugar_lvl = standardize(Sugar,0.390863,1.031078)
    blood_urea_lvl = standardize(BloodUrea,56.637563,49.212389)
    serum_creatinine_lvl = standardize(SerumCreatinine,2.997589,5.649333)
    sodium_lvl = standardize(Sodium,137.755076,9.113033)
    potassium_lvl = standardize(Potassium,4.590102,2.839906)
    haemoglobin_lvl = standardize(Haemoglobin,12.557868,2.731251)
    packed_cell_volume = standardize(PackedCellVolume,39.126904,8.204395)
    white_blood_cell_count = standardize(WhiteBloodCellCount,8280.964467,2499.275841)
    red_blood_cell_count = standardize(RedBloodCellCount,4.744416,0.843344)


    # Convert categorical features to binary
    pus_cell = 1 if PusCell == "normal" else 0
    pus_cell_clumps = 1 if PusCellClumps == "present" else 0
    rbc = 1 if RedBloodCells == "normal" else 0 
    bacteria = 1 if Bacteria == "present" else 0
    hypertension = 1 if Hypertension == "yes" else 0
    dm = 1 if DiabetesMellitus == "yes" else 0
    cad = 1 if CoronaryArteryDisease == "yes" else 0
    appetite = 1 if Appetite == "poor" else 0
    pedaledema = 1 if PedalEdema == "yes" else 0
    anaemia = 1 if Anaemia == "yes" else 0

    # Create input DataFrame
    input_df = pd.DataFrame({
    'age': [float(age)],
    'blood_pressure': [float(blood_pressure)],
    'specific_gravity': [float(specific_gravity)],
    'albumin_lvl': [float(albumin_lvl)],
    'sugar_lvl': [float(sugar_lvl)],
    'red_blood_cells': [float(rbc)],
    'pus_cell': [float(pus_cell)],
    'pus_cell_clumps': [float(pus_cell_clumps)],
    'bacteria': [float(bacteria)],
    'blood_glucose_random_lvl': [float(blood_glucose_random)],
    'blood_urea_lvl': [float(blood_urea_lvl)],
    'serum_creatinine_lvl': [float(serum_creatinine_lvl)],
    'sodium_lvl': [float(sodium_lvl)],
    'potassium_lvl': [float(potassium_lvl)],
    'haemoglobin_lvl': [float(haemoglobin_lvl)],
    'packed_cell_volume': [float(packed_cell_volume)],
    'white_blood_cell_count': [float(white_blood_cell_count)],
    'red_blood_cell_count': [float(red_blood_cell_count)],
    'hypertension': [float(hypertension)],
    'diabetes_mellitus': [float(dm)],
    'coronary_artery_disease': [float(cad)],
    'appetite': [float(appetite)],
    'peda_edema': [float(pedaledema)],
    'anaemia': [float(anaemia)]
    })

    # Make predictions (ckd=0, notckd=1)
    result = gb.predict_proba(input_df)
    yes_ckd = round(float(result[0][0] * 100), 2)
    no_ckd = round(float(result[0][1] * 100), 2)
    return render_template('output.html', Name=Name, no_ckd=no_ckd, yes_ckd=yes_ckd)
