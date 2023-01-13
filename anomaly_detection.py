import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go

from flask import Flask
from flask_mail import Mail, Message
from pathlib import Path
from apscheduler.schedulers.blocking import BlockingScheduler
import uuid
import pyrebase
import os
import csv



def periodical_machine_learning_script():

    print("INIZIO ESECUZIONE SCRIPT")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    # SEZIONE DEDICATA ALLA CREAZIONE ED INOLTRO DELLA MAIL AL DOTTORE
    def send_mail(nome_paziente, valore_misurato, anomalia_completa):
        app = Flask(__name__)
        mail = Mail(app)

        app.config['MAIL_SERVER'] = 'smtp.gmail.com'
        app.config['MAIL_PORT'] = 465
        app.config['MAIL_USERNAME'] = 'sarcapp.anomalydetection@gmail.com'
        app.config['MAIL_PASSWORD'] = 'ljgihbqykhzdwczr'
        app.config['MAIL_USE_TLS'] = False
        app.config['MAIL_USE_SSL'] = True
        mail = Mail(app)

        image_png_path = Path(
            __file__).parent / "images/linear_graph.png"  # SI SPECIFICA IL PERCORSO DELL'IMMAGINE DA ALLEGARE
        image_png_path_2 = Path(
            __file__).parent / "images/isolation_forest.png"  # SI SPECIFICA IL PERCORSO DELL'IMMAGINE DA ALLEGARE
        voucher_id = str(uuid.uuid4())
        html = f"""<html><head></head><body>
            <p>Dear Doctor Bianchi,<br><br> The system has detected anomalies in the analysis of patient data {nome_paziente.upper()}.<br> The anomalies identified are the following: <br><br>{valore_misurato} anomaly<br>{anomalia_completa} 
            <br><br><br> <img src="cid:voucher_png" width=300>
            </p></body></html>"""
        with app.app_context():
            mail = Mail(app)
            message: Message = Message(
                subject="SarcApp - " + nome_paziente.upper() + ", " + valore_misurato + " anomaly detected!",
                sender="sarcapp2022@gmail.com",
                recipients=["inguscioa@gmail.com"], html=html)
            with app.open_resource(image_png_path_2) as fp:
                message.attach(filename="Isolation forest.png", content_type="image/png", data=fp.read(),
                               disposition="inline", headers=[['Content-ID', '<voucher_png>']])
            with app.open_resource(image_png_path) as fp:
                message.attach(filename="Line trend of data.png", content_type="image/png", data=fp.read(),
                               disposition="inline", headers=[['Content-ID', '<voucher_png>']])
            mail.send(message)
        print("LA MAIL E' STATA INOLTRATA CON SUCCESSO")

    # SEZIONE DEDICATA AL PRELIEVO DEI FILE .CSV DALLO STORAGE FIREBASE ED INSERIMENTO NELLA
    # CARTELLA machine_learning
    def download_dataset_CSV(nome_paziente):
        # SI SPECIFICA LA CARTELLA CHE SI DESIDERA CREARE
        directory = nome_paziente

        # SI SPECIFICA DOVE CREARE LA CARTELLA (LASCIANDO IL CAMPO VUOTO, "", SI SOTTOINTENDE CHE SI CREERA' LA CARTELLA NELLA STESSA
        # SORGENTE DEL main PYTHON
        path = os.path.join("Dataset_ML/", directory)

        # SI CREA LA CARTELLA
        try:
            os.mkdir(path)
        except OSError as error:
            print("")

        config = {
            "apiKey": "AIzaSyASI17IrdXGUO0YFCoGljA5_kuBkgfiLbc",
            "authDomain": "sarc-app.firebaseapp.com",
            "databaseURL": "https://sarc-app-default-rtdb.europe-west1.firebasedatabase.app",
            "projectId": "sarc-app",
            "storageBucket": "sarc-app.appspot.com",
            "messagingSenderId": "1065416353614",
            "appId": "1:1065416353614:web:9188ffc176e1419c2b201f",
            "measurementId": "G-6GFZDELGG3"
        }

        firebase = pyrebase.initialize_app(config)
        storage = firebase.storage()

        # MUSCLE_MASS
        # PERCORSO IN CUI VENGONO INSERITI/PRELEVATI I FILE SU FIREBASE
        path_on_firebase = "Dataset_ML/" + nome_paziente + "/Dataset_Muscle_" + nome_paziente + ".csv"

        # PERCORSO IN CUI VENGONO PRELEVATI I FILE DAL PC LOCALE
        # path_laptop_local = "image.jpg"

        # PRELEVA IL FILE DAL PERCORSO path_laptop_local E LO INSERISCE SU FIREBASE NEL PERCORSO path_on_firebase
        # storage.child(path_on_firebase).put(path_laptop_local)

        # SCARICA IL FILE DAL PERCORSO path_on_firebase E LO INSERISCE NEL PERCORSO FIREBASE path_laptop_local
        storage.child(path_on_firebase).download(path="",
                                                 filename="Dataset_ML/" + nome_paziente + "/Dataset_Muscle_" + nome_paziente + ".csv")

        # HAND_GRIP
        # PERCORSO IN CUI VENGONO INSERITI/PRELEVATI I FILE SU FIREBASE
        path_on_firebase = "Dataset_ML/" + nome_paziente + "/Dataset__Hgrip_" + nome_paziente + ".csv"

        # SCARICA IL FILE DAL PERCORSO path_on_firebase E LO INSERISCE NEL PERCORSO FIREBASE path_laptop_local
        storage.child(path_on_firebase).download(path="",
                                                 filename="Dataset_ML/" + nome_paziente + "/Dataset__Hgrip_" + nome_paziente + ".csv")

        # SPEED
        # PERCORSO IN CUI VENGONO INSERITI/PRELEVATI I FILE SU FIREBASE
        path_on_firebase = "Dataset_ML/" + nome_paziente + "/Dataset__Speed_" + nome_paziente + ".csv"

        # SCARICA IL FILE DAL PERCORSO path_on_firebase E LO INSERISCE NEL PERCORSO FIREBASE path_laptop_local
        storage.child(path_on_firebase).download(path="",
                                                 filename="Dataset_ML/" + nome_paziente + "/Dataset__Speed_" + nome_paziente + ".csv")

    # SEZIONE DEDICATA AL PRELIEVO DEL FILE list_of_patient.csv DALLO STORAGE FIREBASE ED INSERIMENTO NELLA
    # CARTELLA machine_learning
    def download_list_of_patient_CSV():
        config = {
            "apiKey": "AIzaSyASI17IrdXGUO0YFCoGljA5_kuBkgfiLbc",
            "authDomain": "sarc-app.firebaseapp.com",
            "databaseURL": "https://sarc-app-default-rtdb.europe-west1.firebasedatabase.app",
            "projectId": "sarc-app",
            "storageBucket": "sarc-app.appspot.com",
            "messagingSenderId": "1065416353614",
            "appId": "1:1065416353614:web:9188ffc176e1419c2b201f",
            "measurementId": "G-6GFZDELGG3"
        }

        firebase = pyrebase.initialize_app(config)
        storage = firebase.storage()

        # PERCORSO IN CUI VENGONO INSERITI I FILE SU FIREBASE
        path_on_firebase = "Dataset_ML/list_of_patient.csv"

        # PERCORSO IN CUI VENGONO PRELEVATI I FILE DAL PC LOCALE
        # path_laptop_local = "image.jpg"

        # PRELEVA IL FILE DAL PERCORSO path_laptop_local E LO INSERISCE SU FIREBASE NEL PERCORSO path_on_firebase
        # storage.child(path_on_firebase).put(path_laptop_local)

        # SCARICA IL FILE DAL PERCORSO path_on_firebase E LO INSERISCE NEL PERCORSO FIREBASE path_laptop_local
        storage.child(path_on_firebase).download(path="", filename="Dataset_ML/list_of_patient.csv")

    # RICHIAMO FUNZIONE
    # Anomaly()
    download_list_of_patient_CSV()

    # Open file
    with open('Dataset_ML/list_of_patient.csv') as file_obj:
        # Skips the heading
        # Using next() method
        heading = next(file_obj)

        # Create reader object by passing the file
        # object to reader method
        reader_obj = csv.reader(file_obj)

        # Select file
        path_of_list_patient = "Dataset_ML/list_of_patient.csv"

        # SI LEGGE IL FILE List_of_patient.csv
        results = pd.read_csv(path_of_list_patient)

        # SI CONTANO IL NUMERO DI RIGHE (ESCLUSA L'INTESTAZIONE) DEL FILE List_of_patient.csv
        number_of_row = len(results)

        single_patient = []
        # SI LEGGE UNA RIGA ALLA VOLTA E SI INSERISCE IL VALORE LETTO ALL'INTERNO DELLA LISTA single_patient
        for row in reader_obj:
            single_patient.append(row[0])

    for i in range(number_of_row):
        download_dataset_CSV(single_patient[i])

        print("VERIFICO LA PRESENZA DI ANOMALIE PER IL PAZIENTE --> " + single_patient[i])

        for j in range(3):

            if j == 0:
                nome_file = "Dataset_ML/" + single_patient[i] + "/Dataset_Muscle_" + single_patient[i] + ".csv"
                valore_misurato = "Muscle Mass"
            elif j == 1:
                nome_file = "Dataset_ML/" + single_patient[i] + "/Dataset__Hgrip_" + single_patient[i] + ".csv"
                valore_misurato = "Hand Grip"
            elif j == 2:
                nome_file = "Dataset_ML/" + single_patient[i] + "/Dataset__Speed_" + single_patient[i] + ".csv"
                valore_misurato = "Speed"

                # SEZIONE DEDICATA ALLO SVOLGIMENTO DELL'ALGORITMO DI ANOMALY DETECTION
            b_anomaly = []

            # Read data
            dataset = pd.read_csv(nome_file, parse_dates=[0])

            # Printing head of the DataFrame
            dataset.head()

            # Limiting DataFrame to specific date
            mask = (dataset['Date'] <= '2023-11-3')  # SI SPECIFICA LA DATA ENTRO CUI ANALIZZARE I DATI (2000-11-01)
            dataset = dataset

            # Plotting a part of DataFrame
            fig = px.line(dataset, x='Date', y='Total',
                          title='Anomaly Detection of ' + valore_misurato + ' - ' + single_patient[i].upper())
            # fig.show()

            # convert the column (it's a string) to datetime type
            datetime_series = pd.to_datetime(dataset['Date'])

            # create datetime index passing the datetime series
            datetime_index = pd.DatetimeIndex(dataset['Date'])

            # datetime_index
            period_index = pd.PeriodIndex(datetime_index, freq='d')

            # period_index
            dataset = dataset.set_index(period_index)

            # we don't need the column anymore
            dataset.drop('Date', axis=1, inplace=True)

            dataset.head()

            # Splitting dataset (test dataset size is last 12 periods/months)
            # y_train, y_test = temporal_train_test_split(dataset, test_size=12)

            # Cloning good dataset
            broken_dataset = dataset.copy()

            # Breaking clonned dataset with random anomaly
            # broken_dataset.loc[datetime(2022, 7, 30), ['Total']] = 26

            # Plotting DataFrame
            fig = px.line(
                broken_dataset,
                x=broken_dataset.index.astype(str),
                y=broken_dataset['Total']
            )
            fig.update_layout(
                yaxis_title='Total',
                xaxis_title='Date',
                title='Anomaly detection (with anomalies) ' + valore_misurato + ' - ' + single_patient[i].upper()
            )
            # fig.show()

            if not os.path.exists("images"):
                os.mkdir("images")
            fig.write_image("images/linear_graph.png")

            # copying dataset
            isf_dataset = broken_dataset.copy()

            # initializing Isolation Forest
            clf = IsolationForest(max_samples='auto', contamination=0.01)

            # training
            clf.fit(isf_dataset)

            # finding anomalies
            isf_dataset['Anomaly'] = clf.predict(isf_dataset)

            # saving anomalies to a separate dataset for visualization purposes
            anomalies_2 = isf_dataset.query('Anomaly == -1')
            anomalies = isf_dataset.query('Anomaly == -1')

            b1 = go.Scatter(x=isf_dataset.index.astype(str),
                            y=isf_dataset['Total'],
                            name="Dataset",
                            mode='markers'
                            )
            b2 = go.Scatter(x=anomalies.index.astype(str),
                            y=anomalies['Total'],
                            name="Anomalies",
                            mode='markers',
                            marker=dict(color='red', size=6,
                                        line=dict(color='red', width=1))
                            )

            layout = go.Layout(
                title="Isolation Forest results " + valore_misurato + ' - ' + single_patient[i].upper(),
                yaxis_title='Total',
                xaxis_title='Date',
                hovermode='closest'
            )

            data = [b1, b2]

            fig = go.Figure(data=data, layout=layout)
            fig.show()

            if not os.path.exists("images"):
                os.mkdir("images")
            fig.write_image("images/isolation_forest.png")

            # SEZIONE CHE PRELEVA SOLO I VALORI IDENTIFICATI COME "ANOMALIE" E LI INSERISCE NEL VETTORE b_anomaly
            b_anomaly = []
            b_anomaly.append(anomalies['Total'])
            # print("L'ANOMALIA A LISTA E'...")
            # print(b_anomaly[0])
            anomalia = str(b_anomaly[0])
            anomalia_completa = anomalia[0:15] + " -> Value: " + anomalia[19:24]
            print("     ANOMALIA RILEVATA PER IL VALORE DI: " + valore_misurato + " \n" + anomalia_completa)

            try:
                anomalia = str(b_anomaly[1])
                anomalia_completa = anomalia_completa + "Date: " + anomalia[0:10] + " -> Value: " + anomalia[20:24]
            except NameError:
                print("LA LISTA CONTIENTE SOLO 1 ANOMALIA")
            except:
                print("")

            # RICHIAMO DI FUNZIONE
            send_mail(single_patient[i], valore_misurato, anomalia_completa)

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("FINE ESECUZIONE SCRIPT")
    print("                                                                                       ")



scheduler = BlockingScheduler()

#SPECIFICA OGNI QUANTI MINUTI VERRA' RIPETUTA L'ESECUZIONE DELLO SCRIPT
scheduler.add_job(periodical_machine_learning_script, 'interval', minutes=1)
scheduler.start()









