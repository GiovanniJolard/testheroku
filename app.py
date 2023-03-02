import requests
import json
import pandas as pd
import numpy as np
import streamlit as st
import datetime
from datetime import date
import hashlib
import ast
import pickle
from PIL import Image


def request_prediction(model_uri, data):
    # sourcery skip: raise-specific-error, use-fstring-for-formatting
    headers = {"Content-Type": "application/json"}

    data_json = {'data': data}
    response = requests.request(method='POST', headers=headers, url=model_uri, json=data_json)

    if response.status_code != 200:
        raise Exception("Request failed with status {}, {}".format(response.status_code, response.text))
    return response.json()

columns_list = ['SK_ID_CURR', 'EXT_SOURCE_3', 'DAYS_LAST_PHONE_CHANGE', 'REGION_POPULATION_RELATIVE',
                'TOTALAREA_MODE', 'HOUR_APPR_PROCESS_START', 'DAYS_BIRTH',
                'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL', 'OBS_60_CNT_SOCIAL_CIRCLE',
                'FLAG_DOCUMENT_3', 'OBS_30_CNT_SOCIAL_CIRCLE', 'EXT_SOURCE_2',
                'DEF_60_CNT_SOCIAL_CIRCLE', 'AMT_ANNUITY', 'DAYS_ID_PUBLISH',
                'DEF_30_CNT_SOCIAL_CIRCLE', 'AMT_GOODS_PRICE','AMT_CREDIT',
                    'CNT_FAM_MEMBERS'
                ]

infos_descrip = ['SK_ID_CURR',
                 "CODE_GENDER",
                 "CNT_CHILDREN",
                 "NAME_FAMILY_STATUS",
                 "NAME_HOUSING_TYPE",
                 "NAME_CONTRACT_TYPE",
                 "NAME_INCOME_TYPE",
                 "OCCUPATION_TYPE",
                 "AMT_INCOME_TOTAL"
                 ]

def main():
    
    st.title("Mon application Streamlit")
    
    pickle_in = open("API_LGBMClassifier.pkl", "rb")
    classifier = pickle.load(pickle_in)     
    "\n"
    
    def load_data(nrows):
        data = pd.read_csv("application_train_echantillon.csv")
        data = data.sample(n=nrows, random_state=1)
        return data

    def load_data1(nrows):
        data = pd.read_csv('application_train_echantillon.csv')
        samp = data.sample(n=nrows, random_state=1)
        data2 = samp[infos_descrip].set_index('SK_ID_CURR')
        data1 = samp[columns_list].set_index('SK_ID_CURR')
        return data1, data2

    data1, data2 = load_data1(1000)

    # Selecting one client
    id_client = st.selectbox('Select ID Client :', data1.index)
    if id_client:
        # Visualizing the personal data of the selected client
        st.subheader('Les informations du client %s : ' % id_client)
        st.table(data2.astype(str).loc[id_client][1:9])
        "\n"
        
    # Afficher les données financières du client sélectionné
    st.subheader('Données financières du client %s :' % id_client)
    st.write(data1.loc[id_client])
     
# Prédiction de la solvabilité du client
    
    if st.button('Vérifier la solvabilité'):
        
        data_pred = data.set_index('SK_ID_CURR').loc[id_client].to_numpy().reshape(1, -1)
        prediction = classifier.predict(data_pred)[0]
        proba = classifier.predict_proba(data_pred)[0, 1]

        # Affichage de la prédiction de solvabilité
        st.subheader('Prédiction de solvabilité :')
        st.write('Le client %s est ' % id_client, 'solvable' if prediction else 'insolvable')
        st.write('avec une probabilité de %0.2f %%' % (proba * 100))
        
        proba_non_remboursement = 1 - proba

        # Créer le graphique
        fig, ax = plt.subplots()
        ax.pie([proba, proba_non_remboursement], labels=['Probabilité de remboursement', 'Probabilité de non-remboursement'], autopct='%1.1f%%')
        ax.set_title('Probabilité de remboursement pour le client %s' % id_client)

        # Afficher le graphique dans Streamlit
        st.pyplot(fig)
        
        
    if st.button('Voulez-vous voir les informations utilisées ?'):
        image = Image.open('image.png')
        st.image(image, caption='Importance des features', use_column_width=True)
        
    # Checkbox pour afficher les graphiques des autres clients similaires
    if st.checkbox('Les autres clients similaires ?'):
        st.subheader('Les autres clients similaires %s :' % id_client)
        feature_name = st.selectbox('Sélectionner le nom de la caractéristique :', [
            "AMT_INCOME_TOTAL",
            "DAYS_EMPLOYED",
            "REGION_POPULATION_RELATIVE",
            "DAYS_BIRTH",
            "DAYS_ID_PUBLISH",
            "AMT_CREDIT"
        ])
        fig, ax = plt.subplots()
        sns.countplot(x=feature_name, data=data1)
        ax.set_title('Distribution de la caractéristique %s pour les clients' % feature_name)
        st.pyplot(fig)

if __name__ == '__main__':
    main()
