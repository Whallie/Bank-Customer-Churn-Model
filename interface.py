import streamlit as st
import pickle
import pandas as pd

with open('bank_churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

def pred(creditscore, geo, gender, age, tenure, balance, num_prod, has_card, active_member, est_salary, complain, sat_score, card_type, points_earned):
    Customer = pd.DataFrame({
        'CreditScore'       :      [creditscore],
        'Geography'         :      [geo],
        'Gender'            :      [gender],
        'Age'               :      [age],
        'Tenure'            :      [tenure],
        'Balance'           :      [balance],
        'NumOfProducts'     :      [num_prod],
        'HasCrCard'         :      [has_card],
        'IsActiveMember'    :      [active_member],
        'EstimatedSalary'   :      [est_salary],
        'Complain'          :      [complain],
        'Satisfaction Score':      [sat_score],
        'Card Type'         :      [card_type],
        'Point Earned'      :      [points_earned],
        'StartAge'          :      [age - tenure]
    })
    
    return model.predict_proba(Customer)

st.set_page_config(page_title="Bank Churn Predictor", page_icon="🏦")
st.title("🏦 Bank Customer Churn Predictor")
st.markdown('''
## To make prediction you need to enter all the detials below:
''')
creditscore = st.text_input('Credit score', value="")

geo = st.selectbox(
    "Customer's Residence",
    ["France", "Spain", "Germany"],
)

gender = st.selectbox(
    "Customer's Gender",
    ["Male", "Female"],
)
