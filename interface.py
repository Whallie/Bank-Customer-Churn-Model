import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np

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
    })
    Customer.loc[Customer['HasCrCard'] == 0, 'Card Type'] = 'Nocard'
    Customer.loc[Customer['HasCrCard'] == 0, 'Point Earned'] = 0

    Customer['Gender'] = Customer['Gender'].map({'Male': 0, 'Female': 1})
    Customer['Geography'] = Customer['Geography'].map({'France': 0, 'Spain': 1, 'Germany': 2})
    Customer['Card Type'] = Customer['Card Type'].map({'DIAMOND': 0, 'GOLD': 1, 'SILVER': 2, 'PLATINUM': 3, 'Nocard': 4})
    Customer['StartAge'] = Customer['Age'] - Customer['Tenure']
    
    return model.predict_proba(Customer), model.predict(Customer)

st.set_page_config(page_title="Bank Churn Predictor", page_icon="🏦")
st.title("🏦 Bank Churn Predictor")
st.markdown('''
## To make prediction you need to fill all the detials below about the customer:
''')
creditscore = st.number_input(
    "Credit Score",
    value=None,
    placeholder="Type a number...",
    max_value = 850,
    step = 1
)

geo = st.selectbox(
    "Residence",
    ["France", "Spain", "Germany"],
    index = None,
    placeholder= 'Select Residence...'
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"],
        index = None,
    placeholder= 'Select Gender...'
)

age = st.number_input(
    "Age",
    value=None,
    placeholder="Type a number...",
    max_value = 200,
    step = 1
)

tenure = st.number_input(
    "Tenure",
    value=None,
    placeholder="Type a number...",
    max_value=age,
    step = 1
)

balance = st.number_input(
    "Balance",
    value=None,
    placeholder="Type a number (Can be decimal)...",
    step = 1.0
)

num_prod = st.number_input(
    "Number of Products",
    value=None,
    placeholder="Type a number...",
    step = 1
)

est_salary = st.number_input(
    "Salary (Can be Estimated)",
    value=None,
    placeholder="Type a number (Can be decimal)...",
    step = 1.0
)

sat_score = st.selectbox(
    "Satisfication Score",
    [1,2,3,4,5],
    index = None,
    placeholder= 'Select Score...'
)

has_card = 1 if st.checkbox("Does the customer have credit card?", value=False) else 0

if (has_card):
    card_type = st.selectbox(
        "Card Type",
        ["DIAMOND", "GOLD", "SILVER", "PLATINUM"],
        index = None,
        placeholder= 'Select Card Type...'
    )
    points_earned = st.number_input(
        "Points Earned",
        value=None,
        placeholder="Type a number...",
        step = 1
    )
else :
    card_type = 'Nocard'
    points_earned = 0

active_member = 1 if st.checkbox("Is the customer an Active member?",value=False) else 0
complain = 1 if st.checkbox("Does the customer have any complaints?",value=False) else 0

check = False

if (creditscore != None and
    geo != None and
    gender != None and 
    age != None and
    tenure != None and
    balance != None and
    num_prod != None and
    has_card != None and
    active_member != None and
    est_salary != None and
    complain != None and
    sat_score != None and
    card_type != None and
    points_earned != None):
    # st.write('YES')
    check = True
else:
    # st.write('NNO')
    # st.caption("If you see this message, it means you haven't filled in all the details yet.")
    check = False

if (check):
    left, middle, right = st.columns(3)
    if middle.button("Make Prediction!", type="primary", icon="🚀", width="stretch"):
        preprob, predic = pred(creditscore, geo, gender, age, tenure, balance, num_prod, has_card, active_member, est_salary, complain, sat_score, card_type, points_earned)
        ex = preprob[0][1]
        nex = preprob[0][0]
        res = predic
        st.markdown(f'### Probability of Exiting: {ex*100:.2f}%')
        st.markdown(f'### Probability of Not Exiting: {nex*100:.2f}%')
        st.markdown(f'### Result: {'Exiting' if res == 1 else 'Not Exiting'}')