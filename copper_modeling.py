from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import streamlit as st
import joblib
import pandas as pd

df = pd.read_csv('.../Copper_Set Result 1.csv', low_memory=False)

st.subheader('Industrial Copper Modelling')

t1, t2 = st.tabs(['Regression model', 'Classification model'])


def set_values(item_type, status):
    item_type_v = {
        'item type_Others': 1 if item_type == 'Others' else 0,
        'item type_W': 1 if item_type == 'W' else 0,
        'item type_WI': 1 if item_type == 'WI' else 0,
        'item type_S': 1 if item_type == 'S' else 0,
        'item type_PL': 1 if item_type == 'PL' else 0,
        'item type_IPL': 1 if item_type == 'IPL' else 0,
        'item type_SLAWR': 1 if item_type == 'SLAWR' else 0
    }
    status_v = {
        'status_Won': 1 if status == 'Won' else 0,
        'status_Draft': 1 if status == 'Draft' else 0,
        'status_To be approved': 1 if status == 'To be approved' else 0,
        'status_Lost': 1 if status == 'Lost' else 0,
        'status_Not lost for AM': 1 if status == 'Not lost for AM' else 0,
        'status_Wonderful': 1 if status == 'Wonderful' else 0,
        'status_Revised': 1 if status == 'Revised' else 0,
        'status_Offered': 1 if status == 'Offered' else 0,
        'status_Offerable': 1 if status == 'Offerable' else 0
    }

    return item_type_v, status_v


def set_values_classifier(item_type):
    item_type_v = {
        'item type_Others': 1 if item_type == 'Others' else 0,
        'item type_W': 1 if item_type == 'W' else 0,
        'item type_WI': 1 if item_type == 'WI' else 0,
        'item type_S': 1 if item_type == 'S' else 0,
        'item type_PL': 1 if item_type == 'PL' else 0,
        'item type_IPL': 1 if item_type == 'IPL' else 0,
        'item type_SLAWR': 1 if item_type == 'SLAWR' else 0
    }

    return item_type_v


with t1:
    quantity_tons = st.number_input('Quantity tons', min_value=0.000001, max_value=1000.00,key='r1')
    customer = st.selectbox('Customer', df['customer'].unique(),key='r2')
    country = st.selectbox('Country',
                           [28.0, 25.0, 30.0, 32.0, 38.0, 78.0, 27.0, 77.0, 113.0, 79.0, 26.0, 39.0, 40.0, 84.0,
                            80.0, 107.0, 89.0],key='r3')
    status = st.selectbox('Status', ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM',
                                     'Wonderful', 'Revised', 'Offered', 'Offerable'])
    item_type = st.selectbox('Item Type', ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR'],key='r4')
    application = st.selectbox('Application',
                               [10.0, 41.0, 28.0, 59.0, 15.0, 4.0, 38.0, 56.0, 42.0, 26.0, 27.0, 19.0, 20.0, 66.0,
                                29.0, 22.0, 40.0, 25.0, 67.0, 79.0, 3.0, 99.0, 2.0, 5.0, 39.0, 69.0, 70.0, 65.0,
                                58.0, 68.0],key='r5')
    thickness = st.number_input('Thickness',key='r6')
    width = st.number_input('Width',key='r7')
    product_ref = st.selectbox('Product Reference',
                               [1670798778, 1668701718, 628377, 640665, 611993, 1668701376, 164141591, 1671863738,
                                1332077137, 640405, 1693867550, 1665572374, 1282007633, 1668701698, 628117,
                                1690738206, 628112, 640400, 1671876026, 164336407, 164337175, 1668701725,
                                1665572032, 611728, 1721130331, 1693867563, 611733, 1690738219, 1722207579,
                                929423819, 1665584320, 1665584662, 1665584642],key='r8')

    item_type_values, status_values = set_values(item_type, status)
    item_type_Others = item_type_values['item type_Others']
    item_type_W = item_type_values['item type_Others']
    item_type_WI = item_type_values['item type_WI']
    item_type_S = item_type_values['item type_S']
    item_type_PL = item_type_values['item type_PL']
    item_type_IPL = item_type_values['item type_IPL']
    item_type_SLAWR = item_type_values['item type_SLAWR']

    status_Won = status_values['status_Won']
    status_Draft = status_values['status_Draft']
    status_To_be_approved = status_values['status_To be approved']
    status_Lost = status_values['status_Lost']
    status_Not_lost_for_AM = status_values['status_Not lost for AM']
    status_Wonderful = status_values['status_Wonderful']
    status_Revised = status_values['status_Revised']
    status_Offered = status_values['status_Offered']
    status_Offerable = status_values['status_Offerable']

    if st.button('Predict',key='s1'):
        model = joblib.load('.../regress_model.joblib')
        independent_var = [item_type_IPL, item_type_Others, item_type_PL, item_type_S, item_type_SLAWR,
                           item_type_W, item_type_WI, status_Draft, status_Lost, status_Not_lost_for_AM,
                           status_Offerable,
                           status_Offered, status_Revised, status_To_be_approved, status_Won, status_Wonderful,
                           quantity_tons,
                           customer, country, application, thickness, width, product_ref]
        price = model.predict([independent_var])
        result = price[0]
        st.write(f'Predicted Selling Price: {result}')

with t2:
    quantity_tons = st.number_input('Quantity tons', min_value=0.000001, max_value=1000.00,key='c1')
    customer = st.selectbox('Customer', df['customer'].unique(),key='c2')
    country = st.selectbox('Country',
                           [28.0, 25.0, 30.0, 32.0, 38.0, 78.0, 27.0, 77.0, 113.0, 79.0, 26.0, 39.0, 40.0, 84.0,
                            80.0, 107.0, 89.0],key='c3')
    application = st.selectbox('Application',
                               [10.0, 41.0, 28.0, 59.0, 15.0, 4.0, 38.0, 56.0, 42.0, 26.0, 27.0, 19.0, 20.0, 66.0,
                                29.0, 22.0, 40.0, 25.0, 67.0, 79.0, 3.0, 99.0, 2.0, 5.0, 39.0, 69.0, 70.0, 65.0,
                                58.0, 68.0],key='c4')
    thickness = st.number_input('Thickness',key='c5')
    width = st.number_input('Width',key='c6')
    product_ref = st.selectbox('Product Reference',
                               [1670798778, 1668701718, 628377, 640665, 611993, 1668701376, 164141591, 1671863738,
                                1332077137, 640405, 1693867550, 1665572374, 1282007633, 1668701698, 628117,
                                1690738206, 628112, 640400, 1671876026, 164336407, 164337175, 1668701725,
                                1665572032, 611728, 1721130331, 1693867563, 611733, 1690738219, 1722207579,
                                929423819, 1665584320, 1665584662, 1665584642],key='c7')
    selling_price = st.number_input('Selling Price', min_value=0.000001, max_value=10.00,key='c8')
    item_type = st.selectbox('Item Type', ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR'],key='c9')

    item_type_values_class = set_values_classifier(item_type)
    item_type_Others = item_type_values_class['item type_Others']
    item_type_W = item_type_values_class['item type_Others']
    item_type_WI = item_type_values_class['item type_WI']
    item_type_S = item_type_values_class['item type_S']
    item_type_PL = item_type_values_class['item type_PL']
    item_type_IPL = item_type_values_class['item type_IPL']
    item_type_SLAWR = item_type_values_class['item type_SLAWR']

    if st.button('Predict',key='s2'):
        model = joblib.load('.../classify_model.joblib')
        independent_var_class = [quantity_tons, customer, country, application, thickness, width, product_ref,
                                 selling_price,
                                 item_type_IPL, item_type_Others, item_type_PL, item_type_S, item_type_SLAWR,
                                 item_type_W, item_type_WI]
        result = model.predict([independent_var_class])
        prediction = result[0]
        st.write(f'Status: {prediction}')