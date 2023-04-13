import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import sklearn
from sklearn.preprocessing import RobustScaler,MinMaxScaler,StandardScaler
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,roc_auc_score,f1_score,recall_score,precision_score,confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


loan_pred = pd.read_csv(r'/Users/hasanzeynalov/Desktop/for_github/deployment/streamlit/Streamlit-Modelling-WebApp/loan_pred.csv')
water_prob = pd.read_csv(r'/Users/hasanzeynalov/Desktop/for_github/deployment/streamlit/Streamlit-Modelling-WebApp/water_potability.csv')

#-------------------------------------------------------------------- Page Title

st.set_page_config(page_title='ML Application', layout = 'wide', initial_sidebar_state = 'auto')
# Disclaimer:
    # Structurally, we need to create a web application with three pages long. We
    # have 2 different datas that contains Water Portability and Long prediction datas.
# Task:
    # Our objective is to create automation process. Pages should be assembled
    # accordingly:


# Sidebar menu

with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=['Homepage', 'EDA', 'Modelling'],
        default_index=0,
        orientation=None,
        menu_icon='cast'
    )

#-------------------------------------------------------------------- page 1
if selected == 'Homepage':
    st.title(selected)
    # 1. The first page (i.e. homepage) should be used to introduce ourselves with the data.
    st.subheader('Deployment with Streamlit:ML Application')

    st.write('''We have 2 datasets named as a Loan prediction and Water potability.At this project,
    we should train model.There are 3 pages:''')

    lst = ['Homepage', 'Exploraty Data Analysis', 'Modelling']
    for i in lst:
        st.markdown("- " + i)
    st.subheader('Homepage')
    st.write('Homepage has all informations about that project')
    st.subheader('Exploraty Data Analysis')
    st.write('''EDA page would be greate to get inside about datas.You can see dataet,then checkin null values,
    describe of datas,imbalance checking,number of unique values and some visualizations
    like Heatmap,Line Chart,Bar Chart,Violin and Strip Plot,Boxplot''')
    st.subheader('Modelling')
    st.write('''There are datasets,options to choose scalling and encoding methods,specifyibg test size for Train Test Splitting,
    chooisng model,see the all significant evaulation metrics and some visualizations like confusion_plotly and roc_auc score''')
#---------------------------------------------------------------------- page 2
elif selected == 'EDA':
    st.title(selected)

    option = st.selectbox('Which data would you like to see now?', (
        'Loan Prediction', 'Water Potability'
    ))

    if option == 'Loan Prediction':
        st.title('Loan Prediction')
        # display dataframe
        st.dataframe(loan_pred)
        option_eda = st.selectbox('Choose the one of the some processes', (
            'Check the null values', 'Describe dataset', 'Number of unique values',
            'Inbalance Checking of Loan Status'
        ))
        if option_eda == 'Check the null values':
            st.title('Checking null values')

            # checking null values
            null_values = loan_pred.isnull().sum()
            col1, col2 = st.columns(2)
            col1.write(pd.DataFrame({'Null Values': null_values}))
            # fill the null values
            for i in loan_pred.columns:
                if loan_pred[i].isnull().sum() != 0:
                    if loan_pred[i].dtype != 'object':
                        loan_pred[i] = loan_pred[i].fillna(loan_pred[i].mean())
                    elif loan_pred[i].dtype == 'object':
                        loan_pred[i] = loan_pred[i].fillna(method='ffill')

            fill_null_values = loan_pred.isnull().sum()
            col2.write(pd.DataFrame({'Number of null values after filling': fill_null_values}))

        elif option_eda == 'Describe dataset':
            # describe dataset
            st.title('Describe of Loan Prediction')
            describing = loan_pred.describe()
            st.write(describing)
        elif option_eda == 'Number of unique values':
            st.title('Number of unique values for each features')
            num_unique = loan_pred.nunique()
            st.write(pd.DataFrame({'Number of unique values': num_unique}))
        elif option_eda == 'Inbalance Checking of Loan Status':
            st.title('Checking Inbalance')
            plt.figure(figsize=(16, 9))
            fig = plt.figure(figsize=(10, 4))
            sns.countplot(x="Loan_Status", data=loan_pred)
            st.pyplot(fig)
        st.subheader('There are some visualizations to get inside about the data.')
        vis_options = st.selectbox(
            'You can choose anyone you want!',
            ('Heatmap', 'Bar Chart', 'Boxplot', '3d Line Plot', 'Viloin and Strip plot'))

        if vis_options == 'Heatmap':
            st.title('Heatmap')
            fig, ax = plt.subplots()
            sns.heatmap(loan_pred.corr(), annot=True, ax=ax)
            st.write(fig)
        elif vis_options == '3d Line Plot':
            st.title('3d Line Plot')
            # line3d = loan_pred.query("Gender=='Male'")
            fig = px.line_3d(loan_pred, x="ApplicantIncome", y="LoanAmount", z="Loan_Amount_Term", color='Gender')
            st.write(fig)
        elif vis_options == 'Viloin and Strip plot':
            st.title('Viloin and Strip Plot')
            co1, co2 = st.columns(2)
            with co1:
                fig = px.violin(loan_pred, y='ApplicantIncome')
                st.write(fig)
            with co2:
                fig = px.strip(loan_pred, y='ApplicantIncome')
                st.write(fig)
        elif vis_options == 'Bar Chart':
            st.title('Applicant Income by Gender')
            fig, ax = plt.subplots()
            sns.barplot(data= loan_pred,x = ['ApplicantIncome'], y = ['Loan_Status'])
            st.write(fig)
        else:
            st.title('Boxplot')
            fig = px.box(loan_pred, y='ApplicantIncome')
            st.write(fig)

    else:
        st.title('Water Potability')
        # display data frame
        st.dataframe(water_prob)
        # checking null values
        null_values = water_prob.isnull().sum()
        st.write(pd.DataFrame({'Null values': null_values}))
        # describe dataset
        st.title('Describe of Dataset')
        describing = water_prob.describe()
        st.write(describing)

#---------------------------------------------------------------------- page 3

#-----Modelling

else:
    st.title(selected)
    # 3. The final page, the modeling page should be included preprocessing, variety of scalers,
    # variety of encoders, train test splitting distribution, building of the model, calculations
    # for evaluation metrics and some visualization.
    df_selection = st.selectbox('Choose one of the following dataframes',
                                options=['Loan Prediction','Water Potatibility'])
    if df_selection == 'Loan Prediction':
        st.dataframe(loan_pred)
    else:
        st.dataframe(water_prob)
#----------------------------preprocessing
    st.subheader('Preprocessing')
#----------------------------Scalling and Encoding

    encoding,scalling = st.columns(2)

    with encoding:
        encoding_options = st.radio('Encoding methods',
                                    ['Label Encoder', 'One Hot Encoder'])
        if encoding_options == 'Label Encoder':
            class MultiColumnLabelEncoder:
                def __init__(self, columns=None):
                    self.columns = columns  # array of column names to encode

                def fit(self, X, y=None):
                    return self  # not relevant here

                def transform(self, X):
                    '''
                    Transforms columns of X specified in self.columns using
                    LabelEncoder(). If no columns specified, transforms all
                    columns in X.
                    '''
                    output = X.copy()
                    if self.columns is not None:
                        for col in self.columns:
                            output[col] = LabelEncoder().fit_transform(output[col])
                    else:
                        for colname, col in output.iteritems():
                            output[colname] = LabelEncoder().fit_transform(col)
                    return output

                def fit_transform(self, X, y=None):
                    return self.fit(X, y).transform(X)
            loan_pred = MultiColumnLabelEncoder(columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area','Loan_Status']).fit_transform(loan_pred)
        else:
            class MultiColumnOneHotEncoder:
                def __init__(self, columns=None):
                    self.columns = columns  # array of column names to encode

                def fit(self, X, y=None):
                    return self  # not relevant here

                def transform(self, X):
                    '''
                    Transforms columns of X specified in self.columns using
                    LabelEncoder(). If no columns specified, transforms all
                    columns in X.
                    '''
                    output = X.copy()
                    if self.columns is not None:
                        for col in self.columns:
                            output[col] = OneHotEncoder().fit_transform(output[col])
                    else:
                        for colname, col in output.iteritems():
                            output[colname] = OneHotEncoder().fit_transform(col)
                    return output

                def fit_transform(self, X, y=None):
                    return self.fit(X, y).transform(X)

            loan_pred = MultiColumnOneHotEncoder(columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area','Loan_Status']).fit_transform(loan_pred)

    X = loan_pred.iloc[:, 0:-1]
    y = loan_pred.drop(loan_pred.iloc[:, 0:-1], axis=1)
    with scalling:
        scalling_options = st.radio('Scalling methods:',
                                    (['Standart Scaler', 'Robust Scaler', 'Min Max Scaler']))
        if scalling_options == 'Standart Scaler':
            # standart_scaler
            scaler = StandardScaler()
        elif scalling_options == 'Robust Scaler':
            # min_max_scaler
            scaler = MinMaxScaler()
        else:
            robust = RobustScaler()
        X = scaler.fit_transform(X)

#-------------------------Train Test Splitting

    st.subheader('Test size would be greate between 20% and 30%')

    train_test_slinder = st.slider(('Test size'))
    st.write(f'Your test size is {train_test_slinder}')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#-------------------------Modelling

    st.subheader('Choose your model')

    models_options = st.selectbox('',
                                  ['Logistic Regression','XGBoost','Random Forest Classsifier'])
    if models_options == 'Logistic Regression':
        log_model = LogisticRegression()
        log_model.fit(X_train, y_train)
        log_model_pre = log_model.predict(X_test)
#-------------------------Evaluation metrics

    st.subheader('Evaluation metrics')
    check = st.button('Check the performance metrics')
    if check:
        accuracy_score(log_model_pre, y_test)

#-------------------------Roc Auc Score and Confusion Matrix

    st.subheader('Roc Auc Score and Confusion matrix to see how performs our model')
    st.button('Check them all')

