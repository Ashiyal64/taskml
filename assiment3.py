import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import PolynomialFeatures
from streamlit_option_menu import option_menu
from sklearn.ensemble import  RandomForestClassifier
from sklearn.model_selection import train_test_split
import csv
import os
from sklearn.metrics import accuracy_score
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# data=pd.read_csv("titanicdataset.csv")
# data.fillna(data.mean(numeric_only=True), inplace=True)
# data.to_csv("titanic.csv")
# data=pd.read_csv("titanic.csv")

USER_FILE="siguodata.csv"


st.sidebar.title("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

# Session state to store the dataset
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    # st.session_state['dataset'] = data


if not os.path.exists(USER_FILE):
    with open(USER_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["email","password"])





if "austhonticade" not in st.session_state:
    st.session_state.authonticated=False
if "user_id" not in st.session_state:
     st.session_state.user_id=None

query_parms=st.query_params

if "auth" in query_parms and query_parms["auth"]=="true":
    st.session_state.authonticated=True
    st.session_state.user_id=query_parms.get("user","")




# selectmodel=st.selectbox("",["Linear Regression","Multiple Linear Regression","Polynomial Linear Regression","Decision Tree"])

with st.sidebar.title("Titanic dataset"):

 if st.session_state.authonticated:
   selected=option_menu("Select dataset",["preloaded data","upload dataset","lable_Encoding","columencoding","Hotcoding","Mlmodel","logout"])
 else:
     selected=option_menu("Login",["login","signup"])



if selected=="login":
    with st.form(key="form1"):
        email = st.text_input("Enetr your email")
        passward = st.text_input("Enter ayour password", type="password")
        submit = st.form_submit_button(label="submit")
        if submit:
            with open(USER_FILE, mode='r') as file:
                users = list(csv.reader(file))

            if [email,passward] in users:
                st.success("Login successful!")
                st.session_state.authenticated = True
                st.session_state.user_id = email
                st.success("login success")
                st.session_state.authonticated = True
                st.session_state.user_id = email
                st.query_params.update(auth="true", user=email)
                st.rerun()
            else:
                st.error("Invalid credentials")

if selected=="signup":
    with st.form(key="signup_form"):
        email = st.text_input("Enter your email")
        password = st.text_input("Enter your password", type="password")
        submit = st.form_submit_button(label="Sign Up")

        if submit:
            with open(USER_FILE, mode='r') as file:
                users = list(csv.reader(file))

            if any(len(row) > 0 and row[0] == email for row in users[1:]):
                st.error("Email already registered!")
            else:
                with open(USER_FILE, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([email, password])

if selected=="preloaded data":
     st.dataframe(data)




if selected=="upload dataset":
    file=st.file_uploader("Upload file",type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        st.dataframe(df)
        df.to_csv(file)
    else:
        st.warning(" select dataset")
if selected=="lable_Encoding":
    # data =pd.read_csv("titanicdataset.csv")
    labletranform=LabelEncoder()


    data["Sex"]=labletranform.fit_transform(data["Sex"])
    data["Embarked"]=labletranform.fit_transform(data["Embarked"])
    data["Survived"]=labletranform.fit_transform(data["Survived"])
    st.dataframe(data)
if selected=="columencoding":
    data=pd.read_csv("titanic.csv")
    categorical_columns = st.multiselect("Select Categorical Columns for Encoding", data.columns.tolist())

    if categorical_columns:
     encoder = LabelEncoder()

     with st.form(key="form7"):
         for col in categorical_columns:
             data[col] = encoder.fit_transform(data[col])
         encoded_df = data[categorical_columns]
         submit = st.form_submit_button(label="submit")
         if submit:
          st.dataframe(encoded_df)

if selected=="Hotcoding":
    encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoded = encoder.fit_transform(data[["Cabin"]])
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(["Cabin"])) # select colom name only in list
    df_encoded = pd.concat([data, one_hot_df], axis=1)
    df_encoded = df_encoded.drop(["Cabin"], axis=1) #select colom name only list
    st.dataframe(df_encoded)

if selected=="Mlmodel":
      selectmodel=st.selectbox("",["Linear Regression","Multiple Linear Regression","Polynomial Linear Regression","Decision Tree","RFM"])

      if selectmodel=="Linear Regression":
         reg=linear_model.LinearRegression()
         reg.fit(data[["Age"]],data["Survived"])
         with st.form(key="form1"):
             age=st.text_input("Enter a age")
             submit=st.form_submit_button(label="submit")
             if submit:
                 AGE=int(age)
                 predict=reg.predict([[AGE]])
                 st.write(predict)

                 fig=px.scatter(data,x="Age",y="Survived")
                 fig.add_trace(
                     go.Scatter(
                         x=[AGE],
                         y=[predict],
                         mode='markers+text',
                         marker=dict(
                             size=12,
                             color='red',
                             symbol='star',
                             line=dict(width=2, color='black')
                         ),
                         text=["Prediction"],
                         textposition="top center",
                         name="Prediction Point"
                     )


                 )
                 fig.update_layout(
                     plot_bgcolor="rgba(0,0,0,0)",
                     paper_bgcolor="rgba(0,0,0,0)",
                     xaxis_title="Age",
                     yaxis_title="Pclass",
                     font=dict(size=14),
                 )

                 st.plotly_chart(fig)

      if selectmodel=="Multiple Linear Regression":
        reg=linear_model.LinearRegression()
        reg.fit(data[["Pclass","Age"]],data["Survived"])


        with st.form(key="form2"):


          Pclass=st.text_input("enter a class")
          age=st.text_input("enetr a ager")
          submit = st.form_submit_button(label="submit")
          if submit:

              Pclass=int(Pclass)

              Age=int(age)
              pre=reg.predict([[Pclass,Age]])
              st.write(pre)
              fig = px.scatter(
                  data,
                  x="Age",
                  y="Pclass",
                  title="Age vs Pclass",
                  color="Pclass",
                  color_continuous_scale="Viridis",  # Color scheme
              )

              fig.add_trace(
                  go.Scatter(
                      x=[Age],
                      y=[pre],
                      mode='markers+text',
                      marker=dict(
                          size=12,
                          color='red',
                          symbol='star',
                          line=dict(width=2, color='black')
                      ),
                      text=["Prediction"],
                      textposition="top center",
                      name="Prediction Point"
                  )
              )

              fig.update_layout(
                  plot_bgcolor="rgba(0,0,0,0)",
                  paper_bgcolor="rgba(0,0,0,0)",
                  xaxis_title="Age",
                  yaxis_title="Pclass",
                  font=dict(size=14),
              )
              st.plotly_chart(fig)


      if selectmodel=="Polynomial Linear Regression":
          x=data[["Age"]]
          y=data["Survived"]
          poly=PolynomialFeatures(degree=3)
          x_poly=poly.fit_transform(x)
          model=LinearRegression()
          model.fit(x_poly,y)
          with st.form(key="form3"):
              age=st.text_input("eneter a age")
              submit=st.form_submit_button(label="submit")
              if submit:
                  Age=int(age)
                  f_day=pd.DataFrame({"Age":[Age]})
                  f_poly=poly.fit_transform(f_day)
                  f_pre=model.predict(f_poly)
                  fig,ax=plt.subplots()
                  ax.scatter(data["Age"],data["Survived"])
                  ax.scatter(Age,f_pre)
                  st.pyplot(fig)
      if selectmodel=="Decision Tree":
         labletranform = LabelEncoder()
         data["Sex"] = labletranform.fit_transform(data["Sex"])
         data["Embarked"] = labletranform.fit_transform(data["Embarked"])
         x=data[["Pclass","Sex","Age","Fare"]]
         y=data["Survived"]
         model=DecisionTreeClassifier()
         model.fit(x,y)
         with st.form(key="for4"):

              pclss=st.text_input("Entre a class")
              gender=st.text_input("Enter a gender")
              age=st.text_input("enetr a age")
              fare=st.text_input("enet a fare")
              submit=st.form_submit_button(label="submit")
              if submit:

                  Pclass=int(pclss)
                  Gender=int(gender)
                  Age=int(age)
                  Fare=int(fare)
                  pre=model.predict([[Pclass,Gender,Age,Fare]])

                  st.write(pre)
      if selectmodel=="RFM":
          labletranform = LabelEncoder()
          data["Sex"] = labletranform.fit_transform(data["Sex"])
          data["Embarked"] = labletranform.fit_transform(data["Embarked"])
          x=data[["Sex","Pclass","Age"]]
          y=data["Survived"]

          x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=43)
          rf_classification = RandomForestClassifier(n_estimators=100, random_state=43)
          rf_classification.fit(x_train, y_train)
          with st.form(key="form4"):
              gender=st.text_input('Enter a gender')
              pclass=st.text_input("Enter a pclass")
              age=st.text_input("enter a age")
              submit=st.form_submit_button(label="submit")
              if submit:
                  Gender=float(gender)
                  Pclass=float(pclass)
                  Age=float(age)

                  pre = rf_classification.predict(x_test)
                  acu = accuracy_score(y_test, pre)
                  predicution = rf_classification.predict([[Gender,Pclass,Age]])

                  st.write(predicution)
                  st.write(acu)
                  fig,ax=plt.subplots()
                  ax.scatter(data["Age"],)


if selected=="logout":
    st.session_state.authonticated = False
    st.session_state.user_id = None
    st.query_params.clear()
    st.rerun()