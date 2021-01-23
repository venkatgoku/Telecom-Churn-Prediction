import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from itertools import combinations
import pickle


df=pd.read_csv("modified_22.csv")
df['churn'].replace(to_replace='Yes', value=1, inplace=True)
df['churn'].replace(to_replace="No",value=0,inplace=True)
df["MonthlyRevenue"].fillna(df["MonthlyRevenue"].mean,inplace=True)
df.dropna(inplace=True)
df_obj=df.select_dtypes(include="object")
df_obj.drop(["ServiceArea"],inplace=True,axis=1)
df_obj.drop(['TruckOwner','MaritalStatus','RespondsToMailOffers','Homeownership'],inplace=True,axis=1)
dummies = pd.get_dummies(df_obj)
df_fin=pd.concat([df, dummies], axis=1)
df_fin.drop(df_fin.select_dtypes(include="object"),axis=1,inplace=True)
df_fin.drop("CustomerID",inplace=True,axis=1)
df_fin.drop(["OverageMinutes","ReceivedCalls","PeakCallsInOut","OffPeakCallsInOut","DroppedBlockedCalls","Handsets","PercChangeRevenues"],inplace=True,axis=1)
df_fin["churn"] = df_fin["churn"].astype(int) 

from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
y = df_fin["churn"].values
X = df_fin[["contract_One year","MonthlyCharges","contract_Month-to-month","contract_Two year","PercChangeMinutes","MonthlyMinutes","OutboundCalls","UnansweredCalls"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)
#sm = SMOTE(random_state=27)
#X_train, y_train = sm.fit_sample(X_train, y_train)
smote = LogisticRegression(solver='liblinear').fit(X_train, y_train)

pickle.dump(smote, open('model.pkl','wb'))