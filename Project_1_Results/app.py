import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, average_precision_score

st.set_page_config(page_title="Fraud Detection Model Lab", page_icon="💳", layout="wide")
st.title("💳 Financial Transaction Anomaly Detection")
st.caption("Educational prototype. Alerts indicate unusual transactions, not confirmed fraud.")

@st.cache_data
def demo_data(n=3000, seed=42):
    rng=np.random.default_rng(seed)
    amount=rng.lognormal(3.2,.8,n); hour=rng.integers(0,24,n)
    distance=np.abs(rng.normal(8,12,n)); velocity=rng.poisson(1.5,n)
    y=(((amount>np.quantile(amount,.985))&(hour<5))|((distance>55)&(velocity>3))).astype(int)
    flip=rng.random(n)<.015; y[flip]=1-y[flip]
    return pd.DataFrame({"Amount":amount.round(2),"Hour":hour,"DistanceFromUsualKm":distance.round(2),"TransactionsLastHour":velocity,"Class":y})

with st.sidebar:
    st.header("Controls")
    source=st.radio("Dataset",["Demo data","Upload CSV"])
    uploaded=st.file_uploader("Upload CSV",type=["csv"]) if source=="Upload CSV" else None
if source=="Upload CSV" and uploaded is None:
    st.info("Upload a CSV to begin."); st.stop()
try:
    df=pd.read_csv(uploaded) if source=="Upload CSV" else demo_data()
except Exception as e:
    st.error(f"Could not read CSV: {e}"); st.stop()

st.write(f"Rows: **{len(df):,}** | Columns: **{len(df.columns)}**")
st.dataframe(df.head(),use_container_width=True,hide_index=True)
opts=["(no label)"]+list(df.columns)
idx=opts.index("Class") if "Class" in opts else 0
label=st.selectbox("Ground-truth label column (0=legitimate, 1=fraud)",opts,index=idx)
label=None if label=="(no label)" else label
X=df.drop(columns=[label],errors="ignore").select_dtypes(include=[np.number]).copy()
X=X.replace([np.inf,-np.inf],np.nan).dropna(axis=1,how="all")
if X.empty: st.error("No usable numeric feature columns found."); st.stop()

if label is None:
    if st.button("Detect unusual transactions",type="primary"):
        pipe=Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler()),
                       ("model",IsolationForest(n_estimators=200,contamination=.03,random_state=42,n_jobs=-1))])
        pipe.fit(X); Xi=pipe.named_steps["imputer"].transform(X); Xs=pipe.named_steps["scale"].transform(Xi)
        raw=pipe.named_steps["model"].predict(Xs); scores=-pipe.named_steps["model"].score_samples(Xs)
        out=df.copy(); out["AnomalyScore"]=scores; out["Prediction"]=np.where(raw==-1,"Review","Typical")
        a,b,c=st.columns(3); a.metric("Rows analysed",f"{len(out):,}"); b.metric("Flagged for review",int((raw==-1).sum())); c.metric("Numeric features",X.shape[1])
        fig,ax=plt.subplots(); ax.hist(scores,bins=30); ax.set_xlabel("Anomaly score (higher = more unusual)"); ax.set_ylabel("Records"); st.pyplot(fig); plt.close(fig)
        st.dataframe(out.sort_values("AnomalyScore",ascending=False).head(100),use_container_width=True,hide_index=True)
        st.download_button("Download anomaly results CSV",out.to_csv(index=False).encode(),"anomaly_results.csv","text/csv")
        st.warning("Unusual does not mean fraudulent. Without labels, fraud performance cannot be measured.")
    st.stop()

try:
    y=pd.to_numeric(df[label],errors="raise")
    valid=y.notna(); X=X.loc[valid]; y=y.loc[valid].astype(int); df=df.loc[valid].copy()
    if y.nunique()!=2 or not set(y.unique()).issubset({0,1}): st.error("Label must be binary 0/1."); st.stop()
    if y.value_counts().min()<2: st.error("Need at least two records from each class."); st.stop()
except Exception as e:
    st.error(f"Could not use label: {e}"); st.stop()

model_choice=st.selectbox("Model results",["Compare both","Random Forest","Isolation Forest"])
if not st.button("Train and evaluate model",type="primary"): st.info("Click the button to train and evaluate."); st.stop()
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)

rf=Pipeline([("imputer",SimpleImputer(strategy="median")),("model",RandomForestClassifier(n_estimators=200,class_weight="balanced_subsample",random_state=42,n_jobs=-1,min_samples_leaf=2))])
with st.spinner("Training Random Forest..."):
    rf.fit(Xtr, ytr)
    joblib.dump(rf, "fraud_detection_pipeline.joblib")
    rfpred = rf.predict(Xte)
    rfprob = rf.predict_proba(Xte)[:, 1]

iso=Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",IsolationForest(n_estimators=200,contamination=.01,random_state=42,n_jobs=-1))])
with st.spinner("Training Isolation Forest baseline..."):
    iso.fit(Xtr)
    itr=iso.named_steps["imputer"].transform(Xtr); ite=iso.named_steps["imputer"].transform(Xte)
    train_scores=-iso.named_steps["model"].score_samples(iso.named_steps["scale"].transform(itr))
    test_scores=-iso.named_steps["model"].score_samples(iso.named_steps["scale"].transform(ite))
    threshold=np.quantile(train_scores,.99); isopred=(test_scores>=threshold).astype(int)

rows=[]
if model_choice in ["Compare both","Random Forest"]:
    rows.append({"Model":"Random Forest","Precision":precision_score(yte,rfpred,zero_division=0),"Recall":recall_score(yte,rfpred,zero_division=0),"F1-score":f1_score(yte,rfpred,zero_division=0),"Average precision (PR-AUC)":average_precision_score(yte,rfprob)})
if model_choice in ["Compare both","Isolation Forest"]:
    rows.append({"Model":"Isolation Forest","Precision":precision_score(yte,isopred,zero_division=0),"Recall":recall_score(yte,isopred,zero_division=0),"F1-score":f1_score(yte,isopred,zero_division=0),"Average precision (PR-AUC)":average_precision_score(yte,test_scores)})
metrics=pd.DataFrame(rows)
st.subheader("Held-out test results")
st.caption(f"Training rows: {len(Xtr):,}; test rows: {len(Xte):,}. Test labels were not used to fit the models.")
st.dataframe(metrics.style.format({c:"{:.4f}" for c in ["Precision","Recall","F1-score","Average precision (PR-AUC)"]}),use_container_width=True,hide_index=True)
predictions=pd.DataFrame({"ActualClass":yte,"RandomForestPrediction":rfpred,"RandomForestFraudProbability":rfprob,"IsolationForestPrediction":isopred,"IsolationForestAnomalyScore":test_scores},index=Xte.index)
for name,pred in [("Random Forest",rfpred),("Isolation Forest",isopred)]:
    if model_choice=="Compare both" or model_choice==name:
        st.subheader(f"{name} confusion matrix")
        cm=confusion_matrix(yte,pred,labels=[0,1])
        st.dataframe(pd.DataFrame(cm,index=["Actual legitimate (0)","Actual fraud (1)"],columns=["Predicted legitimate (0)","Predicted fraud (1)"]))
st.subheader("Random Forest feature importance")
if model_choice!="Isolation Forest":
    st.bar_chart(pd.Series(rf.named_steps["model"].feature_importances_,index=X.columns).sort_values(ascending=False).head(15))
st.subheader("Test predictions")
st.dataframe(predictions.head(100),use_container_width=True)
st.download_button("Download test predictions CSV",predictions.to_csv(index=True).encode(),"fraud_test_predictions.csv","text/csv")
st.download_button("Download metrics CSV",metrics.to_csv(index=False).encode(),"fraud_metrics.csv","text/csv")
st.warning("Prototype only. Fraud is rare; interpret precision/recall and use held-out results. Do not tune repeatedly against the test set or use this system to make real financial decisions.")
