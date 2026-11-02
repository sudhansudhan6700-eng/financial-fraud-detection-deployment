# Improved Fraud Detection Model Lab

1. Open this folder in VS Code.
2. Install packages: `python -m pip install -r requirements.txt`
3. Run: `python -m streamlit run app.py`
4. Upload the public Kaggle credit-card dataset and select `Class` as label, then click **Train and evaluate model**.

The app compares Random Forest with Isolation Forest using a stratified held-out test set. Labels are excluded from features. Use the metrics you actually obtain; no performance is guaranteed. A flagged transaction is not proof of fraud.
