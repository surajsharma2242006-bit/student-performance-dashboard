# Student Performance Dashboard

An interactive Streamlit website for exploring student performance data and trying a sample score prediction model.

## Run it on your computer

```bash
python -m pip install -r requirements.txt
streamlit run "python project.py"
```

## Put it on the web

This project is set up for [Streamlit Community Cloud](https://share.streamlit.io/):

1. Push this repository to GitHub.
2. Sign in to Streamlit Community Cloud with GitHub and choose **Create app**.
3. Select this repository and branch, then set the app file to `python project.py`.
4. Deploy. Streamlit will provide a public `*.streamlit.app` link that you can share.

The app uses the included sample dataset and model. Individual predictions and student IDs are held only in the visitor's active session; they are not saved to a shared database. Do not upload real student records or identifying information.

Predictions are educational estimates and should not be used as final academic or administrative decisions.
