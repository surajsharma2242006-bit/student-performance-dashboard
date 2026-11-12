# Student Performance Dashboard

An interactive Streamlit website for exploring student performance data and trying a sample score prediction model.

## Features

- Filterable cohort dashboard with performance bands and supportive check-in indicators.
- Planner-inspired lavender and teal styling with coordinated charts and rounded dashboard panels.
- Individual and batch predictions, session-only student records, and prediction history.
- Downloadable PDF summaries for individual predictions.
- Study assistant with an offline dashboard helper and optional OpenAI chat.
- OIDC sign-in and administrator tools when authentication is configured.

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

## Enable sign-in, admin tools, and AI chat

Create `.streamlit/secrets.toml` locally (this file is ignored by Git) and configure an OIDC provider. This example uses Google; register the redirect URL with the provider and update it to your deployed app URL when hosting the app.

```toml
[auth]
redirect_uri = "http://localhost:8501/oauth2callback"
cookie_secret = "<long-random-secret>"
client_id = "<oidc-client-id>"
client_secret = "<oidc-client-secret>"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"

[app]
admin_emails = ["admin@example.com"]
openai_api_key = "<optional-openai-api-key>"
openai_model = "gpt-4o-mini"
```

Only email addresses listed in `admin_emails` receive the Admin Console and model retraining access. Without OIDC configured, the app runs in demo mode without sign-in; configure OIDC before deploying with private or real student data. The AI key is optional: without it, the chat page provides a small local helper. When the key is configured, chat prompts are sent to OpenAI, so do not include student names, IDs, or other identifying details. Session records are temporary and are cleared when the app session ends.

Predictions are educational estimates and should not be used as final academic or administrative decisions.
