A trivial AB test designer, approaching from both frequentist and Bayesian perspectives.

1. git clone this repo
2. cd exp-exp-py
3. uv venv -p 3.12
4. source ./.venv/bin/activate
5. uv pip install -r requirements.txt
6. streamlit run app.py

Pretty amazingly, this code was converted by Gemini 2.5 from an R implementation over in https://github.com/schnee/ShinyApps/tree/master/exp-exp, and critically, I only provided the server.R file. Gemini built the UI AND the Bayesian estimates.
