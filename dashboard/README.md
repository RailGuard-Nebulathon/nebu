# Dashboard

Install `pip install -e ".[dashboard]"`, then run `streamlit run dashboard/app.py`. Demo mode is
checkpoint-free and explicitly marks simulated values. Real mode reads outputs produced by inference.
Uploads are supported; production predictions require a trusted trained bundle configured by the user.

