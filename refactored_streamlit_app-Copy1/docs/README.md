# Refactored Streamlit App

This folder contains a cleaned-up copy of the Streamlit cement mix optimizer.

Run it with:

```bash
streamlit run refactored_streamlit_app/streamlit_app.py
```

Main layout:

- `streamlit_app.py`: app entrypoint
- `app/core.py`: merged model, metrics, optimization, sampling, and Pareto utilities
- `app/dashboard.py`: results dashboard
- `app/pdf_report.py`: PDF report generation
- `app/ui_helpers.py`: shared Streamlit UI helpers and page styling
- `app/paths.py`: asset and model path helpers
- `assets/`: static resources
- `models/`: trained model files
