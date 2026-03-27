# Project Structure

```text
refactored_streamlit_app/
├── streamlit_app.py
├── requirements.txt
├── .gitignore
├── app/
│   ├── __init__.py
│   ├── core.py
│   ├── dashboard.py
│   ├── pdf_report.py
│   ├── ui_helpers.py
│   └── paths.py
├── assets/
│   └── psi_logo.png
├── models/
│   └── my_model16082025.h5
├── docs/
│   ├── README.md
│   ├── PROJECT_STRUCTURE.md
│   └── USAGE_EXAMPLES.md
└── notebooks/
    └── cement design.ipynb
```

Code in this folder is self-contained and does not depend on the legacy flat files in the parent directory.
