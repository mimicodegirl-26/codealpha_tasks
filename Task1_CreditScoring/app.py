import streamlit as st
import joblib
import pandas as pd

# Page config
st.set_page_config(
    page_title="Credit Scoring AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a sleek card-based design
st.markdown("""
<style>
    /* Overall background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    /* Main content card */
    .main-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 25px;
        padding: 2.5rem;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
        margin-top: 2rem;
        color: #1e293b;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(30, 41, 59, 0.85);
        backdrop-filter: blur(10px);
        color: white;
    }
    [data-testid="stSidebar"] .stSlider > label,
    [data-testid="stSidebar"] .stSelectbox > label,
    [data-testid="stSidebar"] .stNumberInput > label {
        color: white !important;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 600;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 1rem;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
    }
    .result-good {
        background: #d1fae5;
        color: #065f46;
        padding: 1.5rem;
        border-radius: 15px;
        font-weight: bold;
        border-left: 8px solid #059669;
    }
    .result-bad {
        background: #fee2e2;
        color: #991b1b;
        padding: 1.5rem;
        border-radius: 15px;
        font-weight: bold;
        border-left: 8px solid #dc2626;
    }
    .metric-box {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    model = joblib.load('models/best_rf_model.pkl')
    preprocessor = joblib.load('models/preprocessor.pkl')
    return model, preprocessor

model, preprocessor = load_model()

# ---------- HEADER ----------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<div style='text-align: center; margin-top: 2rem;'>"
                "<h1 style='color: white; font-size: 3rem;'>💳 Credit Scoring AI</h1>"
                "<p style='color: rgba(255,255,255,0.8); font-size: 1.2rem;'>"
                "Instant loan risk prediction powered by machine learning</p></div>",
                unsafe_allow_html=True)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("## 👤 Applicant Profile")
    st.markdown("---")

    with st.expander("📋 Personal Information", expanded=True):
        age = st.slider("Age", 18, 100, 35)
        residence_since = st.slider("Years at current residence", 1, 4, 2)
        num_dependents = st.slider("Number of dependents", 1, 2, 1)
        personal_status = st.selectbox(
            "Personal status & sex",
            ["A91", "A92", "A93", "A94"],
            format_func=lambda x: {"A91": "male : divorced/separated", "A92": "female : divorced/separated/married",
                                   "A93": "male : single", "A94": "male : married/widowed"}.get(x, x)
        )

    with st.expander("💰 Financial Details", expanded=True):
        checking = st.selectbox(
            "Checking account status",
            ["A11", "A12", "A13", "A14"],
            format_func=lambda x: {"A11": "< 0 DM", "A12": "0 to < 200 DM",
                                   "A13": ">= 200 DM / salary", "A14": "no checking account"}.get(x, x)
        )
        savings = st.selectbox(
            "Savings account / bonds",
            ["A61", "A62", "A63", "A64", "A65"],
            format_func=lambda x: {"A61": "< 100 DM", "A62": "100 to < 500 DM",
                                   "A63": "500 to < 1000 DM", "A64": ">= 1000 DM",
                                   "A65": "unknown / no savings"}.get(x, x)
        )
        credit_amount = st.number_input("Credit amount (DM)", 0, 20000, 3000, step=500)
        duration = st.slider("Loan duration (months)", 4, 72, 18)
        installment_rate = st.slider("Installment rate (% of income)", 1, 4, 2)

    with st.expander("🏦 Credit History & Employment", expanded=True):
        credit_hist = st.selectbox(
            "Credit history",
            ["A30", "A31", "A32", "A33", "A34"],
            format_func=lambda x: {"A30": "no credits / all paid back", "A31": "all paid back duly",
                                   "A32": "existing paid back duly", "A33": "delay in past",
                                   "A34": "critical / other credits"}.get(x, x)
        )
        purpose = st.selectbox(
            "Loan purpose",
            ["A40", "A41", "A42", "A43", "A44", "A45", "A46", "A47", "A48", "A49", "A410"]
        )
        employment = st.selectbox(
            "Employment since",
            ["A71", "A72", "A73", "A74", "A75"],
            format_func=lambda x: {"A71": "unemployed", "A72": "< 1 year",
                                   "A73": "1–4 years", "A74": "4–7 years", "A75": ">= 7 years"}.get(x, x)
        )
        existing_credits = st.slider("Existing credits at this bank", 1, 4, 1)

    with st.expander("🏠 Other Details", expanded=False):
        other_debtors = st.selectbox("Other debtors / guarantors", ["A101", "A102", "A103"])
        property_type = st.selectbox("Property", ["A121", "A122", "A123", "A124"])
        other_install = st.selectbox("Other installment plans", ["A141", "A142", "A143"])
        housing = st.selectbox(
            "Housing",
            ["A151", "A152", "A153"],
            format_func=lambda x: {"A151": "rent", "A152": "own", "A153": "for free"}.get(x, x)
        )
        job = st.selectbox("Job", ["A171", "A172", "A173", "A174"])
        telephone = st.selectbox(
            "Telephone",
            ["A191", "A192"],
            format_func=lambda x: {"A191": "none", "A192": "yes, registered"}.get(x, x)
        )
        foreign = st.selectbox(
            "Foreign worker",
            ["A201", "A202"],
            format_func=lambda x: {"A201": "yes", "A202": "no"}.get(x, x)
        )

    predict_btn = st.button("🔮 Predict Creditworthiness")

# ---------- MAIN CONTENT ----------
with st.container():
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)

    # Input summary (before prediction)
    if not predict_btn:
        st.markdown("### 📝 Applicant Summary")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Age", f"{age} yrs")
            st.metric("Credit Amount", f"{credit_amount} DM")
        with col_b:
            st.metric("Duration", f"{duration} months")
            st.metric("Installment Rate", f"{installment_rate}%")
        with col_c:
            st.metric("Existing Credits", existing_credits)
            st.metric("Dependents", num_dependents)
        st.info("👈 Adjust the details in the sidebar and click **Predict Creditworthiness**")
    else:
        # Build input DataFrame
        input_data = {
            'duration': duration,
            'credit_amount': credit_amount,
            'installment_rate': installment_rate,
            'residence_since': residence_since,
            'age': age,
            'existing_credits': existing_credits,
            'num_dependents': num_dependents,
            'checking_account': checking,
            'credit_history': credit_hist,
            'purpose': purpose,
            'savings_account': savings,
            'employment': employment,
            'personal_status_sex': personal_status,
            'other_debtors': other_debtors,
            'property': property_type,
            'other_installment_plans': other_install,
            'housing': housing,
            'job': job,
            'telephone': telephone,
            'foreign_worker': foreign
        }
        input_df = pd.DataFrame([input_data])

        # Preprocess and predict
        input_processed = preprocessor.transform(input_df)
        prediction = model.predict(input_processed)[0]
        proba = model.predict_proba(input_processed)[0, 1]

        # Display result with nice styling
        st.markdown("### 🧾 Credit Decision")
        if prediction == 1:
            st.markdown(
                f"<div class='result-good'>✅ <strong>Good Credit Risk</strong><br>"
                f"Probability of full repayment: <span style='font-size:1.3em;'>{proba:.1%}</span></div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='result-bad'>❌ <strong>Bad Credit Risk</strong><br>"
                f"Probability of good credit: <span style='font-size:1.3em;'>{proba:.1%}</span></div>",
                unsafe_allow_html=True
            )

        # Metrics in a row
        col1, col2, col3 = st.columns(3)
        col1.metric("Risk Probability", f"{proba:.1%}")
        col2.metric("Loan Amount", f"{credit_amount} DM")
        col3.metric("Monthly Installment", f"{(credit_amount/duration):.0f} DM")

        # Show input details in expandable
        with st.expander("🔍 View Full Input Data"):
            st.json(input_data)

    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<div style='text-align: center; color: rgba(255,255,255,0.6); margin-top: 2rem;'>"
            "Built with ❤️ using Streamlit | CodeAlpha ML Internship</div>",
            unsafe_allow_html=True)