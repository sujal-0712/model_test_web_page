import streamlit as st

# 1. Page Configuration (Sets the Global Theme)
st.set_page_config(
    page_title="Sujal's AI Intelligence Hub",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Map the structural pages to clear, non-technical labels
home_view = st.Page("pages/home.py", title="Executive Summary", icon="🏠", default=True)

# Group A: Business Analytics
housing_view = st.Page("pages/housing.py", title="Real Estate Valuation", icon="📊")
diabetes_view = st.Page("pages/diabetes.py", title="Clinical Risk Analytics", icon="🏥")

# Group B: Advanced Deep Learning
traffic_view = st.Page("pages/traffic.py", title="Autonomous Sign Vision", icon="🚗")
malaria_view = st.Page("pages/malaria.py", title="Biomedical Malaria Diagnostic", icon="🔬")

# 3. Compile the structural navigation sidebar
navigation_router = st.navigation({
    "Overview": [home_view],
    "Predictive Machine Learning": [housing_view, diabetes_view],
    "Computer Vision (Deep Learning)": [traffic_view, malaria_view]
})

# 4. Global Sidebar Branding Footer
st.sidebar.markdown("---")
st.sidebar.caption("🤖 Architectural Framework engineered by Sujal")

# 5. Run the routing loop
navigation_router.run()