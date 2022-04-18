import app1_social_detector
import app2_fire_detector
import app3_crash_detector
import app4_fall_alert
import app5_human_activity

import streamlit as st
PAGES = {
    "Social Distance Detector": app1_social_detector,
    "Fire Detector": app2_fire_detector,
    "Vehicle Crash Detector": app3_crash_detector,
    "Fall Alert": app4_fall_alert,
    "Human Activity": app5_human_activity
}
st.sidebar.title('Dashboard')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()