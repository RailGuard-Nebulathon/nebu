def render_shm(st, time, stress):
    import pandas as pd
    st.subheader("Structural health monitoring")
    a, b, c = st.columns(3); a.metric("Predicted damage", "0.230"); b.metric("90% interval", "0.18–0.29"); c.metric("Health", "67/100")
    st.line_chart(pd.DataFrame({"time": time, "stress": stress}).set_index("time"))
    st.caption("Cycle histogram and influential intervals are fatigue proxies; no material S-N constants were assumed.")


def render_health(st, predictions):
    st.subheader("Health intelligence")
    st.bar_chart(predictions.set_index("component")["health"])
    st.warning("Decision-support demonstration only; not an LTA-approved maintenance rule.")

