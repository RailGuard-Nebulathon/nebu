def render_reliability(st):
    import pandas as pd
    st.subheader("Model reliability")
    left, right = st.columns(2)
    left.line_chart(pd.DataFrame({"confidence": [0.1, 0.3, 0.5, 0.7, 0.9], "observed": [0.12, 0.28, 0.51, 0.68, 0.87]}).set_index("confidence"))
    right.bar_chart(pd.DataFrame({"scenario": ["clean", "noise", "dropout", "bias"], "retention": [1.0, 0.93, 0.88, 0.91]}).set_index("scenario"))
    st.info("Calibration and corruption values shown in Demo mode are illustrative simulation, not measured model performance.")

