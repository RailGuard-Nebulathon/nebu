def render_acv(st, frame):
    import pandas as pd
    st.subheader("ACV fault localisation")
    ranking = pd.DataFrame({"car": [f"{i:02d}" for i in [3, 6, 1, 7, 4, 2, 5, 8]], "probability": [0.42, 0.14, 0.11, 0.09, 0.08, 0.07, 0.05, 0.04]})
    left, right = st.columns([1, 2]); left.dataframe(ranking, hide_index=True); right.line_chart(frame.set_index("minutes"))
    st.metric("Top-two confidence margin", "0.28")
    st.caption("Car 03 shows slower thermal recovery relative to the peer median in this simulation.")

