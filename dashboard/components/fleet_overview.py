def render(st, predictions):
    columns = st.columns(5)
    columns[0].metric("Analysed", len(predictions))
    columns[1].metric("Faults detected", int((predictions["health"] < 60).sum()))
    columns[2].metric("Low confidence", int((predictions["confidence"] < 0.8).sum()))
    columns[3].metric("OOD samples", int(predictions["ood"].sum()))
    columns[4].metric("Mean health", f"{predictions['health'].mean():.0f}/100")
    st.dataframe(predictions, use_container_width=True, hide_index=True)

