def render_door(st, frame):
    st.subheader("Door diagnostics")
    a, b, c, d = st.columns(4)
    a.metric("Operation", "Close"); b.metric("Status", "Normal"); c.metric("Probability", "9%"); d.metric("Uncertainty", "0.07")
    st.line_chart(frame.set_index("time")[["motor_current", "voltage", "position"]])
    st.caption("Highlighted signals are model evidence; demo values are simulated.")


def render_corrugation(st, time, vibration):
    import numpy as np
    import pandas as pd
    st.subheader("Rail corrugation")
    st.write("Prediction: **Side II** — probabilities Normal 0.10, Side I 0.12, Side II 0.78")
    left, right = st.columns(2)
    left.line_chart(pd.DataFrame({"time": time, "vibration": vibration}).set_index("time"))
    power = np.abs(np.fft.rfft(vibration)) ** 2; frequency = np.fft.rfftfreq(len(vibration), time[1] - time[0])
    right.line_chart(pd.DataFrame({"frequency": frequency, "power": power}).set_index("frequency"))
    st.caption("Important demo bands: 100–150 Hz and 280–340 Hz; influential channel group: Side II vibration.")

