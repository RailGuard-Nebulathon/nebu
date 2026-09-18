"""RailGuard single-app operator dashboard."""

from pathlib import Path

import pandas as pd

from components import anomaly_view, explanation_view, fleet_overview, health_view, sensor_view
from data_access import demo_data, real_predictions


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise SystemExit("Install the dashboard extra: pip install -e '.[dashboard]'") from exc

    st.set_page_config(page_title="RailGuard AI", page_icon="🚆", layout="wide")
    st.markdown("""<style>
      .stApp {background: #f6f8fb;} [data-testid='stMetric'] {background:white;border:1px solid #e3e8ef;padding:16px;border-radius:12px;}
      h1,h2,h3 {color:#17324d;} .block-container {padding-top:2rem;}
    </style>""", unsafe_allow_html=True)
    st.title("RailGuard AI")
    st.caption("Train condition monitoring · calibrated evidence · maintenance intelligence")
    mode = st.sidebar.radio("Data mode", ["DEMO", "REAL"], help="DEMO is labelled simulation; REAL reads generated outputs.")
    page = st.sidebar.radio("View", ["Fleet Overview", "Door Diagnostics", "ACV Localisation", "Rail Corrugation", "SHM", "Model Reliability", "Health Intelligence"])
    task = st.sidebar.selectbox("Upload subsystem", ["door", "acv", "corrugation", "shm"])
    bundle_path = st.sidebar.text_input("Trusted model bundle (Real mode)", placeholder="outputs/checkpoints/.../bundle")
    uploaded = st.sidebar.file_uploader("Upload or drag a source file", type=["csv", "xlsx", "txt"])
    if uploaded:
        st.sidebar.success(f"Loaded {uploaded.name}. In DEMO mode the displayed result remains explicitly simulated; configure a trusted bundle for production inference.")
    data = demo_data()
    if mode == "REAL":
        outputs = real_predictions()
        if outputs.empty:
            st.warning("No generated prediction CSVs found under outputs/predictions; showing no fabricated real results.")
        else:
            st.download_button("Download combined generated results", outputs.to_csv(index=False), "railguard_results.csv", "text/csv")
            st.dataframe(outputs, use_container_width=True)
        if uploaded and bundle_path:
            import tempfile
            from dataclasses import asdict
            from railguard.inference import RailGuardPredictor

            with tempfile.TemporaryDirectory(prefix="railguard_upload_") as temporary:
                source = Path(temporary) / uploaded.name
                source.write_bytes(uploaded.getvalue())
                results = [asdict(item) for item in RailGuardPredictor.from_bundle(bundle_path).predict(task, source)]
            result_frame = pd.DataFrame(results)
            st.success(f"Generated {len(result_frame)} prediction result(s) from the trusted bundle.")
            st.dataframe(result_frame, use_container_width=True)
            st.download_button("Download prediction result", result_frame.to_csv(index=False), f"{task}_result.csv", "text/csv")
    if page == "Fleet Overview": fleet_overview.render(st, data["predictions"])
    elif page == "Door Diagnostics": sensor_view.render_door(st, data["door"])
    elif page == "ACV Localisation": anomaly_view.render_acv(st, data["acv"])
    elif page == "Rail Corrugation": sensor_view.render_corrugation(st, data["vibration_time"], data["vibration"])
    elif page == "SHM": health_view.render_shm(st, data["stress_time"], data["stress"])
    elif page == "Model Reliability": explanation_view.render_reliability(st)
    else: health_view.render_health(st, data["predictions"])
    if uploaded and mode == "DEMO":
        demo_csv = "file_id,prediction\n" + f"{uploaded.name},SIMULATED_{task.upper()}\n"
        st.download_button("Download simulated result", demo_csv, f"{task}_demo_result.csv", "text/csv")


if __name__ == "__main__":
    main()
