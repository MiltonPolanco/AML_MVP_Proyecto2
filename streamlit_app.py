"""MVP interactivo para priorización de alertas AML."""

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from mvp_model import explain_prediction, load_models, predict_account


ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "artifacts"
DATA_DIR = ROOT / "mvp_assets"

st.set_page_config(
    page_title="Monitoreo transaccional AML",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1250px;}
    h1 {letter-spacing: -0.035em;}
    [data-testid="stMetric"] {
        border: 1px solid #d9e2e8; border-radius: 10px; padding: 0.8rem 1rem;
        background: #f8fafb;
    }
    .result-box {
        border-left: 5px solid #0f766e; background: #eef8f6;
        border-radius: 8px; padding: 1rem 1.1rem; margin: .25rem 0 1rem 0;
    }
    .result-box.alert {border-left-color: #b42318; background: #fff2f0;}
    .small-note {color: #52616b; font-size: .9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_demo_data():
    transactions = pd.read_csv(DATA_DIR / "account_transactions.csv")
    profiles = pd.read_csv(DATA_DIR / "account_profiles.csv")
    return transactions, profiles


@st.cache_resource
def get_models():
    return load_models(ASSET_DIR)


transactions, profiles = load_demo_data()
autoencoder, classifier, stage_a_checkpoint, stage_b_checkpoint = get_models()

st.title("Monitoreo transaccional AML")
st.caption("Priorización de cuentas mediante anomalías de secuencia y clasificación con atención")

profile_lookup = profiles.set_index("nameDest")
case_accounts = profiles.loc[profiles["study_case"], "nameDest"].tolist()
ordered_accounts = case_accounts + [
    account for account in profiles["nameDest"] if account not in case_accounts
]


def account_label(account_id):
    row = profile_lookup.loc[account_id]
    suffix = " · caso del análisis" if bool(row["study_case"]) else ""
    return f"{account_id} · {int(row['total_transactions'])} operaciones{suffix}"


selected_account = st.selectbox(
    "Cuenta a evaluar",
    ordered_accounts,
    format_func=account_label,
    help="La muestra contiene cuentas normales, sospechosas y los cinco casos analizados en el notebook.",
)

account_frame = transactions[transactions["nameDest"] == selected_account].copy()
prediction = predict_account(
    account_frame,
    autoencoder,
    classifier,
    stage_a_checkpoint,
    stage_b_checkpoint,
)

status = "REVISAR" if prediction.is_alert else "SIN ALERTA"
status_class = "alert" if prediction.is_alert else ""
st.markdown(
    f'<div class="result-box {status_class}"><b>Resultado: {status}</b><br>'
    f'{explain_prediction(prediction)}</div>',
    unsafe_allow_html=True,
)

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Probabilidad Etapa B", f"{prediction.probability:.1%}")
metric_2.metric("Umbral de alerta", f"{prediction.threshold:.1%}")
metric_3.metric("Score de anomalía", f"{prediction.anomaly_score:.3f}")
metric_4.metric("Operaciones visibles", len(prediction.transactions), "máximo 24")

st.subheader("Atención por transacción")
chart_data = prediction.transactions[
    ["position", "step", "type", "amount", "attention", "isFraud"]
].copy()
chart_data["secuencia"] = "Peso de atención"
chart_data["etiqueta"] = chart_data["isFraud"].map(
    {0: "Normal en PaySim", 1: "Fraude en PaySim"}
)

heatmap = (
    alt.Chart(chart_data)
    .mark_rect(stroke="white", strokeWidth=1)
    .encode(
        x=alt.X("position:O", title="Posición temporal", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("secuencia:N", title=None, axis=alt.Axis(labels=False, ticks=False)),
        color=alt.Color(
            "attention:Q",
            title="Atención",
            scale=alt.Scale(scheme="yelloworangered", domain=[0, 1]),
        ),
        tooltip=[
            alt.Tooltip("position:O", title="Posición"),
            alt.Tooltip("step:Q", title="Hora simulada"),
            alt.Tooltip("type:N", title="Tipo"),
            alt.Tooltip("amount:Q", title="Monto", format=",.2f"),
            alt.Tooltip("attention:Q", title="Atención", format=".2%"),
            alt.Tooltip("etiqueta:N", title="Referencia"),
        ],
    )
    .properties(height=95)
)

fraud_markers = (
    alt.Chart(chart_data[chart_data["isFraud"] == 1])
    .mark_point(shape="diamond", size=95, color="#6d0015", filled=True)
    .encode(x="position:O", y="secuencia:N", tooltip=["etiqueta:N"])
)
st.altair_chart(heatmap + fraud_markers, use_container_width=True)
st.markdown(
    '<div class="small-note">Los rombos marcan operaciones etiquetadas como fraude en PaySim. '
    'El color muestra cuánto peso asignó el clasificador a cada posición.</div>',
    unsafe_allow_html=True,
)

st.subheader("Secuencia evaluada")
display_frame = prediction.transactions[
    ["position", "step", "type", "amount", "delta_t_hours", "attention", "isFraud"]
].copy()
display_frame.columns = [
    "Posición",
    "Hora simulada",
    "Tipo",
    "Monto",
    "Horas desde la anterior",
    "Atención",
    "Etiqueta PaySim",
]
display_frame["Atención"] = display_frame["Atención"].map(lambda value: f"{value:.2%}")
display_frame["Etiqueta PaySim"] = display_frame["Etiqueta PaySim"].map(
    {0: "Normal", 1: "Fraude"}
)
st.dataframe(
    display_frame,
    hide_index=True,
    use_container_width=True,
    column_config={"Monto": st.column_config.NumberColumn(format="$ %.2f")},
)

with st.expander("Cómo interpretar el resultado"):
    st.write(
        "La Etapa A compara la secuencia con el comportamiento normal aprendido y produce "
        "un error de reconstrucción. La Etapa B combina ese score con una representación "
        "GRU y pesos de atención para estimar la probabilidad calibrada. El umbral fue "
        "elegido exclusivamente en validación."
    )
    st.write(
        "La etiqueta de PaySim se muestra solo como referencia experimental. En una "
        "operación real, la alerta debe complementarse con KYC, listas PEP, información "
        "geográfica y revisión del oficial de cumplimiento."
    )
