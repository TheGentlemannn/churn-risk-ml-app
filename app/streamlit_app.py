import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR / "src"))

from config import FEATURE_IMPORTANCE_PATH, METRICS_PATH, MODEL_PATH  # noqa: E402
from predict import load_model  # noqa: E402


YES_NO_LABELS = {"No": "No", "Yes": "Si"}
GENDER_LABELS = {"Female": "Mujer", "Male": "Hombre"}
INTERNET_LABELS = {"DSL": "Internet DSL", "Fiber optic": "Fibra optica", "No": "Sin internet"}
CONTRACT_LABELS = {
    "Month-to-month": "Mensual, sin permanencia",
    "One year": "Contrato de 1 ano",
    "Two year": "Contrato de 2 anos",
}
PAYMENT_LABELS = {
    "Electronic check": "Pago electronico manual",
    "Mailed check": "Cheque por correo",
    "Bank transfer (automatic)": "Transferencia bancaria automatica",
    "Credit card (automatic)": "Tarjeta de credito automatica",
}
SERVICE_LABELS = {
    "No": "No contratado",
    "Yes": "Contratado",
    "No internet service": "No aplica, sin internet",
    "No phone service": "No aplica, sin telefono",
}

CUSTOMER_PROFILES = {
    "Cliente nuevo con riesgo alto": {
        "gender": "Female",
        "SeniorCitizen": "No",
        "Partner": "No",
        "Dependents": "No",
        "tenure": 3,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.0,
        "TotalCharges": 280.0,
    },
    "Cliente estable": {
        "gender": "Male",
        "SeniorCitizen": "No",
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 48,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 65.0,
        "TotalCharges": 3200.0,
    },
    "Cliente promedio": {
        "gender": "Female",
        "SeniorCitizen": "No",
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 18,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 70.0,
        "TotalCharges": 1260.0,
    },
}

ACTIVE_PROFILE_KEY = "default"


def select_value(label, options, default, labels=None, help_text=None):
    labels = labels or {}
    return st.selectbox(
        label,
        options,
        index=options.index(default),
        format_func=lambda value: labels.get(value, value),
        help=help_text,
        key=f"{ACTIVE_PROFILE_KEY}_{label}",
    )


def risk_message(probability: float) -> tuple[str, str, str]:
    if probability >= 0.65:
        return (
            "Riesgo alto",
            "Este cliente se parece a clientes historicos que cancelaron el servicio.",
            "Prioriza una llamada de retencion, revisa su plan actual y ofrece soporte o descuento personalizado.",
        )
    if probability >= 0.35:
        return (
            "Riesgo medio",
            "Hay senales mixtas: el cliente no esta perdido, pero conviene monitorearlo.",
            "Revisa satisfaccion, beneficios usados y oportunidades de mejorar contrato o metodo de pago.",
        )
    return (
        "Riesgo bajo",
        "El perfil se parece mas a clientes que permanecieron activos.",
        "Mantener seguimiento normal y buscar oportunidades de fidelizacion.",
    )


st.set_page_config(page_title="Churn Risk Predictor", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1180px;}
    .risk-card {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.28);
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .risk-label {
        color: var(--text-color);
        opacity: 0.72;
        font-size: 0.9rem;
        margin-bottom: 0.2rem;
    }
    .risk-value {
        color: var(--text-color);
        font-size: 2.25rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .risk-track {
        background: rgba(128, 128, 128, 0.22);
        border-radius: 999px;
        height: 0.75rem;
        margin-top: 1rem;
        overflow: hidden;
    }
    .risk-fill {
        background: linear-gradient(90deg, #22c55e, #f59e0b, #ef4444);
        border-radius: 999px;
        height: 100%;
    }
    .section-note {
        color: #475569;
        font-size: 0.95rem;
        margin-top: -0.4rem;
        margin-bottom: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Prediccion de riesgo de cancelacion")
st.caption("Ingresa datos simples de un cliente y la app estima que tan probable es que cancele el servicio.")

if not MODEL_PATH.exists():
    st.warning("Todavia no existe un modelo entrenado. Ejecuta `python src/train.py` desde la raiz del proyecto.")
    st.stop()

model = load_model()

profile_name = st.selectbox(
    "Comienza con un ejemplo",
    list(CUSTOMER_PROFILES.keys()),
    index=0,
    help="Elige un perfil prellenado y luego ajusta los campos como quieras.",
)
profile = CUSTOMER_PROFILES[profile_name]
ACTIVE_PROFILE_KEY = profile_name.lower().replace(" ", "_")

left, right = st.columns([1.25, 1], gap="large")

with left:
    st.subheader("1. Informacion basica")
    st.markdown(
        '<p class="section-note">Datos generales que ayudan a entender la relacion del cliente con la empresa.</p>',
        unsafe_allow_html=True,
    )
    basic_a, basic_b = st.columns(2)

    with basic_a:
        gender = select_value("Genero", ["Female", "Male"], profile["gender"], GENDER_LABELS)
        senior_citizen = select_value(
            "Cliente mayor de 65 anos",
            ["No", "Yes"],
            profile["SeniorCitizen"],
            YES_NO_LABELS,
            "En el dataset original esta variable indica si la persona es senior citizen.",
        )
        partner = select_value("Tiene pareja", ["No", "Yes"], profile["Partner"], YES_NO_LABELS)

    with basic_b:
        dependents = select_value("Tiene dependientes", ["No", "Yes"], profile["Dependents"], YES_NO_LABELS)
        tenure = st.slider(
            "Tiempo como cliente",
            min_value=0,
            max_value=72,
            value=profile["tenure"],
            format="%d meses",
            help="Clientes nuevos suelen tener mas riesgo que clientes con muchos meses de permanencia.",
            key=f"{ACTIVE_PROFILE_KEY}_tenure",
        )

    st.divider()
    st.subheader("2. Plan contratado")
    st.markdown(
        '<p class="section-note">Servicios y extras que el cliente tiene actualmente.</p>',
        unsafe_allow_html=True,
    )

    plan_a, plan_b = st.columns(2)
    with plan_a:
        phone_service = select_value("Servicio telefonico", ["No", "Yes"], profile["PhoneService"], YES_NO_LABELS)
        multiple_lines = select_value(
            "Lineas telefonicas adicionales",
            ["No", "Yes", "No phone service"],
            profile["MultipleLines"],
            SERVICE_LABELS,
        )
        internet_service = select_value(
            "Tipo de internet",
            ["DSL", "Fiber optic", "No"],
            profile["InternetService"],
            INTERNET_LABELS,
        )

    no_internet = internet_service == "No"
    internet_default = "No internet service" if no_internet else None

    with plan_b:
        online_security = select_value(
            "Seguridad online",
            ["No", "Yes", "No internet service"],
            internet_default or profile["OnlineSecurity"],
            SERVICE_LABELS,
        )
        online_backup = select_value(
            "Backup online",
            ["No", "Yes", "No internet service"],
            internet_default or profile["OnlineBackup"],
            SERVICE_LABELS,
        )
        tech_support = select_value(
            "Soporte tecnico premium",
            ["No", "Yes", "No internet service"],
            internet_default or profile["TechSupport"],
            SERVICE_LABELS,
        )

    with st.expander("Servicios adicionales"):
        extra_a, extra_b, extra_c = st.columns(3)
        with extra_a:
            device_protection = select_value(
                "Proteccion de dispositivos",
                ["No", "Yes", "No internet service"],
                internet_default or profile["DeviceProtection"],
                SERVICE_LABELS,
            )
        with extra_b:
            streaming_tv = select_value(
                "Streaming TV",
                ["No", "Yes", "No internet service"],
                internet_default or profile["StreamingTV"],
                SERVICE_LABELS,
            )
        with extra_c:
            streaming_movies = select_value(
                "Streaming peliculas",
                ["No", "Yes", "No internet service"],
                internet_default or profile["StreamingMovies"],
                SERVICE_LABELS,
            )

    st.divider()
    st.subheader("3. Contrato y pagos")
    st.markdown(
        '<p class="section-note">Esta parte suele ser muy importante para predecir churn.</p>',
        unsafe_allow_html=True,
    )
    pay_a, pay_b = st.columns(2)

    with pay_a:
        contract = select_value("Tipo de contrato", list(CONTRACT_LABELS), profile["Contract"], CONTRACT_LABELS)
        paperless_billing = select_value(
            "Factura digital",
            ["No", "Yes"],
            profile["PaperlessBilling"],
            YES_NO_LABELS,
        )
        payment_method = select_value(
            "Metodo de pago",
            list(PAYMENT_LABELS),
            profile["PaymentMethod"],
            PAYMENT_LABELS,
        )

    with pay_b:
        monthly_charges = st.number_input(
            "Pago mensual",
            min_value=0.0,
            max_value=200.0,
            value=profile["MonthlyCharges"],
            step=5.0,
            format="%.2f",
            help="Monto que el cliente paga cada mes.",
            key=f"{ACTIVE_PROFILE_KEY}_monthly_charges",
        )
        total_charges = st.number_input(
            "Total pagado historicamente",
            min_value=0.0,
            max_value=10000.0,
            value=profile["TotalCharges"],
            step=50.0,
            format="%.2f",
            help="Suma aproximada de todos los cargos acumulados del cliente.",
            key=f"{ACTIVE_PROFILE_KEY}_total_charges",
        )

customer = {
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
}

with right:
    st.subheader("Resultado")
    probability = float(model.predict_proba(pd.DataFrame([customer]))[:, 1][0])
    risk_level, explanation, recommendation = risk_message(probability)
    probability_pct = probability * 100

    st.markdown(
        f"""
        <div class="risk-card">
            <div class="risk-label">Probabilidad estimada de cancelacion</div>
            <div class="risk-value">{probability_pct:.1f}%</div>
            <div class="risk-track">
                <div class="risk-fill" style="width: {probability_pct:.1f}%"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if probability >= 0.65:
        st.error(risk_level)
    elif probability >= 0.35:
        st.warning(risk_level)
    else:
        st.success(risk_level)

    st.write(explanation)
    st.info(recommendation)

    st.subheader("Resumen del cliente")
    summary = pd.DataFrame(
        [
            {"Campo": "Tiempo como cliente", "Valor": f"{tenure} meses"},
            {"Campo": "Contrato", "Valor": CONTRACT_LABELS[contract]},
            {"Campo": "Internet", "Valor": INTERNET_LABELS[internet_service]},
            {"Campo": "Pago mensual", "Valor": f"${monthly_charges:,.2f}"},
            {"Campo": "Metodo de pago", "Valor": PAYMENT_LABELS[payment_method]},
        ]
    )
    st.dataframe(summary, hide_index=True, use_container_width=True)

    if FEATURE_IMPORTANCE_PATH.exists():
        st.subheader("Factores que mas pesan")
        importance = pd.read_csv(FEATURE_IMPORTANCE_PATH).head(8)
        importance["feature"] = (
            importance["feature"]
            .str.replace("num__", "", regex=False)
            .str.replace("cat__", "", regex=False)
            .str.replace("_", " ", regex=False)
        )
        st.bar_chart(importance.set_index("feature"))

    with st.expander("Ver metricas del modelo"):
        if METRICS_PATH.exists():
            metrics = pd.read_csv(METRICS_PATH)
            st.dataframe(metrics.round(4), use_container_width=True, hide_index=True)
        else:
            st.write("Todavia no hay metricas guardadas.")
