
# RTO FASTAPI SERVICE — MAX AUC VERSION


from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np


# INITIALIZE APP


app = FastAPI(title="RTO Prediction API",
              description="Production-grade RTO Risk Engine",
              version="1.0")



# LOAD ARTIFACTS (LOAD ONCE AT STARTUP)


model        = joblib.load("rto_model.joblib")
calibrator   = joblib.load("prob_calibrator.joblib")
FEATURES     = joblib.load("features.joblib")

user_store   = joblib.load("user_store.joblib")
pin_store    = joblib.load("pin_store.joblib")
GLOBAL_MEAN  = joblib.load("global_mean.joblib")

courier_perf = pd.read_csv("Courier_Percentage_Delivered.csv")
courier_perf = courier_perf.sort_values("PercentageDelivered", ascending=False)



# REQUEST SCHEMA


class OrderRequest(BaseModel):
    mobile_no: str
    address: str
    destination_pincode: str
    order_value: float
    quantity: int
    payment_mode: int  # 1 = COD



# HELPER FUNCTIONS


def normalize_mobile(m):
    return str(m).replace("+91", "").replace(".0", "").strip()


def lookup_user(mobile):
    mobile = normalize_mobile(mobile)

    if mobile in user_store.index:
        row = user_store.loc[mobile]
        return {
            "user_total_orders": int(row.user_total_orders),
            "user_rto_rate": float(row.user_rto_rate),
            "user_last_rto": int(row.user_last_rto),
            "source": "CUSTOMER_HISTORY"
        }

    return {
        "user_total_orders": 0,
        "user_rto_rate": 0.0,
        "user_last_rto": 0,
        "source": "NEW_USER"
    }


def lookup_pincode(pin):
    return float(pin_store.get(str(pin), GLOBAL_MEAN))


def build_features(data: OrderRequest):

    user = lookup_user(data.mobile_no)
    pin_rto = lookup_pincode(data.destination_pincode)

    is_cod = int(data.payment_mode == 1)
    log_value = np.log1p(data.order_value)

    addr = str(data.address)
    addr_len = len(addr)
    addr_digit = sum(c.isdigit() for c in addr)

    user_vs_global = user["user_rto_rate"] - GLOBAL_MEAN
    pin_vs_global  = pin_rto - GLOBAL_MEAN

    risk_logit_pin = np.log((pin_rto + 1e-6)/(1 - pin_rto + 1e-6))
    risk_logit_user = (
        np.log((user["user_rto_rate"] + 1e-6)/(1 - user["user_rto_rate"] + 1e-6))
        if user["user_total_orders"] > 0 else 0
    )

    cod_x_user = is_cod * user["user_rto_rate"]
    cod_x_pin  = is_cod * pin_rto

    row = {
        "is_cod": is_cod,
        "log_value": log_value,
        "Quantity": data.quantity,
        "Courier": "UNKNOWN",
        "DestinationPincode": str(data.destination_pincode),

        "user_total_orders": user["user_total_orders"],
        "user_rto_rate": user["user_rto_rate"],
        "user_last_rto": user["user_last_rto"],

        "pin_rto": pin_rto,
        "cat_rto": GLOBAL_MEAN,

        "addr_len": addr_len,
        "addr_digit": addr_digit,

        "user_vs_global": user_vs_global,
        "pin_vs_global": pin_vs_global,

        "risk_logit_pin": risk_logit_pin,
        "risk_logit_user": risk_logit_user,

        "cod_x_user": cod_x_user,
        "cod_x_pin": cod_x_pin
    }

    df = pd.DataFrame([row])
    df = df.reindex(columns=FEATURES, fill_value=0)

    return df, user, pin_rto


def risk_band(prob):
    pct = prob * 100
    if pct < 30:
        return "LOW"
    elif pct < 50:
        return "MEDIUM"
    elif pct < 70:
        return "HIGH"
    return "VERY_HIGH"


def decision_source(user, pin_rate, prob):

    if prob < 0.30:
        return "LOW_RISK_NO_DOMINANT_FACTOR"

    user_rate = user["user_rto_rate"]

    if user["source"] == "CUSTOMER_HISTORY" and user_rate > pin_rate + 0.05:
        return "CUSTOMER_HISTORY"

    if pin_rate > user_rate + 0.05:
        return "PINCODE_HISTORY"

    return "MIXED_RISK (USER + PINCODE)"


def recommend_top_couriers(n=5):
    return courier_perf.head(n)[
        ["Courier", "PercentageDelivered"]
    ].to_dict("records")



# HEALTH CHECK


@app.get("/")
def health_check():
    return {"status": "RTO API Running"}



# PREDICTION ENDPOINT


@app.post("/predict")
def predict_rto(order: OrderRequest):

    X, user, pin_rto = build_features(order)

    raw_prob = model.predict_proba(X)[0,1]
    prob = calibrator.transform([raw_prob])[0]

    # Cold-start guard
    if user["user_total_orders"] == 0 and pin_rto == GLOBAL_MEAN:
        prob = max(prob, GLOBAL_MEAN)

    band = risk_band(prob)
    src  = decision_source(user, pin_rto, prob)

    action_map = {
        "LOW": "ALLOW_SHIPMENT",
        "MEDIUM": "CALL_CONFIRMATION",
        "HIGH": "OTP_OR_PARTIAL_PREPAID",
        "VERY_HIGH": "PREPAID_ONLY"
    }

    return {
        "rto_probability": round(prob,4),
        "rto_percentage": f"{round(prob*100,2)}%",
        "risk_band": band,
        "decision_source": src,
        "recommended_action": action_map[band],
        "recommended_couriers": recommend_top_couriers()
    }