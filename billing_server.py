import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import stripe

load_dotenv()

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
APP_URL = os.getenv("APP_URL", "http://localhost:8501")

app = FastAPI(title="WiseAGI Billing API")


class CheckoutRequest(BaseModel):
    user_id: str
    email: str | None = None
    plan_code: str
    price_id: str


@app.post("/api/billing/create-checkout-session")
def create_checkout_session(req: CheckoutRequest):
    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            line_items=[{
                "price": req.price_id,
                "quantity": 1,
            }],
            customer_email=req.email,
            success_url=f"{APP_URL}?billing=success&plan={req.plan_code}",
            cancel_url=f"{APP_URL}?billing=cancelled",
            metadata={
                "user_id": req.user_id,
                "plan_code": req.plan_code,
            },
        )

        return {"url": session.url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





