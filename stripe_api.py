import stripe
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# Allow your Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You may tighten this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")  # Use dotenv in local

@app.post("/create-checkout-session/")
async def create_checkout_session(plan_id: str):
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            mode="subscription",
            line_items=[{
                "price": plan_id,
                "quantity": 1,
            }],
            success_url="https://agent.wiseagi.org/success",
            cancel_url="https://agent.wiseagi.org/cancel",
        )
        return {"checkout_url": session.url}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
