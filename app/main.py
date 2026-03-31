from fastapi import FastAPI

app = FastAPI(
    title="TuracoFlow",
    description="Automated WhatsApp Claims Validator — Turaco Insurance",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "TuracoFlow"}
