"""
TuracoFlow — Streamlit Claims Dashboard
"""

import os
import requests
from PIL import Image
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TuracoFlow — Claims Validator",
    page_icon="🏥",
    layout="wide",
)

# ── Header ─────────────────────────────────────────────────────────────────────

st.markdown("""
<h1 style='margin-bottom:0'>🏥 TuracoFlow</h1>
<p style='color:#888; margin-top:4px; font-size:1rem'>
    Automated WhatsApp Claims Validator — Turaco Insurance
</p>
<hr style='margin-top:8px; margin-bottom:24px'>
""", unsafe_allow_html=True)

# ── Layout: left = form, right = result ────────────────────────────────────────

left, right = st.columns([1, 1.4], gap="large")

# ── Left: submission form ──────────────────────────────────────────────────────

with left:
    st.subheader("Submit a Claim")

    uploaded = st.file_uploader(
        "Receipt image",
        type=["png", "jpg", "jpeg"],
        help="Upload the hospital receipt photo",
    )

    if uploaded:
        st.image(Image.open(uploaded), caption="Uploaded receipt", use_container_width=True)
        uploaded.seek(0)  # reset after PIL read

    customer_id = st.text_input("Customer ID", placeholder="e.g. CUST-001")
    policy_id   = st.text_input("Policy ID",   placeholder="e.g. TUR-KE-HC-001-88234")

    submit = st.button("Submit Claim", type="primary", use_container_width=True)


# ── Right: result panel ────────────────────────────────────────────────────────

with right:
    st.subheader("Decision")

    if not submit:
        st.info("Fill in the form and click **Submit Claim** to see the result.")

    elif not uploaded:
        st.warning("Please upload a receipt image.")

    elif not customer_id or not policy_id:
        st.warning("Please fill in both Customer ID and Policy ID.")

    else:
        with st.spinner("Processing claim…"):
            try:
                response = requests.post(
                    f"{API_URL}/claims/submit",
                    data={"customer_id": customer_id, "policy_id": policy_id},
                    files={"image": (uploaded.name, uploaded, uploaded.type)},
                    timeout=300,
                )
                result = response.json()
            except requests.exceptions.ConnectionError:
                st.error(f"Cannot reach the API at {API_URL}. Is TuracoFlow running?")
                st.stop()
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                st.stop()

        if response.status_code != 200:
            st.error(f"API error {response.status_code}: {result.get('detail', 'Unknown error')}")
            st.stop()

        status = result.get("status", "")

        # ── Status badge ──────────────────────────────────────────────────────

        STATUS_STYLE = {
            "APPROVED":       ("✅ APPROVED",       "#1a7f37", "#d4f5d4"),
            "REJECTED":       ("❌ REJECTED",       "#c0392b", "#fde8e8"),
            "REVIEW":         ("🔍 NEEDS REVIEW",   "#b7770d", "#fff3cd"),
            "DUPLICATE_CLAIM":("⚠️ DUPLICATE",      "#6c3483", "#f5eef8"),
        }
        label, fg, bg = STATUS_STYLE.get(
            status, (status, "#555", "#f0f0f0")
        )

        st.markdown(f"""
        <div style='background:{bg}; border-left:5px solid {fg};
                    padding:16px 20px; border-radius:6px; margin-bottom:16px'>
            <span style='color:{fg}; font-size:1.4rem; font-weight:700'>{label}</span>
        </div>
        """, unsafe_allow_html=True)

        # ── Key metrics ───────────────────────────────────────────────────────

        col1, col2, col3 = st.columns(3)
        with col1:
            amount = result.get("approved_amount", 0)
            currency = result.get("currency", "")
            st.metric("Approved Amount",
                      f"{currency} {amount:,.0f}" if amount else "—")
        with col2:
            st.metric("Confidence", f"{result.get('confidence', 0):.0%}")
        with col3:
            st.metric("Extraction", result.get("extraction_method", "—").upper())

        # ── Reason ────────────────────────────────────────────────────────────

        st.markdown("**Reason**")
        st.write(result.get("reason", ""))

        if result.get("policy_matched"):
            st.caption(f"Policy matched: `{result['policy_matched']}`")

        st.caption(f"Claim ID: `{result.get('claim_id', '')}` · "
                   f"Fraud check: `{result.get('fraud_check', '')}`")

        # ── Audit trail ───────────────────────────────────────────────────────

        checks = result.get("checks", {})
        if checks:
            with st.expander("Audit trail", expanded=True):
                for name, info in checks.items():
                    passed = info.get("passed", False)
                    detail = info.get("detail", "")
                    icon   = "✅" if passed else "❌"
                    st.markdown(f"{icon} **{name}** — {detail}")


# ── Footer ─────────────────────────────────────────────────────────────────────

st.markdown("""
<hr style='margin-top:40px'>
<p style='text-align:center; color:#aaa; font-size:0.8rem'>
    TuracoFlow v0.1.0 · Kenya · Uganda · Nigeria
</p>
""", unsafe_allow_html=True)
