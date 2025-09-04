import pandas as pd
import re
import streamlit as st
from datetime import datetime
from textblob import TextBlob
import imaplib
import email
from email.header import decode_header
import os
import heapq
from dataclasses import dataclass, field
import plotly.express as px
from typing import Optional

SUBJECT_FILTER_TERMS = ["support", "query", "request", "help"]
PHONE_RE = re.compile(r'(\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?){1,2}\d{3,4}')
ALT_EMAIL_RE = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b')

@dataclass(order=True)
class PrioritizedEmail:
    priority_score: float
    id: int = field(compare=False)
    row: dict = field(compare=False)

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.strip())

def parse_datetime_safe(x: str) -> Optional[datetime]:
    try:
        return pd.to_datetime(x, errors="coerce")
    except Exception:
        return None

def classify_intent(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ["refund", "return", "money back"]):
        return "Refund Request"
    elif any(w in t for w in ["error", "issue", "bug", "problem"]):
        return "Technical Issue"
    elif any(w in t for w in ["price", "cost", "quote", "subscription"]):
        return "Pricing Inquiry"
    elif any(w in t for w in ["cancel", "unsubscribe", "terminate"]):
        return "Cancellation Request"
    elif any(w in t for w in ["thank", "appreciate", "great service"]):
        return "Appreciation"
    else:
        return "General Query"

def urgency_score(text: str) -> int:
    keywords = {"urgent": 5, "immediately": 5, "asap": 5, "important": 4, "critical": 5, "soon": 3}
    return sum(v for k, v in keywords.items() if k in text.lower())

def sentiment_score(text: str) -> float:
    if not text:
        return 0.0
    return TextBlob(text).sentiment.polarity

def assign_priority(urgency: int, sentiment: float, received_at: Optional[datetime]) -> str:
    if urgency >= 5 or sentiment < -0.5:
        return "P1 - Critical"
    elif urgency >= 3 or sentiment < -0.2:
        return "P2 - High"
    elif received_at and (datetime.utcnow() - received_at).days > 3:
        return "P3 - Medium"
    else:
        return "P4 - Low"

def generate_response(intent: str, sender: str) -> str:
    templates = {
        "Technical Issue": f"Hello {sender},\n\nThank you for reporting the issue. Our technical team is investigating and we will update you soon.\n\nBest regards,\nSupport Team",
        "Pricing Inquiry": f"Hello {sender},\n\nThank you for your interest. Our sales team will share the pricing details with you shortly.\n\nBest regards,\nSupport Team",
        "Cancellation Request": f"Hello {sender},\n\nWe acknowledge your cancellation request. We will confirm the status after verification.\n\nBest regards,\nSupport Team",
        "Appreciation": f"Hello {sender},\n\nThank you for your kind words! We are glad to serve you.\n\nBest regards,\nSupport Team",
        "General Query": f"Hello {sender},\n\nThank you for reaching out. Our support team will get back to you with the requested information.\n\nBest regards,\nSupport Team",
    }
    return templates.get(intent, templates["General Query"])

def check_sla_breach(received_at: Optional[datetime], priority_level: str) -> bool:
    if not received_at:
        return False
    age_hours = (datetime.utcnow() - received_at).total_seconds() / 3600.0
    sla_hours = {"P1 - Critical": 2, "P2 - High": 8, "P3 - Medium": 24, "P4 - Low": 48}.get(priority_level, 24)
    return age_hours > sla_hours

def extract_contact_info(text: str):
    phones = list({m.group(0) for m in PHONE_RE.finditer(text)})
    emails = list({m.group(0) for m in ALT_EMAIL_RE.finditer(text)})
    return {"phones": phones, "emails": emails}

def build_priority_queue(df: pd.DataFrame):
    heap = []
    for _, r in df.iterrows():
        score = {"P1 - Critical": 100, "P2 - High": 70, "P3 - Medium": 40, "P4 - Low": 10}.get(r["priority_level"], 10) + float(r.get("urgency", 0))
        heapq.heappush(heap, PrioritizedEmail(-score, int(r["id"]), r.to_dict()))
    return heap

def pop_next_email(heap):
    if not heap:
        return None
    return heapq.heappop(heap).row

def fetch_emails_imap(host: str, username: str, password: str, mailbox: str = "INBOX", max_messages: int = 200):
    conn = imaplib.IMAP4_SSL(host)
    conn.login(username, password)
    conn.select(mailbox)
    typ, data = conn.search(None, 'ALL')
    ids = data[0].split()[::-1]
    rows = []
    for i, msg_id in enumerate(ids[:max_messages]):
        typ, msg_data = conn.fetch(msg_id, '(RFC822)')
        if typ != 'OK':
            continue
        raw = msg_data[0][1]
        msg = email.message_from_bytes(raw)
        subject, encoding = decode_header(msg.get("Subject", ""))[0]
        if isinstance(subject, bytes):
            try:
                subject = subject.decode(encoding or "utf-8", errors="ignore")
            except:
                subject = subject.decode('utf-8', errors='ignore')
        subject_lower = (subject or "").lower()
        if not any(term in subject_lower for term in SUBJECT_FILTER_TERMS):
            continue
        sender = msg.get("From", "")
        date = msg.get("Date", "")
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                cdisp = str(part.get("Content-Disposition"))
                if ctype == "text/plain" and "attachment" not in cdisp:
                    try:
                        body = part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="ignore")
                    except:
                        body = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                    break
        else:
            try:
                body = msg.get_payload(decode=True).decode(msg.get_content_charset() or "utf-8", errors="ignore")
            except:
                body = str(msg.get_payload())
        rows.append({"sender": sender, "subject": subject, "body": body, "received_at": date})
    conn.logout()
    return pd.DataFrame(rows)

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["sender", "subject", "body", "received_at"]
    for col in required:
        if col not in df.columns:
            df[col] = ""
    if "id" not in df.columns:
        df.insert(0, "id", range(1, len(df) + 1))
    if df["received_at"].isnull().all() or (df["received_at"] == "").all():
        now = datetime.utcnow()
        df["received_at"] = [(now - pd.Timedelta(hours=i * 3)).isoformat() for i in range(len(df))]
    return df

def process_emails(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["clean_text"] = (df["subject"].fillna("") + " " + df["body"].fillna("")).map(clean_text)
    df["intent"] = df["clean_text"].map(classify_intent)
    df["urgency"] = df["clean_text"].map(urgency_score)
    df["sentiment"] = df["clean_text"].map(sentiment_score)
    df["__received_at"] = df["received_at"].map(parse_datetime_safe)
    df["priority_level"] = [assign_priority(u, s, r) for u, s, r in zip(df["urgency"], df["sentiment"], df["__received_at"])]
    df["draft_response"] = [generate_response(intent, sender) for intent, sender in zip(df["intent"], df["sender"])]
    df["sla_breach"] = [check_sla_breach(r, lvl) for r, lvl in zip(df["__received_at"], df["priority_level"])]
    df["extracted"] = df["clean_text"].map(extract_contact_info)
    df["phones"] = df["extracted"].map(lambda x: x.get("phones", []))
    df["alt_emails"] = df["extracted"].map(lambda x: x.get("emails", []))
    return df

st.set_page_config(page_title="AI-Powered Email Assistant", layout="wide")
st.title("📧 AI-Powered Communication Assistant")

with st.sidebar.expander("📥 Connect Email (IMAP)"):
    imap_host = st.text_input("IMAP Host", value=os.environ.get("EMAIL_HOST", ""))
    imap_user = st.text_input("Email", value=os.environ.get("EMAIL_USER", ""))
    imap_pass = st.text_input("Password / App Password", type="password", value=os.environ.get("EMAIL_PASS", ""))
    if st.button("Fetch Emails from IMAP"):
        try:
            fetched_df = fetch_emails_imap(imap_host, imap_user, imap_pass)
            if not fetched_df.empty:
                fetched_df.insert(0, "id", range(1, len(fetched_df) + 1))
                st.session_state["live_emails"] = fetched_df
                st.success(f"Fetched {len(fetched_df)} filtered emails.")
            else:
                st.info("No matching support emails found.")
        except Exception as e:
            st.error(f"Failed to fetch emails: {e}")

uploaded = st.file_uploader("Upload Support Emails CSV", type=["csv"])
if "live_emails" in st.session_state:
    df = st.session_state["live_emails"]
elif uploaded:
    df = load_data(uploaded)
else:
    df = None

if df is not None:
    processed = process_emails(df)
    if "resolved_ids" not in st.session_state:
        st.session_state["resolved_ids"] = set()
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total Emails", len(processed))
    m2.metric("Critical (P1)", (processed["priority_level"] == "P1 - Critical").sum())
    m3.metric("High (P2)", (processed["priority_level"] == "P2 - High").sum())
    m4.metric("SLA Breached", processed["sla_breach"].sum())
    m5.metric("Resolved", sum(eid in st.session_state["resolved_ids"] for eid in processed["id"]))
    m6.metric("Open", len(processed) - sum(eid in st.session_state["resolved_ids"] for eid in processed["id"]))
    processed["__dt"] = pd.to_datetime(processed["__received_at"], errors="coerce")
    last24 = processed[processed["__dt"] >= (datetime.utcnow() - pd.Timedelta(hours=24))]
    counts = last24.groupby(last24["__dt"].dt.hour).size().reindex(range(0, 24), fill_value=0)
    fig = px.bar(x=counts.index, y=counts.values, labels={"x": "Hour (UTC)", "y": "Emails"}, title="Emails Received Last 24 Hours (UTC)")
    st.plotly_chart(fig, use_container_width=True)
    st.sidebar.header("Filters")
    priority_filter = st.sidebar.multiselect("Priority", processed["priority_level"].unique())
    intent_filter = st.sidebar.multiselect("Intent", processed["intent"].unique())
    view = processed.copy()
    if priority_filter:
        view = view[view["priority_level"].isin(priority_filter)]
    if intent_filter:
        view = view[view["intent"].isin(intent_filter)]
    resolved_toggle = st.multiselect("✅ Mark Resolved (by ID)", view["id"].tolist())
    st.session_state["resolved_ids"].update(resolved_toggle)
    view["status"] = view["id"].apply(lambda x: "Resolved" if x in st.session_state["resolved_ids"] else "Open")
    st.subheader("📋 Processed Emails")
    display_cols = ["id", "__received_at", "sender", "subject", "intent", "priority_level", "sla_breach", "phones", "alt_emails", "status", "draft_response"]
    st.dataframe(view[display_cols], use_container_width=True)
    st.subheader("✍️ Edit Draft Response")
    selected_id = st.selectbox("Select Email ID", view["id"].tolist())
    draft_text = view.loc[view["id"] == selected_id, "draft_response"].values[0]
    edited = st.text_area("Draft Response", draft_text, height=200)
    if st.button("Save Response"):
        processed.loc[processed["id"] == selected_id, "draft_response"] = edited
        st.success("Response updated!")
    st.subheader("⬇️ Export Results")
    csv = processed.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "processed_emails.csv", "text/csv")
else:
    st.info("Upload a CSV or connect to IMAP to start.")
