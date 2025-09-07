import streamlit as st
import requests

st.title("GenAI Code Review (Local MVP)")

diff = st.text_area(
    "Paste a unified diff or code snippet",
    height=240,
    value="@@ -1,2 +1,3 @@\n- x = [i for i in range(1000000)]\n+ x = list(range(1000000))\n+ # NOTE: consider generator if not needed all at once\n",
)
focus = st.multiselect(
    "Focus",
    ["security", "performance", "style"],
    default=["security", "performance", "style"],
)

if st.button("Run Review"):
    r = requests.post(
        "http://127.0.0.1:8000/review", json={"diff": diff, "focus": focus}
    )
    if r.ok:
        data = r.json()
        st.subheader("Summary")
        st.write(data["summary"])
        st.subheader("Findings")
        for f in data["findings"]:
            st.markdown(
                f"- **{f['category'].title()} / {f['severity']}** — {f['message']}"
            )
    else:
        st.error(f"Request failed: {r.status_code} - {r.text}")
