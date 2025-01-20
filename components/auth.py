import streamlit as st

class Auth:
    """Handles simple password-based authentication using Streamlit."""
    def __init__(self, required_password: str):
        self.required_password = required_password

    def is_authenticated(self) -> bool:
        return st.session_state.get("authenticated", False)

    def authenticate(self) -> None:
        if "authenticated" not in st.session_state:
            st.session_state["authenticated"] = False
        if not self.is_authenticated():
            password = st.text_input("Enter password", type="password")
            if st.button("Submit Password"):
                if password == self.required_password:
                    st.session_state["authenticated"] = True
                    st.success("Password correct. You can proceed.")
                else:
                    st.warning("Incorrect password.")
                    st.stop()
        else:
            st.markdown("<p style='color:green;'>You are already authenticated.</p>", unsafe_allow_html=True)