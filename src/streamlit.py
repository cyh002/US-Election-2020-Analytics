import streamlit as st

def main():
    st.title("Streamlit Skeleton App")
    user_input = st.text_input("Enter some text:")
    if user_input:
        st.write(f"You entered: {user_input}")

if __name__ == "__main__":
    main()
