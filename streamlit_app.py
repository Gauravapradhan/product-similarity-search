import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Product Similarity Search",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Product Similarity Search")

product_id = st.text_input(
    "Enter Product ID",
    value="26d41bdc1495de290bc8e6062d927729"
)

num_similar = st.slider(
    "Number of Similar Products",
    1,
    10,
    5
)

if st.button("Find Similar Products"):

    response = requests.get(
        f"{API_URL}/find_similar_products",
        params={
            "product_id": product_id,
            "num_similar": num_similar
        }
    )

    if response.status_code != 200:
        st.error(response.json())
    else:

        data = response.json()

        st.subheader("Query Product")

        query = data["query_product"]

        st.write(
            {
                "Product Name": query["product_name"],
                "Brand": query["brand"],
                "Price": query["sales_price"],
                "Rating": query["rating"]
            }
        )

        st.divider()

        st.subheader("Similar Products")

        for idx, product in enumerate(
            data["similar_products"],
            start=1
        ):

            with st.expander(
                f"{idx}. {product['product_name']}"
            ):

                st.write(
                    {
                        "Product ID": product["uniq_id"],
                        "Brand": product["brand"],
                        "Price": product["sales_price"],
                        "Rating": product["rating"]
                    }
                )