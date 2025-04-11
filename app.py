import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

st.title("Market Basket Analysis using Apriori Algorithm")

uploaded_file = st.file_uploader("Upload Online Retail Excel File", type=["xlsx"])

if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)
    
    st.subheader("Raw Data")
    st.write(data.head(10))

    # Drop Duplicates
    st.write("Checking for Duplicates...")
    st.write(f"Duplicate transactions: {data.duplicated().sum()}")
    data = data.drop_duplicates()
    st.write(f"Duplicate transactions after removal: {data.duplicated().sum()}")

    # Null Values
    st.subheader("Missing Values")
    st.write(data.isna().sum())

    # Preprocessing
    st.subheader("Preprocessing")
    try:
        data["Transactions"] = data.iloc[:, 0].apply(lambda x: x.split(","))
        data = data[["Transactions"]]

        te = TransactionEncoder()
        te_ary = te.fit(data["Transactions"]).transform(data["Transactions"])
        data_encoded = pd.DataFrame(te_ary, columns=te.columns_)

        # Apriori
        st.subheader("Frequent Itemsets")
        frequent_itemsets = apriori(data_encoded, min_support=0.02, use_colnames=True)
        st.write(frequent_itemsets)

        st.subheader("Association Rules")
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
        st.write(rules)

        st.write(f"Total number of association rules: {rules.shape[0]}")

        # Top 10 by Lift
        top10 = rules.sort_values('lift', ascending=False).head(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top10.lift, y=top10["antecedents"].astype(str), orient='h')
        plt.xlabel("Lift")
        plt.ylabel("Antecedents")
        plt.title("Top 10 Association Rules by Lift")
        st.pyplot(plt.gcf())

        # Scatter Plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=rules["confidence"], y=rules["lift"], size=rules["support"], alpha=0.6, hue=rules["confidence"], palette="viridis")
        plt.xlabel("Confidence")
        plt.ylabel("Lift")
        plt.title("Lift vs Confidence (Bubble size = Support)")
        st.pyplot(plt.gcf())

        # Heatmap
        st.subheader("Heatmap of Top 20 Rules")
        top_rules_matrix = rules.sort_values("lift", ascending=False).head(20)[["antecedents", "consequents", "lift"]]
        top_rules_matrix["antecedents"] = top_rules_matrix["antecedents"].astype(str)
        top_rules_matrix["consequents"] = top_rules_matrix["consequents"].astype(str)

        pivot_table = top_rules_matrix.pivot(index="antecedents", columns="consequents", values="lift").fillna(0)

        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_table, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Lift Values Between Product Associations")
        plt.xlabel("Consequents (Likely to be Bought)")
        plt.ylabel("Antecedents (Bought First)")
        st.pyplot(plt.gcf())

        # Conclusion
        st.subheader("Conclusion")
        st.markdown("""
        - The Apriori algorithm effectively identifies frequent itemsets and meaningful association rules.
        - Rules with high **lift** indicate strong associations between purchased products.
        - **Confidence** measures the likelihood that a product will be bought if another is bought.
        - **Support** shows how frequently a rule appears in the dataset.
        - This analysis helps retailers understand customer purchase behavior and create strategies for product placement, bundling, and promotions.
        """)

    except Exception as e:
        st.error(f"Error during preprocessing or analysis: {e}")