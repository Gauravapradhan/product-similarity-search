import os
import faiss
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from scipy.sparse import (
    hstack,
    csr_matrix,
    save_npz,
    load_npz
)

from sklearn.feature_extraction.text import (
    TfidfVectorizer
)

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
class ProductSimilarityEngine:

    def __init__(self):

        self.df = None

        self.vectorizer = None
        self.brand_encoder = None
        self.scaler = None

        self.feature_matrix = None

        self.product_lookup = None

        self.faiss_index = None
        self.svd = None
        self.vectors = None 
    def prepare_data(self, df):

        df = df.copy()

        df["brand"] = (
            df["brand"]
            .fillna("unknown")
        )

        df["sales_price"] = (
            df["sales_price"]
            .fillna(
                df["sales_price"].median()
            )
        )

        df["category_text"] = (
            df["parent___child_category__all"]
            .fillna("")
            .astype(str)
        )

        df["combined_text"] = (
            df["product_name"].fillna("")
            + " "
            + df["meta_keywords"].fillna("")
            + " "
            + df["category_text"]
        )

        return df
    def fit(self, df):

        self.df = self.prepare_data(df)

        # --------------------------------------------------
        # TF-IDF
        # --------------------------------------------------

        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2
        )

        text_features = self.vectorizer.fit_transform(
            self.df["combined_text"]
        )

        # --------------------------------------------------
        # Brand
        # --------------------------------------------------

        self.brand_encoder = OneHotEncoder(
            handle_unknown="ignore"
        )

        brand_features = (
            self.brand_encoder.fit_transform(
                self.df[["brand"]]
            )
        )

        # --------------------------------------------------
        # Numeric
        # --------------------------------------------------

        self.scaler = StandardScaler()

        numeric_features = (
            self.scaler.fit_transform(
                self.df[
                    ["sales_price", "rating"]
                ]
            )
        )

        numeric_features = csr_matrix(
            numeric_features
        )

        # --------------------------------------------------
        # Hybrid Feature Matrix
        # --------------------------------------------------

        self.feature_matrix = (
            hstack([
                text_features * 0.85,
                brand_features * 0.10,
                numeric_features * 0.05
            ])
            .tocsr()
        )

        # --------------------------------------------------
        # Dimensionality Reduction
        # --------------------------------------------------

        self.svd = TruncatedSVD(
            n_components=256,
            random_state=42
        )

        self.vectors = (
            self.svd.fit_transform(
                self.feature_matrix
            )
            .astype("float32")
        )

        faiss.normalize_L2(
            self.vectors
        )

        # --------------------------------------------------
        # Product Lookup
        # --------------------------------------------------

        self.product_lookup = {
            pid: idx
            for idx, pid in enumerate(
                self.df["uniq_id"]
            )
        }

        return self
    def build_faiss_index(self):

        self.faiss_index = faiss.IndexFlatIP(
        self.vectors.shape[1]
        )

        self.faiss_index.add(
        self.vectors
        )

        return self
    def find_similar_products(
        self,
        product_id,
        num_similar=5
    ):

        if product_id not in self.product_lookup:
            raise ValueError(
                f"{product_id} not found"
            )

        idx = self.product_lookup[
            product_id
        ]

        query_name = self.df.iloc[idx][
            "product_name"
        ]

        query_vector = (
            self.vectors[idx]
            .reshape(1, -1)
        )

        distances, indices = (
            self.faiss_index.search(
                query_vector,
                num_similar + 10
            )
        )

        results = []

        for i in indices[0]:

            if i == idx:
                continue

            candidate_name = self.df.iloc[i][
                "product_name"
            ]

            if candidate_name == query_name:
                continue

            results.append(
                self.df.iloc[i]["uniq_id"]
            )

            if len(results) == num_similar:
                break

        return results
    def get_product_details(
        self,
        product_ids
        ):

        products = []

        for pid in product_ids:

            row = self.df[
                self.df["uniq_id"] == pid
            ].iloc[0]

            products.append({
                "uniq_id": row["uniq_id"],
                "product_name": row["product_name"],
                "brand": row["brand"],
                "sales_price": (
                    None if pd.isna(row["sales_price"])
                    else float(row["sales_price"])
                ),
                "rating": float(row["rating"])
            })

        return products
    def save(self, artifact_dir):

        os.makedirs(
            artifact_dir,
            exist_ok=True
        )

        joblib.dump(
            self.vectorizer,
            f"{artifact_dir}/tfidf.joblib"
        )

        joblib.dump(
            self.brand_encoder,
            f"{artifact_dir}/brand_encoder.joblib"
        )

        joblib.dump(
            self.scaler,
            f"{artifact_dir}/scaler.joblib"
        )

        joblib.dump(
            self.df,
            f"{artifact_dir}/products.joblib"
        )
        joblib.dump(
            self.svd,
            f"{artifact_dir}/svd.joblib"
        )

        np.save(
            f"{artifact_dir}/vectors.npy",
            self.vectors
        )
        joblib.dump(
            self.product_lookup,
            f"{artifact_dir}/lookup.joblib"
        )

        save_npz(
            f"{artifact_dir}/feature_matrix.npz",
            self.feature_matrix
        )

        faiss.write_index(
            self.faiss_index,
            f"{artifact_dir}/faiss.index"
        )
    @classmethod
    def load(
        cls,
        artifact_dir
    ):

        engine = cls()

        engine.vectorizer = joblib.load(
            f"{artifact_dir}/tfidf.joblib"
        )

        engine.brand_encoder = joblib.load(
            f"{artifact_dir}/brand_encoder.joblib"
        )

        engine.scaler = joblib.load(
            f"{artifact_dir}/scaler.joblib"
        )

        engine.df = joblib.load(
            f"{artifact_dir}/products.joblib"
        )
        engine.svd = joblib.load(
            f"{artifact_dir}/svd.joblib"
        )

        engine.vectors = np.load(
            f"{artifact_dir}/vectors.npy"
        )
        engine.product_lookup = joblib.load(
            f"{artifact_dir}/lookup.joblib"
        )

        engine.feature_matrix = load_npz(
            f"{artifact_dir}/feature_matrix.npz"
        )

        engine.faiss_index = (
            faiss.read_index(
                f"{artifact_dir}/faiss.index"
            )
        )

        return engine