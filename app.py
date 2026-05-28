from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.similarity_engine import (
    ProductSimilarityEngine
)
# ============================================================
# Response Models
# ============================================================

class ProductResponse(BaseModel):
    uniq_id: str
    product_name: str
    brand: str
    sales_price: float | None
    rating: float


class SimilarProductsResponse(BaseModel):
    query_product: ProductResponse
    similar_products: list[ProductResponse]

app = FastAPI(
    title="Product Similarity Search Service",
    version="1.0.0"
)

# Load artifacts once at startup
engine = ProductSimilarityEngine.load(
    "artifacts"
)

@app.get("/")
def root():
    return {
        "service": "Product Similarity Search",
        "status": "running"
    }
@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.get(
    "/find_similar_products",
    response_model=SimilarProductsResponse
)
def find_similar_products(
    product_id: str,
    num_similar: int = 5
):

    try:

        similar_ids = (
        engine.find_similar_products(
        product_id,
        num_similar
        )
    )

        similar_products = (
        engine.get_product_details(
        similar_ids
        )
    )
        query_product = engine.get_product_details(
        [product_id]
        )[0]

        return {
            "query_product": query_product,
            "similar_products": similar_products
        }
    except Exception as e:

        raise HTTPException(
            status_code=404,
            detail=str(e)
        )


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )