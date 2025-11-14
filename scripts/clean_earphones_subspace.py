import os
import pandas as pd

REVIEWS_PATH = "data/raw/AllProductReviews.csv"
PRODUCT_INFO_PATH = "data/raw/ProductInfo.csv"
OUT_PATH = "data/processed/cleaned_reviews_earphones.csv"

os.makedirs("data/processed", exist_ok=True)

print("📥 Loading reviews...")
reviews = pd.read_csv(REVIEWS_PATH)

print("📥 Loading product info...")
products = pd.read_csv(PRODUCT_INFO_PATH)

# Normalize column names
reviews.columns = [c.lower() for c in reviews.columns]
products.columns = [c.lower() for c in products.columns]

# Merge on product name
df = reviews.merge(
    products[["productshortname", "id"]],   # id = product_id
    how="left",
    left_on="product",
    right_on="productshortname"
)

# Rename for consistency
df = df.rename(columns={
    "id": "product_id",
    "product": "product_name",
    "reviewbody": "review_text",
    "reviewtitle": "review_title",
    "reviewstar": "rating"
})

# Add review_id column
df["review_id"] = range(1, len(df) + 1)

# Keep only what we need
df = df[["review_id", "product_id", "product_name", "review_text", "rating"]]

# Drop empty & duplicates
df = df.dropna(subset=["review_text"])
df = df.drop_duplicates(subset=["review_id"])

# Save
df.to_csv(OUT_PATH, index=False)

print(f"✅ Saved cleaned dataset: {OUT_PATH}")
print(f"Total reviews: {len(df):,}")
print("Columns:", list(df.columns))
