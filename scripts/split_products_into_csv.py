import os
import pandas as pd
import re

REVIEWS_PATH = "data/raw/AllProductReviews.csv"
PRODUCT_INFO_PATH = "data/raw/ProductInfo.csv"
OUTPUT_DIR = "data/processed/products"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading reviews...")
reviews = pd.read_csv(REVIEWS_PATH)

print("Loading product info...")
products = pd.read_csv(PRODUCT_INFO_PATH)

reviews.columns = [c.lower() for c in reviews.columns]
products.columns = [c.lower() for c in products.columns]

df = reviews.merge(
    products[["productshortname", "id"]],
    how="left",
    left_on="product",
    right_on="productshortname"
)

df = df.rename(columns={
    "id": "product_id",
    "product": "product_name",
    "reviewbody": "review_text",
    "reviewtitle": "review_title",
    "reviewstar": "rating"
})

df["review_id"] = range(1, len(df) + 1)
df = df[["review_id", "product_id", "product_name", "review_text", "rating"]]
df = df.dropna(subset=["review_text"])

unique_products = df["product_name"].unique()
print(f"Found {len(unique_products)} unique products")

def sanitize_filename(name):
    safe = re.sub(r'[^A-Za-z0-9]+', '_', name)
    return safe.strip("_")

for product in unique_products:
    product_df = df[df["product_name"] == product]
    file_name = sanitize_filename(product) + ".csv"
    out_path = os.path.join(OUTPUT_DIR, file_name)
    product_df.to_csv(out_path, index=False)
    print(f"Saved {len(product_df)} reviews to {out_path}")

print("Done")
