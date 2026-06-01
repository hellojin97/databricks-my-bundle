"""order_items 검증."""
import polars as pl

orders = pl.read_parquet("../../output/raw/orders/**/*.parquet")
items = pl.read_parquet("../../output/raw/order_items/**/*.parquet")

print(f"orders: {len(orders):,}")
print(f"items: {len(items):,}")
print(f"평균 라인/주문: {len(items) / len(orders):.2f}")

# 라인 수 분포
print("\n[1] 주문당 라인 수 분포")
items_per_order = items.group_by("order_id").agg(pl.len().alias("n_lines"))
distribution = (
    items_per_order
    .group_by("n_lines")
    .agg(pl.len().alias("n_orders"))
    .sort("n_lines")
)
print(distribution)

# 할인 분포
print("\n[2] 할인율 분포")
print(items.group_by("discount_pct").len().sort("discount_pct"))

# 수량 분포
print("\n[3] 수량 분포")
print(items.group_by("quantity").len().sort("quantity"))

# total_amount 검증: orders.total = sum(items.line_total)
print("\n[4] orders.total_amount vs items 합계 (정합성 체크)")
agg = items.group_by("order_id").agg(pl.col("line_total").sum().round(2).alias("items_sum"))
merged = orders.join(agg, on="order_id")
diff = merged.with_columns(
    (pl.col("total_amount") - pl.col("items_sum")).abs().alias("diff")
)
print(f"최대 차이: {diff['diff'].max()}")
print(f"평균 차이: {diff['diff'].mean()}")
print("→ 0에 가까워야 정상")

# 인기 상품 Top 5
print("\n[5] 인기 상품 Top 5 (Pareto 검증)")
top_products = items.group_by("product_id").len().sort("len", descending=True).head(5)
print(top_products)

# 전체 매출
print("\n[6] 매출 통계")
total_revenue = items["line_total"].sum()
print(f"총 매출: ${total_revenue:,.2f}")