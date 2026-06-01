"""orders 데이터 검증 — 패턴이 의도대로 들어갔는지 확인."""
import polars as pl

# 전체 파티션 읽기
orders = pl.read_parquet("../../output/raw/orders/**/*.parquet")
users = pl.read_parquet("../../output/raw/users.parquet")

print("=" * 60)
print(f"총 주문 수: {len(orders):,}")
print(f"기간: {orders['created_at'].min()} ~ {orders['created_at'].max()}")
print("=" * 60)

# ① 세그먼트별 주문 수 (Poisson 분포 검증)
print("\n[1] 세그먼트별 주문 분포")
joined = orders.join(users.select(["user_id", "segment"]), on="user_id")
seg_stats = joined.group_by("segment").agg([
    pl.len().alias("n_orders"),
    pl.col("user_id").n_unique().alias("n_users"),
]).with_columns(
    (pl.col("n_orders") / pl.col("n_users")).round(2).alias("orders_per_user")
).sort("orders_per_user", descending=True)
print(seg_stats)

# ② 월별 주문 (시즌성 검증 — 11월/12월 스파이크 보일 것)
print("\n[2] 월별 주문 (블프/12월 스파이크 확인)")
monthly = orders.with_columns(
    pl.col("created_at").dt.strftime("%Y-%m").alias("month")
).group_by("month").len().sort("month")
print(monthly)

# ③ 시간대별 주문 (저녁 피크 검증)
print("\n[3] 시간대별 주문 (저녁 피크)")
hourly = orders.with_columns(
    pl.col("created_at").dt.hour().alias("hour")
).group_by("hour").len().sort("hour")
print(hourly)

# ④ 상태 분포
print("\n[4] Status 분포")
print(orders.group_by("status").len().sort("len", descending=True))

# ⑤ 통화 분포
print("\n[5] Currency 분포")
print(orders.group_by("currency").len().sort("len", descending=True))

# ⑥ 결제수단
print("\n[6] Payment method 분포")
print(orders.group_by("payment_method").len().sort("len", descending=True))

# ⑦ 일별 주문 - 블프 스파이크 직접 확인
print("\n[7] 블프 주간 vs 평소 (2025-11 일별)")
nov_2025 = orders.filter(
    (pl.col("created_at").dt.year() == 2025)
    & (pl.col("created_at").dt.month() == 11)
).with_columns(
    pl.col("created_at").dt.day().alias("day")
).group_by("day").len().sort("day")
print(nov_2025)

with pl.Config(tbl_rows=20):
    print(monthly)