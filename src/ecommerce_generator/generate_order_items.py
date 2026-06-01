"""주문 라인 팩트 테이블 생성기.

orders에 의존 (FK: order_id), products에 의존 (FK: product_id).

실전 느낌:
- 주문당 라인 수: Pareto 분포 (1개가 가장 많음)
- 상품 선택: 인기 상품에 쏠림 (Pareto)
- 할인: 60% 정가, 40% 다양한 할인율
- cancelled/refunded 주문도 라인 그대로 (status만 다름)
- 부수효과: orders.total_amount 업데이트
"""
from typing import Tuple

import numpy as np
import polars as pl

from .base import make_rng


# 주문당 라인 수 분포 (1~5)
ITEMS_PER_ORDER_PROBS = np.array([0.50, 0.25, 0.15, 0.07, 0.03])

# 할인율 분포
DISCOUNT_CHOICES = np.array([0.0, 0.10, 0.20, 0.30])
DISCOUNT_PROBS = np.array([0.60, 0.25, 0.10, 0.05])

# 수량 분포 (대부분 1개, 가끔 여러 개)
QUANTITY_CHOICES = np.array([1, 2, 3], dtype=np.int8)
QUANTITY_PROBS = np.array([0.80, 0.15, 0.05])


def generate(
    orders_df: pl.DataFrame,
    products_df: pl.DataFrame,
    seed: int = 42,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """order_items 생성 + orders의 total_amount 업데이트.

    Args:
        orders_df: 주문 팩트 테이블
        products_df: 상품 차원 (product_id, base_price 필요)
        seed: 재현성

    Returns:
        (order_items_df, updated_orders_df)
    """
    rng = make_rng(seed)

    # --- orders에서 필요한 컬럼 추출 ---
    order_ids = orders_df["order_id"].to_numpy()
    order_created_at = orders_df["created_at"].to_numpy()
    order_dts = orders_df["dt"].to_numpy()
    n_orders = len(order_ids)

    # --- 1. 주문당 라인 수 결정 ---
    n_items_per_order = rng.choice(
        [1, 2, 3, 4, 5],
        size=n_orders,
        p=ITEMS_PER_ORDER_PROBS,
    )
    n_total_items = int(n_items_per_order.sum())
    print(f"    Total order_items: {n_total_items:,}")

    # --- 2. 라인 단위로 펼치기 (np.repeat 패턴) ---
    line_order_ids = np.repeat(order_ids, n_items_per_order)
    line_created_at = np.repeat(order_created_at, n_items_per_order)
    line_dts = np.repeat(order_dts, n_items_per_order)

    # --- 3. 상품 선택 (Pareto: 인기 상품에 쏠림) ---
    product_ids = products_df["product_id"].to_numpy()
    base_prices = products_df["base_price"].to_numpy()

    # 상품별 인기도 가중치
    product_weights = rng.pareto(1.5, size=len(product_ids)) + 0.5
    product_probs = product_weights / product_weights.sum()

    chosen_product_idx = rng.choice(
        len(product_ids),
        size=n_total_items,
        p=product_probs,
    )
    line_product_ids = product_ids[chosen_product_idx]
    line_base_prices = base_prices[chosen_product_idx]

    # --- 4. 할인 + 단가 계산 ---
    discount_pcts = rng.choice(
        DISCOUNT_CHOICES,
        size=n_total_items,
        p=DISCOUNT_PROBS,
    )
    unit_prices = np.round(line_base_prices * (1 - discount_pcts), 2)

    # --- 5. 수량 ---
    quantities = rng.choice(
        QUANTITY_CHOICES,
        size=n_total_items,
        p=QUANTITY_PROBS,
    )

    # --- 6. 라인 합계 ---
    line_totals = np.round(unit_prices * quantities, 2)

    # --- 7. order_item_id 부여 ---
    order_item_ids = np.arange(1, n_total_items + 1, dtype=np.int64)

    # --- 8. DataFrame 조립 ---
    order_items_df = pl.DataFrame({
        "order_item_id": order_item_ids,
        "order_id": line_order_ids,
        "product_id": line_product_ids,
        "quantity": quantities.astype(np.int8),
        "unit_price": unit_prices,
        "discount_pct": discount_pcts,
        "line_total": line_totals,
        "created_at": line_created_at,
        "dt": line_dts,
    })

    # 정렬: order_id 순 (실전 데이터처럼)
    order_items_df = order_items_df.sort(["order_id", "order_item_id"])

    # --- 9. orders.total_amount 업데이트 ---
    # order_id별 line_total 합계를 구해서 orders의 placeholder를 교체
    print("    Updating orders.total_amount with real line totals...")
    real_totals = (
        order_items_df
        .group_by("order_id")
        .agg(pl.col("line_total").sum().round(2).alias("total_amount_real"))
    )

    updated_orders_df = (
        orders_df
        .drop("total_amount")   # placeholder 제거
        .join(real_totals, on="order_id")
        .rename({"total_amount_real": "total_amount"})
    )

    # 컬럼 순서 원복 (total_amount를 dt 앞에 위치)
    updated_orders_df = updated_orders_df.select([
        "order_id", "user_id", "created_at", "status",
        "payment_method", "currency", "total_amount", "dt",
    ])

    return order_items_df, updated_orders_df