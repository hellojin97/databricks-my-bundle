"""주문 라인 팩트 테이블 생성기.

orders에 의존 (FK: order_id), products에 의존 (FK: product_id).

실전 느낌:
- 주문당 라인 수: Pareto 분포 (1개가 가장 많음)
- 상품 선택: 인기 상품에 쏠림 (Pareto)
- 시간적 정합성: 주문일에 이미 존재(created_at ≤ 주문일)하고
  아직 단종되지 않은(주문일 ≤ discontinued_at) 상품만 라인에 담긴다.
- 할인: 60% 정가, 40% 다양한 할인율
- cancelled/refunded 주문도 라인 그대로 (status만 다름)
- 부수효과: orders.total_amount(USD) 업데이트 + amount_local(현지통화) 산출
"""
from datetime import datetime

import numpy as np
import polars as pl

from .base import make_rng
from .generate_orders import ZERO_DECIMAL_CURRENCIES

# discontinued_at NULL(=미단종) 대체용 먼 미래값 (비교 시 항상 "주문 이후")
_FAR_FUTURE = datetime(2999, 1, 1)

# 주문당 라인 수 분포 (1~5)
ITEMS_PER_ORDER_PROBS = np.array([0.50, 0.25, 0.15, 0.07, 0.03])

# 할인율 분포
DISCOUNT_CHOICES = np.array([0.0, 0.10, 0.20, 0.30])
DISCOUNT_PROBS = np.array([0.60, 0.25, 0.10, 0.05])

# 수량 분포 (대부분 1개, 가끔 여러 개)
QUANTITY_CHOICES = np.array([1, 2, 3], dtype=np.int8)
QUANTITY_PROBS = np.array([0.80, 0.15, 0.05])


def _weighted_prefix_choice(
    sorted_created: np.ndarray,
    cum_weights: np.ndarray,
    line_created: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """주문 시점에 '이미 생성된' 상품들 중 인기도 가중으로 1개씩 선택.

    상품을 created_at 오름차순으로 정렬해두면, 주문일 기준 eligible 상품은
    항상 정렬 배열의 prefix [0:k] 가 된다 (k = 주문일까지 생성된 상품 수).
    누적 가중치(cum_weights)에 대해 u ~ U(0, cum_weights[k-1]) 를 뽑고
    searchsorted 하면, prefix 내부에서 가중치 비례 샘플링이 벡터화된다.

    Returns:
        chosen_sorted_idx: 정렬 배열 기준 인덱스 (shape: line 수). k==0(상품
        생성 전 주문)인 경우는 최소 1개로 보정해 첫 상품을 가리킨다.
    """
    k = np.searchsorted(sorted_created, line_created, side="right")
    k = np.maximum(k, 1)  # 모든 상품보다 이른 주문(드묾)은 첫 상품으로 보정
    u = rng.uniform(0, 1, size=len(line_created)) * cum_weights[k - 1]
    chosen = np.searchsorted(cum_weights, u, side="right")
    return np.minimum(chosen, k - 1)  # 부동소수 오차로 prefix 밖을 가리키지 않게


def generate(
    orders_df: pl.DataFrame,
    products_df: pl.DataFrame,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame]:
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

    # --- 3. 상품 선택 (인기도 Pareto + 시간적 정합성) ---
    product_ids = products_df["product_id"].to_numpy()
    base_prices = products_df["base_price"].to_numpy()
    # 단위는 us 고정 (orders/products created_at과 동일). ns는 1678~2262만 표현
    # 가능해 _FAR_FUTURE(2999)가 오버플로우되므로 절대 쓰면 안 된다.
    product_created = products_df["created_at"].to_numpy().astype("datetime64[us]")
    # 미단종 상품은 먼 미래값으로 채워, "주문일 ≤ discontinued_at" 비교를 단순화
    product_disc = (
        products_df["discontinued_at"].fill_null(_FAR_FUTURE).to_numpy().astype("datetime64[us]")
    )
    is_active = products_df["discontinued_at"].is_null().to_numpy()

    line_created = line_created_at.astype("datetime64[us]")

    # 상품별 인기도 가중치
    product_weights = rng.pareto(1.5, size=len(product_ids)) + 0.5

    # (a) created_at ≤ 주문일 제약: 전체 상품을 created_at 순으로 정렬해 prefix 샘플링
    sort_idx = np.argsort(product_created, kind="stable")
    sorted_created = product_created[sort_idx]
    cum_weights = np.cumsum(product_weights[sort_idx])
    chosen_idx = sort_idx[
        _weighted_prefix_choice(sorted_created, cum_weights, line_created, rng)
    ]

    # (b) 단종 정합성: 주문일에 이미 단종된 상품을 고른 라인은 '미단종 상품'으로 재선택
    bad = product_disc[chosen_idx] < line_created
    if bad.any():
        a_orig = np.nonzero(is_active)[0]
        a_sort = np.argsort(product_created[a_orig], kind="stable")
        a_sorted_created = product_created[a_orig][a_sort]
        a_cum_weights = np.cumsum(product_weights[a_orig][a_sort])
        bad_created = line_created[bad]
        # 해당 주문일까지 생성된 미단종 상품이 하나라도 있어야 재선택 가능
        repairable = np.searchsorted(a_sorted_created, bad_created, side="right") >= 1
        bad_lines = np.nonzero(bad)[0][repairable]
        if len(bad_lines) > 0:
            repick = _weighted_prefix_choice(
                a_sorted_created, a_cum_weights, line_created[bad_lines], rng
            )
            chosen_idx[bad_lines] = a_orig[a_sort[repick]]

    line_product_ids = product_ids[chosen_idx]
    line_base_prices = base_prices[chosen_idx]

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

    # --- 9. orders.total_amount(USD) 업데이트 + amount_local(현지통화) 산출 ---
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

    # 현지 통화 금액: total_amount(USD) × fx_rate. KRW/JPY는 소수점 없이 정수 반올림.
    local = pl.col("total_amount") * pl.col("fx_rate")
    updated_orders_df = updated_orders_df.with_columns(
        pl.when(pl.col("currency").is_in(list(ZERO_DECIMAL_CURRENCIES)))
        .then(local.round(0))
        .otherwise(local.round(2))
        .alias("amount_local")
    )

    # 컬럼 순서 정리 (금액 컬럼들을 dt 앞에 위치)
    updated_orders_df = updated_orders_df.select([
        "order_id", "user_id", "created_at", "status",
        "payment_method", "currency", "fx_rate", "total_amount", "amount_local", "dt",
    ])

    return order_items_df, updated_orders_df