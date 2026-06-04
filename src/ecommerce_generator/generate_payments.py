"""결제(payments) 팩트 테이블 생성기.

orders에 1:1로 의존 (FK: order_id). 주문 1건당 결제 1행.

설계:
- payment status는 order.status와 엄격히 정합한다(파생).
    delivered/shipped/processing → captured
    pending                      → authorized
    cancelled                    → failed
    refunded                     → refunded
- 금액은 고객이 실제 청구받은 현지통화 금액(order.amount_local)을 쓰고,
  통화 코드(order.currency)와 일관되게 맞춘다. 수단도 order 값을 그대로 쓴다.
- paid_at은 주문 직후(수초~수분), refunded_at은 환불 건에만 채운다.
"""
from datetime import date

import numpy as np
import polars as pl

from .base import make_rng

# order.status → payment status (모든 주문 상태를 빠짐없이 매핑)
STATUS_MAP = {
    "delivered": "captured",
    "shipped": "captured",
    "processing": "captured",
    "pending": "authorized",
    "cancelled": "failed",
    "refunded": "refunded",
}

# 결제 승인까지 걸리는 시간(초): 주문 직후 1초 ~ 2분
_PAID_MIN_S, _PAID_MAX_S = 1, 120
# 환불까지 걸리는 시간(일): 결제 후 1 ~ 30일
_REFUND_MIN_D, _REFUND_MAX_D = 1, 30


def generate(
    orders_df: pl.DataFrame,
    end_date: date,
    seed: int = 42,
) -> pl.DataFrame:
    """결제 팩트 테이블 생성 (주문당 1행).

    Args:
        orders_df: 주문 팩트 (order_id, created_at, status, payment_method,
                   total_amount, currency 필요)
        end_date: 분석 기간 종료일 (타임스탬프 클리핑용)
        seed: 재현성

    Returns:
        payments DataFrame (paid_at 오름차순, dt 파티션 컬럼 포함)
    """
    rng = make_rng(seed)
    n = len(orders_df)
    print(f"    Total payments: {n:,}")

    order_ids = orders_df["order_id"].to_numpy()
    order_created = orders_df["created_at"].to_numpy().astype("datetime64[us]")
    statuses = orders_df["status"].to_numpy()
    methods = orders_df["payment_method"].to_numpy()
    amounts = orders_df["amount_local"].to_numpy()
    currencies = orders_df["currency"].to_numpy()

    # status 매핑 (벡터화)
    pay_status = (
        orders_df.select(pl.col("status").replace_strict(STATUS_MAP).alias("ps"))["ps"]
        .to_numpy()
    )

    end_ts = (
        np.datetime64(end_date).astype("datetime64[s]") + np.timedelta64(86399, "s")
    ).astype("datetime64[us]")

    # paid_at: 리드타임을 남은 시간 이내로 제한 (클리핑 방지)
    remaining_secs = (
        (end_ts - order_created).astype("timedelta64[s]").astype(np.int64)
    )
    paid_max_secs = np.clip(remaining_secs, _PAID_MIN_S, _PAID_MAX_S)
    paid_offset = np.array([
        rng.integers(_PAID_MIN_S, max_s + 1) if max_s >= _PAID_MIN_S else max_s
        for max_s in paid_max_secs
    ])
    paid_at = order_created + paid_offset.astype("timedelta64[s]")

    # refunded_at: 환불 건에만 (그 외 NaT → polars null)
    refunded_at = np.full(n, np.datetime64("NaT"), dtype="datetime64[us]")
    is_refunded = statuses == "refunded"
    n_ref = int(is_refunded.sum())
    if n_ref:
        paid_ref = paid_at[is_refunded]
        remaining_after_paid = (
            (end_ts - paid_ref).astype("timedelta64[s]").astype(np.int64)
        )
        refund_max_secs = np.clip(
            remaining_after_paid, _REFUND_MIN_D * 86400, _REFUND_MAX_D * 86400
        )
        ref_secs = np.array([
            rng.integers(_REFUND_MIN_D * 86400, max_s + 1)
            if max_s >= _REFUND_MIN_D * 86400 else max_s
            for max_s in refund_max_secs
        ])
        refunded_at[is_refunded] = paid_ref + ref_secs.astype("timedelta64[s]")

    df = pl.DataFrame({
        "payment_id": np.arange(1, n + 1, dtype=np.int64),
        "order_id": order_ids,
        "payment_method": methods.tolist(),
        "amount": amounts,
        "currency": currencies.tolist(),
        "status": pay_status.tolist(),
        "paid_at": paid_at,
        "refunded_at": refunded_at,
        "dt": paid_at.astype("datetime64[D]"),
    })

    return df.sort("paid_at")
