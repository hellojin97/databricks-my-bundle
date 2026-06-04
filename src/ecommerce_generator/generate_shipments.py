"""배송(shipments) 팩트 테이블 생성기.

orders에 의존 (FK: order_id). 실제로 발송된 주문에만 1행을 만든다.

대상: order.status ∈ {shipped, delivered, refunded}
- pending/processing(아직 미발송), cancelled(발송 전 취소)은 배송 행이 없다.

설계:
- shipment status는 order.status와 정합한다.
    shipped            → in_transit
    delivered/refunded → delivered (도착함; 환불은 결제 쪽 사건)
- shipped_at = 주문 + 처리 리드타임(0.5~3일), delivered_at = 발송 + 배송 리드타임(1~7일).
"""
from datetime import date

import numpy as np
import polars as pl

from .base import make_rng, weighted_choice

# 배송 대상 주문 상태
SHIPPED_STATUSES = ["shipped", "delivered", "refunded"]

# order.status → shipment status
STATUS_MAP = {
    "shipped": "in_transit",
    "delivered": "delivered",
    "refunded": "delivered",
}

CARRIERS_WEIGHTED = [
    ("UPS", 0.28), ("FedEx", 0.24), ("USPS", 0.20),
    ("DHL", 0.14), ("CJ Logistics", 0.08), ("Royal Mail", 0.06),
]

# 처리 리드타임(주문 → 발송) 시간 단위: 12h ~ 72h (0.5~3일)
_PROC_MIN_H, _PROC_MAX_H = 12, 72
# 배송 리드타임(발송 → 도착) 시간 단위: 24h ~ 168h (1~7일)
_TRANSIT_MIN_H, _TRANSIT_MAX_H = 24, 168


def generate(
    orders_df: pl.DataFrame,
    end_date: date,
    seed: int = 42,
) -> pl.DataFrame:
    """배송 팩트 테이블 생성 (발송된 주문당 1행).

    Args:
        orders_df: 주문 팩트 (order_id, created_at, status 필요)
        end_date: 분석 기간 종료일 (타임스탬프 클리핑용)
        seed: 재현성

    Returns:
        shipments DataFrame (shipped_at 오름차순, dt 파티션 컬럼 포함)
    """
    rng = make_rng(seed)

    sub = orders_df.filter(pl.col("status").is_in(SHIPPED_STATUSES))
    n = len(sub)
    print(f"    Total shipments: {n:,}")

    order_ids = sub["order_id"].to_numpy()
    order_created = sub["created_at"].to_numpy().astype("datetime64[us]")
    ship_status = sub.select(
        pl.col("status").replace_strict(STATUS_MAP).alias("ss")
    )["ss"].to_numpy()

    carriers = weighted_choice(
        rng,
        [c for c, _ in CARRIERS_WEIGHTED],
        [w for _, w in CARRIERS_WEIGHTED],
        n,
    )

    end_ts = (
        np.datetime64(end_date).astype("datetime64[s]") + np.timedelta64(86399, "s")
    ).astype("datetime64[us]")

    # shipped_at: 리드타임을 남은 시간 이내로 제한 (클리핑 방지)
    remaining_secs = (
        (end_ts - order_created).astype("timedelta64[s]").astype(np.int64)
    )
    # rng.integers는 배열 상한을 받으므로 행별 상한으로 한 번에 벡터화 (Python 루프 제거)
    proc_high = np.clip(remaining_secs, _PROC_MIN_H * 3600, _PROC_MAX_H * 3600) + 1
    proc_secs = rng.integers(_PROC_MIN_H * 3600, proc_high)
    shipped_at = np.minimum(order_created + proc_secs.astype("timedelta64[s]"), end_ts)

    # delivered_at: 도착(delivered) 건에만 (in_transit은 NaT → null)
    delivered_at = np.full(n, np.datetime64("NaT"), dtype="datetime64[us]")
    is_delivered = ship_status == "delivered"
    n_del = int(is_delivered.sum())
    if n_del:
        shipped_del = shipped_at[is_delivered]
        remaining_after_ship = (
            (end_ts - shipped_del).astype("timedelta64[s]").astype(np.int64)
        )
        transit_high = np.clip(
            remaining_after_ship, _TRANSIT_MIN_H * 3600, _TRANSIT_MAX_H * 3600
        ) + 1
        transit_secs = rng.integers(_TRANSIT_MIN_H * 3600, transit_high)
        delivered_at[is_delivered] = np.minimum(
            shipped_del + transit_secs.astype("timedelta64[s]"), end_ts
        )

    df = pl.DataFrame({
        "shipment_id": np.arange(1, n + 1, dtype=np.int64),
        "order_id": order_ids,
        "carrier": carriers.tolist(),
        "status": ship_status.tolist(),
        "shipped_at": shipped_at,
        "delivered_at": delivered_at,
        "dt": shipped_at.astype("datetime64[D]"),
    })

    return df.sort("shipped_at")
