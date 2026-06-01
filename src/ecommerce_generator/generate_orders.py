"""주문 팩트 테이블 생성기.

가장 복잡한 generator. 다음 패턴들이 들어감:
- Poisson 분포로 유저별 주문 수 결정 (세그먼트 가중)
- 가입일 이후로만 주문 가능 (가입 기간 보정)
- 요일/시간대/계절성 시간 패턴
- 세분화된 status (pending → delivered, cancelled, refunded)
"""
from datetime import date

import numpy as np
import polars as pl

from .base import make_rng, weighted_choice


# 세그먼트별 월 평균 주문 수
# 합계가 3M에 맞도록 조정 (100K 유저 기준)
MONTHLY_ORDER_RATE = {
    "vip":     12.0,
    "regular":  2.5,
    "new":      1.0,
    "churned":  0.1,
}

# 시간대 가중치 (24시간, 저녁 피크)
HOUR_WEIGHTS = np.array([
    0.5, 0.3, 0.2, 0.2, 0.3, 0.5,  # 0~5  (새벽)
    1.0, 1.5, 2.0, 2.0, 1.5, 1.5,  # 6~11 (오전)
    2.5, 2.5, 2.0, 2.0, 2.5, 3.0,  # 12~17 (점심+오후)
    4.0, 4.5, 4.0, 3.0, 2.0, 1.0,  # 18~23 (저녁 피크)
])
HOUR_WEIGHTS = HOUR_WEIGHTS / HOUR_WEIGHTS.sum()

STATUS_PROBS = {
    # (final_status_if_old, weight)
    "delivered": 0.75,
    "cancelled": 0.10,
    "refunded":  0.05,
    "shipped":   0.05,
    "processing":0.03,
    "pending":   0.02,
}

PAYMENT_METHODS_WEIGHTED = [
    ("credit_card", 0.55),
    ("debit_card",  0.20),
    ("paypal",      0.12),
    ("apple_pay",   0.08),
    ("bank_transfer", 0.05),
]

COUNTRY_TO_CURRENCY = {
    "US": "USD", "CA": "USD", "MX": "USD", "BR": "USD",
    "UK": "GBP",
    "DE": "EUR", "FR": "EUR", "IT": "EUR",
    "KR": "KRW", "JP": "JPY",
    "AU": "AUD", "IN": "INR",
}


def _compute_orders_per_user(
    users_df: pl.DataFrame,
    end_date: date,
    rng: np.random.Generator,
) -> np.ndarray:
    """각 유저에 대해 생성할 주문 수를 결정.

    유저의 세그먼트(월 기대 주문 수) × 가입 기간(월) 을 람다로
    Poisson 샘플링.

    Returns:
        n_orders_per_user: shape (n_users,), 각 유저의 주문 수
    """
    # 가입일을 numpy datetime으로 변환 후 활동 기간(월) 계산
    created_at = users_df["created_at"].to_numpy()
    end_ts = np.datetime64(end_date)
    
    # 활동 기간을 일 단위로 → 월 단위 (30일 기준)
    days_active = (end_ts - created_at.astype("datetime64[D]")).astype(int)
    months_active = np.maximum(days_active / 30.0, 0.0)  # 음수 방지
    
    # 세그먼트별 월 기대 주문 수를 numpy 배열로 매핑
    segments = users_df["segment"].to_numpy()
    monthly_rates = np.array([MONTHLY_ORDER_RATE[s] for s in segments])
    
    # 람다 = 월 평균 × 활동 개월 수
    lambdas = monthly_rates * months_active
    
    # Poisson 샘플링
    n_orders = rng.poisson(lambdas)
    
    return n_orders

def _black_friday_date(year: int) -> date:
    """해당 연도의 블랙프라이데이 (11월 4번째 목요일 다음 금요일)."""
    # 11/1부터 시작
    nov_1 = date(year, 11, 1)
    # 11/1의 요일 (0=월, 3=목)
    first_thu_offset = (3 - nov_1.weekday()) % 7
    fourth_thursday = date(year, 11, 1 + first_thu_offset + 21)
    return date(year, 11, fourth_thursday.day + 1)  # 금요일


def _generate_timestamps(
    n_orders_per_user: np.ndarray,
    user_ids: np.ndarray,
    created_at: np.ndarray,
    end_date: date,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """각 주문의 user_id와 timestamp를 생성.

    Returns:
        order_user_ids: shape (n_total_orders,)
        order_timestamps: shape (n_total_orders,), datetime64[s]
    """
    # 단계 1: 유저별 정보를 주문 단위로 펼치기
    order_user_ids = np.repeat(user_ids, n_orders_per_user)
    order_starts = np.repeat(
        created_at.astype("datetime64[s]"),
        n_orders_per_user,
    )

    end_ts = (
        np.datetime64(end_date).astype("datetime64[s]")
        + np.timedelta64(86399, "s")  # end_date의 23:59:59
    )
    n_total = len(order_user_ids)

    # 단계 2: 유저 활동 기간 내 uniform 샘플링
    range_seconds = (end_ts - order_starts).astype("int64")
    offsets_seconds = (rng.uniform(0, 1, size=n_total) * range_seconds).astype("int64")
    order_timestamps = order_starts + offsets_seconds.astype("timedelta64[s]")

    # 단계 3: hour를 가중치로 재샘플 (날짜는 유지)
    order_dates = order_timestamps.astype("datetime64[D]").astype("datetime64[s]")
    weighted_hours = rng.choice(24, size=n_total, p=HOUR_WEIGHTS)
    random_minutes_seconds = rng.integers(0, 3600, size=n_total)
    seconds_in_day = weighted_hours * 3600 + random_minutes_seconds
    order_timestamps = order_dates + seconds_in_day.astype("timedelta64[s]")

    # 단계 4: 계절성 - 블랙프라이데이 기간 over-sample
    # 전체의 5%를 블프 주간으로 강제 이동
    bf_fraction = 0.05
    n_bf_orders = int(n_total * bf_fraction)
    bf_indices = rng.choice(n_total, size=n_bf_orders, replace=False)

    # 기간에 걸친 연도들
    start_year = int(order_starts.min().astype("datetime64[Y]").astype(int) + 1970)
    end_year = end_date.year
    bf_dates_pool: list[date] = []
    for y in range(start_year, end_year + 1):
        bf = _black_friday_date(y)
        # 블프 ~ 사이버먼데이 (월요일까지 4일)
        for d_offset in range(4):
            cand = bf.replace(day=bf.day + d_offset) if bf.day + d_offset <= 30 else None
            if cand is not None:
                bf_dates_pool.append(cand)

    bf_dates_np = [d for d in bf_dates_pool if d <= end_date]
    bf_dates_np = np.array(bf_dates_pool, dtype="datetime64[s]")

    # 선택된 인덱스의 날짜를 블프 풀에서 랜덤하게 교체 (hour는 유지)
    chosen_bf_dates = rng.choice(bf_dates_np, size=n_bf_orders)
    current_seconds_in_day = (
        order_timestamps[bf_indices] - order_timestamps[bf_indices].astype("datetime64[D]").astype("datetime64[s]")
    )
    order_timestamps[bf_indices] = chosen_bf_dates + current_seconds_in_day

    # 12월 추가 부스트 (선택된 5%는 그대로, 추가로 3%를 12월로)
    dec_fraction = 0.03
    n_dec_orders = int(n_total * dec_fraction)
    available_idx = np.setdiff1d(np.arange(n_total), bf_indices)
    dec_indices = rng.choice(available_idx, size=n_dec_orders, replace=False)

    # 12월 날짜 풀 (각 연도의 12월 1~31일)
    dec_dates_pool: list[date] = []
    for y in range(start_year, end_year + 1):
        for d in range(1, 32):
            try:
                dec_dates_pool.append(date(y, 12, d))
            except ValueError:
                pass

    dec_dates_pool = [d for d in dec_dates_pool if d <= end_date]
    dec_dates_np = np.array(dec_dates_pool, dtype="datetime64[s]")
    chosen_dec_dates = rng.choice(dec_dates_np, size=n_dec_orders)
    current_seconds_in_day = (
        order_timestamps[dec_indices]
        - order_timestamps[dec_indices].astype("datetime64[D]").astype("datetime64[s]")
    )
    order_timestamps[dec_indices] = chosen_dec_dates + current_seconds_in_day

    # end_date 넘어가는 timestamp는 clipping (블프 풀에 미래 날짜가 섞일 수 있어서)
    order_timestamps = np.minimum(order_timestamps, end_ts)
    # user 가입일보다 이전인 timestamp도 clipping (드물지만 시즌 이동 후 발생 가능)
    order_timestamps = np.maximum(order_timestamps, order_starts)

    return order_user_ids, order_timestamps


def _assign_status(
    order_timestamps: np.ndarray,
    end_date: date,
    rng: np.random.Generator,
) -> np.ndarray:
    """주문 timestamp 기반으로 status 결정.

    오래된 주문 → delivered/cancelled/refunded
    최근 주문 → pending/processing/shipped 가능
    """
    end_ts = np.datetime64(end_date).astype("datetime64[s]")
    age_seconds = (end_ts - order_timestamps).astype("int64")
    age_hours = age_seconds / 3600.0

    n = len(order_timestamps)
    statuses = np.empty(n, dtype=object)

    # 시간 버킷별 마스크
    mask_very_recent = age_hours < 1            # 1시간 미만
    mask_recent = (age_hours >= 1) & (age_hours < 24)   # 1시간 ~ 1일
    mask_short = (age_hours >= 24) & (age_hours < 72)   # 1~3일
    mask_medium = (age_hours >= 72) & (age_hours < 168) # 3~7일
    mask_old = age_hours >= 168                  # 7일 이상

    # 각 버킷에서 가중 샘플링
    statuses[mask_very_recent] = "pending"

    statuses[mask_recent] = rng.choice(
        ["pending", "processing"],
        size=int(mask_recent.sum()),
        p=[0.4, 0.6],
    )

    statuses[mask_short] = rng.choice(
        ["processing", "shipped", "cancelled"],
        size=int(mask_short.sum()),
        p=[0.4, 0.5, 0.1],
    )

    statuses[mask_medium] = rng.choice(
        ["shipped", "delivered", "cancelled"],
        size=int(mask_medium.sum()),
        p=[0.3, 0.6, 0.1],
    )

    statuses[mask_old] = rng.choice(
        ["delivered", "cancelled", "refunded"],
        size=int(mask_old.sum()),
        p=[0.85, 0.10, 0.05],
    )

    return statuses


def _assign_currencies(
    order_user_ids: np.ndarray,
    users_df: pl.DataFrame,
) -> np.ndarray:
    """유저의 country를 기반으로 통화 결정."""
    # user_id → country lookup
    user_country = dict(
        zip(
            users_df["user_id"].to_list(),
            users_df["country"].to_list(),
        )
    )
    # 주문별 country
    countries = np.array([user_country[uid] for uid in order_user_ids])
    # country → currency
    currencies = np.array([COUNTRY_TO_CURRENCY.get(c, "USD") for c in countries])
    return currencies


def generate(
    users_df: pl.DataFrame,
    end_date: date,
    seed: int = 42,
) -> pl.DataFrame:
    """주문 팩트 테이블 생성.

    Args:
        users_df: 유저 차원 테이블 (segment, created_at, country 필요)
        end_date: 분석 기간 종료일
        seed: 재현성

    Returns:
        orders DataFrame (총 주문 수는 유저 분포에 따라 결정)
    """
    rng = make_rng(seed)

    # 7a: 유저별 주문 수
    n_orders_per_user = _compute_orders_per_user(users_df, end_date, rng)

    user_ids = users_df["user_id"].to_numpy()
    created_at = users_df["created_at"].to_numpy()

    # 7b: timestamp 생성
    order_user_ids, order_timestamps = _generate_timestamps(
        n_orders_per_user, user_ids, created_at, end_date, rng,
    )

    n_total = len(order_user_ids)
    print(f"    Total orders: {n_total:,}")

    # 7c: 나머지 속성
    order_ids = np.arange(1, n_total + 1, dtype=np.int64)

    statuses = _assign_status(order_timestamps, end_date, rng)

    payment_methods = weighted_choice(
        rng,
        [p for p, _ in PAYMENT_METHODS_WEIGHTED],
        [w for _, w in PAYMENT_METHODS_WEIGHTED],
        n_total,
    )

    currencies = _assign_currencies(order_user_ids, users_df)

    # total_amount: 일단 placeholder (order_items 만든 후 업데이트 예정)
    # 대략 $20 ~ $500 사이 log-uniform
    total_amounts = np.round(
        np.exp(rng.uniform(np.log(20), np.log(500), size=n_total)),
        2,
    )

    # 파티션 키 (dt = created_at의 날짜)
    order_dts = order_timestamps.astype("datetime64[D]")

    df = pl.DataFrame({
        "order_id": order_ids,
        "user_id": order_user_ids,
        "created_at": order_timestamps.astype("datetime64[us]"),  # ← [s] → [us]
        "status": statuses.tolist(),
        "payment_method": payment_methods.tolist(),
        "currency": currencies.tolist(),
        "total_amount": total_amounts,
        "dt": order_dts,
    })

    # 정렬: created_at 오름차순 (실전 데이터처럼 시간순으로)
    df = df.sort("created_at")

    return df