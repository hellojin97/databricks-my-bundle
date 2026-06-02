"""클릭스트림(events) 팩트 테이블 생성기.

orders/order_items(구매 퍼널) + users/products(브라우징 세션)에 의존.

이 테이블은 "왜·어떻게 구매에 이르렀나"(행동·퍼널·전환)를 분석 가능하게 한다.
두 종류의 세션을 만든다:

- 구매 세션(purchase): 주문 1건당 5-이벤트 퍼널 1개.
  page_view → product_view → add_to_cart → begin_checkout → purchase
  purchase 이벤트는 실제 order_id 를 참조하고, event_ts == order.created_at.
  앞 4단계는 그 직전 시각으로 백데이팅(상품/가입일 floor로 클립).
  product_view / add_to_cart 는 주문의 '대표 상품'(첫 라인 상품) 1개를 참조.

- 브라우징 세션(browse): 구매로 이어지지 않는 탐색 세션(view/search/cart 후 이탈).
  유저당 평균 browse_sessions_per_user(Poisson, 세그먼트 가중) 만큼 생성.
  전환율 = 구매세션 / 전체세션 분석을 가능하게 한다.

재현성 계약(중요): 같은 seed → 동일 출력. 이를 위해 RNG draw 순서를 고정한다.
  ① 구매 백데이팅 gap → ② 브라우즈 세션 수 → ③ 세션 시작 시각 →
  ④ 세션당 이벤트 수 → ⑤ 이벤트 간 gap → ⑥ 브라우즈 event_type →
  ⑦ 브라우즈 상품 선택(가중·prefix·단종 repair) → ⑧ search_query(+NULL 주입)
이 순서를 바꾸면 .equals() 재현성이 깨진다.
"""
from datetime import date

import numpy as np
import polars as pl

from .base import inject_nulls, make_rng, weighted_choice
from .generate_order_items import _FAR_FUTURE, _weighted_prefix_choice
from .generate_orders import HOUR_WEIGHTS

# 구매 세션 퍼널 (고정 5단계)
PURCHASE_FUNNEL = ["page_view", "product_view", "add_to_cart", "begin_checkout", "purchase"]
# product_id 가 설정되는 퍼널 단계 인덱스 (product_view, add_to_cart, purchase)
_PRODUCT_STEPS = (1, 2, 4)
_PURCHASE_STEP = 4

# 브라우징 세션 이벤트 타입 (purchase 제외 — 구매로 이어지지 않음)
BROWSE_EVENT_TYPES = ["page_view", "product_view", "search", "add_to_cart", "begin_checkout"]
# pos>=1 에서의 가중치 (탐색 위주, 깊은 퍼널은 드물게)
BROWSE_EVENT_WEIGHTS = [0.30, 0.30, 0.20, 0.13, 0.07]

# 세그먼트별 브라우징 활성도 (vip가 더 활발, churned는 거의 없음)
SEGMENT_BROWSE_MULT = {"new": 0.8, "regular": 1.0, "vip": 2.5, "churned": 0.2}

# 구매 퍼널 단계 간 백데이팅 간격(초): 30초 ~ 30분
_MIN_GAP_S, _MAX_GAP_S = 30, 1800
# 브라우징 세션 내 이벤트 간격(초): 10초 ~ 5분
_MIN_STEP_S, _MAX_STEP_S = 10, 300

# search 이벤트용 정적 쿼리 풀 (카테고리/상품군 기반, 현실감용)
SEARCH_QUERIES = [
    "laptop", "wireless earbuds", "gaming chair", "4k tv", "smartphone",
    "running shoes", "winter jacket", "office desk", "coffee maker", "air fryer",
    "yoga mat", "backpack", "sunglasses", "bluetooth speaker", "monitor",
    "mechanical keyboard", "desk lamp", "water bottle", "headphones", "sneakers",
    "kitchen knife set", "bed sheets", "skincare set", "board game", "dumbbells",
    "tablet", "phone case", "camera", "smartwatch", "vacuum cleaner",
]


def _within_group_index(counts: np.ndarray) -> np.ndarray:
    """그룹별 0-기반 인덱스를 평탄화해 반환.

    예) counts=[2,3] → [0,1, 0,1,2]. np.repeat 로 펼친 행에 세션 내 위치를 부여할 때 사용.
    """
    counts = np.asarray(counts)
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    starts = np.repeat(np.cumsum(counts) - counts, counts)
    return np.arange(total, dtype=np.int64) - starts


def _segmented_cumsum(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """그룹(세션) 내부에서만 누적합. 세션별 단조 증가 timestamp 오프셋 산출용."""
    total = len(values)
    if total == 0:
        return np.zeros(0, dtype="int64")
    global_cum = np.cumsum(values)
    starts = np.cumsum(counts) - counts  # 각 그룹의 시작 인덱스
    prefix = np.zeros(len(counts), dtype=global_cum.dtype)
    nonzero = starts > 0
    prefix[nonzero] = global_cum[starts[nonzero] - 1]  # 그룹 직전까지의 누적
    base = np.repeat(prefix, counts)
    return global_cum - base


def _empty_block() -> dict[str, np.ndarray]:
    """이벤트 없는(빈) 블록. concat 시 dtype 호환을 위해 빈 배열을 dtype별로 준비."""
    return {
        "user_id": np.zeros(0, dtype=np.int64),
        "session_id": np.zeros(0, dtype=object),
        "event_type": np.zeros(0, dtype=object),
        "product_id": np.zeros(0, dtype=object),
        "search_query": np.zeros(0, dtype=object),
        "order_id": np.zeros(0, dtype=object),
        "event_ts": np.zeros(0, dtype="datetime64[us]"),
    }


def _representative_products(
    orders_df: pl.DataFrame, order_items_df: pl.DataFrame
) -> np.ndarray:
    """주문별 '대표 상품'(가장 작은 order_item_id의 product_id)을 orders 순서로 반환.

    모든 주문은 라인이 최소 1개 보장되므로 NULL이 없다.
    """
    rep = (
        order_items_df.sort(["order_id", "order_item_id"])
        .group_by("order_id", maintain_order=False)
        .agg(pl.col("product_id").first())
    )
    joined = orders_df.select("order_id").join(rep, on="order_id", how="left")
    return joined["product_id"].to_numpy()


def _build_purchase_events(
    orders_df: pl.DataFrame,
    users_df: pl.DataFrame,
    rep_product_ids: np.ndarray,
    rep_product_created: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """주문 1건당 5-이벤트 구매 퍼널을 생성.

    purchase 는 order.created_at/order_id/대표상품을 그대로 사용하고,
    앞 4단계는 누적 gap 만큼 백데이팅한 뒤 floor=max(가입일, 대표상품 생성일)로 클립한다.
    floor 로 클립하면 단조성(비감소)이 유지되면서 product_view/add_to_cart 의
    시간정합(상품 생성일 ≤ 이벤트 시각)이 보장된다.
    """
    n_orders = len(orders_df)
    if n_orders == 0:
        return _empty_block()

    order_ids = orders_df["order_id"].to_numpy()
    order_user_ids = orders_df["user_id"].to_numpy()
    purchase_ts = orders_df["created_at"].to_numpy().astype("datetime64[us]")

    # 주문별 유저 가입일
    signup = (
        orders_df.select("user_id")
        .join(
            users_df.select(["user_id", "created_at"]).rename({"created_at": "signup"}),
            on="user_id",
            how="left",
        )["signup"]
        .to_numpy()
        .astype("datetime64[us]")
    )

    n_steps = len(PURCHASE_FUNNEL)

    # 앞 4단계 백데이팅: 단계별 gap(초) → 구매로부터의 역방향 누적 오프셋
    gaps = rng.integers(_MIN_GAP_S, _MAX_GAP_S, size=(n_orders, n_steps - 1))
    rev_cum = np.cumsum(gaps[:, ::-1], axis=1)[:, ::-1]  # [:,0]=총합, [:,3]=마지막 gap
    offsets = np.zeros((n_orders, n_steps), dtype="int64")
    offsets[:, : n_steps - 1] = rev_cum

    ts = purchase_ts[:, None] - offsets.astype("timedelta64[s]")
    floor = np.maximum(signup, rep_product_created)  # 가입일·상품생성일 중 늦은 것
    ts = np.maximum(ts, floor[:, None])
    ts_flat = ts.reshape(-1)  # row-major: [o0s0..o0s4, o1s0..]

    pos = np.tile(np.arange(n_steps), n_orders)
    event_type = np.tile(np.array(PURCHASE_FUNNEL, dtype=object), n_orders)
    user_flat = np.repeat(order_user_ids, n_steps)
    order_rep = np.repeat(order_ids, n_steps)
    rep_prod_rep = np.repeat(rep_product_ids, n_steps)

    # product_id: product_view/add_to_cart/purchase 단계에만
    product_id = np.full(len(ts_flat), None, dtype=object)
    prod_mask = np.isin(pos, _PRODUCT_STEPS)
    product_id[prod_mask] = rep_prod_rep[prod_mask]

    # order_id: purchase 단계에만
    order_id = np.full(len(ts_flat), None, dtype=object)
    purch_mask = pos == _PURCHASE_STEP
    order_id[purch_mask] = order_rep[purch_mask]

    search_query = np.full(len(ts_flat), None, dtype=object)
    session_id = np.char.add(
        np.char.add(user_flat.astype(str), "-p"), order_rep.astype(str)
    ).astype(object)

    return {
        "user_id": user_flat,
        "session_id": session_id,
        "event_type": event_type,
        "product_id": product_id,
        "search_query": search_query,
        "order_id": order_id,
        "event_ts": ts_flat,
    }


def _browse_sessions_per_user(
    users_df: pl.DataFrame, rng: np.random.Generator, browse_sessions_per_user: float
) -> np.ndarray:
    """세그먼트 가중 Poisson 으로 유저별 브라우징 세션 수 결정."""
    segs = users_df["segment"].to_numpy()
    mult = np.array([SEGMENT_BROWSE_MULT[s] for s in segs])
    lam = browse_sessions_per_user * mult
    return rng.poisson(lam)


def _build_browse_events(
    users_df: pl.DataFrame,
    n_sessions_per_user: np.ndarray,
    end_date: date,
    max_events_per_session: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """브라우징 세션을 유저→세션→이벤트로 2단 확장. (상품/검색어는 이후 단계에서 채움)"""
    user_ids = users_df["user_id"].to_numpy()
    signup = users_df["created_at"].to_numpy().astype("datetime64[us]")

    total_sessions = int(n_sessions_per_user.sum())
    if total_sessions == 0:
        return _empty_block()

    # 세션 단위 확장
    sess_user = np.repeat(user_ids, n_sessions_per_user)
    sess_signup = np.repeat(signup, n_sessions_per_user)
    sess_idx = _within_group_index(n_sessions_per_user)

    end_ts = (
        np.datetime64(end_date).astype("datetime64[s]") + np.timedelta64(86399, "s")
    ).astype("datetime64[us]")

    # 세션 시작 시각: [가입일, end_ts] uniform → HOUR_WEIGHTS 로 hour 재샘플
    range_us = np.maximum((end_ts - sess_signup).astype("int64"), 0)
    start_offset = (rng.uniform(0, 1, size=total_sessions) * range_us).astype("int64")
    sess_start = sess_signup + start_offset.astype("timedelta64[us]")

    sess_day = sess_start.astype("datetime64[D]").astype("datetime64[us]")
    weighted_hours = rng.choice(24, size=total_sessions, p=HOUR_WEIGHTS)
    minute_second = rng.integers(0, 3600, size=total_sessions)
    seconds_in_day = weighted_hours.astype("int64") * 3600 + minute_second
    sess_start = sess_day + seconds_in_day.astype("timedelta64[s]")
    # hour 재샘플로 가입일 이전/ end_date 초과 가능 → 재클립
    sess_start = np.minimum(np.maximum(sess_start, sess_signup), end_ts)

    # 세션당 이벤트 수: 1 + Poisson(2), [1, max] 로 clip (짧은 세션 위주, 상한 보장)
    n_events = np.clip(1 + rng.poisson(2, size=total_sessions), 1, max_events_per_session)
    total_events = int(n_events.sum())

    # 이벤트 단위 확장
    ev_user = np.repeat(sess_user, n_events)
    ev_sess_idx = np.repeat(sess_idx, n_events)
    ev_sess_start = np.repeat(sess_start, n_events)
    ev_pos = _within_group_index(n_events)

    # 세션 내 단조 증가 timestamp (pos==0 은 세션 시작, 이후 gap 누적)
    step_gaps = rng.integers(_MIN_STEP_S, _MAX_STEP_S, size=total_events)
    step_gaps[ev_pos == 0] = 0
    within = _segmented_cumsum(step_gaps, n_events)
    ev_ts = ev_sess_start + within.astype("timedelta64[s]")
    ev_ts = np.minimum(np.maximum(ev_ts, ev_sess_start), end_ts)

    # event_type: 첫 이벤트는 page_view, 이후는 가중 선택(purchase 제외 → 구매 발생 불가)
    event_type = weighted_choice(
        rng, BROWSE_EVENT_TYPES, BROWSE_EVENT_WEIGHTS, total_events
    ).astype(object)
    event_type[ev_pos == 0] = "page_view"

    session_id = np.char.add(
        np.char.add(ev_user.astype(str), "-b"), ev_sess_idx.astype(str)
    ).astype(object)

    return {
        "user_id": ev_user,
        "session_id": session_id,
        "event_type": event_type,
        "product_id": np.full(total_events, None, dtype=object),
        "search_query": np.full(total_events, None, dtype=object),
        "order_id": np.full(total_events, None, dtype=object),  # 브라우즈는 항상 None
        "event_ts": ev_ts.astype("datetime64[us]"),
    }


def _assign_browse_products(
    block: dict[str, np.ndarray], products_df: pl.DataFrame, rng: np.random.Generator
) -> None:
    """브라우즈의 product_view/add_to_cart 행에 시간정합 상품을 부여(블록을 in-place 수정).

    generate_order_items 의 prefix-choice + 단종 repair 패턴을 재사용한다.
    이벤트 시각에 (아직) 적합한 상품이 없으면(생성 전 / 단종됨) product_id=None +
    event_type을 page_view 로 강등한다.
    """
    event_type = block["event_type"]
    need_mask = np.isin(event_type, ["product_view", "add_to_cart"])
    if not need_mask.any():
        return

    line_created = block["event_ts"][need_mask].astype("datetime64[us]")

    product_ids = products_df["product_id"].to_numpy()
    # us 고정 — ns는 _FAR_FUTURE(2999) 오버플로우(~2262 한계). order_items 와 동일.
    product_created = products_df["created_at"].to_numpy().astype("datetime64[us]")
    product_disc = (
        products_df["discontinued_at"].fill_null(_FAR_FUTURE).to_numpy().astype("datetime64[us]")
    )
    is_active = products_df["discontinued_at"].is_null().to_numpy()
    weights = rng.pareto(1.5, size=len(product_ids)) + 0.5

    sort_idx = np.argsort(product_created, kind="stable")
    sorted_created = product_created[sort_idx]
    cum_weights = np.cumsum(weights[sort_idx])
    chosen_idx = sort_idx[
        _weighted_prefix_choice(sorted_created, cum_weights, line_created, rng)
    ]

    # 단종 정합성 repair (이미 단종된 상품을 고른 라인은 미단종 상품으로 재선택)
    bad = product_disc[chosen_idx] < line_created
    if bad.any():
        a_orig = np.nonzero(is_active)[0]
        if len(a_orig) > 0:
            a_sort = np.argsort(product_created[a_orig], kind="stable")
            a_sorted_created = product_created[a_orig][a_sort]
            a_cum = np.cumsum(weights[a_orig][a_sort])
            bad_created = line_created[bad]
            repairable = np.searchsorted(a_sorted_created, bad_created, side="right") >= 1
            bad_lines = np.nonzero(bad)[0][repairable]
            if len(bad_lines) > 0:
                repick = _weighted_prefix_choice(
                    a_sorted_created, a_cum, line_created[bad_lines], rng
                )
                chosen_idx[bad_lines] = a_orig[a_sort[repick]]

    # 적합 상품이 없는 행은 강등: ① 생성 전(k==0) ② repair 불가 단종행
    k_raw = np.searchsorted(sorted_created, line_created, side="right")
    drop = (k_raw == 0) | (product_disc[chosen_idx] < line_created)

    assigned = product_ids[chosen_idx].astype(object)
    assigned[drop] = None

    need_idx = np.nonzero(need_mask)[0]
    block["product_id"][need_idx] = assigned
    block["event_type"][need_idx[drop]] = "page_view"


def _assign_browse_search(
    block: dict[str, np.ndarray], null_rate_search: float, rng: np.random.Generator
) -> None:
    """search 이벤트에 쿼리 부여 + 일부 NULL(dirty data). 블록을 in-place 수정."""
    search_mask = block["event_type"] == "search"
    n_search = int(search_mask.sum())
    if n_search == 0:
        return
    queries = rng.choice(np.array(SEARCH_QUERIES, dtype=object), size=n_search)
    queries = inject_nulls(queries, null_rate=null_rate_search, rng=rng)
    block["search_query"][np.nonzero(search_mask)[0]] = queries


def generate(
    users_df: pl.DataFrame,
    products_df: pl.DataFrame,
    orders_df: pl.DataFrame,
    order_items_df: pl.DataFrame,
    end_date: date,
    seed: int = 42,
    browse_sessions_per_user: float = 3,
    max_events_per_session: int = 8,
    null_rate_search: float = 0.10,
) -> pl.DataFrame:
    """클릭스트림 events 팩트 테이블 생성.

    Args:
        users_df: 유저 차원 (user_id, segment, created_at 필요)
        products_df: 상품 차원 (product_id, created_at, discontinued_at 필요)
        orders_df: 주문 팩트 (order_id, user_id, created_at 필요)
        order_items_df: 주문 라인 (order_id, order_item_id, product_id 필요)
        end_date: 분석 기간 종료일
        seed: 재현성
        browse_sessions_per_user: 유저당 평균 브라우징 세션 수 (Poisson, 세그먼트 가중)
        max_events_per_session: 브라우징 세션당 이벤트 상한
        null_rate_search: search_query NULL 비율 (dirty data)

    Returns:
        events DataFrame (event_ts 오름차순 정렬, dt 파티션용 컬럼 포함)
    """
    rng = make_rng(seed)

    # --- 구매 세션 (주문당 퍼널 1개) ---
    rep_product_ids = _representative_products(orders_df, order_items_df)
    rep_product_created = (
        pl.DataFrame({"product_id": rep_product_ids})
        .join(products_df.select(["product_id", "created_at"]), on="product_id", how="left")[
            "created_at"
        ]
        .to_numpy()
        .astype("datetime64[us]")
    )
    purchase = _build_purchase_events(
        orders_df, users_df, rep_product_ids, rep_product_created, rng
    )

    # --- 브라우징 세션 ---
    n_sessions = _browse_sessions_per_user(users_df, rng, browse_sessions_per_user)
    browse = _build_browse_events(
        users_df, n_sessions, end_date, max_events_per_session, rng
    )
    if len(browse["event_ts"]) > 0:
        _assign_browse_products(browse, products_df, rng)
        _assign_browse_search(browse, null_rate_search, rng)

    n_purchase = len(purchase["event_ts"])
    n_browse = len(browse["event_ts"])
    print(
        f"    Total events: {n_purchase + n_browse:,} "
        f"(purchase {n_purchase:,} / browse {n_browse:,})"
    )
    print(
        f"    Sessions: purchase {len(orders_df):,} / browse {int(n_sessions.sum()):,}"
    )

    # --- 합치기 (구매 먼저, 그다음 브라우즈 — 재현성을 위해 순서 고정) ---
    cols = [
        "user_id", "session_id", "event_type", "product_id",
        "search_query", "order_id", "event_ts",
    ]
    merged = {c: np.concatenate([purchase[c], browse[c]]) for c in cols}

    n_total = len(merged["event_ts"])
    event_id = np.arange(1, n_total + 1, dtype=np.int64)
    event_ts = merged["event_ts"].astype("datetime64[us]")
    dt = event_ts.astype("datetime64[D]")

    df = pl.DataFrame(
        {
            "event_id": event_id,
            "user_id": merged["user_id"].astype(np.int64),
            "session_id": merged["session_id"].tolist(),
            "event_type": merged["event_type"].tolist(),
            "product_id": merged["product_id"].tolist(),
            "search_query": merged["search_query"].tolist(),
            "order_id": merged["order_id"].tolist(),
            "event_ts": event_ts,
            "dt": dt,
        },
        # object+None 배열이 Float64 로 추론되는 것 방지 (nullable Int64/Utf8 고정)
        schema_overrides={
            "product_id": pl.Int64,
            "order_id": pl.Int64,
            "search_query": pl.Utf8,
            "session_id": pl.Utf8,
            "event_type": pl.Utf8,
        },
    )

    return df.sort("event_ts")
