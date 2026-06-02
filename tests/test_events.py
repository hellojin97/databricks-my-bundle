"""클릭스트림 events 팩트 테이블 검증."""
from datetime import date

import polars as pl

from ecommerce_generator import (
    generate_categories,
    generate_events,
    generate_order_items,
    generate_orders,
    generate_products,
    generate_users,
)

EXPECTED_COLUMNS = {
    "event_id", "user_id", "session_id", "event_type",
    "product_id", "search_query", "order_id", "event_ts", "dt",
}
VALID_EVENT_TYPES = {
    "page_view", "product_view", "search", "add_to_cart", "begin_checkout", "purchase",
}
PRODUCT_EVENT_TYPES = {"product_view", "add_to_cart", "purchase"}


# --- 스키마 ---

def test_schema(events):
    assert set(events.columns) == EXPECTED_COLUMNS


def test_schema_dtypes(events):
    """nullable Int64 가 Float64 로 추론되지 않도록 고정 + 파티션/시각 타입."""
    schema = events.schema
    assert schema["event_id"] == pl.Int64
    assert schema["user_id"] == pl.Int64
    assert schema["product_id"] == pl.Int64
    assert schema["order_id"] == pl.Int64
    assert schema["search_query"] == pl.Utf8
    assert schema["session_id"] == pl.Utf8
    assert schema["event_type"] == pl.Utf8
    assert schema["event_ts"] == pl.Datetime("us")
    assert schema["dt"] == pl.Date


def test_event_id_unique(events):
    assert events["event_id"].n_unique() == len(events)


# --- FK 유효성 ---

def test_user_fk_valid(events, users):
    user_ids = set(users["user_id"].to_list())
    assert set(events["user_id"].unique().to_list()) <= user_ids


def test_product_fk_valid(events, products):
    """non-null product_id 는 실제 상품이어야 한다."""
    product_ids = set(products["product_id"].to_list())
    seen = events.filter(pl.col("product_id").is_not_null())["product_id"]
    assert set(seen.unique().to_list()) <= product_ids


def test_order_fk_valid(events, order_items_and_orders):
    """non-null order_id 는 실제 주문이어야 한다."""
    _, updated_orders = order_items_and_orders
    order_ids = set(updated_orders["order_id"].to_list())
    seen = events.filter(pl.col("order_id").is_not_null())["order_id"]
    assert set(seen.unique().to_list()) <= order_ids


# --- 컬럼 의미 규칙 ---

def test_event_type_valid(events):
    assert set(events["event_type"].unique().to_list()) <= VALID_EVENT_TYPES


def test_product_id_set_only_on_product_events(events):
    """product_id 가 non-null 인 행은 product_view/add_to_cart/purchase 뿐이어야 한다."""
    with_product = events.filter(pl.col("product_id").is_not_null())
    assert set(with_product["event_type"].unique().to_list()) <= PRODUCT_EVENT_TYPES


def test_search_query_only_on_search(events):
    """search_query 가 non-null 인 행은 search 이벤트뿐이어야 한다."""
    with_query = events.filter(pl.col("search_query").is_not_null())
    assert set(with_query["event_type"].unique().to_list()) <= {"search"}


def test_order_id_only_on_purchase(events):
    """order_id 가 non-null 인 행은 purchase 이벤트뿐이어야 한다."""
    with_order = events.filter(pl.col("order_id").is_not_null())
    assert set(with_order["event_type"].unique().to_list()) <= {"purchase"}


def test_every_purchase_has_order_id(events):
    """모든 purchase 이벤트는 order_id 를 가져야 한다 (브라우즈엔 purchase 없음)."""
    purchases = events.filter(pl.col("event_type") == "purchase")
    assert purchases["order_id"].null_count() == 0


def test_dt_matches_event_ts_date(events):
    """파티션 키 dt 는 event_ts 의 날짜와 일치해야 한다."""
    mismatch = events.filter(events["event_ts"].dt.date() != events["dt"])
    assert len(mismatch) == 0


# --- 시간적 정합성 (FK) ---

def test_purchase_ts_equals_order_created_at(events, order_items_and_orders):
    """purchase 이벤트 시각은 해당 주문의 created_at 과 정확히 일치해야 한다."""
    _, updated_orders = order_items_and_orders
    purchases = events.filter(pl.col("event_type") == "purchase").select(
        ["order_id", "event_ts"]
    )
    joined = purchases.join(
        updated_orders.select(["order_id", "created_at"]), on="order_id"
    )
    assert (joined["event_ts"] == joined["created_at"]).all()


def test_event_after_user_signup(events, users):
    """모든 이벤트는 유저 가입일 이후여야 한다 (시간적 FK)."""
    joined = events.join(
        users.select(["user_id", "created_at"]).rename({"created_at": "signup_at"}),
        on="user_id",
    )
    assert (joined["event_ts"] >= joined["signup_at"]).all()


def test_referenced_product_created_before_event(events, products):
    """참조 상품은 이벤트 시각에 이미 생성돼 있어야 한다 (created_at ≤ event_ts)."""
    joined = (
        events.filter(pl.col("product_id").is_not_null())
        .select(["product_id", "event_ts"])
        .join(
            products.select(["product_id", "created_at"]).rename(
                {"created_at": "product_created_at"}
            ),
            on="product_id",
        )
    )
    assert (joined["product_created_at"] <= joined["event_ts"]).all()


def test_referenced_product_not_discontinued_before_event(events, products):
    """참조 상품은 이벤트 시각에 아직 단종되지 않았어야 한다 (event_ts ≤ discontinued_at)."""
    joined = (
        events.filter(pl.col("product_id").is_not_null())
        .select(["product_id", "event_ts"])
        .join(products.select(["product_id", "discontinued_at"]), on="product_id")
    )
    discontinued = joined.filter(pl.col("discontinued_at").is_not_null())
    assert (discontinued["event_ts"] <= discontinued["discontinued_at"]).all()


# --- 퍼널 / 세션 구조 ---

def test_purchase_session_has_five_events(events):
    """구매 세션(-p)은 정확히 5 이벤트, purchase 1개여야 한다."""
    purchase_sessions = events.filter(pl.col("session_id").str.contains("-p"))
    per_session = purchase_sessions.group_by("session_id").agg(
        pl.len().alias("n"),
        (pl.col("event_type") == "purchase").sum().alias("n_purchase"),
    )
    assert (per_session["n"] == 5).all()
    assert (per_session["n_purchase"] == 1).all()


def test_no_purchase_in_browse_sessions(events):
    """브라우징 세션(-b)에는 purchase 이벤트가 없어야 한다."""
    browse = events.filter(pl.col("session_id").str.contains("-b"))
    assert (browse["event_type"] == "purchase").sum() == 0


def test_purchase_is_last_and_preceded_by_checkout(events):
    """구매 세션 내에서 purchase 가 마지막이고 직전이 begin_checkout 이어야 한다."""
    sample_id = (
        events.filter(pl.col("event_type") == "purchase")["session_id"][0]
    )
    session = events.filter(pl.col("session_id") == sample_id).sort("event_ts")
    types = session["event_type"].to_list()
    assert types[-1] == "purchase"
    assert types[-2] == "begin_checkout"


def test_conversion_rate_sane(events, order_items_and_orders):
    """전환율 = 구매세션 / 전체세션 이 타당한 범위(0~1)여야 한다."""
    _, updated_orders = order_items_and_orders
    n_purchase_sessions = len(updated_orders)
    n_browse_sessions = (
        events.filter(pl.col("session_id").str.contains("-b"))["session_id"].n_unique()
    )
    total = n_purchase_sessions + n_browse_sessions
    rate = n_purchase_sessions / total
    assert n_browse_sessions > 0
    assert 0.0 < rate < 1.0


# --- 재현성 ---

def _make_events(seed: int):
    """작은 규모로 users→products→orders→order_items→events 체인을 생성."""
    start, end = date(2025, 1, 1), date(2025, 12, 31)
    cats = generate_categories.generate(seed=seed)
    users = generate_users.generate(
        n_users=800, start_date=start, end_date=end, seed=seed
    )
    products = generate_products.generate(
        n_products=200, categories_df=cats, start_date=start, end_date=end, seed=seed
    )
    orders = generate_orders.generate(users_df=users, end_date=end, seed=seed)
    items, updated_orders = generate_order_items.generate(
        orders_df=orders, products_df=products, seed=seed
    )
    return generate_events.generate(
        users_df=users, products_df=products,
        orders_df=updated_orders, order_items_df=items,
        end_date=end, seed=seed,
    )


def test_same_seed_same_events():
    assert _make_events(42).equals(_make_events(42))


def test_different_seed_different_events():
    assert not _make_events(42).equals(_make_events(43))
