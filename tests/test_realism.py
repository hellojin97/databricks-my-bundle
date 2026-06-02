"""현실성(시간적 정합성 + 통화/금액) 검증."""
import polars as pl

from ecommerce_generator.generate_orders import FX_RATES, ZERO_DECIMAL_CURRENCIES


def test_order_items_reference_existing_products(order_items_and_orders, products, orders):
    """라인의 상품은 주문일에 이미 생성돼 있어야 한다 (created_at ≤ 주문일)."""
    items, _ = order_items_and_orders
    joined = (
        items.select(["order_id", "product_id"])
        .join(orders.select(["order_id", "created_at"]), on="order_id")
        .join(
            products.select(["product_id", "created_at"]).rename(
                {"created_at": "product_created_at"}
            ),
            on="product_id",
        )
    )
    assert (joined["product_created_at"] <= joined["created_at"]).all()


def test_order_items_not_discontinued_before_order(order_items_and_orders, products, orders):
    """라인의 상품은 주문일에 아직 단종되지 않았어야 한다 (주문일 ≤ discontinued_at)."""
    items, _ = order_items_and_orders
    joined = (
        items.select(["order_id", "product_id"])
        .join(orders.select(["order_id", "created_at"]), on="order_id")
        .join(
            products.select(["product_id", "discontinued_at"]),
            on="product_id",
        )
    )
    # 단종 상품을 참조한 라인만 검사 (미단종은 discontinued_at NULL → 제약 없음)
    discontinued_lines = joined.filter(pl.col("discontinued_at").is_not_null())
    assert (
        discontinued_lines["created_at"] <= discontinued_lines["discontinued_at"]
    ).all()


def test_fx_rate_matches_currency(orders):
    """fx_rate 는 통화별 고정 환율 테이블과 일치해야 한다."""
    distinct = orders.select(["currency", "fx_rate"]).unique()
    for currency, fx_rate in zip(distinct["currency"], distinct["fx_rate"], strict=False):
        assert fx_rate == FX_RATES[currency]


def test_amount_local_equals_total_times_fx(order_items_and_orders):
    """amount_local = total_amount(USD) × fx_rate (통화별 반올림 규칙 적용)."""
    _, updated_orders = order_items_and_orders
    raw = updated_orders["total_amount"] * updated_orders["fx_rate"]
    expected = (
        pl.when(updated_orders["currency"].is_in(list(ZERO_DECIMAL_CURRENCIES)))
        .then(raw.round(0))
        .otherwise(raw.round(2))
    )
    expected = updated_orders.select(expected.alias("e"))["e"]
    assert (updated_orders["amount_local"] - expected).abs().max() < 0.01


def test_zero_decimal_currencies_have_integer_local_amount(order_items_and_orders):
    """KRW/JPY 의 현지 금액은 정수여야 한다 (소수부 0)."""
    _, updated_orders = order_items_and_orders
    zero_dec = updated_orders.filter(
        pl.col("currency").is_in(list(ZERO_DECIMAL_CURRENCIES))
    )
    frac = (zero_dec["amount_local"] - zero_dec["amount_local"].round(0)).abs().max()
    assert frac is None or frac < 1e-9


def test_usd_orders_local_equals_usd(order_items_and_orders):
    """USD 주문은 fx_rate=1 이라 amount_local == total_amount."""
    _, updated_orders = order_items_and_orders
    usd = updated_orders.filter(pl.col("currency") == "USD")
    assert (usd["amount_local"] - usd["total_amount"]).abs().max() < 0.01
