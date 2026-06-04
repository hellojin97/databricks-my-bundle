"""결제 payments 팩트 테이블 검증."""
from datetime import date

import polars as pl

from ecommerce_generator import (
    generate_categories,
    generate_order_items,
    generate_orders,
    generate_payments,
    generate_products,
    generate_users,
)

EXPECTED_COLUMNS = {
    "payment_id", "order_id", "payment_method", "amount",
    "currency", "status", "paid_at", "refunded_at", "dt",
}
VALID_STATUS = {"captured", "authorized", "failed", "refunded"}

# order.status → 기대 payment status (generate_payments.STATUS_MAP와 동일해야 함)
STATUS_MAP = {
    "delivered": "captured", "shipped": "captured", "processing": "captured",
    "pending": "authorized", "cancelled": "failed", "refunded": "refunded",
}


def test_schema(payments):
    assert set(payments.columns) == EXPECTED_COLUMNS


def test_schema_dtypes(payments):
    s = payments.schema
    assert s["payment_id"] == pl.Int64
    assert s["order_id"] == pl.Int64
    assert s["amount"] == pl.Float64
    assert s["paid_at"] == pl.Datetime("us")
    assert s["refunded_at"] == pl.Datetime("us")
    assert s["dt"] == pl.Date


def test_payment_id_unique(payments):
    assert payments["payment_id"].n_unique() == len(payments)


def test_one_payment_per_order(payments, order_items_and_orders):
    """모든 주문에 정확히 1개의 결제가 있어야 한다 (1:1)."""
    _, orders = order_items_and_orders
    assert len(payments) == len(orders)
    assert payments["order_id"].n_unique() == len(payments)
    assert set(payments["order_id"].to_list()) == set(orders["order_id"].to_list())


def test_status_valid(payments):
    assert set(payments["status"].unique().to_list()) <= VALID_STATUS


def test_status_consistent_with_order(payments, order_items_and_orders):
    """payment status가 order.status에서 파생된 값과 정확히 일치해야 한다."""
    _, orders = order_items_and_orders
    joined = payments.select(["order_id", "status"]).join(
        orders.select(["order_id", pl.col("status").alias("order_status")]),
        on="order_id",
    )
    expected = joined["order_status"].replace_strict(STATUS_MAP)
    assert (joined["status"] == expected).all()


def test_amount_matches_order_local(payments, order_items_and_orders):
    """결제 금액은 고객이 청구받은 현지통화 금액(order.amount_local)과 일치해야 한다."""
    _, orders = order_items_and_orders
    joined = payments.select(["order_id", "amount"]).join(
        orders.select(["order_id", "amount_local"]), on="order_id"
    )
    assert (joined["amount"] - joined["amount_local"]).abs().max() < 0.01


def test_amount_currency_consistent(payments, order_items_and_orders):
    """amount(현지통화)와 currency가 같은 통화 기준이어야 한다 (USD 라벨에 USD 금액 등).

    회귀 방지: 과거 amount가 USD(total_amount)인데 currency는 현지통화라 불일치했음.
    USD 주문은 amount_local == total_amount 이므로, 비-USD 주문에서 amount가
    total_amount(USD)와 다름을 확인해 현지통화로 환산됐는지 검증한다.
    """
    _, orders = order_items_and_orders
    joined = payments.select(["order_id", "amount", "currency"]).join(
        orders.select(["order_id", "total_amount"]), on="order_id"
    )
    non_usd = joined.filter(pl.col("currency") != "USD")
    if len(non_usd) > 0:
        # 현지통화로 환산됐다면 USD 원금과 (대부분) 다른 값이어야 한다
        assert (non_usd["amount"] != non_usd["total_amount"]).any()


def test_paid_after_order_created(payments, order_items_and_orders):
    _, orders = order_items_and_orders
    joined = payments.select(["order_id", "paid_at"]).join(
        orders.select(["order_id", "created_at"]), on="order_id"
    )
    assert (joined["paid_at"] >= joined["created_at"]).all()


def test_refunded_at_only_on_refunded(payments):
    """refunded_at은 status=refunded 행에만 채워지고, 그 외엔 null."""
    with_refund = payments.filter(pl.col("refunded_at").is_not_null())
    assert set(with_refund["status"].unique().to_list()) <= {"refunded"}
    refunded = payments.filter(pl.col("status") == "refunded")
    assert refunded["refunded_at"].null_count() == 0
    assert (refunded["refunded_at"] >= refunded["paid_at"]).all()


def test_dt_matches_paid_at(payments):
    assert len(payments.filter(payments["paid_at"].dt.date() != payments["dt"])) == 0


# --- 재현성 ---

def _make_payments(seed: int):
    """payments는 금액 확정된 orders가 필요하므로 order_items까지 체인을 만든다."""
    start, end = date(2025, 1, 1), date(2025, 12, 31)
    cats = generate_categories.generate(seed=seed)
    users = generate_users.generate(n_users=800, start_date=start, end_date=end, seed=seed)
    products = generate_products.generate(
        n_products=200, categories_df=cats, start_date=start, end_date=end, seed=seed
    )
    orders = generate_orders.generate(users_df=users, end_date=end, seed=seed)
    _, orders = generate_order_items.generate(orders_df=orders, products_df=products, seed=seed)
    return generate_payments.generate(orders_df=orders, end_date=end, seed=seed)


def test_same_seed_same_payments():
    assert _make_payments(42).equals(_make_payments(42))


def test_different_seed_different_payments():
    assert not _make_payments(42).equals(_make_payments(43))
