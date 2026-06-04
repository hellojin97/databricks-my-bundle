"""배송 shipments 팩트 테이블 검증."""
from datetime import date

import polars as pl

from ecommerce_generator import generate_orders, generate_shipments, generate_users

from .conftest import END_DATE

EXPECTED_COLUMNS = {
    "shipment_id", "order_id", "carrier", "status",
    "shipped_at", "delivered_at", "dt",
}
VALID_STATUS = {"in_transit", "delivered"}
SHIPPED_ORDER_STATUSES = {"shipped", "delivered", "refunded"}


def test_schema(shipments):
    assert set(shipments.columns) == EXPECTED_COLUMNS


def test_schema_dtypes(shipments):
    s = shipments.schema
    assert s["shipment_id"] == pl.Int64
    assert s["order_id"] == pl.Int64
    assert s["shipped_at"] == pl.Datetime("us")
    assert s["delivered_at"] == pl.Datetime("us")
    assert s["dt"] == pl.Date


def test_shipment_id_unique(shipments):
    assert shipments["shipment_id"].n_unique() == len(shipments)


def test_only_shipped_orders_have_shipment(shipments, order_items_and_orders):
    """배송 행은 shipped/delivered/refunded 주문에만, 그런 주문은 모두 1행씩."""
    _, orders = order_items_and_orders
    eligible = set(
        orders.filter(pl.col("status").is_in(list(SHIPPED_ORDER_STATUSES)))["order_id"].to_list()
    )
    shipped_orders = set(shipments["order_id"].to_list())
    assert shipped_orders == eligible
    assert shipments["order_id"].n_unique() == len(shipments)


def test_order_fk_valid(shipments, order_items_and_orders):
    _, orders = order_items_and_orders
    assert set(shipments["order_id"].unique().to_list()) <= set(orders["order_id"].to_list())


def test_status_valid(shipments):
    assert set(shipments["status"].unique().to_list()) <= VALID_STATUS


def test_carrier_non_empty(shipments):
    assert shipments["carrier"].null_count() == 0
    assert (shipments["carrier"].str.len_chars() > 0).all()


def test_shipped_after_order_created(shipments, order_items_and_orders):
    _, orders = order_items_and_orders
    joined = shipments.select(["order_id", "shipped_at"]).join(
        orders.select(["order_id", "created_at"]), on="order_id"
    )
    assert (joined["shipped_at"] >= joined["created_at"]).all()


def test_delivered_at_rules(shipments):
    """delivered_at은 status=delivered 행에만, shipped_at 이후, end_date 이내."""
    delivered = shipments.filter(pl.col("status") == "delivered")
    in_transit = shipments.filter(pl.col("status") == "in_transit")
    assert delivered["delivered_at"].null_count() == 0
    assert in_transit["delivered_at"].null_count() == len(in_transit)
    assert (delivered["delivered_at"] >= delivered["shipped_at"]).all()
    assert (delivered["delivered_at"].dt.date() <= END_DATE).all()


def test_dt_matches_shipped_at(shipments):
    mismatch = shipments.filter(shipments["shipped_at"].dt.date() != shipments["dt"])
    assert len(mismatch) == 0


# --- 재현성 ---

def _make_shipments(seed: int):
    start, end = date(2025, 1, 1), date(2025, 12, 31)
    users = generate_users.generate(n_users=800, start_date=start, end_date=end, seed=seed)
    orders = generate_orders.generate(users_df=users, end_date=end, seed=seed)
    return generate_shipments.generate(orders_df=orders, end_date=end, seed=seed)


def test_same_seed_same_shipments():
    assert _make_shipments(42).equals(_make_shipments(42))


def test_different_seed_different_shipments():
    assert not _make_shipments(42).equals(_make_shipments(43))
