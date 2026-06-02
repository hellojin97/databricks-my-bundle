"""주문 라인 팩트 테이블 검증 + orders.total_amount 갱신 검증."""
import polars as pl

EXPECTED_COLUMNS = {
    "order_item_id", "order_id", "product_id", "quantity",
    "unit_price", "discount_pct", "line_total", "created_at", "dt",
}


def test_schema(order_items_and_orders):
    items, _ = order_items_and_orders
    assert set(items.columns) == EXPECTED_COLUMNS


def test_order_item_id_unique(order_items_and_orders):
    items, _ = order_items_and_orders
    assert items["order_item_id"].n_unique() == len(items)


def test_order_fk_valid(order_items_and_orders):
    items, updated_orders = order_items_and_orders
    order_ids = set(updated_orders["order_id"].to_list())
    assert set(items["order_id"].unique().to_list()) <= order_ids


def test_product_fk_valid(order_items_and_orders, products):
    items, _ = order_items_and_orders
    product_ids = set(products["product_id"].to_list())
    assert set(items["product_id"].unique().to_list()) <= product_ids


def test_every_order_has_at_least_one_item(order_items_and_orders):
    items, updated_orders = order_items_and_orders
    orders_with_items = set(items["order_id"].unique().to_list())
    assert set(updated_orders["order_id"].to_list()) <= orders_with_items


def test_line_total_equals_unit_price_times_quantity(order_items_and_orders):
    items, _ = order_items_and_orders
    expected = (items["unit_price"] * items["quantity"]).round(2)
    assert (items["line_total"] - expected).abs().max() < 0.01


def test_orders_total_amount_equals_sum_of_lines(order_items_and_orders):
    """갱신된 orders.total_amount 가 해당 주문 라인 합계와 일치해야 한다."""
    items, updated_orders = order_items_and_orders
    line_sums = items.group_by("order_id").agg(
        pl.col("line_total").sum().round(2).alias("expected_total")
    )
    merged = updated_orders.join(line_sums, on="order_id")
    diff = (merged["total_amount"] - merged["expected_total"]).abs().max()
    assert diff < 0.01
