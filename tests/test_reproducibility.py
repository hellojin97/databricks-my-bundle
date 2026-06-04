"""재현성(seed) 검증 — 이 프로젝트의 핵심 보장.

같은 seed 는 항상 같은 데이터를, 다른 seed 는 다른 데이터를 만들어야 한다.
"""
from datetime import date

from ecommerce_generator import (
    generate_categories,
    generate_order_items,
    generate_orders,
    generate_products,
    generate_users,
)

START_DATE = date(2025, 1, 1)
END_DATE = date(2025, 12, 31)
N_USERS = 1000
N_PRODUCTS = 200


def _make_users(seed: int):
    return generate_users.generate(
        n_users=N_USERS,
        start_date=START_DATE,
        end_date=END_DATE,
        seed=seed,
    )


def test_same_seed_same_users():
    assert _make_users(42).equals(_make_users(42))


def test_different_seed_different_users():
    assert not _make_users(42).equals(_make_users(43))


def test_same_seed_same_products():
    cats = generate_categories.generate(seed=42)
    p1 = generate_products.generate(
        n_products=N_PRODUCTS, categories_df=cats,
        start_date=START_DATE, end_date=END_DATE, seed=42,
    )
    p2 = generate_products.generate(
        n_products=N_PRODUCTS, categories_df=cats,
        start_date=START_DATE, end_date=END_DATE, seed=42,
    )
    assert p1.equals(p2)


def test_categories_deterministic():
    """카테고리는 seed 와 무관하게 항상 동일한 결정적 트리."""
    assert generate_categories.generate(seed=1).equals(
        generate_categories.generate(seed=999)
    )


def _make_orders(seed: int):
    users = _make_users(seed)
    return generate_orders.generate(users_df=users, end_date=END_DATE, seed=seed)


def test_same_seed_same_orders():
    assert _make_orders(42).equals(_make_orders(42))


def test_different_seed_different_orders():
    assert not _make_orders(42).equals(_make_orders(43))


def _make_order_items(seed: int):
    cats = generate_categories.generate(seed=seed)
    users = _make_users(seed)
    products = generate_products.generate(
        n_products=N_PRODUCTS, categories_df=cats,
        start_date=START_DATE, end_date=END_DATE, seed=seed,
    )
    orders = generate_orders.generate(users_df=users, end_date=END_DATE, seed=seed)
    # order_items.generate 는 (items, 갱신된 orders) 튜플을 반환
    return generate_order_items.generate(orders_df=orders, products_df=products, seed=seed)


def test_same_seed_same_order_items():
    a_items, a_orders = _make_order_items(42)
    b_items, b_orders = _make_order_items(42)
    assert a_items.equals(b_items)
    assert a_orders.equals(b_orders)


def test_different_seed_different_order_items():
    a_items, _ = _make_order_items(42)
    b_items, _ = _make_order_items(43)
    assert not a_items.equals(b_items)
