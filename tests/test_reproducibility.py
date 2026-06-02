"""재현성(seed) 검증 — 이 프로젝트의 핵심 보장.

같은 seed 는 항상 같은 데이터를, 다른 seed 는 다른 데이터를 만들어야 한다.
"""
from datetime import date

from ecommerce_generator import (
    generate_categories,
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
