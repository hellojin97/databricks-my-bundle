"""테스트 공용 픽스처.

generator들은 메모리에서 DataFrame을 반환하므로 파일 IO 없이 빠르게 검증한다.
규모는 비율(NULL/단종 등) 검증이 안정적으로 통과할 만큼만 키운다.
"""
from datetime import date

import pytest

from ecommerce_generator import (
    generate_categories,
    generate_events,
    generate_order_items,
    generate_orders,
    generate_products,
    generate_users,
)

SEED = 42
START_DATE = date(2025, 1, 1)
END_DATE = date(2025, 12, 31)
N_USERS = 5000
N_PRODUCTS = 1000
NULL_RATE_GENDER = 0.05
NULL_RATE_BRAND = 0.02
DISCONTINUED_RATE = 0.12
BROWSE_SESSIONS_PER_USER = 3
MAX_EVENTS_PER_SESSION = 8
NULL_RATE_SEARCH = 0.10


@pytest.fixture(scope="session")
def categories():
    return generate_categories.generate(seed=SEED)


@pytest.fixture(scope="session")
def users():
    return generate_users.generate(
        n_users=N_USERS,
        start_date=START_DATE,
        end_date=END_DATE,
        null_rate_gender=NULL_RATE_GENDER,
        seed=SEED,
    )


@pytest.fixture(scope="session")
def products(categories):
    return generate_products.generate(
        n_products=N_PRODUCTS,
        categories_df=categories,
        start_date=START_DATE,
        end_date=END_DATE,
        null_rate_brand=NULL_RATE_BRAND,
        discontinued_rate=DISCONTINUED_RATE,
        seed=SEED,
    )


@pytest.fixture(scope="session")
def orders(users):
    return generate_orders.generate(users_df=users, end_date=END_DATE, seed=SEED)


@pytest.fixture(scope="session")
def order_items_and_orders(orders, products):
    """order_items 생성 시 orders.total_amount도 갱신되므로 둘을 함께 반환."""
    items, updated_orders = generate_order_items.generate(
        orders_df=orders, products_df=products, seed=SEED
    )
    return items, updated_orders


@pytest.fixture(scope="session")
def events(users, products, order_items_and_orders):
    """클릭스트림 events. order_items로 갱신된 orders(업데이트본)를 전달한다."""
    items, updated_orders = order_items_and_orders
    return generate_events.generate(
        users_df=users,
        products_df=products,
        orders_df=updated_orders,
        order_items_df=items,
        end_date=END_DATE,
        seed=SEED,
        browse_sessions_per_user=BROWSE_SESSIONS_PER_USER,
        max_events_per_session=MAX_EVENTS_PER_SESSION,
        null_rate_search=NULL_RATE_SEARCH,
    )
