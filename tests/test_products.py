"""상품 차원 테이블 검증."""
import polars as pl

from .conftest import DISCONTINUED_RATE, N_PRODUCTS, NULL_RATE_BRAND

EXPECTED_COLUMNS = {
    "product_id", "sku", "name", "category_id", "brand",
    "base_price", "cost", "created_at", "discontinued_at", "is_active",
}


def test_schema(products):
    assert set(products.columns) == EXPECTED_COLUMNS


def test_row_count(products):
    assert len(products) == N_PRODUCTS


def test_product_id_unique(products):
    assert products["product_id"].n_unique() == N_PRODUCTS


def test_category_fk_points_to_leaf(products, categories):
    """상품의 category_id 는 리프(depth=3) 카테고리만 가리켜야 한다."""
    leaf_ids = set(categories.filter(pl.col("depth") == 3)["category_id"].to_list())
    used_ids = set(products["category_id"].to_list())
    assert used_ids <= leaf_ids


def test_cost_below_base_price(products):
    """원가는 항상 판매가보다 낮아야 한다 (40~70% 비율)."""
    assert (products["cost"] < products["base_price"]).all()


def test_brand_null_rate_in_range(products):
    rate = products["brand"].null_count() / len(products)
    assert abs(rate - NULL_RATE_BRAND) < 0.02


def test_discontinued_rate_in_range(products):
    """is_active=False 비율이 단종 목표치 부근(±3%p)이어야 한다."""
    inactive_rate = (~products["is_active"]).sum() / len(products)
    assert abs(inactive_rate - DISCONTINUED_RATE) < 0.03


def test_discontinued_at_consistency(products):
    """단종 상품만 discontinued_at 이 채워져 있어야 한다."""
    discontinued = products.filter(~products["is_active"])
    active = products.filter(products["is_active"])
    assert discontinued["discontinued_at"].null_count() == 0
    assert active["discontinued_at"].null_count() == len(active)
