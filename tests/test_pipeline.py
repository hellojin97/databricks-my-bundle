"""파이프라인 통합 검증.

generator 단위 테스트가 못 잡는 영역만 본다:
- `main()` end-to-end 실행(인자 전달·6단계 연결)이 오류 없이 끝나는가
- 6개 산출물이 올바른 경로에 영속되는가 (차원=파일, 팩트=dt 파티션 디렉터리)
- 디스크에서 다시 읽었을 때 스키마·파티션(dt)·교차 FK가 정합한가
- CLI 오버라이드가 실제로 반영되는가
재현성·분포 등 디테일은 각 generator 단위 테스트가 담당한다(중복 회피).
"""
import polars as pl
import pytest

from ecommerce_generator.main import main

N_USERS = 200
N_PRODUCTS = 50
START = "2025-01-01"
END = "2025-12-31"

# 영속된 차원 파일 → 기대 컬럼셋
DIM_FILES = {
    "categories.parquet": {"category_id", "name", "parent_id", "depth", "path"},
    "users.parquet": {
        "user_id", "email", "first_name", "last_name", "gender",
        "birth_date", "country", "segment", "is_active", "created_at",
    },
    "products.parquet": {
        "product_id", "sku", "name", "category_id", "brand",
        "base_price", "cost", "created_at", "discontinued_at", "is_active",
    },
}

# 영속된 팩트 디렉터리(dt 파티션) → 기대 컬럼셋(파티션 복원 후 dt 포함)
FACT_DIRS = {
    "orders": {
        "order_id", "user_id", "created_at", "status", "payment_method",
        "currency", "fx_rate", "total_amount", "amount_local", "dt",
    },
    "order_items": {
        "order_item_id", "order_id", "product_id", "quantity",
        "unit_price", "discount_pct", "line_total", "created_at", "dt",
    },
    "events": {
        "event_id", "user_id", "session_id", "event_type", "product_id",
        "search_query", "order_id", "event_ts", "dt",
    },
}


@pytest.fixture(scope="module")
def out_dir(tmp_path_factory):
    """파이프라인을 소규모로 1회 실행하고 출력 디렉터리를 반환."""
    out = tmp_path_factory.mktemp("raw")
    main([
        "--start-date", START,
        "--end-date", END,
        "--output-volume", str(out),
        "--users", str(N_USERS),
        "--products", str(N_PRODUCTS),
    ])
    return out


@pytest.fixture(scope="module")
def frames(out_dir):
    """디스크에서 다시 읽은 6개 테이블 (팩트는 hive 파티션 복원)."""
    return {
        "categories": pl.read_parquet(out_dir / "categories.parquet"),
        "users": pl.read_parquet(out_dir / "users.parquet"),
        "products": pl.read_parquet(out_dir / "products.parquet"),
        "orders": pl.read_parquet(str(out_dir / "orders"), hive_partitioning=True),
        "order_items": pl.read_parquet(str(out_dir / "order_items"), hive_partitioning=True),
        "events": pl.read_parquet(str(out_dir / "events"), hive_partitioning=True),
    }


# --- 산출물 존재 / 파티션 구조 ---

def test_dimension_files_exist(out_dir):
    for fname in DIM_FILES:
        assert (out_dir / fname).is_file(), f"{fname} 미생성"


def test_fact_dirs_are_partitioned(out_dir):
    """팩트는 디렉터리이고 dt= 파티션 하위 디렉터리를 가져야 한다."""
    for name in FACT_DIRS:
        d = out_dir / name
        assert d.is_dir(), f"{name} 디렉터리 미생성"
        parts = [p for p in d.iterdir() if p.is_dir() and p.name.startswith("dt=")]
        assert parts, f"{name} 에 dt= 파티션이 없음"


# --- 영속 스키마 ---

def test_dimension_schemas(frames):
    for fname, cols in DIM_FILES.items():
        key = fname.removesuffix(".parquet")
        assert set(frames[key].columns) == cols, f"{key} 스키마 불일치"


def test_fact_schemas(frames):
    for name, cols in FACT_DIRS.items():
        assert set(frames[name].columns) == cols, f"{name} 스키마 불일치"


# --- CLI 오버라이드 반영 ---

def test_cli_overrides_applied(frames):
    assert len(frames["users"]) == N_USERS
    assert len(frames["products"]) == N_PRODUCTS


# --- 팩트 비어있지 않음 ---

def test_facts_are_non_empty(frames):
    for name in FACT_DIRS:
        assert len(frames[name]) > 0, f"{name} 가 비어있음"


# --- 파티션 키 정합 (dt == 타임스탬프의 날짜) ---

def test_partition_dt_matches_timestamp(frames):
    bad_orders = frames["orders"].filter(
        pl.col("created_at").dt.date() != pl.col("dt")
    )
    assert len(bad_orders) == 0
    bad_events = frames["events"].filter(
        pl.col("event_ts").dt.date() != pl.col("dt")
    )
    assert len(bad_events) == 0


# --- 교차 테이블 FK (디스크 산출물 기준) ---

def test_cross_table_fk_at_rest(frames):
    user_ids = set(frames["users"]["user_id"].to_list())
    product_ids = set(frames["products"]["product_id"].to_list())
    order_ids = set(frames["orders"]["order_id"].to_list())
    leaf_cat_ids = set(
        frames["categories"].filter(pl.col("depth") == 3)["category_id"].to_list()
    )

    # 차원 ← 차원
    assert set(frames["products"]["category_id"].to_list()) <= leaf_cat_ids
    # 팩트 ← 차원/팩트
    assert set(frames["orders"]["user_id"].unique().to_list()) <= user_ids
    assert set(frames["order_items"]["order_id"].unique().to_list()) <= order_ids
    assert set(frames["order_items"]["product_id"].unique().to_list()) <= product_ids
    assert set(frames["events"]["user_id"].unique().to_list()) <= user_ids

    # events 의 nullable FK 는 non-null 값만 검사
    ev = frames["events"]
    seen_products = ev.filter(pl.col("product_id").is_not_null())["product_id"]
    seen_orders = ev.filter(pl.col("order_id").is_not_null())["order_id"]
    assert set(seen_products.unique().to_list()) <= product_ids
    assert set(seen_orders.unique().to_list()) <= order_ids
