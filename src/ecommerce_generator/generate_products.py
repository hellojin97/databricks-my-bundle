"""상품 차원 테이블 생성기.

카테고리에 의존 (FK: category_id). 리프 카테고리에만 상품 배치.

실전 느낌 포인트:
- 가격: 카테고리별 로그-균등 분포 (전자제품은 비싸고, 책은 저렴)
- 카테고리별 상품 분포는 Pareto (인기 카테고리에 상품 쏠림)
- 원가는 판매가의 40~70%
- 12% 상품은 단종 (discontinued_at 있음)
- 2% 상품은 brand NULL
"""
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl

from .base import inject_nulls, make_rng

# L1 카테고리별 브랜드 풀
BRANDS_BY_L1 = {
    "Electronics": [
        "TechPro", "ElectraMax", "DigiCore", "Voltix", "Nexgen",
        "PixelForge", "CircuitPlus", "ZenithTech",
    ],
    "Fashion": [
        "UrbanThread", "LuxeStyle", "ModePoint", "VelvetRow",
        "Atelier9", "Northwind", "TerraDenim",
    ],
    "Home & Kitchen": [
        "HearthCo", "Lumico", "NestPro", "PaloHome",
        "BrightHaven", "Kindling Co",
    ],
    "Sports & Outdoors": [
        "TrailHead", "PeakLine", "FlexFit", "Atlas Outdoor",
        "Summit7", "Velora",
    ],
    "Books & Media": [
        "Penguin Press", "Quill House", "BlueInk", "PageTurner",
    ],
    "Beauty & Personal Care": [
        "Aurora", "Pure Petal", "GlowLab", "Soft & Bare",
        "Mira Skin", "Botanique",
    ],
    "Toys & Games": [
        "PlayPals", "FunForge", "Bricknest", "WonderBox",
    ],
}

# L1 카테고리별 가격 범위 (USD): (min, max)
PRICE_RANGES = {
    "Electronics":             (15,   2500),
    "Fashion":                 (10,    400),
    "Home & Kitchen":          (8,    1200),
    "Sports & Outdoors":       (12,    800),
    "Books & Media":           (5,     100),
    "Beauty & Personal Care":  (6,     150),
    "Toys & Games":            (8,     300),
}

ADJECTIVES = [
    "Pro", "Ultra", "Lite", "Max", "Plus", "Classic", "Essential",
    "Premium", "Mini", "Compact", "Smart", "Eco", "Heritage", "Bold",
    "Modern", "Elite", "Standard", "Signature",
]
MODEL_PREFIXES = ["Mk", "Series", "Model", "Edition", "Gen"]


def generate(
    n_products: int,
    categories_df: pl.DataFrame,
    start_date: date,
    end_date: date,
    null_rate_brand: float = 0.02,
    discontinued_rate: float = 0.12,
    seed: int = 42,
) -> pl.DataFrame:
    """상품 차원 테이블 생성."""
    rng = make_rng(seed)

    # 리프 카테고리만 사용 (depth = 3)
    leaf_cats = categories_df.filter(pl.col("depth") == 3)
    leaf_ids = leaf_cats["category_id"].to_numpy()
    leaf_names = leaf_cats["name"].to_list()
    leaf_paths = leaf_cats["path"].to_list()

    # 카테고리별 상품 수: Pareto (인기 카테고리에 더 많은 상품)
    cat_weights = rng.pareto(1.5, size=len(leaf_ids)) + 0.5
    cat_probs = cat_weights / cat_weights.sum()

    chosen_idx = rng.choice(len(leaf_ids), size=n_products, p=cat_probs)
    chosen_cat_ids = leaf_ids[chosen_idx]
    chosen_cat_names = [leaf_names[i] for i in chosen_idx]
    chosen_l1 = [leaf_paths[i].split(" > ")[0] for i in chosen_idx]

    product_ids = np.arange(1, n_products + 1, dtype=np.int64)

    # 브랜드: L1 카테고리 기반
    brands = np.array(
        [rng.choice(BRANDS_BY_L1[l1]) for l1 in chosen_l1],
        dtype=object,
    )

    # 상품명: "<Brand> <Adjective> <Category> <Model X>"
    adj_choices = rng.choice(ADJECTIVES, size=n_products)
    model_prefixes = rng.choice(MODEL_PREFIXES, size=n_products)
    model_numbers = rng.integers(1, 50, size=n_products)
    names = [
        f"{b} {a} {c} {mp} {mn}"
        for b, a, c, mp, mn in zip(
            brands, adj_choices, chosen_cat_names, model_prefixes, model_numbers, strict=False
        )
    ]

    # 가격: L1 카테고리별 로그-균등 분포
    base_prices = np.zeros(n_products)
    for i, l1 in enumerate(chosen_l1):
        lo, hi = PRICE_RANGES[l1]
        log_p = rng.uniform(np.log(lo), np.log(hi))
        base_prices[i] = round(float(np.exp(log_p)), 2)
    
    # 원가: 판매가의 40 ~ 70%
    cost_ratios = rng.uniform(0.4, 0.7, size=n_products)
    costs = np.round(base_prices * cost_ratios, 2)

    # 생성일: 분석 기간 앞쪽 80% 구간에 분산
    total_days = (end_date - start_date).days
    created_offsets = rng.uniform(0, total_days * 0.8, size=n_products)
    start_dt = datetime.combine(start_date, datetime.min.time())
    created_at = [start_dt + timedelta(days=float(d)) for d in created_offsets]

    # 단종 여부
    is_discontinued = rng.random(n_products) < discontinued_rate
    discontinued_at: list = []
    for ca, disc in zip(created_at, is_discontinued, strict=False):
        if disc:
            # 생성 후 30일 이후 ~ end_date 사이에 단종
            max_offset = max(31, (end_date - ca.date()).days)
            offset = int(rng.integers(30, max_offset + 1))
            discontinued_at.append(ca + timedelta(days=offset))
        else:
            discontinued_at.append(None)

    # SKU 코드
    skus = [
        f"{str(b)[:3].upper()}-{pid:08d}"
        for b, pid in zip(brands, product_ids, strict=False)
    ]

    # 브랜드 NULL 일부 (dirty data)
    brands_dirty = inject_nulls(brands, null_rate=null_rate_brand, rng=rng)

    df = pl.DataFrame({
        "product_id": product_ids,
        "sku": skus,
        "name": names,
        "category_id": chosen_cat_ids.astype(np.int32),
        "brand": brands_dirty.tolist(),
        "base_price": base_prices,
        "cost": costs,
        "created_at": created_at,
        "discontinued_at": discontinued_at,
        "is_active": (~is_discontinued).tolist()
    })

    return df