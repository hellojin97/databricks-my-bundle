"""카테고리 차원 테이블 생성기.

3단계 계층 구조 (L1 > L2 > L3)를 adjacency list 형태로 출력.
parent_id로 self-join, depth로 필터링 연습 가능.
"""
import polars as pl
from .base import make_rng


# 카테고리 트리
CATEGORY_TREE = {
    "Electronics": {
        "Computers & Tablets": ["Laptops", "Desktops", "Tablets", "Monitors", "Computer Accessories"],
        "Phones": ["Smartphones", "Phone Cases", "Chargers", "Wireless Earbuds"],
        "TV & Audio": ["Televisions", "Speakers", "Soundbars", "Headphones"],
        "Cameras": ["DSLR", "Mirrorless", "Action Cameras", "Camera Accessories"],
        "Gaming": ["Consoles", "Video Games", "Gaming Accessories", "Gaming Chairs"],
    },
    "Fashion": {
        "Men's Clothing": ["T-Shirts", "Shirts", "Pants", "Outerwear", "Suits"],
        "Women's Clothing": ["Dresses", "Tops", "Pants", "Skirts", "Outerwear"],
        "Shoes": ["Sneakers", "Boots", "Heels", "Sandals", "Athletic"],
        "Accessories": ["Bags", "Watches", "Belts", "Sunglasses", "Jewelry"],
    },
    "Home & Kitchen": {
        "Furniture": ["Sofas", "Beds", "Tables", "Chairs", "Storage"],
        "Kitchen": ["Cookware", "Bakeware", "Small Appliances", "Cutlery"],
        "Bedding": ["Sheets", "Pillows", "Duvets", "Blankets"],
        "Decor": ["Lighting", "Rugs", "Wall Art", "Plants"],
    },
    "Sports & Outdoors": {
        "Fitness": ["Yoga", "Weights", "Cardio Equipment", "Activewear"],
        "Outdoor": ["Camping", "Hiking", "Cycling", "Water Sports"],
        "Team Sports": ["Soccer", "Basketball", "Baseball", "Football"],
    },
    "Books & Media": {
        "Books": ["Fiction", "Non-Fiction", "Children's", "Textbooks"],
        "Movies": ["DVDs", "Blu-ray", "4K Ultra HD"],
        "Music": ["Vinyl", "CDs", "Sheet Music"],
    },
    "Beauty & Personal Care": {
        "Skincare": ["Cleansers", "Moisturizers", "Serums", "Sunscreen"],
        "Makeup": ["Face", "Eyes", "Lips", "Tools"],
        "Hair Care": ["Shampoo", "Conditioner", "Styling", "Color"],
    },
    "Toys & Games": {
        "Toys": ["Action Figures", "Dolls", "Building Sets", "Stuffed Animals"],
        "Games": ["Board Games", "Card Games", "Puzzles"],
    },
}

def generate(seed: int = 42) -> pl.DataFrame:
    """카테고리 계층 구조를 평탄화된 테이블로 반환.

    Schema:  
    - category_id  : Int32, PK.  
    - name         : Utf8.  
    - parent_id    : Int32, FK (null if depth=1).  
    - depth        : Int8 (1, 2, or 3).  
    - path         : Utf8 (e.g., "Electronics > Phones > Smartphones").  
    """
    _ = make_rng(seed) # 현재는 결정적이라 RNG 안 씀, 일관성 위해 시그니처는 유지
    rows = []
    next_id = 1

    for l1_name, l1_subs in CATEGORY_TREE.items():
        l1_id = next_id
        next_id += 1
        rows.append({
            "category_id": l1_id,
            "name": l1_name,
            "parent_id": None,
            "depth": 1,
            "path": l1_name,
        })
        for l2_name, l2_subs in l1_subs.items():
            l2_id = next_id
            next_id += 1
            rows.append({
                "category_id": l2_id,
                "name": l2_name,
                "parent_id": l1_id,
                "depth": 2,
                "path": f"{l1_name} > {l2_name}",
            })
            for l3_name in l2_subs:
                l3_id = next_id
                next_id += 1
                rows.append({
                    "category_id": l3_id,
                    "name": l3_name,
                    "parent_id": l2_id,
                    "depth": 3,
                    "path": f"{l1_name} > {l2_name} > {l3_name}",
                })
    
    return pl.DataFrame(rows).with_columns([
        pl.col("category_id").cast(pl.Int32),
        pl.col("parent_id").cast(pl.Int32),
        pl.col("depth").cast(pl.Int8),
    ])