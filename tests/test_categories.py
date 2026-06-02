"""카테고리 차원 테이블 검증."""


EXPECTED_COLUMNS = {"category_id", "name", "parent_id", "depth", "path"}


def test_schema(categories):
    assert set(categories.columns) == EXPECTED_COLUMNS


def test_category_id_unique(categories):
    assert categories["category_id"].n_unique() == len(categories)


def test_depths_are_1_to_3(categories):
    assert set(categories["depth"].unique().to_list()) == {1, 2, 3}


def test_root_categories_have_no_parent(categories):
    roots = categories.filter(categories["depth"] == 1)
    assert roots["parent_id"].null_count() == len(roots)


def test_non_root_parent_ids_exist(categories):
    """depth>1 의 parent_id 는 반드시 존재하는 category_id 여야 한다 (self-FK)."""
    ids = set(categories["category_id"].to_list())
    non_root = categories.filter(categories["depth"] > 1)
    parent_ids = non_root["parent_id"].to_list()
    assert all(pid in ids for pid in parent_ids)


def test_path_depth_matches(categories):
    """path 의 ' > ' 구분자 수가 depth-1 과 일치해야 한다."""
    for depth, path in zip(categories["depth"], categories["path"], strict=False):
        assert path.count(" > ") == depth - 1
