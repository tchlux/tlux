from tlux.search.hkm.builder.recursive_index_builder import _resolve_max_depth


def test_tree_depth_scales_with_embedding_count() -> None:
    small = _resolve_max_depth(1024, 1024, 8, 0)
    large = _resolve_max_depth(1_000_000_000, 1024, 8, 0)
    assert small == 1
    assert large >= 7
