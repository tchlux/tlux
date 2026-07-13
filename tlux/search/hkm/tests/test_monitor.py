from tlux.search.hkm.monitor import _parse_ps_usage


def test_parse_ps_usage_accepts_macos_whitespace() -> None:
    assert _parse_ps_usage(" 123456  94.7\n") == (123456 * 1024, 94.7)
    assert _parse_ps_usage("") == (0, 0.0)
