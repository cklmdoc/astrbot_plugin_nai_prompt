"""单元测试：SSRF URL 校验、分数防护、缓存键与角色上限。

运行方式（仓库根目录）：
    pip install -r requirements-dev.txt
    pytest tests/ -q
"""

import json

import pytest

from prompt_engine import (
    MAX_CHARACTERS,
    DanbooruSearchLookup,
    _safe_score,
    is_safe_url,
    parse_llm_response,
)


class TestSafeScore:
    """DanbooruSearch final_score 安全提取。"""

    def test_normal_float(self):
        assert _safe_score({"final_score": 0.8}) == 0.8

    def test_none_or_zero(self):
        assert _safe_score({"final_score": None}) == 0.0
        assert _safe_score({"final_score": 0}) == 0.0

    def test_non_numeric_string(self):
        assert _safe_score({"final_score": "abc"}) == 0.0

    def test_missing_key(self):
        assert _safe_score({}) == 0.0


class TestIsSafeUrl:
    """SSRF URL 校验（纯离线：IP 直连 + 白名单主机名 + monkeypatch 解析）。"""

    def test_public_ip_ok(self):
        assert is_safe_url("http://8.8.8.8/1.png")

    def test_public_https_ip_ok(self):
        assert is_safe_url("https://8.8.8.8/1.png")

    def test_non_http_scheme_rejected(self):
        assert not is_safe_url("ftp://example.com/a.png")
        assert not is_safe_url("file:///etc/passwd")

    def test_no_host_rejected(self):
        assert not is_safe_url("http:///a.png")

    def test_not_a_url(self):
        assert not is_safe_url("")
        assert not is_safe_url("not a url")

    def test_loopback_rejected(self):
        assert not is_safe_url("http://127.0.0.1:8765/1.png")
        assert not is_safe_url("http://localhost:8765/1.png")

    def test_private_ip_rejected(self):
        assert not is_safe_url("http://192.168.1.1/1.png")
        assert not is_safe_url("http://10.0.0.1/1.png")

    def test_link_local_rejected(self):
        assert not is_safe_url("http://169.254.169.254/latest/meta-data/1.png")

    def test_allowed_hostname_exact(self):
        assert is_safe_url("http://media.local/1.png", allowed_hosts=frozenset({"media.local"}))

    def test_allowed_ip_exact(self):
        assert is_safe_url("http://192.168.1.10/1.png", allowed_hosts=frozenset({"192.168.1.10"}))

    def test_allowed_cidr(self):
        assert is_safe_url("http://192.168.1.10/1.png", allowed_hosts=frozenset({"192.168.1.0/24"}))

    def test_allowed_cidr_not_matching(self):
        assert not is_safe_url("http://192.168.2.10/1.png", allowed_hosts=frozenset({"192.168.1.0/24"}))

    def test_allowed_hostname_case_insensitive(self):
        assert is_safe_url("http://Media.Local/1.png", allowed_hosts=frozenset({"media.local"}))

    def test_resolved_private_ip_rejected(self, monkeypatch):
        def fake_getaddrinfo(host, port, *args, **kwargs):
            return [(2, 1, 6, "", ("10.0.0.5", 0))]

        monkeypatch.setattr("prompt_engine.socket.getaddrinfo", fake_getaddrinfo)
        assert not is_safe_url("http://evil.example.com/1.png")

    def test_resolved_public_ip_allowed(self, monkeypatch):
        def fake_getaddrinfo(host, port, *args, **kwargs):
            return [(2, 1, 6, "", ("93.184.216.34", 0))]

        monkeypatch.setattr("prompt_engine.socket.getaddrinfo", fake_getaddrinfo)
        assert is_safe_url("http://ok.example.com/1.png")

    def test_dns_failure_rejected(self, monkeypatch):
        def fake_getaddrinfo(host, port, *args, **kwargs):
            raise OSError("name resolution failed")

        monkeypatch.setattr("prompt_engine.socket.getaddrinfo", fake_getaddrinfo)
        assert not is_safe_url("http://nonexistent.example/1.png")

    def test_allowed_hostname_skips_resolution(self, monkeypatch):
        """白名单精确主机名在 DNS 解析前即放行（离线可测）。"""
        def fake_getaddrinfo(host, port, *args, **kwargs):
            raise AssertionError("白名单命中不应触发 DNS 解析")

        monkeypatch.setattr("prompt_engine.socket.getaddrinfo", fake_getaddrinfo)
        assert is_safe_url("http://media.local/1.png", allowed_hosts=frozenset({"media.local"}))


class TestCacheKey:
    """DanbooruSearch 缓存键必须包含 show_nsfw。"""

    def test_key_normalizes_and_keeps_nsfw(self):
        assert DanbooruSearchLookup._key(" Hatsune Miku ", True) == ("hatsune miku", True)
        assert DanbooruSearchLookup._key("Hatsune Miku", False) == ("hatsune miku", False)

    def test_key_differs_by_nsfw(self):
        assert DanbooruSearchLookup._key("Hatsune Miku", True) != DanbooruSearchLookup._key("Hatsune Miku", False)


class TestMaxCharacters:
    """LLM 解析阶段角色上限截断。"""

    @staticmethod
    def _payload(char_count: int) -> str:
        characters = [
            {"display_name": f"char_{i}", "danbooru_tag": "", "tags": [], "position": ""}
            for i in range(char_count)
        ]
        return json.dumps({
            "characters": characters,
            "shared_tags": [],
            "outfit_tags": [],
            "action_tags": [],
            "scene_tags": [],
            "style_tags": [],
            "nsfw_level": "safe",
            "weights": [],
        })

    def test_cap_at_max_characters(self):
        parsed = parse_llm_response(self._payload(8))
        assert parsed is not None
        assert len(parsed.characters) == MAX_CHARACTERS

    def test_under_cap_unchanged(self):
        parsed = parse_llm_response(self._payload(3))
        assert parsed is not None
        assert len(parsed.characters) == 3
