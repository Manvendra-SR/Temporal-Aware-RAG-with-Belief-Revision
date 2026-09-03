"""
Version ordering is the backbone of lineage resolution, the version-pinning
boost and the Library's lineage strip. These tests pin down the policy defined
in services/version_resolver.py.
"""

from __future__ import annotations

import pytest

from services.version_resolver import is_version_greater, version_sort_key


class TestTrailingZerosAreInsignificant:
    """"2.2" and "2.2.0" name the same release and must compare equal."""

    @pytest.mark.parametrize(
        "a, b",
        [
            ("2.2", "2.2.0"),
            ("2.2", "2.2.0.0"),
            ("1", "1.0"),
            ("1", "1.0.0"),
            ("v3.1", "3.1.0"),
        ],
    )
    def test_equal_versions(self, a: str, b: str) -> None:
        assert version_sort_key(a) == version_sort_key(b)

    def test_equal_versions_are_not_greater(self) -> None:
        # The lineage guard requires STRICTLY greater, so uploading "2.2" on
        # top of a "2.2.0" parent must be rejected.
        assert not is_version_greater("2.2", "2.2.0")
        assert not is_version_greater("2.2.0", "2.2")


class TestNumericNotLexicalComparison:
    """Components compare as integers — "2.10" is newer than "2.9"."""

    def test_double_digit_minor_beats_single_digit(self) -> None:
        assert is_version_greater("2.10", "2.9")
        assert not is_version_greater("2.9", "2.10")

    def test_double_digit_patch(self) -> None:
        assert is_version_greater("1.13.10", "1.13.9")

    def test_major_dominates_minor(self) -> None:
        assert is_version_greater("2.0", "1.99")


class TestPrefixesAndSeparators:
    def test_v_prefix_is_ignored(self) -> None:
        assert version_sort_key("v1.13.1") == version_sort_key("1.13.1")

    def test_underscores_behave_like_dots(self) -> None:
        assert version_sort_key("1_13_1") == version_sort_key("1.13.1")

    def test_date_style_versions_order_correctly(self) -> None:
        assert is_version_greater("2024-03-01", "2024-02-28")
        assert is_version_greater("2024-10", "2024-9")


class TestMissingAndMalformedVersions:
    @pytest.mark.parametrize("bad", [None, "", "   ", "bad!!", "alpha"])
    def test_unusable_versions_share_the_lowest_key(self, bad) -> None:
        assert version_sort_key(bad) == version_sort_key(None)

    @pytest.mark.parametrize("bad", [None, "", "bad!!"])
    def test_unusable_versions_sort_below_real_ones(self, bad) -> None:
        assert version_sort_key(bad) < version_sort_key("0.0.1")

    @pytest.mark.parametrize("bad", [None, "", "bad!!"])
    def test_unusable_version_is_never_greater(self, bad) -> None:
        # Including "greater than another missing version" — otherwise a
        # version-less upload could pass the lineage guard.
        assert not is_version_greater(bad, "1.0")
        assert not is_version_greater(bad, None)

    def test_any_real_version_beats_a_missing_one(self) -> None:
        assert is_version_greater("0.1", None)


class TestSortingAChain:
    def test_lineage_sorts_oldest_first(self) -> None:
        chain = ["2.10", "1.13", "2.2", "1.9", "2.2.1"]
        assert sorted(chain, key=version_sort_key) == [
            "1.9", "1.13", "2.2", "2.2.1", "2.10",
        ]
