"""
Version ordering is the backbone of lineage resolution, the version filter for
"according to v2.0" questions, and the Library's lineage strip. These tests pin
down the single policy defined in services/version_resolver.py.
"""

from __future__ import annotations

import pytest

from services.version_resolver import (
    is_version_greater,
    version_matches,
    version_sort_key,
)


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


class TestVersionMatching:
    """
    version_matches() is the filter behind "according to v2.0, ...". It reuses
    the ordering policy above, so a version query matches exactly the releases
    that compare equal to what was asked for.
    """

    @pytest.mark.parametrize("chunk_version, requested", [
        ("2.0", "2.0"),
        ("2.0.0", "2.0"),
        ("1.0", "1"),
        ("v3.1", "3.1"),
    ])
    def test_the_same_release_matches(self, chunk_version, requested) -> None:
        assert version_matches(chunk_version, requested)

    @pytest.mark.parametrize("chunk_version, requested", [
        ("1.13", "2.2"),
        ("2.0.1", "2.0"),     # a different release, not a longer spelling of 2.0
        ("3.0", "2.0"),
    ])
    def test_a_different_release_does_not_match(self, chunk_version, requested) -> None:
        assert not version_matches(chunk_version, requested)

    def test_components_compare_numerically_not_as_strings(self) -> None:
        # Regression: a string-prefix match served v2.10 to someone asking for v2.1.
        assert not version_matches("2.10", "2.1")
        assert version_matches("2.1", "2.1")

    @pytest.mark.parametrize("chunk_version, requested", [
        (None, "2.2"),
        ("bad!!", "2.2"),
        ("2.2", None),
        ("2.2", "bad!!"),
        (None, None),
    ])
    def test_an_unusable_version_matches_nothing(self, chunk_version, requested) -> None:
        assert not version_matches(chunk_version, requested)
