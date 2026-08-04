"""Tests for the plotting module."""

from collections.abc import Iterator

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from puffins.plotting import panel_label  # noqa: E402


@pytest.fixture
def ax() -> Iterator[matplotlib.axes.Axes]:
    """A bare Axes on a figure that is closed after the test."""
    fig, ax_ = plt.subplots()
    yield ax_
    plt.close(fig)


def _texts(ax_: matplotlib.axes.Axes) -> list[str]:
    """The strings of every Text artist added to the Axes."""
    return [t.get_text() for t in ax_.texts]


class TestPanelLabel:
    """Tests for panel_label."""

    def test_default_is_lowercase_in_parentheses(
        self, ax: matplotlib.axes.Axes
    ) -> None:
        """The historical default formatting is unchanged."""
        panel_label(0, ax=ax)
        assert _texts(ax) == ["(a)"]

    def test_counts_up_the_alphabet(self, ax: matplotlib.axes.Axes) -> None:
        """panel_num indexes the alphabet from zero."""
        panel_label(3, ax=ax)
        assert _texts(ax) == ["(d)"]

    def test_uppercase_gives_capital_letters(self, ax: matplotlib.axes.Axes) -> None:
        """uppercase=True switches to A, B, C..."""
        panel_label(1, ax=ax, uppercase=True)
        assert _texts(ax) == ["(B)"]

    def test_brackets_false_drops_the_parentheses(
        self, ax: matplotlib.axes.Axes
    ) -> None:
        """brackets=False leaves a bare letter."""
        panel_label(2, ax=ax, brackets=False)
        assert _texts(ax) == ["c"]

    def test_bold_sets_the_font_weight(self, ax: matplotlib.axes.Axes) -> None:
        """bold=True renders the label bold; the default does not."""
        panel_label(0, ax=ax, bold=True)
        assert ax.texts[0].get_fontweight() == "bold"

    def test_not_bold_by_default(self, ax: matplotlib.axes.Axes) -> None:
        """Without bold=True the weight is left at the rcParams default."""
        panel_label(0, ax=ax)
        assert ax.texts[0].get_fontweight() != "bold"

    def test_explicit_fontweight_wins_over_bold(self, ax: matplotlib.axes.Axes) -> None:
        """bold only supplies a default, so a passed fontweight takes priority."""
        panel_label(0, ax=ax, bold=True, fontweight="light")
        assert ax.texts[0].get_fontweight() == "light"

    def test_extra_text_is_appended(self, ax: matplotlib.axes.Axes) -> None:
        """extra_text follows the letter in the same Text artist."""
        panel_label(0, ax=ax, extra_text="JJAS")
        assert _texts(ax) == ["(a) JJAS"]

    def test_combined_options(self, ax: matplotlib.axes.Axes) -> None:
        """uppercase and brackets compose."""
        panel_label(0, ax=ax, uppercase=True, brackets=False, extra_text="rain")
        assert _texts(ax) == ["A rain"]

    def test_none_panel_num_labels_each_axes_in_turn(self) -> None:
        """Passing an iterable of Axes labels them a, b, c..."""
        fig, axarr = plt.subplots(1, 3)
        panel_label(ax=axarr)
        assert [_texts(a) for a in axarr] == [["(a)"], ["(b)"], ["(c)"]]
        plt.close(fig)

    def test_none_panel_num_propagates_the_style_options(self) -> None:
        """Style options reach each Axes in the iterable, not just the first."""
        fig, axarr = plt.subplots(1, 2)
        panel_label(ax=axarr, uppercase=True, brackets=False, bold=True)
        assert [_texts(a) for a in axarr] == [["A"], ["B"]]
        assert all(a.texts[0].get_fontweight() == "bold" for a in axarr)
        plt.close(fig)

    def test_rejects_panel_num_past_the_alphabet(
        self, ax: matplotlib.axes.Axes
    ) -> None:
        """Running off the end of the alphabet is an error, not an IndexError."""
        with pytest.raises(ValueError, match="panel_num must be between"):
            panel_label(26, ax=ax)

    @pytest.mark.parametrize("panel_num", [-1, -27])
    def test_rejects_negative_panel_num(
        self, panel_num: int, ax: matplotlib.axes.Axes
    ) -> None:
        """Negative indices are rejected rather than wrapping round the alphabet.

        Unguarded, -1 indexes the string from the back and silently labels the
        panel "(z)", while -27 raises the IndexError the check exists to
        replace.

        """
        with pytest.raises(ValueError, match="panel_num must be between"):
            panel_label(panel_num, ax=ax)

    def test_vertical_alignment_defaults_to_top(self, ax: matplotlib.axes.Axes) -> None:
        """The label anchors from its top edge unless told otherwise."""
        panel_label(0, ax=ax)
        assert ax.texts[0].get_verticalalignment() == "top"

    def test_vertical_alignment_can_be_overridden(
        self, ax: matplotlib.axes.Axes
    ) -> None:
        """An explicit va is honored, via either spelling."""
        panel_label(0, ax=ax, va="bottom")
        assert ax.texts[0].get_verticalalignment() == "bottom"
