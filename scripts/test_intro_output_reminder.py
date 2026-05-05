"""
Introduction section_goal icinde OUTPUT REMINDER metninin bulundugunu dogrular.

Calistir: python scripts/test_intro_output_reminder.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orchestration.paper_blueprint import DEFAULT_PAPER_SECTIONS


def main() -> None:
    title, goal = DEFAULT_PAPER_SECTIONS[0]
    assert title == "Introduction and Motivation", title
    assert "OUTPUT REMINDER" in goal, "Introduction tuple'da OUTPUT REMINDER eksik."
    print("OK:", title)
    print("... tail:", goal[-220:])


if __name__ == "__main__":
    main()
