"""Tests for belief updates about quantities of interest of a linear system."""

import pathlib

case_modules = (pathlib.Path(__file__).parent / "cases").stem
cases_belief_updates = case_modules + ".belief_updates"
cases_states = case_modules + ".states"
