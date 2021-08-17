from pathlib import Path

TOXINI_FILE = Path("tox.ini")
GLOBAL_DISABLES = {}


# Parse ./tox.ini
tox_lines = TOXINI_FILE.read_text().splitlines()
tox_pylint_lines = [
    l for l in tox_lines if l.strip().startswith("pylint src") and "--disable" in l
]
tox_global_disables = {m for m in tox_pylint_lines[0].split('"')[1].split(",")}
tox_per_package_disables = {
    m for l in tox_pylint_lines[1:] for m in l.split('"')[1].split(",")
}


# Check correctness and raise errors
try:
    global_but_not_per_package = tox_global_disables.difference(
        tox_per_package_disables
    ).difference(GLOBAL_DISABLES)
    assert not global_but_not_per_package
except AssertionError:
    raise Exception(
        f"""
    The following pylint messages seem to be disabled in the global linting pass, but
    not in the per-package linting passes in `./tox.ini`: {global_but_not_per_package}.
    This could have two reasons with different solutions:
    1. You fixed one or many pylint errors in one of the packages and there is no
       package left that needs this specific message disabled. Then, please also remove
       it from the global linting pass in `tox.ini` such that pylint runs in the
       strictest configuration possible.
    2. You decided to disable a message in the global linting pass in `./tox.ini`..
       In this case either add it to the list of messages that should be disabled in the
       long term (`pyproject.toml`) or add it to the `GLOBAL_DISABLES` variable in this
       python script (`./github/workflows/pylint_check.py`) if the message should only
       be disabled in the short term.
    If you are not sure what exactly you are supposed to do, or if you think that this
    message is wrong please feel free to open an issue on GitHub.
    """
    )


print(
    "The disabled pylint messages of the global and per-package linting passes in "
    "`./tox.ini` seem to be correctly synchronized."
)
