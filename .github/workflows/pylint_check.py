from pathlib import Path

TOXINI_FILE = Path("tox.ini")
PYPROJECTTOML_FILE = Path("pyproject.toml")
GLOBAL_DISABLES = {
    "invalid-name",
    "fixme",
    "bad-continuation",
    "no-else-raise",
    "no-else-return",
    "no-member",
    "redefined-variable-type",
}


# Parse ./tox.ini
tox_lines = TOXINI_FILE.read_text().splitlines()
tox_pylint_lines = [
    l for l in tox_lines if l.strip().startswith("pylint") and "--disable" in l
]
tox_disables = {i for l in tox_pylint_lines for i in l.split('"')[1].split(",")}


# Parse ./pyproject.toml
pyproject_lines = PYPROJECTTOML_FILE.read_text().splitlines()
is_in_disable_string = False
pyproject_disables = set()
for line in pyproject_lines:
    if is_in_disable_string and line.startswith('"""'):
        is_in_disable_string = False

    if is_in_disable_string and line:
        pyproject_disables.add(line.strip().strip(","))

    if line.startswith("disable"):
        is_in_disable_string = True


# Check correctness and raise errors
try:
    in_pyproject_butnot_tox = pyproject_disables.difference(tox_disables).difference(
        GLOBAL_DISABLES
    )
    assert not in_pyproject_butnot_tox
except AssertionError:
    raise Exception(
        f"""
    The following pylint messages seem to be disabled in `./pyproject.toml`, but not in
    any of the pylint calls in `./tox.ini`: {in_pyproject_butnot_tox}. This could have
    two reasons with different solutions:
    1. You fixed one or many pylint errors in one of the modules and there is no module
       left that needs this specific message disabled. Then, please also remove it from
       the `./pyproject.toml` file such that pylint uses the most up-to-date
       configuration file.
    2. You added a new global exception to `./pyproject.toml` after deciding that this
       is a message that we do not want to enforce anywhere currently. Then, please add
       this exception also to the `GLOBAL_DISABLES` variable in this this python script
       (`./github/workflows/pylint_check.py`).

    If you are not sure what exactly you are supposed to do, or if you think that this
    message is wrong please feel free to ping @nathanaelbosch.
    """
    )


try:
    in_tox_butnot_pyproject = tox_disables.difference(pyproject_disables)
    assert not in_tox_butnot_pyproject
except AssertionError:
    raise Exception(
        f"""
    The following pylint messages seem to be disabled in `./tox.ini`, but not in
    `./pyproject.toml`: {in_tox_butnot_pyproject}. Please make sure to add them to
    `./pyproject.toml` such that pylint does not raise any warnings that are not
    supposed to be fixed right now.

    If you are not sure what exactly you are supposed to do, or if you think that this
    message is wrong please feel free to ping @nathanaelbosch.
    """
    )


print(
    "The pylint exceptions in `./tox.ini` and `./pyproject.toml` seem to be correctly synchronized."
)
