from pathlib import Path

toxini = Path("tox.ini")
pyprojecttoml = Path("pyproject.toml")


tox_lines = toxini.read_text().splitlines()
tox_pylint_lines = [l for l in tox_lines if l.strip().startswith("pylint")]
tox_disables = set([i for l in tox_pylint_lines for i in l.split('"')[1].split(",")])


pyproject_lines = pyprojecttoml.read_text().splitlines()
is_in_disable_string = False
pyproject_disables = set()
for line in pyproject_lines:
    if is_in_disable_string and line.startswith('"""'):
        is_in_disable_string = False

    if is_in_disable_string and line:
        pyproject_disables.add(line.strip().strip(","))

    if line.startswith("disable"):
        is_in_disable_string = True


assert pyproject_disables.difference(tox_disables) == {
    "invalid-name",
    "no-else-raise",
    "no-else-return",
}
assert len(tox_disables.difference(pyproject_disables)) == 0
