from docutils import nodes
from docutils.statemachine import StringList
from sphinx.util.docutils import SphinxDirective, switch_source_input

import probnum as pn


class ProbNumConfigOptions(SphinxDirective):
    """Sphinx plugin that automatically generates the docs for ProbNum config options.

    The ``name``s, ``default_value``s, and ``description``s of the pre-defined
    ProbNum configuration options are read and dynamically added to a table in the docs.
    This plugin [1]_ avoids maintaining the table in the config docstring manually.

    See Also
    --------
    probnum.config : Register and set configs to control specific ProbNum behavior.

    References
    ----------
    .. [1] ``https://www.sphinx-doc.org/en/master/development/tutorials/helloworld.html``
    """

    def run(self):
        table = nodes.table(
            "",
            classes=[
                "longtable",
                "colwidths-auto",
                # "colwidths-given",
            ],
        )

        group = nodes.tgroup("", cols=3)
        table.append(group)
        group.append(nodes.colspec("", colwidth=35))
        group.append(nodes.colspec("", colwidth=15))
        group.append(nodes.colspec("", colwidth=50))

        thead = nodes.thead("")
        group.append(thead)

        headrow = nodes.row("")
        thead.append(headrow)

        headrow.append(nodes.entry("", nodes.paragraph(text="Config Option")))
        headrow.append(nodes.entry("", nodes.paragraph(text="Default Value")))
        headrow.append(nodes.entry("", nodes.paragraph(text="Description")))

        tbody = nodes.tbody("")
        group.append(tbody)

        for config_option in pn.config._options_registry.values():
            row = nodes.row("")
            tbody.append(row)

            name = config_option.name
            default_value = config_option.default_value
            description = config_option.description

            row.append(nodes.entry("", nodes.literal(text=name)))
            row.append(nodes.entry("", nodes.literal(text=repr(default_value))))
            row.append(nodes.entry("", self._parse_string(description)))

        return [table]

    def _parse_string(self, s: str):
        """Adapted from https://github.com/sphinx-doc/sphinx/blob/5559e5af1ff6f5fc2dc70679bdd6dc089cfff388/sphinx/ext/autosummary/__init__.py#L425"""
        node = nodes.paragraph("")

        vl = StringList()

        source, line = self.state_machine.get_source_and_line()
        vl.append(s, f"{source}:{line}:<probnum-config-options>")

        with switch_source_input(self.state, vl):
            self.state.nested_parse(vl, 0, node)

            try:
                if isinstance(node[0], nodes.paragraph):
                    node = node[0]
            except IndexError:
                pass

        return node


def setup(app):
    app.add_directive("probnum-config-options", ProbNumConfigOptions)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
