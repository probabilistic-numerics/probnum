# Makefile for Sphinx documentation


# Guide to sphinx docs on github pages:
# https://jefflirion.github.io/sphinx-github-pages.html

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = probnum
SOURCEDIR     = source
BUILDDIR      = _build
AUTOMODAPIDIR = source/automod

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Remove the automodapidir as it does not update with e.g. "make html"
clean:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	$(info Deleting contents of $(AUTOMODAPIDIR) directory...)
	-rm -r $(AUTOMODAPIDIR)/*