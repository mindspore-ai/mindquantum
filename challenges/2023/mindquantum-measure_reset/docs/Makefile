# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# User-friendly check for sphinx-build
ifeq ($(shell which $(SPHINXBUILD) >/dev/null 2>&1; echo $$?), 1)
$(error The '$(SPHINXBUILD)' command was not found. Make sure you have Sphinx installed, then set the SPHINXBUILD environment variable to point to the full path of the '$(SPHINXBUILD)' executable. Alternatively you can add the directory with the executable to your PATH. If you do not have Sphinx installed, grab it from http://sphinx-doc.org/)
endif


ifeq ($(OS),Windows_NT)
    MKDIR = mkdir
    MKDIR_POSTARGS = -ea 0
else
    MKDIR = mkdir -p
endif


# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean: clean_doxygen
	rm -rf $(BUILDDIR)/*

clean_doxygen:
	rm -rf xml

.PHONY: help Makefile doxygen

doxygen:
	$(MKDIR) doxygen $(MKDIR_POSTARGS)
	doxygen

%_only: Makefile
	@$(SPHINXBUILD) -M $(subst _only,,$@) "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile doxygen
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
