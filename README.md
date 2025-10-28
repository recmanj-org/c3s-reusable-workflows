## c3s-reusable workflows

This repository contains reusable GitHub Actions workflows for C3S projects. There is currently one CI workflow for Jupyter Notebook QA automation.


### notebook-qa

This workflow implements QA automation for Jupyter Notebooks with following automation checks:

1. Code linting with `nbqa` and `ruff`
2. Code formatting with `nbqa` and `ruff`
3. Notebook linting with `pynblint`
4. Custom DOI checker
5. Link availability testing with `pytest-check-links`
6. Notebook execution and memory profilling with `ploomber-engine`
7. Custom metadata version check
8. Custom test and coverage checks
9. Custom accessibility check
10. Custom figure labels check
11. License file check
12. Changelog file check

#### How to use `notebook-qa.yml` workflow

Configure the target repository which you want to run the QA check against using this format:

```
.github/workflows/qa.yml

------------------------

name: Notebook QA

on:
  push:
    branches:
      - develop
  pull_request:
    branches:
      - develop
  workflow_dispatch:
    inputs:
      notebooks:
        description: 'Comma-separated list of notebook paths to check (e.g., ./notebook1.ipynb,./folder/notebook2.ipynb). Leave empty to check all notebooks.'
        required: false
        type: string
        default: ''

jobs:
  notebook-qa:
    uses: recmanj-org/c3s-reusable-workflows/.github/workflows/notebook-qa.yml@main
    with:
      notebooks: ${{ inputs.notebooks || '' }}
    secrets: inherit
```

This sets up automated checks on new pull requests and merges/pushes into `develop` branch. It also allow manual Action run in GitHub Actions UI.


### How to configure access to cdsapi for notebook execution check

The action responsible for notebook execution allows setting a `cdsapi` key via `CDSAPI_KEY` secret set either on repository or organisation level.


### How to setup c3s-reusable-workflows repository in GitHub organisation

1. Fork this repository into your organisation
2. Leave the fork network in the newly forked repository settings
