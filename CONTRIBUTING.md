# Contributing to Conecsa System Vision

Thank you for helping improve Conecsa System Vision. This repository is a
filtered export of Conecsa's private development repository, so a contribution
takes a slightly different path than in a typical open-source project.

## Before you start

- Open an issue to discuss a larger change before you write it.
- Do not report a security vulnerability in a public issue; contact Conecsa
  privately first.

## Licensing your contribution

There is no contributor license agreement. A contribution is licensed inbound
= outbound: by opening a pull request you confirm that the change is your own
work and that you license it under the license of the files it changes
(Apache-2.0 or AGPL-3.0-only, see below). If part of it is not your own work,
say so in the pull request and name its source and license. A maintainer asks
for this statement as a comment on the pull request when it is missing, and
the pull request is not accepted without it.

## How a pull request lands

This repository receives exported snapshots, never the private history. When a
pull request is accepted, a maintainer applies it to the private repository,
where the full test suite runs, and the change reaches this repository with the
next export. The pull request is then closed with a reference to that export.

## Licensing layers

Every file declares its license with an SPDX header or an annotation in
[`REUSE.toml`](REUSE.toml):

- **Apache-2.0**: `proto/`, `styles/`, `i18n/`, `os-base/conecsa_shm`,
  `os-base/conecsa_common`, `flow/nodes/conecsa-system-vision`, `scripts/`,
  `yocto/` and `docs/`.
- **AGPL-3.0-only**: everything else.

A contribution takes the license of the directory it lands in, which is the
license you grant it under, and Apache-2.0 code must never import AGPL code.
Give every new source file a header, for example:

    reuse annotate --copyright Conecsa --year 2026 --copyright-prefix spdx \
        --license AGPL-3.0-only path/to/new_file.py

using `Apache-2.0` for files in the Apache-2.0 directories.

## Development

- `./scripts/init.sh` creates the Python virtualenv, compiles the protobuf
  stubs and checks the Rust toolchain.
- `scripts/test.sh` runs the same suites as CI, including
  `scripts/check-licenses.sh`; run it before you open a pull request.
  `ruff check .` and `.venv/bin/pyright` must report zero findings.
- Write code, comments, documentation and commit messages in English.
- Commit subjects follow `type(scope): imperative lowercase subject`, for
  example `fix(api-gateway): reject an empty task`, with a short prose body
  that explains why the change is needed.
- When you add user-facing text, keep the `en`, `pt-BR` and `es` translation
  catalogs in parity.

The `docs/` directory covers the architecture, configuration and each service.
