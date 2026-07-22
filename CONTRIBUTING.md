# Contributing Guide

InfiniCore is a version manifest for the InfiniTensor core components. Changes
to component implementations belong in the corresponding component
repositories; changes here should be limited to component revisions and
manifest documentation.

## General

1. Keep changes minimal. Do not add what is not necessary for the manifest.
2. Prefer self-explanatory content over abundant comments.
3. Use Markdown syntax, such as backticks, when referring to identifiers.
4. Write comments and error messages in English and follow the conventions of
   the file format or language.
5. End every file with a newline.
6. Review all changes before committing, especially AI-generated changes.

## Submodule Updates

1. Update a gitlink only to a reviewed, reachable component revision.
2. Keep independent component updates in separate commits when possible. When
   multiple revisions must move together for compatibility, update them in one
   commit and explain the relationship in the pull request.
3. Do not copy component source code, build logic, or runtime implementations
   into InfiniCore.
4. Record the old and new revisions and link the corresponding upstream change
   in the pull request.

## Commits

Commit messages must follow
[Conventional Commits](https://www.conventionalcommits.org/):

```text
<type>[optional scope][!]: <description>
```

Use the type that best describes the change, such as `docs`, `fix`, `refactor`,
or `chore`. Mark breaking changes with `!` and describe the impact in the
commit body or a `BREAKING CHANGE` footer.

Small changes should be represented by a single commit. Larger changes may use
multiple commits, but each commit must be meaningful, focused, and well-formed.

## Branches

Branch names use the format `<type>/xxx-yyyy-zzzz`, where `<type>` matches the
pull request title's Conventional Commits type and words are joined with
hyphens. For example:

```text
docs/contribution-guidelines
refactor/component-manifest
```

## Pull Requests

1. Use Conventional Commits format for the pull request title. Its type must
   match the branch type.
2. Explain what changed, why it changed, which component revisions are
   affected, and how the change was validated.
3. Keep small pull requests squashed. Large pull requests may retain multiple
   commits when every commit is meaningful and well-formed.
4. Include validation evidence for every affected component and platform. Use
   broader validation for high-risk changes, release preparation, shared build
   changes, or cross-platform behavior changes.
5. State which checks were not run and why.

## Validation

InfiniCore has no build or test suite of its own. Validate the manifest itself
before opening a pull request:

```shell
git diff --check
git ls-files --stage submodules
git submodule sync --recursive
git submodule update --init --recursive
git submodule status --recursive
```

Confirm that each component entry is a gitlink, each pinned revision can be
checked out, and the recursive submodule state is clean. For revision updates,
also include the relevant component or integration build and test evidence in
the pull request.
