# Contributing to Charl

Thank you for your interest in contributing to Charl! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Set up your development environment**:
   ```bash
   git clone https://github.com/YOUR-USERNAME/charlcode.git
   cd charlcode
   cargo build
   cargo test
   ```

## How to Contribute

### Reporting Bugs

- Check if the bug has already been reported in [Issues](https://github.com/charlcoding-stack/charlcode/issues)
- If not, create a new issue with:
  - Clear title and description
  - Steps to reproduce
  - Expected vs actual behavior
  - Charl version and platform information

### Suggesting Features

- Open an issue with the `enhancement` label
- Clearly describe the feature and its use case
- Explain why this feature would be useful to most users

### Code Contributions

#### Before You Start

1. **Discuss major changes** by opening an issue first
2. **Sign the Contributor License Agreement** (CLA) - see CLA.md
3. **Follow the coding style** used throughout the project

#### Development Workflow

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes:
   - Write clear, concise commit messages
   - Add tests for new functionality
   - Update documentation as needed
   - Ensure all tests pass: `cargo test`
   - Run the linter: `cargo clippy`
   - Format code: `cargo fmt`

3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Create a Pull Request**:
   - Provide a clear description of the changes
   - Reference any related issues
   - Ensure CI checks pass

#### Code Style

- Follow Rust conventions and idioms
- Use `cargo fmt` for formatting
- Address `cargo clippy` warnings
- Write descriptive variable and function names
- Add comments for complex logic
- Keep functions focused and small

#### Testing

- All new features must include tests
- Aim for high test coverage
- Tests should be clear and maintainable
- Run the full test suite before submitting:
  ```bash
  cargo test --all-features
  ```

#### Documentation

- Update relevant documentation for any changes
- Add doc comments for public APIs
- Include code examples where appropriate
- Keep the README.md up to date

### Pull Request Process

1. **Ensure your PR**:
   - Passes all CI checks
   - Includes tests
   - Updates documentation
   - Follows code style guidelines

2. **PR Review**:
   - Maintainers will review your PR
   - Address any requested changes
   - Be patient - reviews may take time

3. **After Approval**:
   - Your PR will be merged by a maintainer
   - You'll be added to CONTRIBUTORS.md

## Contributor License Agreement

By contributing to Charl, you agree that your contributions will be licensed under the MIT License. You also certify that:

- You created the contribution entirely yourself
- You have the right to submit the contribution
- The contribution does not violate any third-party rights

See [CLA.md](CLA.md) for the full Contributor License Agreement.

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Questions?

- Open an issue with the `question` label
- Join discussions in existing issues
- Check the [documentation](https://github.com/charlcoding-stack/charlcode/tree/main/docs)

## Recognition

Contributors who have their PRs merged will be listed in [CONTRIBUTORS.md](CONTRIBUTORS.md).

Thank you for helping make Charl better!
