# Contributing to Gift Recommendation Platform

Thank you for your interest in contributing to the Gift Recommendation Platform! This document provides guidelines and information about contributing to this project.

## 🎯 How to Contribute

### 1. Fork and Clone
1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/gift-recommendation-platform.git
   cd gift-recommendation-platform
   ```

### 2. Set Up Development Environment
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

### 3. Make Your Changes
1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following our code style guidelines
3. Add tests for any new functionality
4. Update documentation if needed

## 📝 Code Style Guidelines

### Python Code Style
- Use **Black** for code formatting:
  ```bash
  black src/ tests/ scripts/ app/
  ```

- Use **Flake8** for linting:
  ```bash
  flake8 src/ tests/ scripts/ app/
  ```

- Follow PEP 8 conventions
- Use type hints for all function parameters and return values
- Use descriptive variable and function names

### Docstring Format
Use NumPy-style docstrings:

```python
def calculate_confidence(self, gift: Dict, sentiment: float, size: str) -> float:
    """
    Calculate confidence score for gift recommendation.

    Parameters
    ----------
    gift : Dict
        Gift information dictionary
    sentiment : float
        Sentiment score between 0 and 1
    size : str
        Hand size category ('small', 'medium', 'large')

    Returns
    -------
    float
        Confidence score between 0 and 1

    Examples
    --------
    >>> engine = RecommendationEngine(gift_db)
    >>> confidence = engine.calculate_confidence(gift, 0.8, "small")
    >>> print(confidence)
    0.85
    """
```

## 🧪 Testing

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_nlp/test_preprocessor.py
```

### Writing Tests
- Write tests for all new functionality
- Aim for >80% code coverage
- Use descriptive test names
- Include both positive and negative test cases
- Use fixtures for common test data

Example test structure:
```python
import pytest
from src.nlp.preprocessor import TextPreprocessor

class TestTextPreprocessor:

    def setup_method(self):
        self.processor = TextPreprocessor()

    def test_clean_text_removes_urls(self):
        text = "Check this out https://example.com"
        result = self.processor.clean_text(text)
        assert "https://" not in result
```

## 📋 Pull Request Process

### Before Submitting
1. **Run the full test suite**: `pytest`
2. **Check code formatting**: `black --check src/ tests/ scripts/ app/`
3. **Run linting**: `flake8 src/ tests/ scripts/ app/`
4. **Update documentation** if you've made API changes
5. **Add tests** for any new functionality

### Pull Request Guidelines
1. **Create descriptive PR titles** and descriptions
2. **Reference related issues** using `Fixes #123` or `Closes #123`
3. **Keep PRs focused** - one feature or fix per PR
4. **Include screenshots** for UI changes
5. **Update the changelog** if applicable

### PR Template
```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] All new and existing tests pass locally with my changes

## Checklist
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my own code
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
```

## 🐛 Reporting Bugs

### Bug Report Template
When reporting bugs, please include:

1. **Environment information**:
   - OS (Windows/macOS/Linux)
   - Python version
   - Package versions

2. **Steps to reproduce**:
   - Clear, numbered steps
   - Expected behavior
   - Actual behavior

3. **Code samples** (if applicable)
4. **Error messages** (full traceback)
5. **Screenshots** (for UI issues)

## 💡 Suggesting Enhancements

### Feature Request Template
1. **Problem description**: What problem does this solve?
2. **Proposed solution**: Describe your suggested approach
3. **Alternatives considered**: Other approaches you've thought about
4. **Additional context**: Screenshots, mockups, examples

## 📚 Development Areas

### Priority Areas for Contributions
1. **NLP Improvements**:
   - Better text preprocessing
   - Additional sentiment analysis models
   - Multi-language support

2. **Computer Vision Enhancements**:
   - Improved hand size estimation
   - Better landmark detection
   - Additional hand measurements

3. **Recommendation Engine**:
   - More sophisticated scoring algorithms
   - Additional gift categories
   - User preference learning

4. **User Interface**:
   - Better visualization
   - Mobile-responsive design
   - Accessibility improvements

5. **Testing & Documentation**:
   - Increase test coverage
   - API documentation
   - Tutorial notebooks

## 🎯 Good First Issues

Look for issues labeled:
- `good first issue`
- `beginner-friendly`
- `documentation`
- `help wanted`

## 📞 Getting Help

- **GitHub Discussions**: For questions and general discussion
- **GitHub Issues**: For bug reports and feature requests
- **Documentation**: Check the `docs/` directory

## 🤝 Code of Conduct

### Our Standards
- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Be patient with newcomers

### Unacceptable Behavior
- Harassment or discrimination
- Trolling or inflammatory comments
- Publishing private information
- Unprofessional conduct

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

---

**Happy Contributing! 🎉**

Thank you for helping make the Gift Recommendation Platform better for everyone!