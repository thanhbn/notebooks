# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with the DeepEval evaluation framework in this repository.

## Project Overview

This is a DeepEval educational and development repository focused on teaching AI model evaluation techniques for code review systems. The project contains both learning materials and practical implementation examples for evaluating Large Language Models (LLMs) in code review scenarios.

## Project Structure

### Phase-based Learning Approach
- **Phase 1**: Environment Setup & Data Loading (`Phase1_Environment_Setup_Data_Loading.ipynb`)
- **Phase 2**: Basic Evaluation Framework (`Phase2_Basic_Evaluation_Framework.ipynb`)
- **Phase 3**: Advanced Metrics & Analysis (`Phase3_Advanced_Metrics_Analysis.ipynb`)
- **Future Phases**: Advanced visualization, production deployment, real-world case studies

### Key Directories
- `data/raw/`: Raw datasets and benchmarks
- `data/processed/`: Cleaned and preprocessed evaluation data
- `deep_evals/`: Original evaluation notebooks (5 comprehensive tutorials)
- `deepeval_claude_created/`: Claude-enhanced versions with additional features

## Key Files

### Evaluation Notebooks
- `Phase1_Environment_Setup_Data_Loading.ipynb`: Complete setup for evaluation environment
- `Phase2_Basic_Evaluation_Framework.ipynb`: Core evaluation pipeline implementation
- `Phase3_Advanced_Metrics_Analysis.ipynb`: Advanced metrics, statistical analysis, and model comparison
- `01_Foundation_and_Core_Concepts.ipynb`: DeepEval fundamentals
- `02_Advanced_RAG_Evaluation.ipynb`: RAG-specific evaluation techniques
- `03_Evaluating_Code_Generation_and_Review.ipynb`: Code quality assessment
- `04_Evaluating_AI_Agents_and_CoT_with_LangGraph.ipynb`: Agent evaluation
- `05_Feedback_Loop_ReGenerating_CoT.ipynb`: Iterative improvement techniques

### Documentation
- `benchmarks.md`: Comprehensive benchmark reference (40+ AI benchmarks)
- `benchmark_table.md`: Structured benchmark comparison table
- `all_about_benchmark.txt`: Detailed benchmark descriptions

### Data Files
- `code_samples.json`: Test code samples with various complexity levels
- `rag_document.txt`: RAG evaluation document in Vietnamese

## Development Commands

### Working with Jupyter Notebooks
```bash
# Start Jupyter notebook server
jupyter notebook

# Execute specific phase notebook
jupyter nbconvert --to notebook --execute Phase1_Environment_Setup_Data_Loading.ipynb
jupyter nbconvert --to notebook --execute Phase2_Basic_Evaluation_Framework.ipynb
jupyter nbconvert --to notebook --execute Phase3_Advanced_Metrics_Analysis.ipynb

# Run all notebooks in sequence
jupyter nbconvert --to notebook --execute Phase*.ipynb
```

### Environment Setup
```bash
# Install core dependencies (from Phase 1)
pip install langchain deepeval datasets transformers pandas matplotlib tqdm nltk scikit-learn

# Install additional dependencies for advanced features
pip install accelerate sentencepiece chromadb pinecone-client faiss-cpu

# Install Phase 3 dependencies for advanced analysis
pip install scipy statsmodels textstat radon

# Install API client libraries
pip install openai anthropic
```

### Model Configuration
```bash
# Set API keys for model evaluation
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

# For testing without API keys, notebooks will use mock responses
```

## Architecture Notes

### Evaluation Pipeline Design
1. **Data Loading**: Flexible dataset loaders for various code review formats
2. **Model Wrappers**: Unified interface for GPT-4, Claude, and other LLMs
3. **Metrics Engine**: Multiple evaluation metrics (BLEU, accuracy, similarity)
4. **Advanced Analysis**: Security, style, and statistical analysis frameworks
5. **Batch Processing**: Automated evaluation of multiple samples
6. **Statistical Testing**: Significance testing and multi-model comparison
7. **Results Storage**: Structured JSON output with comprehensive reports

### Core Components

#### CodeReviewModel Class
```python
class CodeReviewModel:
    def generate_review(self, code): 
        # AI-generated review with sentiment analysis
        
    def _extract_sentiment(self, review):
        # Automatic sentiment classification
```

#### EvaluationPipeline Class  
```python
def evaluate_model(model, dataset):
    results = []
    for sample in dataset:
        prediction = model.generate_review(sample['code'])
        scores = calculate_metrics(prediction, sample['expected'])
        results.append(scores)
    return aggregate_results(results)
```

#### Supported Metrics
- **Basic Metrics**: BLEU score, accuracy, precision/recall, F1-score
- **Semantic Metrics**: Review similarity, sentiment matching
- **Security Metrics**: Vulnerability detection, false positive analysis, security review assessment
- **Style Metrics**: Code quality analysis, readability assessment, improvement coverage
- **Statistical Metrics**: Confidence intervals, significance testing, correlation analysis
- **DeepEval Metrics**: Answer relevancy, hallucination detection
- **Performance Metrics**: Generation time, token usage

### Dataset Support
- **HumanEval**: Python coding problems with canonical solutions
- **CodeReviewer**: Microsoft's code review dataset format
- **Synthetic Data**: Generated code samples for testing
- **Custom Formats**: Extensible loader for new datasets

## Key Dependencies

### Core Packages
- `langchain`, `langchain-openai`, `langchain-anthropic`: LLM integration
- `deepeval`: Standardized evaluation framework
- `datasets`, `transformers`: Data loading and model utilities
- `pandas`, `numpy`: Data manipulation
- `matplotlib`, `seaborn`: Visualization

### Evaluation Packages
- `nltk`: Natural language processing metrics
- `scikit-learn`: Machine learning evaluation metrics
- `scipy`: Statistical analysis and significance testing
- `statsmodels`: Advanced statistical modeling
- `textstat`: Text readability and complexity metrics
- `radon`: Code complexity analysis
- `tqdm`: Progress tracking for batch operations

### Optional Packages
- `chromadb`, `pinecone-client`, `faiss-cpu`: Vector databases for RAG
- `openai`, `anthropic`: Direct API clients
- `pytest`: Testing framework integration

## Usage Patterns

### Quick Start Evaluation
```python
# Load pre-processed data
datasets = load_phase1_data()

# Initialize model
model = CodeReviewModel(ModelConfig('gpt-4'))

# Run evaluation
pipeline = EvaluationPipeline(model, EvaluationMetrics())
results = pipeline.evaluate_dataset(datasets['humaneval'], max_samples=10)

# Get aggregated metrics
metrics = pipeline.aggregate_results()
```

### Custom Dataset Integration
```python
def load_custom_dataset(path, sample_size=100):
    # Load your custom code review data
    return {
        'code': [...],
        'reviews': [...], 
        'labels': [...]
    }
```

### Adding New Metrics
```python
class CustomMetrics(EvaluationMetrics):
    @staticmethod
    def calculate_code_complexity(code):
        # Your custom metric implementation
        pass
```

### Advanced Analysis Patterns
```python
# Security analysis
security_metrics = SecurityMetrics()
vulnerabilities = security_metrics.detect_vulnerabilities(code)
security_analysis = security_metrics.analyze_security_review(review)

# Style analysis
style_metrics = StyleMetrics()
code_style = style_metrics.analyze_code_style(code)
quality_focus = style_metrics.analyze_review_quality_focus(review)

# Statistical comparison
model_comparison = ModelComparison()
model_comparison.add_model_results('gpt-4', results_gpt4)
model_comparison.add_model_results('claude-2', results_claude)
rankings = model_comparison.rank_models(['bleu_score', 'similarity'])

# Error analysis
error_analysis = ErrorAnalysis()
error_categories = error_analysis.categorize_errors(evaluation_results)
improvement_recs = error_analysis.generate_improvement_recommendations(error_patterns)
```

### Multi-Model Comparison
```python
# Compare models statistically
comparison_results = {}
for metric in ['bleu_score', 'similarity', 'sentiment_match']:
    comparisons = model_comparison.pairwise_comparison(metric)
    comparison_results[metric] = comparisons

# Generate insights
insights = model_comparison.generate_comparison_insights(rankings, comparisons)
```

## Best Practices

### Model Evaluation
1. **Start Small**: Use `max_samples=10` for initial testing
2. **Mock Testing**: Test pipeline with mock responses before using real APIs
3. **Rate Limiting**: Include delays between API calls to avoid rate limits
4. **Error Handling**: Always check for successful generation before calculating metrics
5. **Reproducibility**: Set random seeds and save exact model configurations
6. **Statistical Rigor**: Use appropriate sample sizes for significance testing (n≥30 recommended)
7. **Multiple Comparisons**: Apply correction methods when testing multiple hypotheses
8. **Effect Size**: Report effect sizes alongside p-values for practical significance

### Data Management
1. **Version Control**: Track dataset versions and preprocessing steps
2. **Data Validation**: Always validate data quality before evaluation
3. **Incremental Processing**: Save intermediate results to avoid re-computation
4. **Backup Results**: Store evaluation results with timestamps

### Performance Optimization
1. **Batch Processing**: Process multiple samples in parallel when possible
2. **Caching**: Cache model responses to avoid redundant API calls
3. **Sampling**: Use representative samples for large datasets
4. **Resource Monitoring**: Track token usage and generation costs
5. **Incremental Analysis**: Run advanced analysis on subsets before full evaluation
6. **Statistical Sampling**: Use power analysis to determine minimum sample sizes

### Advanced Analysis Best Practices
1. **Security Focus**: Always analyze code for common vulnerability patterns
2. **Style Consistency**: Evaluate both code quality and review quality
3. **Statistical Validation**: Use appropriate statistical tests for data distributions
4. **Error Categorization**: Systematically categorize and analyze failures
5. **Comparative Analysis**: Always compare multiple models when possible
6. **Insight Generation**: Focus on actionable recommendations from analysis

## Troubleshooting

### Common Issues
1. **API Key Errors**: Ensure environment variables are set correctly
2. **Rate Limiting**: Add delays between requests or use smaller batch sizes
3. **Memory Issues**: Process datasets in chunks for large evaluations
4. **Metric Calculation**: Check for empty responses before calculating metrics
5. **Statistical Test Failures**: Ensure sufficient sample sizes and check data distributions
6. **Security Pattern Matching**: Validate regex patterns for different code styles
7. **AST Parsing Errors**: Handle syntax errors gracefully in code analysis

### Debugging Tips
1. **Enable Logging**: Set logging level to INFO or DEBUG
2. **Mock Mode**: Use mock responses to test pipeline logic
3. **Sample Inspection**: Manually review first few results
4. **Metric Validation**: Test metrics with known inputs
5. **Statistical Debugging**: Check data distributions and outliers
6. **Component Testing**: Test security, style, and error analysis components separately
7. **Comparison Validation**: Verify statistical test assumptions before comparison

## Important Instructions

### File Management
- **ALWAYS prefer editing existing files** to creating new ones
- **DO NOT create documentation files** unless explicitly requested
- **Use the established phase-based structure** for new notebooks

### Evaluation Standards
- **Include error handling** in all evaluation pipelines
- **Validate data quality** before running evaluations
- **Save results in structured JSON format** with metadata
- **Apply statistical rigor** in all comparative analyses
- **Document assumptions** for statistical tests used
- **Generate comprehensive reports** with executive summaries
- **Focus on actionable insights** rather than just metrics

### Code Quality
- **Follow existing patterns** for model wrappers and metrics
- **Add proper docstrings** to all classes and functions
- **Include type hints** for better code clarity
- **Handle edge cases** in metric calculations
- **Implement robust error handling** for advanced analysis components
- **Use appropriate statistical tests** for data types and distributions
- **Generate meaningful insights** from analytical results

### Phase 3 Specific Guidelines
- **Security Analysis**: Always check for common vulnerability patterns (SQL injection, XSS, etc.)
- **Style Analysis**: Use AST parsing for robust code structure analysis
- **Statistical Testing**: Apply multiple comparison corrections when testing multiple models
- **Error Categorization**: Use the established 7-category error framework
- **Model Comparison**: Include effect size interpretation alongside significance testing
- **Report Generation**: Create both detailed analysis and executive summary reports

This repository provides a comprehensive framework for evaluating AI models in code review scenarios, with emphasis on statistical rigor, security analysis, and actionable insights for production deployment.