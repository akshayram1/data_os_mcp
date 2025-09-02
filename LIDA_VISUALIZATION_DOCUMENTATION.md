# LIDA Visualization Flow Documentation

## Overview

This document describes the comprehensive LIDA (Lightweight Interactive Data Analysis) visualization flow implemented in the MCP Streamable HTTP application. The system provides intelligent, context-aware data visualizations with automatic persona generation and styling instructions.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Initialization and Setup](#initialization-and-setup)
3. [Main Visualization Pipeline](#main-visualization-pipeline)
4. [Persona and Instructions Generation](#persona-and-instructions-generation)
5. [Code Processing and Execution](#code-processing-and-execution)
6. [Library Selection Strategy](#library-selection-strategy)
7. [Error Handling and Fallbacks](#error-handling-and-fallbacks)
8. [Integration Points](#integration-points)
9. [API Reference](#api-reference)
10. [Configuration](#configuration)
11. [Troubleshooting](#troubleshooting)

## Architecture Overview

The LIDA visualization system follows a multi-layered architecture designed to provide beautiful, intelligent visualizations with complete autonomy:

```
User Query → Data Parsing → LIDA Pipeline → Visualization Generation → Display
     ↓              ↓              ↓                    ↓              ↓
Context Analysis → DataFrame → Goal Generation → Code/Chart → Streamlit UI
                              ↓
                    Persona & Instructions → Enhanced Styling
```

### Key Design Principles

1. **LIDA Authority**: LIDA has complete control over library selection, chart type, and styling
2. **Context Awareness**: System uses user queries and data characteristics for intelligent decisions
3. **Graceful Degradation**: Multiple fallback mechanisms ensure functionality even when LIDA fails
4. **Multi-Library Support**: Supports Plotly, Seaborn, Matplotlib, Altair, Bokeh, and Plotnine

## Initialization and Setup

### LIDA Manager Initialization

```python
# Initialize LIDA with OpenAI integration
lida_manager = None
if LIDA_AVAILABLE and os.getenv("OPENAI_API_KEY"):
    try:
        lida_manager = Manager(text_gen=llm("openai"))
    except Exception as e:
        st.warning(f"Failed to initialize LIDA: {str(e)}")
        lida_manager = None
```

### Required Dependencies

- **LIDA**: Core visualization generation
- **OpenAI**: LLM integration for goals and instructions
- **Streamlit**: UI framework
- **Pandas**: Data manipulation
- **Multiple viz libraries**: plotly, seaborn, matplotlib, altair, bokeh, plotnine

### Environment Variables

```env
OPENAI_API_KEY=your_openai_api_key_here
MCP_SERVER_URL=http://localhost:8000/mcp
```

## Main Visualization Pipeline

### 1. Enhanced Visualization Pipeline (`enhanced_visualization_pipeline()`)

This is the primary entry point that gives LIDA complete authority over visualization decisions.

```python
def enhanced_visualization_pipeline(df, user_query="", enable_lida=True):
    """
    Enhanced visualization pipeline with LIDA having COMPLETE AUTHORITY 
    over library selection for creating beautiful visualizations.
    """
```

#### Flow Steps:

1. **Input Validation**: Check if DataFrame is valid and not empty
2. **LIDA Authority Check**: Verify LIDA manager availability
3. **Summary Generation**: Create data summary using LIDA
4. **Goal Enhancement**: Generate enhanced goals based on user query
5. **Visualization Generation**: Let LIDA choose the best approach
6. **Code Processing**: Clean and execute generated code
7. **Display**: Render the final visualization

#### Goal Generation Logic:

```python
if user_query:
    enhanced_goal = user_query
    
    # Chart type detection
    chart_type_keywords = ["pie chart", "bar chart", "line chart", "scatter plot", 
                          "histogram", "heatmap", "treemap", "donut chart", "area chart",
                          "box plot", "violin plot", "sunburst", "waterfall", "radar"]
    
    user_requested_specific_chart = any(keyword in user_query.lower() 
                                      for keyword in chart_type_keywords)
    
    if user_requested_specific_chart:
        # Honor specific chart type request
        enhanced_goal += ". Create a beautiful, visually appealing [chart_type] with proper colors and styling."
    else:
        # Give LIDA complete freedom
        enhanced_goal += ". Create the most beautiful, visually appealing, and professional-looking visualization for this data. Choose the best chart type, library, colors, and styling to make it stunning and insightful."
```

### 2. Advanced Chart Creation (`create_advanced_chart()`)

Alternative entry point with simpler interface for direct chart creation.

```python
def create_advanced_chart(df, chart_type="auto", title=None, user_query=""):
    """Create advanced charts using LIDA with COMPLETE AUTHORITY for beautiful visualizations."""
```

#### Process Flow:

1. **LIDA Invocation**: Call LIDA with enhanced prompt
2. **Code Extraction**: Extract generated code from LIDA response
3. **Code Preprocessing**: Clean and prepare code for execution
4. **Execution**: Run code with all available libraries
5. **Display**: Show the resulting visualization

## Persona and Instructions Generation

### Persona Generation

The system automatically generates contextual personas to better understand who would be interested in the visualization:

```python
def generate_lida_instructions_and_persona(df, user_query=""):
    """Generate intelligent instructions and persona for LIDA using LLM."""
```

#### Persona Prompt Template:

```
Based on this data analysis request, generate a specific persona who would be interested in this visualization:

USER QUERY: "{user_query}"
DATASET CONTEXT: Data about {columns}

Generate a specific persona (role, goals, context) who would want to see this data visualization. 
Be specific about their role, why they need this data, and what decisions they might make.

Return only the persona description in 1-2 sentences, focusing on their role and data needs.
```

#### Example Generated Personas:

- **Sales Manager**: "A regional sales manager who needs to track product performance across categories to optimize inventory and identify top-performing products for the upcoming quarter."
- **Marketing Analyst**: "A digital marketing analyst who analyzes customer engagement metrics to optimize campaign targeting and improve conversion rates."
- **Operations Director**: "An operations director who monitors efficiency metrics to identify bottlenecks and optimize resource allocation across different departments."

### Instructions Generation

The system generates 2-3 specific styling and formatting instructions:

```python
instruction_prompt = f"""
Based on this data analysis request and dataset characteristics, generate 2-3 specific visualization instructions:

USER QUERY: "{user_query}"
DATASET INFO:
- Columns: {data_info['columns']}
- Numeric columns: {data_info['numeric_cols']}
- Categorical columns: {data_info['categorical_cols']}
- Data shape: {data_info['shape']}
- Sample: {data_info['sample_data']}

Generate 2-3 specific instructions to make the visualization more beautiful and insightful. Focus on:
1. Visual aesthetics (colors, styling, layout)
2. Data clarity (labels, formatting, readability)
3. Professional appearance (titles, legends, annotations)

Return only the instructions as a simple list, one per line, starting with "- ".
"""
```

#### Example Generated Instructions:

- "Use professional color palette with high contrast"
- "Add clear axis labels and title"
- "Ensure the chart is publication-ready with proper formatting"
- "Include data value annotations for clarity"
- "Apply consistent font sizing and legend positioning"

### Fallback Instructions

When OpenAI API is unavailable, the system provides sensible defaults:

```python
return [
    "use professional color palette with high contrast",
    "add clear axis labels and title",
    "ensure the chart is publication-ready with proper formatting"
], f"data analyst interested in {user_query.lower()}"
```

## Code Processing and Execution

### Code Preprocessing (`preprocess_lida_code()`)

LIDA-generated code often contains return statements and other constructs that need cleaning for Streamlit execution:

```python
def preprocess_lida_code(lida_code, df_name="data"):
    """
    Enhanced preprocessing for LIDA-generated code to handle common issues:
    - Remove top-level return statements
    - Handle function wrapping
    - Clean semicolons and whitespace
    - Extract function calls if needed
    """
```

#### Cleaning Patterns:

```python
return_patterns = [
    r'^\s*return\s+[^;]*;?\s*$',        # Return at start of line
    r'\n\s*return\s+[^;]*;?\s*$',       # Return at end after newline
    r'\s*return\s+[^;]*;?\s*$',         # Any trailing return
    r'return\s+plt\s*;?\s*',            # Specific plt return
    r'return\s+fig\s*;?\s*',            # Specific fig return
    r'return\s+chart\s*;?\s*',          # Specific chart return
    r'return\s+[a-zA-Z_][a-zA-Z0-9_]*\s*;?\s*$',  # Any variable return
]
```

#### Function Detection and Handling:

```python
is_function = bool(re.search(r'def\s+\w+\s*\([^)]*\):', clean_code))

if is_function:
    # Extract function name and add function call
    func_match = re.search(r'def\s+(\w+)\s*\([^)]*\):', clean_code)
    func_name = func_match.group(1) if func_match else "plot"
    clean_code += f"\n{func_name}({df_name})"
```

### Execution Environment

```python
exec_globals = {
    'data': df, 'df': df, 'pd': pd, 'plt': plt,
    'sns': sns, 'px': px, 'alt': alt, 'np': np,
    'go': go if 'go' in globals() else None,
    'make_subplots': make_subplots if 'make_subplots' in globals() else None,
}
exec(clean_code, exec_globals)
```

## Library Selection Strategy

### Best Library Recommendations

The system maintains a comprehensive mapping of chart types to optimal libraries:

```python
def get_best_library_for_chart_type(chart_type):
    """Get the best Python library for each visualization type based on best practices."""
    library_recommendations = {
        "bar": "seaborn",           # High-level API, beautiful default styles
        "line": "matplotlib",       # Flexible, publication-quality, time series
        "scatter": "plotly",        # Interactivity, zoom, selection
        "histogram": "seaborn",     # Statistical features, easy syntax
        "box": "seaborn",          # Statistical features, easy grouping
        "heatmap": "seaborn",      # Annotated, clustering, color palettes
        "pie": "plotly",           # Interactive, easy labels
        "donut": "plotly",         # Interactive, easy labels
        "area": "plotly",          # Stacked areas, interactivity
        "violin": "seaborn",       # Statistical, easy grouping
        "3d": "plotly",            # True 3D support, interactive
        "treemap": "plotly",       # Hierarchical, interactive
        "sunburst": "plotly",      # Hierarchical, interactive
        "facet": "altair",         # Declarative grammar, simple faceting
        "network": "plotly",       # NetworkX for data, Plotly for visualization
        "map": "plotly",           # Easy maps, choropleths, scatter geo
        "statistical": "seaborn",   # Built-in stats, regression, correlation
        "interactive": "plotly",    # Best overall for interactivity
        "waterfall": "plotly",     # Good support for waterfall charts
        "radar": "plotly"          # Good radar chart support
    }
    
    return library_recommendations.get(chart_type.lower(), "plotly")
```

### Library Selection Criteria

1. **Visual Appeal**: Libraries known for beautiful default styling
2. **Functionality**: Best feature set for specific chart types
3. **Interactivity**: Support for zoom, hover, selection
4. **Statistical Features**: Built-in statistical computations
5. **Web Compatibility**: Streamlit integration quality

## Error Handling and Fallbacks

### Multi-Level Fallback System

1. **LIDA Primary**: Full LIDA integration with complete authority
2. **Simple Fallback**: Basic Plotly charts when LIDA fails
3. **Data Table**: Clean data display when visualization fails
4. **Error Messages**: Informative error reporting

### Debug and Diagnostic Tools

```python
def debug_lida_integration(df, user_query=""):
    """Debug LIDA integration and return diagnostic information."""
```

#### Diagnostic Steps:

1. **LIDA Availability Check**: Verify manager initialization
2. **Summary Generation Test**: Test data summarization
3. **Goal Generation Test**: Test goal creation
4. **Visualization Test**: Test chart generation
5. **Code Analysis**: Analyze generated code quality

### Comprehensive Diagnostics

```python
def run_lida_diagnostics():
    """Run comprehensive LIDA diagnostics."""
```

#### Tests Performed:

- LIDA Manager initialization status
- OpenAI API key validation
- Sample data processing
- Code generation and preprocessing
- Library availability checks

## Integration Points

### Main Application Integration

LIDA is integrated at two primary points:

#### 1. Automatic Data Visualization

```python
# After successful data parsing
df = parse_table_from_text(final_response)
if df is not None:
    st.session_state.last_data = df
    with st.spinner("🎨 Creating intelligent visualization..."):
        enhanced_visualization_pipeline(df, user_query=user_text)
```

#### 2. Manual Chart Creation

```python
# When users explicitly request visualizations
create_advanced_chart(df, chart_type="auto", title=None, user_query=user_query)
```

### Data Flow Integration

```
MCP Tool Call → Data Response → Table Parsing → DataFrame Creation → LIDA Pipeline → Visualization Display
```

## API Reference

### Core Functions

#### `enhanced_visualization_pipeline(df, user_query="", enable_lida=True)`

**Primary visualization entry point**

- **Parameters**:
  - `df`: pandas DataFrame with data to visualize
  - `user_query`: User's natural language query for context
  - `enable_lida`: Boolean to enable/disable LIDA processing
- **Returns**: Dictionary with success status and description
- **Raises**: Exception on critical failures

#### `create_advanced_chart(df, chart_type="auto", title=None, user_query="")`

**Direct chart creation interface**

- **Parameters**:
  - `df`: pandas DataFrame with data
  - `chart_type`: Specific chart type or "auto"
  - `title`: Chart title (optional)
  - `user_query`: Context for styling decisions
- **Returns**: None (displays chart directly)

#### `generate_lida_instructions_and_persona(df, user_query="")`

**Generate contextual personas and styling instructions**

- **Parameters**:
  - `df`: Data for analysis
  - `user_query`: User context
- **Returns**: Tuple of (instructions_list, persona_string)

#### `preprocess_lida_code(lida_code, df_name="data")`

**Clean and prepare LIDA-generated code**

- **Parameters**:
  - `lida_code`: Raw code from LIDA
  - `df_name`: Variable name for DataFrame
- **Returns**: Tuple of (cleaned_code, is_function)

#### `debug_lida_integration(df, user_query="")`

**Comprehensive debugging and testing**

- **Parameters**:
  - `df`: Test data
  - `user_query`: Test query
- **Returns**: Dictionary with diagnostic information

### Configuration Functions

#### `get_best_library_for_chart_type(chart_type)`

**Get optimal library recommendation**

- **Parameters**: `chart_type`: Type of chart
- **Returns**: String library name

#### `validate_load_query_args(args)`

**Validate and fix query arguments**

- **Parameters**: `args`: Query arguments dictionary
- **Returns**: Validated arguments or error dictionary

## Configuration

### Environment Setup

```env
# Required
OPENAI_API_KEY=sk-your-key-here

# Optional
MCP_SERVER_URL=http://localhost:8000/mcp
```

### Streamlit Configuration

```python
st.set_page_config(
    page_title="Agentic MCP Chat (OpenAI)", 
    layout="wide"
)
```

### LIDA Configuration

```python
lida_manager = Manager(text_gen=llm("openai"))
```

### Library Availability Detection

```python
# Graceful imports with availability flags
try:
    from lida import Manager, TextGenerationConfig, llm
    LIDA_AVAILABLE = True
except ImportError:
    LIDA_AVAILABLE = False
    st.warning("LIDA library not available.")
```

## Troubleshooting

### Common Issues and Solutions

#### 1. LIDA Not Initializing

**Symptoms**: "LIDA Manager is not initialized" error

**Solutions**:
- Verify OpenAI API key is set correctly
- Check internet connectivity
- Validate API key permissions
- Ensure LIDA library is installed: `pip install lida`

#### 2. Code Execution Errors

**Symptoms**: "Error executing LIDA code" messages

**Solutions**:
- Check if all required libraries are installed
- Review preprocessed code for syntax issues
- Verify DataFrame structure matches code expectations
- Enable debug mode for detailed error information

#### 3. Empty Visualizations

**Symptoms**: "No plot was generated" warning

**Solutions**:
- Verify data has appropriate numeric/categorical columns
- Check if LIDA generated valid plotting code
- Try manual chart creation as fallback
- Review data types and ensure proper formatting

#### 4. Library Import Errors

**Symptoms**: Missing library warnings in sidebar

**Solutions**:
```bash
pip install plotly seaborn matplotlib altair bokeh plotnine
```

#### 5. API Rate Limiting

**Symptoms**: OpenAI API errors during persona/instruction generation

**Solutions**:
- Implement exponential backoff
- Use fallback instructions
- Monitor API usage quotas
- Consider caching generated personas

### Debug Mode

Enable comprehensive debugging:

```python
# Set logging level
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use diagnostic functions
debug_info = debug_lida_integration(df, user_query)
run_lida_diagnostics()
```

### Performance Optimization

1. **Caching**: Cache LIDA results for repeated queries
2. **Preprocessing**: Pre-validate data before LIDA calls
3. **Library Loading**: Lazy load visualization libraries
4. **Error Handling**: Fast-fail for invalid data structures

## Best Practices

### For Users

1. **Clear Queries**: Use specific, descriptive queries for better results
2. **Data Quality**: Ensure clean, well-structured data
3. **Chart Type Specification**: Specify chart types when you have preferences
4. **Context Provision**: Provide business context for better personas

### For Developers

1. **Error Handling**: Always implement fallbacks
2. **Library Support**: Support multiple visualization libraries
3. **Code Safety**: Sanitize and validate all executed code
4. **User Feedback**: Provide clear status messages and progress indicators
5. **Performance**: Monitor API usage and implement caching

### Code Organization

1. **Separation of Concerns**: Keep LIDA logic separate from UI logic
2. **Modularity**: Use small, focused functions
3. **Documentation**: Document all complex algorithms
4. **Testing**: Implement comprehensive test coverage

## Future Enhancements

### Planned Features

1. **Custom Personas**: Allow users to define custom personas
2. **Style Templates**: Pre-defined styling templates
3. **Export Options**: Save visualizations in multiple formats
4. **Collaboration**: Share visualizations and personas
5. **Advanced Analytics**: Statistical analysis integration

### Technical Improvements

1. **Caching Layer**: Redis/Memory caching for LIDA results
2. **Async Processing**: Non-blocking visualization generation
3. **Real-time Updates**: Live data visualization updates
4. **Custom Libraries**: Support for custom visualization libraries
5. **A/B Testing**: Compare different visualization approaches

---

## Conclusion

The LIDA visualization flow provides a comprehensive, intelligent approach to data visualization that combines the power of AI-driven chart generation with contextual awareness and graceful error handling. By giving LIDA complete authority over visualization decisions while providing robust fallbacks, the system ensures both beautiful results and reliable functionality.

For additional support or feature requests, please refer to the project repository or contact the development team.
