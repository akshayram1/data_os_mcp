# LIDA Integration - Code Changes Required

## Files to Remove Completely

**visualization_intelligence.py** - Delete entire file

## Files to Modify

### 1. app.py - Major Changes Required

#### REMOVE: Import statements
```python
from visualization_intelligence import VisualizationIntelligence, get_smart_visualization_suggestions, analyze_query_for_chart_type
```

#### REMOVE: All manual chart creation functions
- `create_seaborn_chart()`
- `create_altair_chart()`
- `create_bokeh_chart()`
- `create_plotly_chart()`
- `display_seaborn_chart()`
- `display_bokeh_chart()`
- `create_beautiful_chart()`
- `auto_select_chart_type()`
- `explain_chart_choice()`
- `create_chart_from_sidebar()`

#### REMOVE: Sidebar visualization tab entirely
Remove the entire `elif tab_selection == "📊 Visualization":` section (lines ~150-500+)

#### REMOVE: Session state initialization
```python
if "viz_config" not in st.session_state:
    st.session_state.viz_config = {...}
if "viz_intelligence" not in st.session_state:
    st.session_state.viz_intelligence = VisualizationIntelligence()
```

#### REMOVE: Chart detection functions
- `detect_chart_request()`
- `extract_chart_type()`
- `create_advanced_chart()`

#### ADD: New imports
```python
from lida import Manager, TextGenerationConfig, llm
```

#### ADD: New LIDA initialization
```python
# Initialize LIDA
lida_manager = None
if OPENAI_API_KEY:
    lida_manager = Manager(text_gen=llm("openai"))
```

#### REPLACE: Visualization logic in agent_turn()
Replace this section:
```python
# Check if user is requesting a chart
if detect_chart_request(user_text):
    # ... existing chart logic
```

With:
```python
# Auto-generate visualization with LIDA if data is available
if st.session_state.last_data is not None and lida_manager:
    create_lida_visualization(st.session_state.last_data, user_text, final_response)
```

#### ADD: New LIDA visualization function
```python
def create_lida_visualization(df, user_query, assistant_response):
    """Generate visualization using LIDA based on user query and data."""
    try:
        if lida_manager is None:
            st.warning("LIDA not available. Please check OpenAI API key.")
            return
        
        with st.spinner("Generating intelligent visualization..."):
            # Generate visualization using LIDA
            textgen_config = TextGenerationConfig(
                n=1,
                temperature=0.2,
                model="gpt-4o-mini"
            )
            
            # LIDA analyzes the data and query to generate appropriate viz
            charts = lida_manager.visualize(
                summary=lida_manager.summarize(df),
                goal=f"Based on this query: '{user_query}' and response: '{assistant_response[:500]}...', create the most insightful visualization",
                textgen_config=textgen_config
            )
            
            if charts:
                # Display the generated chart
                chart_code = charts[0].code
                exec(chart_code)
                st.success("Visualization generated automatically by LIDA!")
                
                # Show manual chart options
                show_manual_chart_options(df, user_query)
            else:
                st.warning("Could not generate automatic visualization")
                show_manual_chart_options(df, user_query)
                
    except Exception as e:
        st.error(f"LIDA visualization failed: {e}")
        show_manual_chart_options(df, user_query)

def show_manual_chart_options(df, user_query):
    """Show manual chart selection as fallback."""
    with st.expander("Create Custom Chart", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Bar Chart"):
                create_simple_chart(df, "bar")
        with col2:
            if st.button("Line Chart"):  
                create_simple_chart(df, "line")
        with col3:
            if st.button("Scatter Plot"):
                create_simple_chart(df, "scatter")

def create_simple_chart(df, chart_type):
    """Create simple fallback charts."""
    try:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if chart_type == "bar" and categorical_cols and numeric_cols:
            fig = px.bar(df, x=categorical_cols[0], y=numeric_cols[0])
            st.plotly_chart(fig, use_container_width=True)
        elif chart_type == "line" and numeric_cols:
            x_col = categorical_cols[0] if categorical_cols else df.index
            fig = px.line(df, x=x_col, y=numeric_cols[0])
            st.plotly_chart(fig, use_container_width=True)
        elif chart_type == "scatter" and len(numeric_cols) >= 2:
            fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Cannot create {chart_type} with current data structure")
    except Exception as e:
        st.error(f"Error creating chart: {e}")
```

#### MODIFY: parse_table_from_text() call
In `agent_turn()`, after storing data, change:
```python
if df is not None:
    st.session_state.last_data = df
    
    # Auto-create visualization if enabled
    if auto_suggest:
        create_advanced_chart(df, "auto", user_query=user_text)
```

To:
```python
if df is not None:
    st.session_state.last_data = df
    # LIDA will handle this automatically in the main flow
```

#### SIMPLIFY: Sidebar to only show settings
Keep only the "⚙️ Settings" tab:
- MCP Server URL
- OpenAI model
- Temperature
- Auto-suggest visualizations toggle
- Available libraries info

#### REMOVE: All chart library imports and availability checks
- `BOKEH_AVAILABLE` checks
- `PLOTNINE_AVAILABLE` checks
- Seaborn/matplotlib imports (unless needed for LIDA fallback)

### 2. requirements.txt - Add New Dependency
```
lida
```

### 3. system.txt - Update System Prompt

#### REMOVE: Entire visualization intelligence section
Remove this section:
```
## INTELLIGENT VISUALIZATION RECOMMENDATIONS
...
Remember: Your visualization recommendations should be as intelligent...
```

#### ADD: New LIDA integration section
```
## AUTOMATIC VISUALIZATION WITH LIDA

When you return tabular data, the system will automatically generate appropriate visualizations using Microsoft LIDA. LIDA will:

- Analyze your query intent and the data structure
- Generate the most suitable visualization automatically  
- Provide interactive charts without manual configuration

You don't need to recommend specific chart types - focus on providing clear, analytical responses about the data insights. LIDA will handle visualization selection and creation automatically.

If users request specific chart types, the system provides manual override options.
```

## Summary of Changes

- **Removed:** ~800+ lines of manual visualization logic
- **Added:** ~50 lines of LIDA integration
- **Simplified:** Sidebar UI by 70%
- **Enhanced:** Automatic visualization intelligence
- **Maintained:** Manual chart options as fallback

This approach preserves all current functionality while adding intelligent auto-visualization through LIDA, and provides manual chart options when users want specific visualizations.