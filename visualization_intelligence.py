import pandas as pd
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ChartRecommendation:
    """Structure for chart recommendations"""
    name: str
    chart_type: str
    description: str
    analytical_value: str
    when_to_use: str
    priority: int  # 1 = primary, 2 = secondary, 3 = alternative

class VisualizationIntelligence:
    """Intelligent visualization recommendation engine"""
    
    def __init__(self):
        self.intent_keywords = {
            'trend': ['over time', 'trending', 'growth', 'decline', 'month', 'year', 'daily', 'weekly', 'quarterly', 'seasonal'],
            'comparison': ['by country', 'by region', 'by city', 'top', 'highest', 'lowest', 'compare', 'versus', 'best', 'worst'],
            'distribution': ['segmentation', 'breakdown', 'categories', 'groups', 'distribution', 'share', 'proportion'],
            'geographic': ['country', 'region', 'city', 'state', 'location', 'geographic', 'map', 'spatial'],
            'correlation': ['relationship', 'correlation', 'impact', 'versus', 'dependent', 'influence'],
            'performance': ['kpi', 'target', 'performance', 'metrics', 'goal', 'benchmark', 'achievement']
        }
        
        self.chart_library = {
            'line': {
                'name': 'Line Chart with Trend Analysis',
                'best_for': ['temporal data', 'continuous trends', 'growth patterns'],
                'analytical_value': 'Reveals trends, patterns, and temporal relationships',
                'min_rows': 3, 'max_rows': 1000,
                'requires': ['time_column', 'numeric_column']
            },
            'area': {
                'name': 'Area Chart',
                'best_for': ['temporal magnitude', 'cumulative impact', 'volume emphasis'],
                'analytical_value': 'Emphasizes magnitude of change and cumulative effects',
                'min_rows': 3, 'max_rows': 500,
                'requires': ['time_column', 'numeric_column']
            },
            'horizontal_bar': {
                'name': 'Horizontal Bar Chart',
                'best_for': ['category comparison', 'ranking', 'many categories'],
                'analytical_value': 'Clear comparison with readable labels for many items',
                'min_rows': 2, 'max_rows': 50,
                'requires': ['categorical_column', 'numeric_column']
            },
            'column': {
                'name': 'Column Chart',
                'best_for': ['category comparison', 'few categories', 'traditional comparison'],
                'analytical_value': 'Direct numerical comparison between categories',
                'min_rows': 2, 'max_rows': 15,
                'requires': ['categorical_column', 'numeric_column']
            },
            'treemap': {
                'name': 'Treemap',
                'best_for': ['hierarchical data', 'proportional relationships', 'space efficiency'],
                'analytical_value': 'Shows hierarchy and proportional relationships efficiently',
                'min_rows': 3, 'max_rows': 100,
                'requires': ['categorical_column', 'numeric_column']
            },
            'donut': {
                'name': 'Donut Chart',
                'best_for': ['part-to-whole', 'proportional data', 'focus on categories'],
                'analytical_value': 'Highlights proportional relationships and category focus',
                'min_rows': 2, 'max_rows': 10,
                'requires': ['categorical_column', 'numeric_column']
            },
            'scatter': {
                'name': 'Scatter Plot with Trend Line',
                'best_for': ['correlation analysis', 'relationship patterns', 'outlier detection'],
                'analytical_value': 'Reveals relationships, correlations, and data patterns',
                'min_rows': 10, 'max_rows': 1000,
                'requires': ['two_numeric_columns']
            },
            'bubble': {
                'name': 'Bubble Chart',
                'best_for': ['three-variable analysis', 'size relationships', 'complex comparisons'],
                'analytical_value': 'Shows relationships between three variables simultaneously',
                'min_rows': 5, 'max_rows': 100,
                'requires': ['three_numeric_columns']
            },
            'heatmap': {
                'name': 'Heatmap',
                'best_for': ['correlation matrix', 'pattern identification', 'density visualization'],
                'analytical_value': 'Reveals patterns, correlations, and intensity distributions',
                'min_rows': 3, 'max_rows': 50,
                'requires': ['multiple_numeric_columns']
            },
            'waterfall': {
                'name': 'Waterfall Chart',
                'best_for': ['sequential changes', 'contribution analysis', 'variance breakdown'],
                'analytical_value': 'Shows cumulative effect of sequential changes',
                'min_rows': 3, 'max_rows': 20,
                'requires': ['categorical_column', 'numeric_column']
            },
            'slope': {
                'name': 'Slope Chart',
                'best_for': ['period comparison', 'change emphasis', 'before/after analysis'],
                'analytical_value': 'Emphasizes change and comparison between two time periods',
                'min_rows': 2, 'max_rows': 20,
                'requires': ['two_time_periods', 'numeric_column']
            },
            'lollipop': {
                'name': 'Lollipop Chart',
                'best_for': ['clean comparison', 'reduced visual clutter', 'ranking'],
                'analytical_value': 'Clean comparison without visual noise',
                'min_rows': 3, 'max_rows': 30,
                'requires': ['categorical_column', 'numeric_column']
            }
        }

    def analyze_query_context(self, user_query: str, assistant_response: str = "") -> List[str]:
        """Extract analytical intent from user query and assistant response"""
        detected_intents = []
        combined_text = f"{user_query} {assistant_response}".lower()
        
        for intent, keywords in self.intent_keywords.items():
            if any(keyword in combined_text for keyword in keywords):
                detected_intents.append(intent)
        
        return detected_intents

    def analyze_data_characteristics(self, df: pd.DataFrame) -> Dict[str, any]:
        """Analyze data structure and characteristics"""
        if df is None or df.empty:
            return {}
        
        characteristics = {
            'row_count': len(df),
            'column_count': len(df.columns),
            'has_datetime': len(df.select_dtypes(include=['datetime64']).columns) > 0,
            'datetime_columns': list(df.select_dtypes(include=['datetime64']).columns),
            'numeric_columns': list(df.select_dtypes(include=['number']).columns),
            'categorical_columns': list(df.select_dtypes(include=['object']).columns),
            'has_geographic': any('country' in col.lower() or 'city' in col.lower() or 'state' in col.lower() 
                                for col in df.columns),
            'geographic_columns': [col for col in df.columns 
                                 if any(geo in col.lower() for geo in ['country', 'city', 'state', 'region'])],
            'is_hierarchical': len(df.columns) > 2 and len(df.select_dtypes(include=['object']).columns) > 1,
            'has_multiple_measures': len(df.select_dtypes(include=['number']).columns) > 1
        }
        
        return characteristics

    def get_chart_recommendations(self, df: pd.DataFrame, user_query: str, 
                                assistant_response: str = "") -> List[ChartRecommendation]:
        """Generate intelligent chart recommendations based on context and data"""
        
        if df is None or df.empty:
            return []
        
        intents = self.analyze_query_context(user_query, assistant_response)
        characteristics = self.analyze_data_characteristics(df)
        recommendations = []
        
        # Trend Analysis Intent
        if 'trend' in intents and characteristics['has_datetime']:
            recommendations.append(ChartRecommendation(
                name="Line Chart with Trend Analysis",
                chart_type="line",
                description="Shows temporal patterns and growth trends",
                analytical_value="Reveals growth rates, seasonal patterns, and performance inflection points",
                when_to_use="Perfect for time-series data with consistent intervals",
                priority=1
            ))
            
            if len(characteristics['numeric_columns']) == 1:
                recommendations.append(ChartRecommendation(
                    name="Area Chart",
                    chart_type="area",
                    description="Emphasizes magnitude and cumulative impact",
                    analytical_value="Highlights volume changes and cumulative effects over time",
                    when_to_use="When you want to emphasize the magnitude of change",
                    priority=2
                ))

        # Comparison Analysis Intent
        elif 'comparison' in intents or (len(characteristics['categorical_columns']) > 0 and len(characteristics['numeric_columns']) > 0):
            if characteristics['row_count'] > 10:
                recommendations.append(ChartRecommendation(
                    name="Horizontal Bar Chart",
                    chart_type="horizontal_bar",
                    description="Clean comparison with readable labels",
                    analytical_value="Clear ranking and performance comparison for many categories",
                    when_to_use="Best for comparing many categories with long labels",
                    priority=1
                ))
            else:
                recommendations.append(ChartRecommendation(
                    name="Column Chart",
                    chart_type="column",
                    description="Traditional categorical comparison",
                    analytical_value="Direct numerical comparison between categories",
                    when_to_use="Ideal for comparing few categories with clear values",
                    priority=1
                ))
                
            recommendations.append(ChartRecommendation(
                name="Lollipop Chart",
                chart_type="lollipop",
                description="Modern, clean comparison without visual clutter",
                analytical_value="Reduces visual noise while maintaining comparison clarity",
                when_to_use="When you want a cleaner alternative to bar charts",
                priority=2
            ))

        # Distribution Analysis Intent
        elif 'distribution' in intents or characteristics['is_hierarchical']:
            recommendations.append(ChartRecommendation(
                name="Treemap",
                chart_type="treemap",
                description="Hierarchical proportional relationships",
                analytical_value="Shows market share, segments, and proportional hierarchy efficiently",
                when_to_use="Perfect for showing parts of a whole with hierarchy",
                priority=1
            ))
            
            if characteristics['row_count'] <= 10:
                recommendations.append(ChartRecommendation(
                    name="Donut Chart",
                    chart_type="donut",
                    description="Modern proportional visualization",
                    analytical_value="Highlights category proportions with modern aesthetic",
                    when_to_use="Best for showing proportional data with few categories",
                    priority=2
                ))

        # Geographic Analysis Intent
        elif 'geographic' in intents or characteristics['has_geographic']:
            recommendations.append(ChartRecommendation(
                name="Bubble Map Visualization",
                chart_type="bubble_map",
                description="Geographic data with quantity context",
                analytical_value="Shows both location and magnitude relationships",
                when_to_use="When geographic context is important for understanding data",
                priority=1
            ))
            
            recommendations.append(ChartRecommendation(
                name="Horizontal Bar Chart",
                chart_type="horizontal_bar",
                description="Geographic comparison without map complexity",
                analytical_value="Clear regional comparison focusing on values rather than geography",
                when_to_use="When you want to focus on values rather than geographic patterns",
                priority=2
            ))

        # Correlation Analysis Intent
        elif 'correlation' in intents and len(characteristics['numeric_columns']) >= 2:
            recommendations.append(ChartRecommendation(
                name="Scatter Plot with Trend Line",
                chart_type="scatter",
                description="Two-variable relationship analysis",
                analytical_value="Reveals correlations, patterns, and outliers in relationships",
                when_to_use="Perfect for identifying relationships between two numeric variables",
                priority=1
            ))
            
            if len(characteristics['numeric_columns']) >= 3:
                recommendations.append(ChartRecommendation(
                    name="Bubble Chart",
                    chart_type="bubble",
                    description="Three-variable relationship analysis",
                    analytical_value="Shows complex relationships between three variables simultaneously",
                    when_to_use="When you need to analyze three variables and their relationships",
                    priority=2
                ))

        # Performance Analysis Intent
        elif 'performance' in intents:
            recommendations.append(ChartRecommendation(
                name="Bullet Chart",
                chart_type="bullet",
                description="Performance vs targets visualization",
                analytical_value="Shows actual performance against targets and benchmarks",
                when_to_use="Perfect for KPI dashboards and performance monitoring",
                priority=1
            ))

        # Default recommendations if no specific intent detected
        if not recommendations:
            if characteristics['has_datetime']:
                recommendations.append(ChartRecommendation(
                    name="Line Chart",
                    chart_type="line",
                    description="Time-based data visualization",
                    analytical_value="Shows trends and patterns over time",
                    when_to_use="Default for temporal data",
                    priority=1
                ))
            elif len(characteristics['categorical_columns']) > 0 and len(characteristics['numeric_columns']) > 0:
                recommendations.append(ChartRecommendation(
                    name="Bar Chart",
                    chart_type="bar",
                    description="Category comparison",
                    analytical_value="Compares values across categories",
                    when_to_use="Default for categorical comparisons",
                    priority=1
                ))
        
        return sorted(recommendations, key=lambda x: x.priority)

    def generate_recommendation_text(self, recommendations: List[ChartRecommendation], 
                                   query_context: str, data_summary: str = "") -> str:
        """Generate intelligent recommendation text"""
        
        if not recommendations:
            return "I don't have specific visualization recommendations for this data structure."
        
        primary_rec = recommendations[0]
        secondary_recs = [r for r in recommendations[1:3] if r.priority <= 3]  # Top 2 alternatives
        
        recommendation_text = f"""Based on your query analysis, this data reveals {data_summary}. I recommend:

**Primary Visualization**: {primary_rec.name}
- Why: {primary_rec.when_to_use}
- Shows: {primary_rec.analytical_value}

**Alternative Options**:"""
        
        for rec in secondary_recs:
            recommendation_text += f"\n- {rec.name}: {rec.analytical_value}"
        
        recommendation_text += f"\n\nWould you like me to create the {primary_rec.name} to highlight the key insights, or would you prefer one of the alternative visualizations?"
        
        return recommendation_text

    def detect_key_insights(self, df: pd.DataFrame, user_query: str) -> str:
        """Detect key insights from data to enhance recommendations"""
        
        if df is None or df.empty:
            return "data patterns"
        
        insights = []
        
        # Check for trends in time-based data
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(datetime_cols) > 0 and len(numeric_cols) > 0:
            insights.append("temporal performance patterns")
        
        # Check for distribution patterns
        if len(df) > 1 and len(numeric_cols) > 0:
            for col in numeric_cols:
                if df[col].nunique() > 1:
                    insights.append("performance variations")
                    break
        
        # Check for geographic patterns
        if any('country' in col.lower() or 'city' in col.lower() for col in df.columns):
            insights.append("geographic distribution patterns")
        
        # Check for comparative data
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            insights.append("comparative performance across categories")
        
        return " and ".join(insights) if insights else "key data patterns"

# Usage example functions
def get_smart_visualization_suggestions(df: pd.DataFrame, user_query: str, 
                                      assistant_response: str = "") -> str:
    """Main function to get intelligent visualization suggestions"""
    
    viz_intelligence = VisualizationIntelligence()
    recommendations = viz_intelligence.get_chart_recommendations(df, user_query, assistant_response)
    key_insights = viz_intelligence.detect_key_insights(df, user_query)
    
    return viz_intelligence.generate_recommendation_text(recommendations, user_query, key_insights)

def analyze_query_for_chart_type(user_message: str) -> Dict[str, any]:
    """Analyze user message to determine if they want a specific chart type"""
    chart_requests = {
        'line': ['line chart', 'line graph', 'trend chart', 'time series'],
        'bar': ['bar chart', 'bar graph', 'column chart'],
        'pie': ['pie chart', 'donut chart', 'proportional'],
        'scatter': ['scatter plot', 'correlation chart', 'relationship chart'],
        'area': ['area chart', 'filled line'],
        'heatmap': ['heatmap', 'heat map', 'correlation matrix'],
        'treemap': ['treemap', 'tree map', 'hierarchical'],
        'bubble': ['bubble chart', 'bubble plot']
    }
    
    message_lower = user_message.lower()
    requested_charts = []
    
    for chart_type, keywords in chart_requests.items():
        if any(keyword in message_lower for keyword in keywords):
            requested_charts.append(chart_type)
    
    return {
        'has_chart_request': len(requested_charts) > 0,
        'requested_types': requested_charts,
        'is_generic_request': any(word in message_lower for word in ['chart', 'graph', 'visualize', 'plot'])
    }