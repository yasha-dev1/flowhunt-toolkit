"""
Simple chart generator that avoids binary encoding issues with Plotly.
"""
import json
import uuid
import pandas as pd
from typing import Dict, Any, List


class SimpleChartGenerator:
    """Generate charts using direct JavaScript to avoid binary encoding issues."""
    
    def create_bar_chart(self, categories: List[str], counts: List[int], 
                        title: str = "Bar Chart", colors: List[str] = None) -> str:
        """Create a bar chart with explicit data."""
        chart_id = str(uuid.uuid4())
        
        if colors is None:
            colors = ['#3498db'] * len(categories)
        
        return f'''
        <div id="{chart_id}" class="plotly-graph-div" style="height:400px; width:100%;"></div>
        <script type="text/javascript">
            if (document.getElementById("{chart_id}")) {{
                var data = [{{
                    x: {json.dumps(categories)},
                    y: {json.dumps(counts)},
                    type: 'bar',
                    marker: {{
                        color: {json.dumps(colors)}
                    }}
                }}];
                
                var layout = {{
                    title: '{title}',
                    xaxis: {{ title: 'Category' }},
                    yaxis: {{ title: 'Count' }},
                    height: 400
                }};
                
                Plotly.newPlot("{chart_id}", data, layout);
            }}
        </script>
        '''
    
    def create_pie_chart(self, labels: List[str], values: List[int], 
                        title: str = "Pie Chart", colors: List[str] = None) -> str:
        """Create a pie chart with explicit data."""
        chart_id = str(uuid.uuid4())
        
        if colors is None:
            colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'][:len(labels)]
        
        return f'''
        <div id="{chart_id}" class="plotly-graph-div" style="height:400px; width:100%;"></div>
        <script type="text/javascript">
            if (document.getElementById("{chart_id}")) {{
                var data = [{{
                    labels: {json.dumps(labels)},
                    values: {json.dumps(values)},
                    type: 'pie',
                    marker: {{
                        colors: {json.dumps(colors)}
                    }},
                    textinfo: 'percent+label',
                    textposition: 'inside'
                }}];
                
                var layout = {{
                    title: '{title}',
                    height: 400
                }};
                
                Plotly.newPlot("{chart_id}", data, layout);
            }}
        </script>
        '''
    
    def create_histogram(self, values: List[float], title: str = "Histogram", 
                        color: str = '#3498db', mean_value: float = None) -> str:
        """Create a histogram with explicit data."""
        chart_id = str(uuid.uuid4())
        
        histogram_script = f'''
        <div id="{chart_id}" class="plotly-graph-div" style="height:400px; width:100%;"></div>
        <script type="text/javascript">
            if (document.getElementById("{chart_id}")) {{
                var data = [{{
                    x: {json.dumps(values)},
                    type: 'histogram',
                    marker: {{
                        color: '{color}'
                    }}
                }}];
                
                var layout = {{
                    title: '{title}',
                    xaxis: {{ title: 'Value' }},
                    yaxis: {{ title: 'Frequency' }},
                    height: 400
                }};
        '''
        
        if mean_value is not None:
            histogram_script += f'''
                layout.shapes = [{{
                    type: 'line',
                    x0: {mean_value},
                    x1: {mean_value},
                    y0: 0,
                    y1: 1,
                    yref: 'paper',
                    line: {{
                        color: 'red',
                        width: 2,
                        dash: 'dash'
                    }}
                }}];
                
                layout.annotations = [{{
                    x: {mean_value},
                    y: 0.9,
                    yref: 'paper',
                    text: 'Mean: {mean_value:.2f}',
                    showarrow: false,
                    font: {{ color: 'red' }}
                }}];
            '''
        
        histogram_script += f'''
                Plotly.newPlot("{chart_id}", data, layout);
            }}
        </script>
        '''
        
        return histogram_script


def create_correctness_charts(df: pd.DataFrame) -> Dict[str, str]:
    """Create correctness bar and pie charts from evaluation data."""
    chart_gen = SimpleChartGenerator()
    
    # Clean correctness data
    correctness_mapping = {
        'correct': 'Correct', 'Correct': 'Correct', 'true': 'Correct', '1': 'Correct',
        'incorrect': 'Incorrect', 'Incorrect': 'Incorrect', 'false': 'Incorrect', '0': 'Incorrect'
    }
    
    df_clean = df.copy()
    df_clean['judge_correctness_clean'] = df_clean['judge_correctness'].map(correctness_mapping).fillna('Unknown')
    correctness_counts = df_clean['judge_correctness_clean'].value_counts()
    
    # Extract data as native Python types
    categories = [str(x) for x in correctness_counts.index.tolist()]
    counts = [int(x) for x in correctness_counts.values.tolist()]
    colors = ['#2ecc71' if cat == 'Correct' else '#e74c3c' if cat == 'Incorrect' else '#95a5a6' for cat in categories]
    
    return {
        'bar_chart': chart_gen.create_bar_chart(
            categories, counts, 
            title="Answer Correctness Distribution", 
            colors=colors
        ),
        'pie_chart': chart_gen.create_pie_chart(
            categories, counts,
            title="Answer Correctness Proportion",
            colors=colors
        )
    }


def create_score_histogram(df: pd.DataFrame) -> str:
    """Create score distribution bar chart with individual score values."""
    chart_gen = SimpleChartGenerator()
    
    scores = pd.to_numeric(df['judge_score'], errors='coerce').dropna()
    if len(scores) == 0:
        return "<p>No valid score data available for score distribution.</p>"
    
    # Count occurrences of each score value
    score_counts = scores.value_counts().sort_index()
    
    # Convert to lists for the bar chart
    score_labels = [str(int(score)) if score == int(score) else str(score) for score in score_counts.index]
    counts = score_counts.values.tolist()
    
    # Create colors - highlight mean score in different color
    mean_score = float(scores.mean())
    colors = []
    for score in score_counts.index:
        if abs(score - mean_score) < 0.1:  # Close to mean
            colors.append('#e74c3c')  # Red for mean
        else:
            colors.append('#3498db')  # Blue for others
    
    return chart_gen.create_bar_chart(
        categories=score_labels,
        counts=counts,
        title=f"Score Distribution (Mean: {mean_score:.1f})",
        colors=colors
    )


def create_answer_length_histogram(df: pd.DataFrame) -> str:
    """Create answer length distribution bar chart with intelligent length ranges."""
    chart_gen = SimpleChartGenerator()
    
    if 'answer_length' not in df.columns:
        df = df.copy()
        df['answer_length'] = df['actual_answer'].astype(str).str.len()
    
    lengths = df['answer_length'].dropna()
    if len(lengths) == 0:
        return "<p>No valid answer length data available for length analysis.</p>"
    
    # Create intelligent length ranges based on data distribution
    min_length = int(lengths.min())
    max_length = int(lengths.max())
    mean_length = float(lengths.mean())
    
    # Determine appropriate bin size based on data range
    data_range = max_length - min_length
    if data_range <= 50:
        bin_size = 5  # Small ranges: 5-char bins
    elif data_range <= 200:
        bin_size = 10  # Medium ranges: 10-char bins
    elif data_range <= 500:
        bin_size = 25  # Large ranges: 25-char bins
    else:
        bin_size = 50  # Very large ranges: 50-char bins
    
    # Create bins and count occurrences
    bins = {}
    for length in lengths:
        bin_start = (int(length) // bin_size) * bin_size
        bin_end = bin_start + bin_size - 1
        bin_key = f"{bin_start}-{bin_end}"
        bins[bin_key] = bins.get(bin_key, 0) + 1
    
    # Sort bins by range start
    sorted_bins = sorted(bins.items(), key=lambda x: int(x[0].split('-')[0]))
    
    categories = [bin_range for bin_range, _ in sorted_bins]
    counts = [count for _, count in sorted_bins]
    
    # Create colors - highlight bins containing the mean
    colors = []
    for bin_range, _ in sorted_bins:
        bin_start, bin_end = map(int, bin_range.split('-'))
        if bin_start <= mean_length <= bin_end:
            colors.append('#e74c3c')  # Red for bin containing mean
        else:
            colors.append('#9b59b6')  # Purple for others
    
    return chart_gen.create_bar_chart(
        categories=categories,
        counts=counts,
        title=f"Answer Length Distribution (Mean: {mean_length:.0f} chars)",
        colors=colors
    )
