"""HTML report generation for evaluation results with interactive visualizations."""

import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from .chart_generator import create_correctness_charts, create_score_histogram, create_answer_length_histogram
import json


class EvaluationReportGenerator:
    """Generates interactive HTML reports for evaluation results."""
    
    def __init__(self):
        """Initialize report generator."""
        pass
    
    def generate_html_report(self, results: List[Dict[str, Any]], output_path: Path, 
                           summary_stats: Dict[str, Any]) -> Path:
        """Generate comprehensive HTML report with interactive visualizations.
        
        Args:
            results: List of evaluation results
            output_path: Path where to save the HTML report
            summary_stats: Summary statistics from evaluation
            
        Returns:
            Path to generated HTML report
        """
        if not results:
            raise ValueError("No results provided for report generation")
        
        # Convert results to DataFrame
        df = pd.DataFrame(results)
        
        # Ensure we have the required columns
        required_cols = ['judge_score', 'judge_correctness', 'actual_answer']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 'N/A'
        
        # Add answer length column
        df['answer_length'] = df['actual_answer'].astype(str).str.len()
        
        # Generate all visualizations
        charts = self._generate_all_charts(df, summary_stats)
        
        # Create HTML report
        html_content = self._create_html_template(charts, summary_stats, df)
        
        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path
    
    def _generate_all_charts(self, df: pd.DataFrame, summary_stats: Dict[str, Any]) -> Dict[str, str]:
        """Generate all required charts and return as HTML strings.
        
        Args:
            df: DataFrame with evaluation results
            summary_stats: Summary statistics
            
        Returns:
            Dictionary with chart names as keys and HTML strings as values
        """
        charts = {}
        
        # Generate correctness charts using new chart generator
        correctness_charts = create_correctness_charts(df)
        charts['correctness_bar'] = correctness_charts['bar_chart']
        charts['correctness_pie'] = correctness_charts['pie_chart']
        
        # 3. Score distribution histogram
        charts['score_distribution'] = create_score_histogram(df)
        
        # 4. Top lowest scoring answers
        charts['lowest_scores'] = self._create_top_answers_table(df, lowest=True, top_n=10)
        # 5. Top highest scoring answers
        charts['highest_scores'] = self._create_top_answers_table(df, lowest=False, top_n=10)
        
        # 6. Answer length distribution
        charts['answer_length'] = create_answer_length_histogram(df)
        
        return charts
    
    def _create_top_answers_table(self, df: pd.DataFrame, lowest: bool = True, top_n: int = 5) -> str:
        """Create table showing top answers with lowest or highest scores."""
        if 'judge_score' not in df.columns:
            return "<p>No score data available for table generation.</p>"
        
        # Convert scores to numeric and handle errors
        df_clean = df.copy()
        df_clean['judge_score_numeric'] = pd.to_numeric(df_clean['judge_score'], errors='coerce')
        df_clean = df_clean.dropna(subset=['judge_score_numeric'])
        
        if len(df_clean) == 0:
            return "<p>No valid score data available for table generation.</p>"
        
        # Sort by score
        if lowest:
            df_sorted = df_clean.nsmallest(top_n, 'judge_score_numeric')
            title = f"Top {top_n} Lowest Scoring Answers"
        else:
            df_sorted = df_clean.nlargest(top_n, 'judge_score_numeric')
            title = f"Top {top_n} Highest Scoring Answers"
        
        # Create HTML table
        table_html = f'''
        <div class="table-container">
            <h3>{title}</h3>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Question</th>
                        <th>Expected Answer</th>
                        <th>Actual Answer</th>
                        <th>Score</th>
                        <th>Judge Reasoning</th>
                    </tr>
                </thead>
                <tbody>
        '''
        
        import html
        for _, row in df_sorted.iterrows():
            question = html.escape(str(row.get('question', 'N/A')))
            expected = html.escape(str(row.get('expected_answer', 'N/A')))
            actual = html.escape(str(row.get('actual_answer', 'N/A')))
            score = row['judge_score_numeric']
            reasoning = html.escape(str(row.get('judge_reasoning', 'N/A')))
            
            table_html += f'''
                    <tr>
                        <td>{question}</td>
                        <td>{expected}</td>
                        <td>{actual}</td>
                        <td>{score:.1f}</td>
                        <td>{reasoning}</td>
                    </tr>
            '''
        
        table_html += '''
                </tbody>
            </table>
        </div>
        '''
        
        return table_html
    
    def _create_html_template(self, charts: Dict[str, str], summary_stats: Dict[str, Any], 
                            df: pd.DataFrame) -> str:
        """Create complete HTML template with all charts and statistics."""
        
        # Calculate additional stats
        total_questions = len(df)
        accuracy = summary_stats.get('accuracy', 0) * 100
        avg_score = summary_stats.get('average_score', 0)
        median_score = summary_stats.get('median_score', 0)
        error_rate = summary_stats.get('error_rate', 0) * 100
        avg_answer_length = df['answer_length'].mean() if 'answer_length' in df.columns else 0
        
        html_template = f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FlowHunt Evaluation Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 40px;
            border-bottom: 2px solid rgb(59, 130, 246);
            padding-bottom: 20px;
        }}
        .header-logo {{
            height: 60px;
            width: auto;
        }}
        .header-content {{
            flex: 1;
            text-align: center;
        }}
        .header h1 {{
            color: rgb(59, 130, 246);
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            color: #7f8c8d;
            margin: 10px 0 0 0;
            font-size: 1.1em;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .stat-card h3 {{
            margin: 0 0 10px 0;
            font-size: 2.2em;
            font-weight: bold;
        }}
        .stat-card p {{
            margin: 0;
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section h2 {{
            color: rgb(59, 130, 246);
            border-bottom: 2px solid rgb(59, 130, 246);
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .chart-container {{
            background-color: #fff;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .table-container {{
            background-color: #fff;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow-x: auto;
        }}
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        .data-table th, .data-table td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
            vertical-align: top;
            word-wrap: break-word;
            max-width: 300px;
        }}
        .data-table th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        .data-table tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .data-table tr:hover {{
            background-color: #f5f5f5;
        }}
        @media (max-width: 768px) {{
            .container {{
                padding: 15px;
            }}
            .stats-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <svg class="header-logo" width="530" height="110" viewBox="0 0 530 110" fill="none" xmlns="http://www.w3.org/2000/svg">
                <g clip-path="url(#clip0_8_2)">
                    <path d="M36.369 175.282L24.2163 203.986C22.1071 208.969 23.073 214.948 27.1337 219.014C29.8048 221.688 33.3037 223.02 36.8027 223.02C40.3016 223.02 43.8006 221.688 46.4716 219.014L58.023 207.449L101.627 163.787C103.647 161.764 102.218 158.32 99.3599 158.32H74.5815C74.4336 158.32 74.2858 158.3 74.1281 158.3C48.0289 158.3 26.8578 136.8 27.3506 110.563C27.8335 84.9175 49.2905 64.6304 74.9067 64.6304H127.785C128.633 64.6304 129.451 64.295 130.052 63.6931L151.006 42.7153C153.027 40.6925 151.598 37.2488 148.739 37.2488H75.1531C34.0134 37.2488 -0.365082 70.9455 -0.000396729 112.131C0.236145 138.98 14.7839 162.454 36.3591 175.302L36.369 175.282ZM199.992 158.31C225.608 158.31 247.065 138.023 247.548 112.378C248.031 86.7331 226.87 64.6403 200.77 64.6403C200.613 64.6403 200.445 64.6206 200.287 64.6206H175.529C172.68 64.6206 171.251 61.167 173.262 59.1541L219.103 13.2615H219.093L228.121 4.20336C233.276 -0.957219 241.664 -1.50979 247.124 3.33504C251.707 7.39048 252.88 13.7154 250.662 18.945L238.51 47.639C260.105 60.4763 274.662 83.9505 274.909 110.799C275.273 151.985 240.895 185.692 199.755 185.692H126.159C123.31 185.692 121.881 182.238 123.892 180.225L144.846 159.248C145.447 158.646 146.266 158.31 147.113 158.31H200.002H199.992ZM186.617 87.1771C199.696 87.1771 210.301 97.7943 210.301 110.888C210.301 123.982 199.696 134.599 186.617 134.599C173.538 134.599 162.932 123.982 162.932 110.888C162.932 97.7943 173.538 87.1771 186.617 87.1771ZM89.829 87.1673C102.908 87.1673 113.513 97.7844 113.513 110.878C113.513 123.972 102.908 134.589 89.829 134.589C76.7498 134.589 66.1445 123.972 66.1445 110.878C66.1445 97.7844 76.7498 87.1673 89.829 87.1673Z" fill="url(#paint0_linear_8_2)"/>
                    <path d="M1058.62 83.453V100.945H1008.11V83.453H1058.62ZM1019.58 63.3373H1042.84V141.614C1042.84 143.764 1043.16 145.44 1043.82 146.643C1044.48 147.809 1045.39 148.629 1046.55 149.103C1047.75 149.576 1049.13 149.813 1050.7 149.813C1051.79 149.813 1052.88 149.722 1053.98 149.54C1055.07 149.321 1055.91 149.157 1056.49 149.048L1060.15 166.376C1058.98 166.74 1057.34 167.159 1055.23 167.633C1053.12 168.143 1050.55 168.453 1047.53 168.562C1041.93 168.781 1037.01 168.034 1032.79 166.321C1028.6 164.608 1025.35 161.948 1023.02 158.34C1020.69 154.733 1019.54 150.178 1019.58 144.675V63.3373Z" fill="#111928"/>
                    <path d="M940.083 118.874V167.414H916.823V83.4531H938.991V98.2666H939.974C941.83 93.3834 944.942 89.5206 949.311 86.6782C953.679 83.7993 958.975 82.3599 965.199 82.3599C971.024 82.3599 976.101 83.6353 980.433 86.1862C984.765 88.7371 988.132 92.3813 990.534 97.1187C992.937 101.82 994.138 107.432 994.138 113.955V167.414H970.878V118.109C970.914 112.971 969.604 108.962 966.947 106.083C964.289 103.168 960.631 101.71 955.972 101.71C952.841 101.71 950.075 102.385 947.672 103.733C945.306 105.081 943.45 107.049 942.103 109.636C940.793 112.187 940.119 115.267 940.083 118.874Z" fill="#111928"/>
                    <path d="M873.42 131.665V83.4531H896.68V167.415H874.348V152.164H873.474C871.582 157.083 868.433 161.037 864.028 164.025C859.66 167.014 854.328 168.508 848.03 168.508C842.425 168.508 837.492 167.232 833.233 164.681C828.975 162.13 825.644 158.505 823.241 153.804C820.875 149.103 819.674 143.472 819.638 136.913V83.4531H842.898V132.759C842.934 137.715 844.263 141.632 846.884 144.511C849.505 147.39 853.017 148.829 857.422 148.829C860.225 148.829 862.845 148.192 865.284 146.916C867.723 145.604 869.689 143.673 871.181 141.122C872.71 138.571 873.456 135.419 873.42 131.665Z" fill="#111928"/>
                    <path d="M703.82 167.414V55.4659H727.462V101.656H775.457V55.4659H799.044V167.414H775.457V121.17H727.462V167.414H703.82Z" fill="#111928"/>
                    <path d="M587.607 167.415L564.783 83.4531H588.317L601.312 139.865H602.076L615.617 83.4531H638.713L652.473 139.537H653.183L665.959 83.4531H689.438L666.669 167.415H642.044L627.629 114.611H626.592L612.177 167.415H587.607Z" fill="#111928"/>
                    <path d="M515.224 169.054C506.743 169.054 499.408 167.251 493.22 163.643C487.068 159.999 482.318 154.933 478.969 148.447C475.62 141.924 473.946 134.362 473.946 125.762C473.946 117.089 475.62 109.509 478.969 103.022C482.318 96.4992 487.068 91.4338 493.22 87.8261C499.408 84.1819 506.743 82.3599 515.224 82.3599C523.706 82.3599 531.022 84.1819 537.174 87.8261C543.362 91.4338 548.131 96.4992 551.479 103.022C554.828 109.509 556.503 117.089 556.503 125.762C556.503 134.362 554.828 141.924 551.479 148.447C548.131 154.933 543.362 159.999 537.174 163.643C531.022 167.251 523.706 169.054 515.224 169.054ZM515.333 151.016C519.192 151.016 522.413 149.923 524.998 147.736C527.582 145.513 529.53 142.488 530.84 138.662C532.187 134.836 532.86 130.481 532.86 125.598C532.86 120.715 532.187 116.36 530.84 112.533C529.53 108.707 527.582 105.682 524.998 103.46C522.413 101.237 519.192 100.125 515.333 100.125C511.439 100.125 508.163 101.237 505.505 103.460C502.884 105.682 500.901 108.707 499.554 112.533C498.243 116.36 497.588 120.715 497.588 125.598C497.588 130.481 498.243 134.836 499.554 138.662C500.901 142.488 502.884 145.513 505.505 147.736C508.163 149.923 511.439 151.016 515.333 151.016Z" fill="#111928"/>
                    <path d="M457.161 55.4659V167.414H433.901V55.4659H457.161Z" fill="#111928"/>
                    <path d="M342.859 167.414V55.4659H416.898V74.9804H366.501V101.656H411.983V121.17H366.501V167.414H342.859Z" fill="#111928"/>
                </g>
                <defs>
                    <linearGradient id="paint0_linear_8_2" x1="-0.000215382" y1="-8.42837e-05" x2="304.637" y2="162.153" gradientUnits="userSpaceOnUse">
                        <stop stop-color="#0084FF"/>
                        <stop offset="1" stop-color="#1A56DB"/>
                    </linearGradient>
                    <clipPath id="clip0_8_2">
                        <rect width="1060" height="223" fill="white"/>
                    </clipPath>
                </defs>
            </svg>
            <div>
                <h1>FlowHunt Evaluation Report</h1>
                <p>Comprehensive analysis of flow evaluation results</p>
            </div>
            <div style="width: 60px;"></div> <!-- Spacer for balance -->
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <h3>{total_questions}</h3>
                <p>Total Questions</p>
            </div>
            <div class="stat-card">
                <h3>{accuracy:.1f}%</h3>
                <p>Accuracy</p>
            </div>
            <div class="stat-card">
                <h3>{avg_score:.1f}</h3>
                <p>Average Score</p>
            </div>
            <div class="stat-card">
                <h3>{median_score:.1f}</h3>
                <p>Median Score</p>
            </div>
            <div class="stat-card">
                <h3>{error_rate:.1f}%</h3>
                <p>Error Rate</p>
            </div>
            <div class="stat-card">
                <h3>{avg_answer_length:.0f}</h3>
                <p>Avg Answer Length</p>
            </div>
        </div>
        
        <div class="section">
            <h2>📈 Answer Correctness Analysis</h2>
            <div class="chart-container">
                {charts.get('correctness_bar', '<p>Bar chart not available</p>')}
            </div>
            <div class="chart-container">
                {charts.get('correctness_pie', '<p>Pie chart not available</p>')}
            </div>
        </div>
        
        <div class="section">
            <h2>📊 Score Distribution</h2>
            <div class="chart-container">
                {charts.get('score_distribution', '<p>Score distribution chart not available</p>')}
            </div>
        </div>
        
        <div class="section">
            <h2>📋 Top Scoring Answers</h2>
            {charts.get('lowest_scores', '<p>Lowest scores table not available</p>')}
            {charts.get('highest_scores', '<p>Highest scores table not available</p>')}
        </div>
        
        <div class="section">
            <h2>📏 Answer Length Analysis</h2>
            <div class="chart-container">
                {charts.get('answer_length', '<p>Answer length chart not available</p>')}
            </div>
        </div>
        
        <div class="section">
            <h2>📝 Report Summary</h2>
            <div class="chart-container">
                <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p><strong>Total Evaluations:</strong> {total_questions}</p>
                <p><strong>Overall Performance:</strong> {accuracy:.1f}% accuracy with an average score of {avg_score:.1f}</p>
                <p><strong>Data Quality:</strong> {error_rate:.1f}% error rate indicates {"excellent" if error_rate < 5 else "good" if error_rate < 15 else "moderate"} data quality</p>
            </div>
        </div>
    </div>
</body>
</html>
        '''
        
        return html_template
