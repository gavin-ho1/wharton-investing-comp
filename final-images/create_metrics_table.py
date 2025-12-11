import pandas as pd
import dataframe_image as dfi

def create_metrics_table(data, output_image='metrics_table.png'):
    """
    Creates a metrics comparison table from a dictionary or DataFrame.
    
    Args:
        data: Dictionary with metric names as keys and lists of values, 
              or a pandas DataFrame with metrics as rows
        output_image: Path for output PNG image
    """
    # Convert data to DataFrame if it's a dictionary
    if isinstance(data, dict):
        df = pd.DataFrame(data)
    else:
        df = data.copy()
    
    # Style the dataframe
    styles = [
        {'selector': 'th', 'props': [
            ('background-color', '#4472C4'),
            ('color', 'white'),
            ('font-weight', 'bold'),
            ('text-align', 'center'),
            ('font-size', '11pt'),
            ('border', '1px solid #4472C4'),
            ('padding', '8px 12px')
        ]},
        {'selector': 'td', 'props': [
            ('border', '1px solid #ddd'),
            ('padding', '8px 12px'),
            ('text-align', 'center'),
            ('font-size', '11pt')
        ]},
        {'selector': '', 'props': [
            ('border-collapse', 'separate'),
            ('border-spacing', '0')
        ]},
        # Left-align the first column (Metric names)
        {'selector': 'td:nth-child(1)', 'props': [
            ('text-align', 'left'),
            ('font-weight', '500')
        ]}
    ]
    
    styled = df.style.hide(axis='index').set_table_styles(styles)
    
    # Export the image with higher DPI for better quality
    dfi.export(styled, output_image, max_cols=-1, max_rows=-1, dpi=150)
    
    print(f"✓ {output_image} created")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")

if __name__ == "__main__":
    # Example data matching the user's request
    data = {
        'Metric': ['CAGR', 'Sharpe Ratio'],
        'Mean (10k Sims)': ['39.18%', '1.56'],
        '10th Percentile Sim': ['24.81%', '0.98'],
        'S&P': ['14.84%', '0.88']
    }
    
    create_metrics_table(data, "metrics_comparison.png")
