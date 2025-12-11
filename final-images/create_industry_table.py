import pandas as pd
import dataframe_image as dfi
import numpy as np

def create_industry_table(input_csv, output_image='industry_table.png', max_rows=None, two_column_layout=True):
    """
    Reads a CSV file with industry allocations and creates a table image.
    
    Args:
        input_csv: Path to input CSV file with industry allocations
        output_image: Path for output PNG image
        max_rows: Maximum number of rows to display (None = all rows)
        two_column_layout: If True, split table into two columns side-by-side. If False, single unified table.
    """
    # Read the CSV
    df_raw = pd.read_csv(input_csv)
    
    # Parse the data - the CSV has a specific structure with two sets of columns
    # Columns: Ideal Industry Weights,,, Final Industry Weights,
    # We need to extract Industry and Weight from both sections
    
    # Get the first set (Ideal Industry Weights)
    ideal_industry = df_raw.iloc[:, 0].values  # Industry column
    ideal_weights = df_raw.iloc[:, 1].values   # Weight column
    
    # Get the second set (Final Industry Weights) - skip empty columns
    final_industry = df_raw.iloc[:, 3].values  # Industry column
    final_weights = df_raw.iloc[:, 4].values   # Weight column
    
    # Create a clean dataframe
    df = pd.DataFrame({
        'Industry': ideal_industry,
        'Recommended Portfolio': ideal_weights,
        'WInS Final Portfolio': final_weights
    })
    
    # Skip the header row if it contains "Industry" text
    if df.iloc[0, 0] == 'Industry':
        df = df.iloc[1:].reset_index(drop=True)
    
    # Convert percentage strings to floats for sorting
    def parse_percentage(val):
        if isinstance(val, str) and '%' in val:
            return float(val.replace('%', ''))
        return float(val) if val else 0.0
    
    df['Recommended Portfolio Numeric'] = df['Recommended Portfolio'].apply(parse_percentage)
    
    # Sort by Recommended Portfolio (largest first)
    df = df.sort_values('Recommended Portfolio Numeric', ascending=False).reset_index(drop=True)
    
    # Remove the numeric column used for sorting
    df = df.drop('Recommended Portfolio Numeric', axis=1)
    
    # Limit rows if specified
    if max_rows is not None and max_rows > 0:
        df = df.head(max_rows)
    
    total_rows = len(df)
    
    if two_column_layout:
        # Calculate rows per column (split total rows in half, rounding up for left column)
        rows_per_col = (total_rows + 1) // 2
        
        # Split into left and right halves
        df_left = df.iloc[:rows_per_col].copy()
        df_right = df.iloc[rows_per_col:].copy() if rows_per_col < total_rows else pd.DataFrame(columns=df.columns)
        
        # Reset indices
        df_left.reset_index(drop=True, inplace=True)
        df_right.reset_index(drop=True, inplace=True)
        
        # Rename columns for right half to avoid duplicates (using zero-width space)
        df_right.columns = [col + '\u200b' for col in df_right.columns]
        
        # Create a spacer column between the two halves
        spacer = pd.DataFrame({'': [''] * len(df_left)})
        
        # Concatenate horizontally with spacer in the middle
        combined = pd.concat([df_left, spacer, df_right], axis=1).fillna('')
        
        # Get the number of columns in the left half (plus 1 for spacer)
        num_cols_left = len(df.columns)
        spacer_position = num_cols_left + 1
    else:
        # Single unified table
        combined = df.copy()
        spacer_position = None
    
    # Identify columns for right alignment (percentage columns)
    def get_right_align_selectors(dataframe):
        selectors = []
        for idx, col in enumerate(dataframe.columns, start=1):
            # Right-align the portfolio columns (not Industry)
            if 'Portfolio' in col or '%' in str(col):
                selectors.append(f'td:nth-child({idx})')
        return selectors
    
    # Style the dataframe
    right_align_selectors = get_right_align_selectors(combined)
    
    styles = [
        {'selector': 'th', 'props': [
            ('background-color', '#4472C4'),
            ('color', 'white'),
            ('font-weight', 'bold'),
            ('text-align', 'center'),
            ('font-size', '9pt'),
            ('border', '1px solid #4472C4'),
            ('padding', '4px')
        ]},
        {'selector': 'td', 'props': [
            ('border', '1px solid #ddd'),
            ('padding', '4px'),
            ('text-align', 'left')
        ]},
        {'selector': '', 'props': [
            ('border-collapse', 'separate'),
            ('border-spacing', '0')
        ]}
    ]
    
    # Add spacer column styling only for two-column layout
    if two_column_layout and spacer_position is not None:
        styles.extend([
            # Hide the spacer column header
            {'selector': f'th:nth-child({spacer_position})', 'props': [
                ('background-color', 'white'),
                ('border', 'none'),
                ('width', '20px'),
                ('min-width', '20px'),
                ('padding', '0'),
                ('color', 'white')
            ]},
            # Hide the spacer column cells
            {'selector': f'td:nth-child({spacer_position})', 'props': [
                ('border', 'none'),
                ('width', '20px'),
                ('min-width', '20px'),
                ('background-color', 'white')
            ]}
        ])
    
    # Add right alignment for numeric columns
    for selector in right_align_selectors:
        styles.append({
            'selector': selector,
            'props': [('text-align', 'right')]
        })
    
    styled = combined.style.hide(axis='index').set_properties(**{
        'font-size': '9pt'
    }).set_table_styles(styles).apply(lambda x: ['background-color: white; border: none; color: white' if x.name == '' else '' for i in x], axis=0)
    
    # Export the image
    dfi.export(styled, output_image, max_cols=-1, max_rows=-1, dpi=150)
    
    print(f"✓ {output_image} created")
    if two_column_layout:
        empty_string = '\u200b'
        left_rows = len(df_left)
        right_rows = len(df_right) if len(df_right) > 0 else 0
        print(f"  Left column: {left_rows} rows")
        print(f"  Right column: {right_rows} rows")
    print(f"  Total rows: {total_rows}")
    if max_rows:
        print(f"  (Limited to top {max_rows} industries by Recommended Portfolio %)")

if __name__ == "__main__":
    # Example usage:
    # Two-column layout (default) - top 20 industries
    create_industry_table("Portfolio Allocations - Sheet5.csv", "industry_table_full.png", two_column_layout=True)
    
    # Single unified column layout - top 20 industries
    # create_industry_table("Portfolio Allocations - Sheet5.csv", "industry_table_single.png", max_rows=20, two_column_layout=False)
