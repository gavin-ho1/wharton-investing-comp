import pandas as pd
import dataframe_image as dfi
import numpy as np

def create_two_column_table(input_csv, output_image='table_output.png'):
    """
    Reads a CSV file and creates a single table image with rows displayed in a two-column layout.
    
    Args:
        input_csv: Path to input CSV file
        output_image: Path for output PNG image
    """
    # Read the CSV
    df = pd.read_csv(input_csv)
    
    total_rows = len(df)
    
    # Calculate rows per column (split total rows in half, rounding up for left column)
    rows_per_col = (total_rows + 1) // 2
    
    # Split into left and right halves
    df_left = df.iloc[:rows_per_col].copy()
    df_right = df.iloc[rows_per_col:].copy() if rows_per_col < total_rows else pd.DataFrame(columns=df.columns)
    
    # Reset indices
    df_left.reset_index(drop=True, inplace=True)
    df_right.reset_index(drop=True, inplace=True)
    
    # Rename columns for right half to avoid duplicates
    df_right.columns = [col + '\u200b' for col in df_right.columns]
    
    # Create a spacer column between the two halves
    spacer = pd.DataFrame({'': [''] * len(df_left)})  # Empty string as column name
    
    # Concatenate horizontally with spacer in the middle
    combined = pd.concat([df_left, spacer, df_right], axis=1).fillna('')
    
    # Get the number of columns in the left half (plus 1 for spacer)
    num_cols_left = len(df.columns)
    spacer_position = num_cols_left + 1
    
    # Identify numeric columns for right alignment
    def get_right_align_selectors(dataframe):
        selectors = []
        for idx, col in enumerate(dataframe.columns, start=1):
            if any(keyword in col for keyword in ['Value', '%', 'Percent', 'Total']):
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
        ]},
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
    ]
    
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
    
    empty_string = '\u200b'
    left_rows = len([c for c in combined.columns if empty_string not in c]) // len(df.columns) * len(df_left)
    right_rows = len(df_right) if len(df_right) > 0 else 0
    
    print(f"✓ {output_image} created")
    print(f"  Left column: {len(df_left)} rows")
    print(f"  Right column: {right_rows} rows")
    print(f"  Total rows: {total_rows}")

if __name__ == "__main__":
    create_two_column_table("Final Table Wharton - Sheet1.csv", "table_output.png")