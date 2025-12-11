import pandas as pd
import dataframe_image as dfi

def create_single_column_table(input_csv, output_image='etf_table_output.png'):
    """
    Reads a CSV file and creates a single-column table image (no split).
    
    Args:
        input_csv: Path to input CSV file
        output_image: Path for output PNG image
    """
    # Read the CSV
    df = pd.read_csv(input_csv)
    
    total_rows = len(df)
    
    # Identify numeric columns for right alignment
    def get_right_align_selectors(dataframe):
        selectors = []
        for idx, col in enumerate(dataframe.columns, start=1):
            if any(keyword in col for keyword in ['Value', '%', 'Percent', 'Total']):
                selectors.append(f'td:nth-child({idx})')
        return selectors
    
    # Style the dataframe
    right_align_selectors = get_right_align_selectors(df)
    
    styles = [
        {'selector': 'th', 'props': [
            ('background-color', '#49467F'),
            ('color', 'white'),
            ('font-weight', 'bold'),
            ('text-align', 'center'),
            ('font-size', '9pt'),
            ('border', '1px solid #49467F'),
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
    
    # Add right alignment for numeric columns
    for selector in right_align_selectors:
        styles.append({
            'selector': selector,
            'props': [('text-align', 'right')]
        })
    
    styled = df.style.hide(axis='index').set_properties(**{
        'font-size': '9pt'
    }).set_table_styles(styles)
    
    # Export the image
    dfi.export(styled, output_image, max_cols=-1, max_rows=-1, dpi=150)
    
    print(f"✓ {output_image} created")
    print(f"  Total rows: {total_rows}")

if __name__ == "__main__":
    create_single_column_table("ETF.csv", "etf_table_output.png")
