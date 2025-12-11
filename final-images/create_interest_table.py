import pandas as pd
import dataframe_image as dfi

def create_metrics_table(data, output_image='metrics_interest.png'):
    """
    Creates a metrics comparison table from a dictionary or DataFrame,
    styled to match the specific reference image.
    """
    # Convert data to DataFrame if it's a dictionary
    if isinstance(data, dict):
        df = pd.DataFrame(data)
    else:
        df = data.copy()
    
    # Define the specific footnote text
    footnote = "*Some stocks fit more than one value that Mr. Barwin cherishes. Only the primary value is counted for those stocks."

    # Style the dataframe
    styles = [
        # Table wide settings
        {'selector': '', 'props': [
            ('border-collapse', 'collapse'),
            ('font-family', '"Times New Roman", Times, serif'), # Match image font
        ]},
        # Header Styling
        {'selector': 'th', 'props': [
            ('background-color', '#49467F'), # Purple header
            ('color', 'white'),
            ('font-weight', 'bold'),
            ('text-align', 'center'),
            ('font-size', '13pt'),
            ('border', '1px solid #ddd'),
            ('padding', '15px 40px'),
            ('min-width', '150px')
        ]},
        # Body Cell Styling
        {'selector': 'td', 'props': [
            ('border', '1px solid #ddd'),
            ('padding', '15px 40px'),
            ('text-align', 'center'),
            ('font-size', '12pt'),
            ('color', 'black'),
            ('min-width', '150px')
        ]},
        # Left-align and Bold the first column (Client Values)
        {'selector': 'td:nth-child(1)', 'props': [
            ('text-align', 'left'),
            ('font-weight', 'bold')
        ]},
        # Style the caption (footnote) to appear at the bottom
        {'selector': 'caption', 'props': [
            ('caption-side', 'bottom'),
            ('text-align', 'left'),
            ('font-size', '10pt'),
            ('font-weight', 'bold'),
            ('color', 'black'),
            ('padding-top', '5px')
        ]}
    ]
    
    # Apply styles and set the caption
    styled = (df.style
              .hide(axis='index')
              .set_table_styles(styles)
              .set_caption(footnote)) # Adds the footnote
    
    # Export the image
    dfi.export(styled, output_image, max_cols=-1, max_rows=-1, dpi=150)
    
    print(f"✓ {output_image} created")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")

if __name__ == "__main__":
    # Data updated to match the image order and values exactly
    data = {
        "Client Values": [
            "Community",
            "B-Corps/ESG/Environment",
            "Healthcare",
            "Sports",
            "Education",
            "Other"
        ],
        "Recommended Portfolio": [
            "49.13%",
            "28.25%",
            "11.53%",
            "5.90%",
            "1.38%",
            "3.83%"
        ],
        "Final WInS Portfolio": [
            "48.40%",
            "28.68%",
            "11.71%",
            "5.77%",
            "1.52%",
            "3.90%"
        ],
        "Profit/Loss of Category": [
            "1.69%",
            "4.24%",
            "3.43%",
            "0.21%",
            "15.54%",
            "3.85%"
        ]
    }
    
    create_metrics_table(data, "metrics_interest.png")