import pandas as pd
import re
from typing import List

def clean_text_field(text: str, keywords: List[str], case_sensitive: bool = False) -> str:
    """
    Clean a single text field by replacing it with a matching keyword.
    If any keyword is found in the text, replace the entire text with just that keyword.
    
    Parameters:
    -----------
    text : str
        Input text to clean
    keywords : List[str]
        List of keywords to search for
    case_sensitive : bool
        Whether matching should be case sensitive
        
    Returns:
    --------
    str
        Either the matching keyword or the original text if no keyword found
    """
    if pd.isna(text) or text == '':
        return ''
    
    text = str(text).strip()
    
    # Check each keyword to see if it exists in the text
    for keyword in keywords:
        if case_sensitive:
            if keyword in text:
                return keyword
        else:
            if keyword.lower() in text.lower():
                return keyword
    
    # If no keyword found, return original text
    return text


def create_multi_onehot_encoding(
    df: pd.DataFrame, 
    column_name: str, 
    predefined_options: List[str],
    prefix: str = None
) -> pd.DataFrame:
    """
    Create multi one-hot encoding based on predefined options list.
    Only predefined options will get columns, new entries are ignored.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    column_name : str
        Name of the text column to encode
    predefined_options : List[str]
        Predefined list of options to create columns for
    prefix : str
        Prefix for new column names (defaults to column_name)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with additional one-hot encoded columns
    """
    result_df = df.copy()
    
    if prefix is None:
        prefix = column_name
    
    # Create one-hot encoded columns for each predefined option
    for option in predefined_options:
        column_name_encoded = f"{prefix}_{option.replace(' ', '_').replace('-', '_')}"
        
        result_df[column_name_encoded] = result_df[column_name].apply(
            lambda x: 1 if check_option_in_text(str(x), option) else 0
        )
    
    return result_df


def check_option_in_text(text: str, option: str) -> bool:
    """
    Helper function to check if an option exists in comma-separated text.
    
    Parameters:
    -----------
    text : str
        Text to search in (comma-separated values)
    option : str
        Option to search for
        
    Returns:
    --------
    bool
        True if option found, False otherwise
    """
    if pd.isna(text) or text == '':
        return False
    
    # Split by comma and strip whitespace
    text_options = [item.strip() for item in str(text).split(',') if item.strip()]
    
    # Check if option exists in the list
    return option in text_options


# Example usage:
if __name__ == "__main__":
    # Sample data
    data = {
        'id': [1, 2, 3, 4, 5, 6],
        'text_field': [
            'C-TX per ashley, abc',
            'C-TX',
            'C-TX, as per ash, dd',
            'abc, dd, df',
            'dd,df',
            'some new entry, xyz'  # This won't get encoded unless in predefined_options
        ]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print()
    
    # Step 1: Clean the text fields
    # Define your keywords that should replace entire text field when found
    keywords_to_clean = ['C-TX', 'abc', 'dd', 'df']
    
    df['text_field_cleaned'] = df['text_field'].apply(
        lambda x: clean_text_field(x, keywords_to_clean)
    )
    
    print("After cleaning:")
    print(df[['text_field', 'text_field_cleaned']])
    print()
    
    # Step 2: Create multi one-hot encoding
    # Define your predefined options (based on your current data analysis)
    predefined_options = ['C-TX', 'abc', 'dd', 'df']  # Only these will get columns
    
    result_df = create_multi_onehot_encoding(
        df, 
        'text_field_cleaned', 
        predefined_options,
        prefix='feature'
    )
    
    print("Final result with one-hot encoding:")
    print(result_df)
