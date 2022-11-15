from IPython.display import HTML, display
import pandas as pd


def html_table(df):
    if(isinstance(df, pd.Series)):
        df = df.to_frame()
    display(HTML(df.to_html()))

