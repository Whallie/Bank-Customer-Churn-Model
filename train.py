import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
import pandas as pd

#Pytorch

DATA_PATH = Path(__file__).parent / "Data" / "Customer-Churn-Records.csv"
df = pd.read_csv(DATA_PATH)
print(df.head())
