# preprocess_data.py
import pandas as pd
import numpy as np

def process_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    # ================= USER ID =================
    data['ID'] = data['ID'].replace(-2147483648, np.nan)
    data['ID'] = pd.to_numeric(data['ID'], errors="coerce")
    data = data.dropna(subset=["ID"])
    data = data[data["ID"] != 0]
    data['ID'] = data['ID'].astype(int)

    # ================= PRODUCT ID =================
    # KEEP ProdID AS STRING (VERY IMPORTANT)
    data['ProdID'] = data['ProdID'].astype(str)

    # Internal numeric index for ML
    data['ProductIndex'] = data['ProdID'].astype("category").cat.codes

    # ================= RATINGS =================
    data['Rating'] = pd.to_numeric(data['Rating'], errors="coerce").fillna(0)

    # ================= REVIEW COUNT =================
    data["ReviewCount"] = pd.to_numeric(
        data["ReviewCount"], errors='coerce'
    ).fillna(0).astype(int)

    # ================= TEXT COLUMNS =================
    for col in ['Category', 'Brand', 'Description', 'Tags', 'Name']:
        if col in data.columns:
            data[col] = data[col].fillna('').astype(str)
        else:
            data[col] = ''

    # ================= IMAGE URL CLEAN =================
    if 'ImageURL' in data.columns:
        data['ImageURL'] = (
            data['ImageURL']
            .astype(str)
            .str.split('|')
            .str[0]
        )

    # ================= DROP JUNK COLUMN =================
    if 'Unnamed: 0' in data.columns:
        data.drop(columns=['Unnamed: 0'], inplace=True)

    return data
