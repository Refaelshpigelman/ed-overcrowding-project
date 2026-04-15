import pandas as pd

DEPARTMENT_MAPPING = {
    "מיון כירורגיה - מ.הלכים": "Surgery_Walking",
    "מיון פנימי": "Internal",
    "A": "Area_A",
    "טראומה חדש": "New_Trauma",
    "מלר\"ד אגף מהלכים": "ED_Walking_Wing",
    "C": "Area_C",
    "B": "Area_B",
    "טריאז' מיון חדש": "New_ED_Triage",
    "מלרד טראומה בנין 16": "ED_Trauma_Building_16",
    "מלרד מהלכים בנין 16": "ED_Walking_Building_16",
    "A מלרד אגף": "Area_A_ED_Wing",
    "מיון כירורגי": "Surgical_ED",
    "מיון אורתופדי": "Orthopedic_ED",
    "המחלקה לרפואה דחופה": "Emergency_Medicine_Department",
    "B מלרד אגף": "Area_B_ED_Wing",
    "מיון מהלכים": "Walking_ED",
    "דימות": "Imaging",
    "C מלרד אגף": "Area_C_ED_Wing",
    "מיון פנימי - לא פעיל": "Internal_Inactive",
    "מיון כירורגי - לא פעיל": "Surgical_Inactive",
    "מיון מהלכים - לא פעיל": "Walking_Inactive",
    "מיון אורתופדי - לא פעיל": "Orthopedic_Inactive",
    "המחלקה לרפואה דחופה - לא פעיל": "Emergency_Medicine_Inactive",
    "חדר צוות": "Staff_Room",
    "משרד קבלה": "Reception_Office",
    "חדר סמינרים": "Seminar_Room",
    "מיון כירורגי ישן": "Old_Surgical_ED",
    "מלרד טראומה": "ED_Trauma",
    "מלרד מהלכים": "ED_Walking",
    "D מלרד מהלכים": "Area_D_ED_Walking",
    "מיון פנימי ישן": "Old_Internal_ED",
    "חדר הלם": "Shock_Room",
    "מחסן שוכבים": "Patient_Storage"
}

STATUS_MAPPING = {
    "not_required": 0,
    "required": 1,
    "in_progress": 2,
    "partial": 3,
    "satisfied": 4,
    "taken": 5,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5
}

def load_data(csv_path: str) -> pd.DataFrame:
    """Load the raw ED dataset."""
    return pd.read_csv(csv_path)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and transform the ED dataset for modeling."""
    data = df.copy()

    # Remove exact duplicates
    data = data.drop_duplicates()

    # Convert timestamp
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce", utc=True)

    # Remove rows with invalid timestamps
    data = data.dropna(subset=["timestamp"])

    # Standardize department names to English
    data["department_en"] = data["department"].map(DEPARTMENT_MAPPING).fillna("Unknown")

    # Encode status values to numeric
    data["status_num"] = data["status"].astype(str).map(STATUS_MAPPING).fillna(-1).astype(int)

    # Basic time-based features
    data["hour"] = data["timestamp"].dt.hour
    data["day_of_week"] = data["timestamp"].dt.dayofweek
    data["month"] = data["timestamp"].dt.month
    data["is_weekend"] = data["day_of_week"].isin([4, 5]).astype(int)

    return data

def create_hourly_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate records hourly to create a simple overcrowding target:
    number of records per hour.
    """
    hourly = (
        df.set_index("timestamp")
          .groupby(pd.Grouper(freq="H"))
          .agg(
              arrivals=("status_num", "size"),
              avg_status=("status_num", "mean")
          )
          .dropna()
          .reset_index()
    )

    hourly["hour"] = hourly["timestamp"].dt.hour
    hourly["day_of_week"] = hourly["timestamp"].dt.dayofweek
    hourly["month"] = hourly["timestamp"].dt.month
    hourly["is_weekend"] = hourly["day_of_week"].isin([4, 5]).astype(int)

    # Lag and rolling features
    hourly["arrivals_lag_1"] = hourly["arrivals"].shift(1)
    hourly["arrivals_lag_2"] = hourly["arrivals"].shift(2)
    hourly["arrivals_roll_3"] = hourly["arrivals"].rolling(3).mean()
    hourly["arrivals_roll_6"] = hourly["arrivals"].rolling(6).mean()

    hourly = hourly.dropna().reset_index(drop=True)
    return hourly
