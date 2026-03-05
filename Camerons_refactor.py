import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import pandas as pd



# CONFIG

@dataclass(frozen=True)
class Settings:
    input_file: str = r"C:\Users\CameronWise\Downloads\CommunityEngagement-main\CommunityEngagement-main\data.csv"

    master_out: str = "community_engagement_v2_MASTER.csv"
    city_year_out: str = "community_engagement_v2_CITY_YEAR_TOTALS.csv"
    month_org_city_out: str = "community_engagement_v2_MONTH_ORG_CITY.csv"
    suggested_aliases_out: str = "suggested_org_aliases.csv"  # reserved for later

    org_aliases_file: str = "org_aliases.csv"       # required headers: alias_clean, canonical_clean
    grouped_names_file: str = "grouped_names.csv"   # required headers: canonical_clean, reporting_org

    desired_cols: Tuple[str, ...] = (
        "City",
        "Date",
        "Contact Name",
        "Contact Info",
        "Organization Name",
        "Organization Type",
    )

    exclude_keywords_in_orgname: Tuple[str, ...] = ("tms", "seed", "internal")

    city_alias_map: Dict[str, str] = None

    legal_suffixes: Tuple[str, ...] = (
        "inc", "ltd", "llc", "corp", "corporation", "co", "company",
        "limited", "incorporated"
    )

    target_year: int = 2025

    def __post_init__(self):
        if self.city_alias_map is None:
            object.__setattr__(
                self,
                "city_alias_map",
                {
                    "red deer, alberta": "red deer",
                    "red deer alberta": "red deer",
                    "other tms city": "calgary",
                    "central services": "calgary",
                },
            )


SETTINGS = Settings()



# TEXT / FIELD HELPERS

def clean_text_basic(value) -> str:
    """
    Perform basic text cleaning: handle NaN, normalize whitespace, lowercase, and remove punctuation.
    Args:
        value: The raw text value to clean.
    Returns:
        A cleaned string with normalized whitespace, lowercase, and no punctuation. Returns empty string if input is NaN.
    """
    if pd.isna(value):
        return ""
    s = str(value).replace("\u00A0", " ")
    s = s.strip().lower()
    s = re.sub(r"&", " and ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def clean_dataframe_strings(df: pd.DataFrame, skip_cols=None) -> pd.DataFrame:
    """
    Apply clean_text_basic to every string column in the dataframe once.
    Args:
        df: The DataFrame to clean.
        skip_cols: An optional iterable of column names to skip from cleaning.
    Returns:
        The DataFrame with all string columns cleaned, except those in skip_cols.
    """
    if skip_cols is None:
        skip_cols = []

    str_cols = [
        c for c in df.columns
        if df[c].dtype == "object" and c not in skip_cols
    ]

    df[str_cols] = df[str_cols].applymap(clean_text_basic)

    return df


def clean_city(value) -> str:
    """
    Clean the city name by applying basic text cleaning and then normalizing common variations of city names.
    Args:
        value: The raw city name to clean.
    Returns:
        A cleaned and normalized city name string.
    """
    return clean_text_basic(value)


def extract_email_domain(contact_info) -> str:
    """
    Extract the email domain from the contact info string. Returns empty string if no valid email is found.
    Args:
        contact_info: A string that may contain an email address.
    Returns:
        The domain part of the email address if found, otherwise an empty string.
    """
    if pd.isna(contact_info):
        return ""
    s = str(contact_info).lower()
    m = re.search(r"@([a-z0-9\.-]+\.[a-z]{2,})", s)
    return m.group(1) if m else ""


def bucket_type(org_type) -> str:
    """
    Bucket the organization type into simplified categories based on keywords. Defaults to "Organization" if no keywords match.
    Args:
        org_type: The raw organization type string to bucket.
    Returns:
        A string representing the bucketed organization type ("School", "Church", "Business", or "Organization").
    """
    t = clean_text_basic(org_type)
    if not t:
        return "Organization"
    if any(x in t for x in ("university", "college", "school")):
        return "School"
    if any(x in t for x in ("church", "chapel")):
        return "Church"
    if any(x in t for x in ("business", "company", "inc", "ltd")):
        return "Business"
    return "Organization"


def strip_legal_suffix(name, legal_suffixes: Iterable[str]) -> str:
    """
    Strip legal suffixes from an organization name for cleaner matching. Only removes suffixes that appear as whole words at the end of the name.
    Args:
        name: The organization name to clean.
        legal_suffixes: An iterable of legal suffixes to remove (e.g. "inc", "ltd").
    Returns:
        The cleaned organization name with legal suffixes removed.
    """
    if not name:
        return ""
    words = [w for w in name.split() if w not in set(legal_suffixes)]
    return " ".join(words).strip()


def format_date_dmy(dt) -> str:
    """
    Format a datetime object into D/M/YYYY format for display. Returns empty string if input is NaT.
    Args:
        dt: A datetime object or NaT.
    Returns:
        A string in D/M/YYYY format if dt is a valid date, or an empty string if dt is NaT.
    """
    if pd.isna(dt):
        return ""
    return f"{int(dt.day)}/{int(dt.month)}/{int(dt.year)}"



# CSV MAPPING LOADER

def _try_read_csv_any_delim(path: str) -> pd.DataFrame:
    """
    Try reading a CSV file with various delimiters to find one that produces multiple columns. Raises an error if none work.
    Args:
        path: The file path to the CSV file.
    Returns:
        A DataFrame read from the CSV file using the first successful delimiter.
    """
    enc = "utf-8-sig"
    try:
        df = pd.read_csv(path, encoding=enc)
        if df.shape[1] > 1:
            return df
    except Exception:
        pass

    for sep in (";", "\t"):
        try:
            df = pd.read_csv(path, encoding=enc, sep=sep)
            if df.shape[1] > 1:
                return df
        except Exception:
            continue

    raise ValueError("Unable to parse CSV with expected delimiters.")


def load_optional_map(path: str, key_col: str, val_col: str) -> Dict[str, str]:
    """
    Load a mapping from a CSV file with specified key and value columns. If the file is missing or invalid, return an empty dict.
    Args:
        path: The file path to the CSV mapping file.
        key_col: The column name to use as keys in the mapping.
        val_col: The column name to use as values in the mapping.
    Returns:
        A dictionary mapping from cleaned key_col values to cleaned val_col values.
    """
    try:
        dfm = _try_read_csv_any_delim(path)

        dfm.columns = (
            dfm.columns.astype(str)
            .str.replace("\ufeff", "", regex=False)
            .str.strip()
            .str.lower()
        )

        k = key_col.strip().lower()
        v = val_col.strip().lower()

        if k not in dfm.columns or v not in dfm.columns:
            print(f"WARNING: {path} missing columns {key_col}/{val_col}. Skipping.")
            return {}

        dfm = dfm[dfm[k] != ""]

        return dict(zip(dfm[k], dfm[v]))

    except FileNotFoundError:
        print(f"WARNING: {path} not found. Skipping.")
        return {}
    except Exception as e:
        print(f"WARNING: could not read {path} ({e}). Skipping.")
        return {}



# PIPELINE STEPS

def load_raw(settings: Settings) -> pd.DataFrame:
    """
    Load the raw data CSV with basic cleaning of column names.
    Args:
        settings: Settings object containing configuration.
    Returns:
        A DataFrame with the raw data and cleaned column names.
    """
    raw = pd.read_csv(settings.input_file, encoding="latin1")
    raw.columns = [c.strip() for c in raw.columns]
    return raw


def select_columns(raw: pd.DataFrame, desired_cols: Iterable[str]) -> pd.DataFrame:
    """
    Select only the desired columns from the raw DataFrame, adding a Row ID.
    Args:
        raw: The raw DataFrame loaded from CSV.
        desired_cols: An iterable of column names to select if they exist.
    Returns:
        A DataFrame with only the desired columns and a Row ID.
    """
    use_cols = [c for c in desired_cols if c in raw.columns]
    df = raw[use_cols].copy()
    df.insert(0, "Row ID", range(1, len(df) + 1))
    return df


def add_date_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the Date column into datetime, and create display and year fields.
    Args:
        df: DataFrame with a "Date" column.
    Returns:
        DataFrame with added "Date_dt", "Date Display", "Year", and "YearMonth" columns.
    """
    df["Date_dt"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df["Date Display"] = df["Date_dt"].apply(format_date_dmy)
    df["Year"] = df["Date_dt"].dt.year
    df["YearMonth"] = df["Date_dt"].dt.to_period("M").astype(str)
    return df


def add_city_fields(df: pd.DataFrame, city_alias_map: Dict[str, str]) -> pd.DataFrame:
    """
    Clean the City field and apply canonical mapping.
    Args:
        df: DataFrame with a "City" column.
        city_alias_map: A dictionary mapping cleaned city names to canonical names.
    Returns:
        DataFrame with added "City Clean" and "City Canonical" columns.
    """

    # If you've already cleaned strings globally, City is already clean.
    # But we can still be defensive:
    if "City" not in df.columns:
        raise KeyError("Missing required column: City")

    if "City Clean" not in df.columns:
        df["City Clean"] = df["City"].map(clean_city)

    df["City Canonical"] = df["City Clean"].map(city_alias_map).fillna(df["City Clean"])
    return df


def add_org_fields(
    df: pd.DataFrame,
    alias_map: Dict[str, str],
    group_map: Dict[str, str],
    legal_suffixes: Iterable[str],
) -> pd.DataFrame:
    """
    Clean the Organization Name, apply alias and group mappings, and strip legal suffixes.
    Args:
        df: DataFrame with an "Organization Name" column.
        alias_map: A dictionary mapping cleaned org names to canonical names.
        group_map: A dictionary mapping canonical org names to reporting org names.
        legal_suffixes: An iterable of legal suffixes to strip from org names.
    Returns:
        DataFrame with added "Org Clean", "Canonical Org Clean", "Reporting Org Name", and "Reporting Org Clean" columns.
    """

    if "Organization Name" not in df.columns:
        raise KeyError("Missing required column: Organization Name")

    # Create Org Clean if it doesn't exist (works whether or not you pre-cleaned df)
    if "Org Clean" not in df.columns:
        df["Org Clean"] = df["Organization Name"].map(clean_text_basic)

    # Apply alias mapping
    df["Canonical Org Clean"] = df["Org Clean"].map(alias_map).fillna(df["Org Clean"])
    df["Canonical Org Clean"] = df["Canonical Org Clean"].map(
        lambda s: strip_legal_suffix(s, legal_suffixes)
    )

    # Apply grouping (canonical -> reporting)
    df["Reporting Org Name"] = df["Canonical Org Clean"]
    if group_map:
        df["Reporting Org Name"] = df["Canonical Org Clean"].map(group_map).fillna(df["Reporting Org Name"])

    df["Reporting Org Name"] = df["Reporting Org Name"].map(
        lambda s: strip_legal_suffix(s, legal_suffixes)
    )

    df["Reporting Org Clean"] = df["Reporting Org Name"].map(clean_text_basic)
    return df


def add_contact_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the Contact Name and extract email domain from Contact Info.
    Args:
        df: DataFrame with "Contact Name" and "Contact Info" columns.
    Returns:
        DataFrame with added "Contact Clean" and "Email Domain" columns.
    """
    df["Email Domain"] = df["Contact Info"].apply(extract_email_domain)
    return df


def add_type_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bucket the Organization Type into simplified categories.
    Args:
        df: DataFrame with an "Organization Type" column.
    Returns:
        DataFrame with added "Type Bucket" column.
    """
    df["Organization Type"] = df["Organization Type"].apply(bucket_type)
    df["Type Bucket"] = df["Organization Type"]
    return df


def apply_exclusions(
    df: pd.DataFrame,
    raw: pd.DataFrame,
    exclude_keywords: Iterable[str],
    target_year: int,
) -> pd.DataFrame:
    """
    Apply exclusion rules to determine which rows should be excluded from counting.
    Exclusion rules:
     - Missing Org Name or Date
     - Not in target year
     - TMS Internal (if "Organization Type" is present and indicates internal)
     - Internal keyword in Org Name
    Args:
        df: DataFrame with cleaned fields.
        raw: The original raw DataFrame, used for checking "Organization Type".
        exclude_keywords: An iterable of keywords that, if present in the Org Name, should trigger exclusion.
        target_year: The year that valid contacts must be in to avoid exclusion.
    Returns:
        DataFrame with added "Excluded" and "Exclusion Reason" columns.
    """
    df["Excluded"] = 0
    df["Exclusion Reason"] = ""

    missing_org = df["Org Clean"].eq("")
    missing_date = df["Date_dt"].isna()
    mask_missing = missing_org | missing_date
    df.loc[mask_missing, "Excluded"] = 1
    df.loc[mask_missing, "Exclusion Reason"] = "Missing Org Name or Date"

    not_target_year = df["Year"].ne(target_year)
    df.loc[not_target_year, "Excluded"] = 1
    df.loc[not_target_year & (df["Exclusion Reason"] == ""), "Exclusion Reason"] = f"Not in {target_year}"

    if "Organization Type" in raw.columns:
        org_type_raw = raw["Organization Type"].astype(str).str.strip().str.lower()
        tms_internal_type = org_type_raw.eq("tms internal")
        df.loc[tms_internal_type, "Excluded"] = 1
        df.loc[tms_internal_type & (df["Exclusion Reason"] == ""), "Exclusion Reason"] = "TMS Internal (Org Type)"

    kw = tuple(exclude_keywords)
    if kw:
        kw_pattern = r"\b(" + "|".join(map(re.escape, kw)) + r")\b"
        has_kw = df["Org Clean"].str.contains(kw_pattern, na=False, regex=True)
        df.loc[has_kw, "Excluded"] = 1
        df.loc[has_kw & (df["Exclusion Reason"] == ""), "Exclusion Reason"] = "Internal keyword in Org Name"

    return df


def add_first_contact_and_countable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Determine which rows represent the first contact for each organization and create a countable flag.
    Args:
        df: DataFrame with cleaned and exclusion-applied fields.
    Returns:
        DataFrame with added "First Contact" and "Countable" columns.
    """
    df["Org Key"] = df["Reporting Org Clean"] + " | " + df["City Canonical"]

    df["First Contact"] = 0
    valid = df[df["Excluded"] == 0].copy().sort_values(["Date_dt", "Row ID"])
    first_idx = valid.drop_duplicates(subset=["Org Key"], keep="first").index
    df.loc[first_idx, "First Contact"] = 1

    df["Countable"] = ((df["Excluded"] == 0) & (df["First Contact"] == 1)).astype(int)
    return df



# OUTPUT BUILDERS


def build_master(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the master output DataFrame with all relevant fields.
    Args:
        df: The fully processed DataFrame with all added fields.
    Returns:
        A DataFrame with selected columns in the desired order for master output.
    """
    master = pd.DataFrame({
        "Row ID": df["Row ID"],
        "City": df["City Canonical"],
        "Date": df["Date Display"],
        "Contact Name": df["Contact Name"],
        "Contact Info": df["Contact Info"],
        "Organization Name": df["Organization Name"],
        "Reporting Org Name": df["Reporting Org Name"],
        "First Contact": df["First Contact"],
        "Organization Type": df["Organization Type"],
        "Excluded": df["Excluded"],
        "Exclusion Reason": df["Exclusion Reason"],
        "Org Key": df["Org Key"],
        "Email Domain": df["Email Domain"],
    })

    col_order = [
        "Row ID",
        "City",
        "Date",
        "Contact Name",
        "Contact Info",
        "Organization Name",
        "Reporting Org Name",
        "First Contact",
        "Organization Type",
        "Excluded",
        "Exclusion Reason",
        "Org Key",
        "Email Domain",
    ]
    return master[col_order]


def build_city_year_totals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the city-year totals DataFrame, counting unique organizations per city and year, and a total row.
    Args:
        df: The fully processed DataFrame with all added fields.
    Returns:
        A DataFrame with columns "Year", "City", and "Year Total (All Types)", including a total row.
    """
    city_year = (
        df[df["Countable"] == 1]
        .groupby(["Year", "City Canonical"], as_index=False)["Org Key"]
        .nunique()
        .rename(columns={"City Canonical": "City", "Org Key": "Year Total (All Types)"})
    )

    year_total = city_year.groupby("Year", as_index=False)["Year Total (All Types)"].sum()
    year_total["City"] = "Total"

    return pd.concat([city_year, year_total], ignore_index=True)


def build_month_org_city(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the month-org-city DataFrame, counting first contacts per month, city, and organization.
    Args:
        df: The fully processed DataFrame with all added fields.
    Returns:
        A DataFrame with columns "Month", "City", "Organization", "First Contacts (Count)", and "First Contact Date".
    """
    out = (
        df[df["Countable"] == 1]
        .groupby(["YearMonth", "City Canonical", "Reporting Org Name"], as_index=False)
        .agg(
            First_Contacts=("Org Key", "nunique"),
            First_Contact_Date=("Date_dt", "min"),
        )
    )

    # out = out.drop(columns=["First_Contact_Date"])

    out = out.rename(columns={
        "YearMonth": "Month",
        "City Canonical": "City",
        "Reporting Org Name": "Organization",
        "First_Contacts": "First Contacts (Count)",
        "First_Contact_Date": "First Contact Date",

    })

    out = out[["Month", "City", "Organization", "First Contacts (Count)", "First Contact Date"]]
    return out.sort_values(["Month", "City", "Organization"])


def save_csv(df: pd.DataFrame, path: str) -> None:
    """
    Save a DataFrame to CSV without the index and print the saved path.
    Args:
        df: The DataFrame to save.
        path: The file path to save the CSV to.
    """
    df.to_csv(path, index=False)
    print("Saved:", path)



# MAIN


def run_pipeline(settings: Settings) -> None:
    """
    Run the entire data processing pipeline from loading raw data to saving outputs.
    Args:
        settings: A Settings object containing all configuration for the pipeline.
    """
    raw = load_raw(settings)
    df = select_columns(raw, settings.desired_cols)

    df = clean_dataframe_strings(
        df,
        skip_cols=["Organization Type"]   # keep raw if needed
    )

    df = add_date_fields(df)
    df = add_city_fields(df, settings.city_alias_map)

    alias_map = load_optional_map(settings.org_aliases_file, "alias_clean", "canonical_clean")
    group_map = load_optional_map(settings.grouped_names_file, "canonical_clean", "reporting_org")

    df = add_org_fields(df, alias_map, group_map, settings.legal_suffixes)
    df = add_type_fields(df)
    df = add_contact_fields(df)
    df = apply_exclusions(df, raw, settings.exclude_keywords_in_orgname, settings.target_year)
    df = add_first_contact_and_countable(df)

    master = build_master(df)
    city_year = build_city_year_totals(df)
    month_org_city = build_month_org_city(df)

    save_csv(master, settings.master_out)
    save_csv(city_year, settings.city_year_out)
    save_csv(month_org_city, settings.month_org_city_out)

    print("RUNNING FROM:", os.getcwd())
    print("org_aliases path:", os.path.abspath(settings.org_aliases_file))
    print("grouped_names path:", os.path.abspath(settings.grouped_names_file))


if __name__ == "__main__":
    run_pipeline(SETTINGS)