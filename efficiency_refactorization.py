import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import pandas as pd


# Constants and regular expressions for text cleaning and extraction
NBSP = "\u00A0"
RE_AMP = re.compile(r"&")
RE_PUNCT = re.compile(r"[^\w\s]")
RE_WS = re.compile(r"\s+")
RE_EMAIL_DOMAIN = re.compile(r"@([a-z0-9\.-]+\.[a-z]{2,})", re.IGNORECASE)


def clean_series_text(series: pd.Series) -> pd.Series:
    """
    Cleans text data in a pandas Series by applying a series of transformations:
        1. Fill missing values with empty strings
        2. Convert to string type
        3. Replace non-breaking spaces with regular spaces
        4. Strip leading/trailing whitespace and convert to lowercase
        5. Replace ampersands with "and"
        6. Remove punctuation
        7. Normalize whitespace to single spaces
    Args:
        series (pd.Series): The input Series to clean.
    Returns:
        pd.Series: The cleaned Series with text transformations applied.
    """
    series = series.fillna("")
    series = series.astype(str)
    series = series.str.replace(NBSP, " ", regex=False)
    series = series.str.strip().str.lower()
    series = series.str.replace(RE_AMP, " and ", regex=True)
    series = series.str.replace(RE_PUNCT, " ", regex=True)
    series = series.str.replace(RE_WS, " ", regex=True).str.strip()
    return series


def clean_dataframe_text(
    frame: pd.DataFrame,
    *,
    include_columns: list[str] | None = None,
    exclude_columns: list[str] | None = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Cleans text data in specified columns of a DataFrame by applying a series of transformations.
    Args:
        frame (pd.DataFrame): The input DataFrame to clean.
        include_columns (list[str] | None): List of column names to include for cleaning. If None, all object-type columns are included.
        exclude_columns (list[str] | None): List of column names to exclude from cleaning. Applied after include_columns selection.
        inplace (bool): If True, modifies the original DataFrame. If False, returns a cleaned copy.
    Returns:
        pd.DataFrame: The cleaned DataFrame with specified columns processed.
    """
    if not inplace:
        frame = frame.copy()

    exclude_columns = exclude_columns or []

    if include_columns is None:
        text_columns = frame.select_dtypes(include=["object"]).columns
    else:
        text_columns = pd.Index(include_columns)

    text_columns = text_columns.difference(exclude_columns)

    for col in text_columns:
        frame[col] = clean_series_text(frame[col])

    return frame


def extract_email_domain_series(contact_info: pd.Series) -> pd.Series:
    """
    Extracts email domains from the contact information series using a regular expression.
    Args:
        contact_info (pd.Series): A pandas Series containing contact information, which may include email addresses.
    Returns:
        pd.Series: A Series containing the extracted email domains, or empty strings for rows without valid email addresses.
    """
    return contact_info.str.extract(RE_EMAIL_DOMAIN, expand=False).fillna("")


def bucket_org_type_series(org_type_clean: pd.Series) -> pd.Series:
    """
    Buckets organization types into broader categories (School, Church, Business, Organization) based on keywords.
    Args:
        org_type_clean (pd.Series): A pandas Series containing cleaned organization type text.
    Returns:
        pd.Series: A Series of organization type buckets based on the presence of keywords.
    """
    org_type_clean = org_type_clean.fillna("")

    is_school = org_type_clean.str.contains(r"university|college|school", regex=True)
    is_church = org_type_clean.str.contains(r"church|chapel", regex=True)
    is_business = org_type_clean.str.contains(r"business|company|inc|ltd", regex=True)

    bucket = pd.Series("Organization", index=org_type_clean.index)
    bucket[is_school] = "School"
    bucket[is_church] = "Church"
    bucket[is_business] = "Business"
    return bucket


def remove_legal_suffixes_series(name_clean: pd.Series, legal_suffixes: Iterable[str]) -> pd.Series:
    """
    Removes legal suffixes from organization names for better grouping and analysis.
    Args:
        name_clean (pd.Series): A pandas Series containing cleaned organization names.
        legal_suffixes (Iterable[str]): A list of legal suffixes to remove (e.g., "inc", "ltd", "llc").
    Returns:
        pd.Series: A Series of organization names with legal suffixes removed.
    """
    suffix_pattern = r"\b(" + "|".join(map(re.escape, legal_suffixes)) + r")\b"
    name_clean = name_clean.str.replace(suffix_pattern, "", regex=True)
    name_clean = name_clean.str.replace(RE_WS, " ", regex=True).str.strip()
    return name_clean


def format_date_dmy_series(date_series: pd.Series) -> pd.Series:
    """
    Formats a datetime series into "D/M/YYYY" format, handling missing values gracefully.
    Args:
        date_series (pd.Series): A pandas Series containing datetime objects.
    Returns:
        pd.Series: A Series of formatted date strings in "D/M/YYYY" format, with missing values as empty strings.
    """
    day = date_series.dt.day.astype("Int64").astype(str)
    month = date_series.dt.month.astype("Int64").astype(str)
    year = date_series.dt.year.astype("Int64").astype(str)
    return day + "/" + month + "/" + year


def load_optional_mapping(csv_path: str, key_column: str, value_column: str) -> Dict[str, str]:
    """
    Loads an optional mapping from a CSV file and returns it as a dictionary.
    Args:
        csv_path (str): Path to the CSV file containing the mapping.
        key_column (str): Name of the column to use as keys in the mapping.
        value_column (str): Name of the column to use as values in the mapping.
    Returns:
        Dict[str, str]: A dictionary mapping keys to values based on the specified columns in the CSV.
    """
    try:
        mapping_df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except FileNotFoundError:
        print(f"WARNING: {csv_path} not found. Skipping.")
        return {}

    mapping_df.columns = (
        mapping_df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
    )

    key_col = key_column.strip().lower()
    val_col = value_column.strip().lower()

    if key_col not in mapping_df.columns or val_col not in mapping_df.columns:
        print(f"WARNING: {csv_path} missing columns {key_column}/{value_column}. Skipping.")
        return {}

    mapping_df[key_col] = clean_series_text(mapping_df[key_col])
    mapping_df[val_col] = clean_series_text(mapping_df[val_col])

    mapping_df = mapping_df[mapping_df[key_col] != ""]
    return dict(zip(mapping_df[key_col], mapping_df[val_col]))


@dataclass(frozen=True)
class Settings:
    # File paths
    input_csv: str = r"C:\Users\CameronWise\Downloads\CommunityEngagement-main\CommunityEngagement-main\data.csv"

    # Output file paths
    output_master_csv: str = "community_engagement_v2_MASTER.csv"
    output_city_year_csv: str = "community_engagement_v2_CITY_YEAR_TOTALS.csv"
    output_month_org_city_csv: str = "community_engagement_v2_MONTH_ORG_CITY.csv"

    # Optional mapping files
    org_aliases_csv: str = "org_aliases.csv"
    grouped_names_csv: str = "grouped_names.csv"

    # Processing settings
    required_columns: Tuple[str, ...] = (
        "City",
        "Date",
        "Contact Name",
        "Contact Info",
        "Organization Name",
        "Organization Type",
    )

    # Keywords that if found in the cleaned organization name will lead to exclusion
    exclude_keywords_in_org_name: Tuple[str, ...] = ("tms", "seed", "internal")

    # Mapping of city name aliases to canonical names for standardization
    city_alias_map: Dict[str, str] = None

    # Suffixes to remove from organization names for better grouping and analysis
    legal_suffixes: Tuple[str, ...] = (
        "inc", "ltd", "llc", "corp", "corporation", "co", "company",
        "limited", "incorporated",
    )

    # Reporting year for filtering contacts (only contacts in this year will be included in the analysis)
    reporting_year: int = 2025

    # Initialization to set default city alias mapping if not provided
    def __post_init__(self):
        if self.city_alias_map is None:
            object.__setattr__(
                self,
                "city_alias_map",
                {
                    "red deer alberta": "red deer",
                    "red deer, alberta": "red deer",
                    "other tms city": "calgary",
                    "central services": "calgary",
                },
            )


SETTINGS = Settings()


def load_raw_frame(settings: Settings) -> pd.DataFrame:
    """
    Loads the raw data from the specified CSV file and prepares it for processing.
    Args:
        settings (Settings): Configuration containing the input CSV file path.
    Returns:
        pd.DataFrame: The loaded raw dataframe with stripped column names.
    """
    raw_frame = pd.read_csv(settings.input_csv, encoding="latin1")
    raw_frame.columns = raw_frame.columns.str.strip()
    return raw_frame


def select_working_columns(raw_frame: pd.DataFrame, required_columns: Iterable[str]) -> pd.DataFrame:
    """
    Selects the required columns from the raw dataframe and prepares a working dataframe.
    Args:
        raw_frame (pd.DataFrame): The original raw dataframe loaded from CSV.
        required_columns (Iterable[str]): List of required column names to select.
    Returns:
        pd.DataFrame: A new dataframe containing only the required columns and a Row ID for traceability.
    """
    available_columns = [c for c in required_columns if c in raw_frame.columns]
    working_frame = raw_frame[available_columns].copy()
    working_frame.insert(0, "Row ID", range(1, len(working_frame) + 1))
    return working_frame


def add_date_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Adds date-related columns to the dataframe, including a datetime object, formatted display date, year, and year-month period.
    Args:
        frame (pd.DataFrame): The dataframe with the original "Date" column.
    Returns:
        pd.DataFrame: The dataframe with added "Date_dt", "Date Display", "Year", and "YearMonth" columns.
    """
    frame["Date_dt"] = pd.to_datetime(frame["Date"], errors="coerce", dayfirst=True)
    frame["Date Display"] = format_date_dmy_series(frame["Date_dt"])
    frame["Year"] = frame["Date_dt"].dt.year
    frame["YearMonth"] = frame["Date_dt"].dt.to_period("M").astype(str)
    return frame


def add_city_columns(frame: pd.DataFrame, city_alias_map: Dict[str, str]) -> pd.DataFrame:
    """
    Adds city-related columns to the dataframe, including cleaned city names and canonical city names based on an alias mapping.
    Args:
        frame (pd.DataFrame): The dataframe with original city columns.
        city_alias_map (Dict[str, str]): Mapping of city name aliases to canonical names.
    Returns:
        pd.DataFrame: The dataframe with added "City Clean" and "City Canonical" columns.
    """
    frame["City Clean"] = frame["City"]
    frame["City Canonical"] = frame["City Clean"].map(city_alias_map).fillna(frame["City Clean"])
    return frame


def add_organization_columns(
    frame: pd.DataFrame,
    org_alias_map: Dict[str, str],
    org_group_map: Dict[str, str],
    legal_suffixes: Iterable[str],
) -> pd.DataFrame:
    """
    Adds organization-related columns to the dataframe, including cleaned organization names, canonical names, and reporting names.
    Args:
        frame (pd.DataFrame): The dataframe with original organization columns.
        org_alias_map (Dict[str, str]): Mapping of organization name aliases to canonical names.
        org_group_map (Dict[str, str]): Mapping of canonical organization names to reporting group names.
        legal_suffixes (Iterable[str]): List of legal suffixes to remove from organization names.
    Returns:
        pd.DataFrame: The dataframe with added organization-related
    """
    frame["Org Clean"] = frame["Organization Name"]

    frame["Canonical Org Clean"] = frame["Org Clean"].map(org_alias_map).fillna(frame["Org Clean"])
    frame["Canonical Org Clean"] = remove_legal_suffixes_series(frame["Canonical Org Clean"], legal_suffixes)

    reporting_name = frame["Canonical Org Clean"]
    if org_group_map:
        reporting_name = frame["Canonical Org Clean"].map(org_group_map).fillna(reporting_name)

    frame["Reporting Org Name"] = remove_legal_suffixes_series(reporting_name, legal_suffixes)
    frame["Reporting Org Clean"] = frame["Reporting Org Name"]

    return frame


def add_contact_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Adds contact-related columns to the dataframe, including a cleaned contact name and extracted email domain.
    Args:
        frame (pd.DataFrame): The dataframe with original contact columns.
    Returns:
        pd.DataFrame: The dataframe with added "Contact Clean" and "Email Domain" columns.
    """
    frame["Contact Clean"] = frame["Contact Name"]
    frame["Email Domain"] = extract_email_domain_series(frame["Contact Info"])
    return frame


def add_type_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Adds an "Organization Type" column based on the original "Organization Type" column.
    Args:
        frame (pd.DataFrame): The dataframe with the original "Organization Type" column.
    Returns:
        pd.DataFrame: The dataframe with the new "Organization Type" and "Type Bucket" columns.
    """
    frame["Organization Type"] = bucket_org_type_series(frame["Organization Type"])
    frame["Type Bucket"] = frame["Organization Type"]
    return frame


def apply_exclusion_rules(frame: pd.DataFrame, raw_frame: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """
    Applies exclusion rules to the dataframe and flags rows for exclusion with reasons.
    Exclusion rules:
        1. Missing Org Name or Date
        2. Not in reporting year
        3. TMS Internal (Org Type)
        4. Internal keyword in Org Name
    Args:
        frame (pd.DataFrame): The processed dataframe with contact flags and date columns.
        raw_frame (pd.DataFrame): The original raw dataframe for reference to uncleaned columns.
        settings (Settings): Configuration for exclusion rules and keywords.
    Returns:
        pd.DataFrame: Dataframe with added "Excluded" and "Exclusion Reason" columns based on the rules.
    """
    frame["Excluded"] = 0
    frame["Exclusion Reason"] = ""

    is_missing_org = frame["Org Clean"] == ""
    is_missing_date = frame["Date_dt"].isna()
    missing_required = is_missing_org | is_missing_date

    frame.loc[missing_required, "Excluded"] = 1
    frame.loc[missing_required, "Exclusion Reason"] = "Missing Org Name or Date"

    is_wrong_year = frame["Year"] != settings.reporting_year
    frame.loc[is_wrong_year, "Excluded"] = 1
    frame.loc[is_wrong_year & (frame["Exclusion Reason"] == ""), "Exclusion Reason"] = f"Not in {settings.reporting_year}"

    if "Organization Type" in raw_frame.columns:
        raw_org_type_lower = raw_frame["Organization Type"].astype(str).str.strip().str.lower()
        is_tms_internal_type = raw_org_type_lower == "tms internal"

        frame.loc[is_tms_internal_type, "Excluded"] = 1
        frame.loc[
            is_tms_internal_type & (frame["Exclusion Reason"] == ""),
            "Exclusion Reason",
        ] = "TMS Internal (Org Type)"

    keyword_pattern = r"\b(?:" + "|".join(map(re.escape, settings.exclude_keywords_in_org_name)) + r")\b"
    contains_internal_keyword = frame["Org Clean"].str.contains(keyword_pattern, regex=True)
    frame.loc[contains_internal_keyword, "Excluded"] = 1
    frame.loc[
        contains_internal_keyword & (frame["Exclusion Reason"] == ""),
        "Exclusion Reason",
    ] = "Internal keyword in Org Name"

    return frame


def add_first_contact_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Flags the first contact for each unique organization-city combination.
    Creates an "Org Key" by combining the cleaned reporting organization name and canonical city.
    Args:
        frame (pd.DataFrame): The processed dataframe with contact flags and date columns.
    Returns:
        pd.DataFrame: Dataframe with added "Org Key", "First Contact", and "Countable" columns.
    """
    frame["Org Key"] = frame["Reporting Org Clean"] + " | " + frame["City Canonical"]

    frame["First Contact"] = 0
    valid_rows = frame[frame["Excluded"] == 0].sort_values(["Date_dt", "Row ID"])
    first_contact_index = valid_rows.drop_duplicates(subset=["Org Key"], keep="first").index
    frame.loc[first_contact_index, "First Contact"] = 1

    frame["Countable"] = ((frame["Excluded"] == 0) & (frame["First Contact"] == 1)).astype(int)
    return frame


def build_master_output(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the MASTER output dataframe with all relevant columns for analysis.
    Args:
        frame (pd.DataFrame): The processed dataframe with contact flags and date columns.
    Returns:
        pd.DataFrame: Dataframe with selected columns in a specific order for the MASTER output.
    """
    master = pd.DataFrame({
        "Row ID": frame["Row ID"],  # Include Row ID for traceability
        "City": frame["City Canonical"],  # Use canonical city for consistency
        "Date": frame["Date Display"],  # Use formatted date for better readability
        "Contact Name": frame["Contact Name"],  # Use original contact name for accuracy
        "Contact Info": frame["Contact Info"],  # Use original contact info for accuracy
        "Organization Name": frame["Organization Name"],  # Use original organization name for accuracy
        "Reporting Org Name": frame["Reporting Org Name"],  # Use reporting org name for grouping and analysis
        "First Contact": frame["First Contact"],  # Include first contact flag for analysis
        "Organization Type": frame["Organization Type"],  # Use original organization type for accuracy
        "Excluded": frame["Excluded"],  # Include exclusion flag for analysis
        "Exclusion Reason": frame["Exclusion Reason"],  # Include exclusion reason for analysis
        "Org Key": frame["Org Key"],  # Include org key for traceability and analysis
        "Email Domain": frame["Email Domain"],  # Include email domain for potential analysis of contact patterns
    })

    column_order = [
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
    return master[column_order]


def build_city_year_totals(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the CITY_YEAR_TOTALS output dataframe with unique first contact counts by city and year.
    Groups by Year and City Canonical to calculate:
        - Year Total (All Types): Number of unique first contacts for that city and year
    Args:
        frame (pd.DataFrame): The processed dataframe with contact flags and date columns.
    Returns:
        pd.DataFrame: Dataframe with columns Year, City, Year Total (All Types).
    """
    city_year = (
        frame[frame["Countable"] == 1]
        .groupby(["Year", "City Canonical"], as_index=False)["Org Key"]
        .nunique()
        .rename(columns={"City Canonical": "City", "Org Key": "Year Total (All Types)"})
    )

    year_totals = city_year.groupby("Year", as_index=False)["Year Total (All Types)"].sum()
    year_totals["City"] = "Total"

    return pd.concat([city_year, year_totals], ignore_index=True)


def build_month_org_city(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the MONTH_ORG_CITY output dataframe with first contact counts and dates.
    Groups by YearMonth, City Canonical, and Reporting Org Name to calculate:
        - First Contacts (Count): Number of unique first contacts
        - First Contact Date: Earliest contact date for the group
    Args:
        frame (pd.DataFrame): The processed dataframe with contact flags and date columns.
    Returns:
        pd.DataFrame: Dataframe with columns Month, City, Organization, First Contacts (Count), First Contact Date.
    """
    summary = (
        frame[frame["Countable"] == 1]
        .groupby(["YearMonth", "City Canonical", "Reporting Org Name"], as_index=False)
        .agg(
            First_Contacts=("Org Key", "nunique"),
            First_Contact_Date=("Date_dt", "min"),
        )
    )

    summary["First Contact Date"] = format_date_dmy_series(summary["First_Contact_Date"])

    summary = summary.rename(columns={
        "YearMonth": "Month",
        "City Canonical": "City",
        "Reporting Org Name": "Organization",
        "First_Contacts": "First Contacts (Count)",
    })

    summary = summary[
        ["Month", "City", "Organization", "First Contacts (Count)", "First Contact Date"]
    ]

    return summary.sort_values(["Month", "City", "Organization"])


def run_pipeline(settings: Settings) -> None:
    """
    Main function to run the data processing pipeline.
    Steps:
        1. Load raw data
        2. Select and prepare working columns
        3. Clean text data
        4. Add derived columns (dates, cities, organizations, types, contacts)
        5. Apply exclusion rules
        6. Flag first contacts
        7. Build output dataframes
        8. Save outputs to CSV
    Args:
        settings (Settings): Configuration for file paths, column names, and processing rules.
    """

    raw_frame = load_raw_frame(settings)

    working_frame = select_working_columns(raw_frame, settings.required_columns)

    working_frame = clean_dataframe_text(
        working_frame,
        exclude_columns=["Organization Type"],
        inplace=True,
    )

    working_frame = add_date_columns(working_frame)
    working_frame = add_city_columns(working_frame, settings.city_alias_map)

    org_alias_map = load_optional_mapping(settings.org_aliases_csv, "alias_clean", "canonical_clean")
    org_group_map = load_optional_mapping(settings.grouped_names_csv, "canonical_clean", "reporting_org")

    working_frame = add_organization_columns(
        working_frame,
        org_alias_map,
        org_group_map,
        settings.legal_suffixes,
    )

    working_frame = add_type_columns(working_frame)
    working_frame = add_contact_columns(working_frame)

    working_frame = apply_exclusion_rules(working_frame, raw_frame, settings)
    working_frame = add_first_contact_flags(working_frame)

    master_output = build_master_output(working_frame)
    city_year_output = build_city_year_totals(working_frame)
    month_org_city_output = build_month_org_city(working_frame)

    master_output.to_csv(settings.output_master_csv, index=False)
    city_year_output.to_csv(settings.output_city_year_csv, index=False)
    month_org_city_output.to_csv(settings.output_month_org_city_csv, index=False)

    print("Saved:", settings.output_master_csv)
    print("Saved:", settings.output_city_year_csv)
    print("Saved:", settings.output_month_org_city_csv)

    print("RUNNING FROM:", os.getcwd())
    print("org_aliases path:", os.path.abspath(settings.org_aliases_csv))
    print("grouped_names path:", os.path.abspath(settings.grouped_names_csv))


if __name__ == "__main__":
    run_pipeline(SETTINGS)