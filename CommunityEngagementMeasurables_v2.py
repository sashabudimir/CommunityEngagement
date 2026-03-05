import re
import os
import pandas as pd



# SETTINGS

INPUT_FILE = "data_raw.csv"

MASTER_OUT = "community_engagement_v2_MASTER.csv"
CITY_YEAR_OUT = "community_engagement_v2_CITY_YEAR_TOTALS.csv"
SUGGESTED_ALIASES_FILE = "suggested_org_aliases.csv"

ORG_ALIASES_FILE = "org_aliases.csv"       # required headers: alias_clean, canonical_clean
GROUPED_NAMES_FILE = "grouped_names.csv"   # required headers: canonical_clean, reporting_org

DESIRED_COLS = [
    "City",
    "Date",
    "Contact Name",
    "Contact Info",
    "Organization Name",
    "Organization Type",
]

EXCLUDE_KEYWORDS_IN_ORGNAME = ["tms", "seed", "internal"]

CITY_ALIAS_MAP = {
    "red deer, alberta": "red deer",
    "red deer alberta": "red deer",
    "other tms city": "calgary",
    "central services": "calgary",
}

TYPE_BUCKETS = ["School", "Business", "Church", "Organization"]

LEGAL_SUFFIXES = {
    "inc", "ltd", "llc", "corp", "corporation", "co", "company",
    "limited", "incorporated"
}

# RapidFuzz tuning
HIGH_CONF = 92
LOW_CONF = 87
FUZZY_LIMIT = 15

# HELPERS

def clean_text_basic(s) -> str:
    """Lowercase, normalize spaces, remove punctuation, normalize non-breaking spaces."""
    # if pd.isna(s):
    #     return ""
    s = str(s).replace("\u00A0", " ")  # Excel non breaking space
    s = s.strip().lower()
    s = re.sub(r"&", " and ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def clean_city(s) -> str:
    return clean_text_basic(s)

def extract_email_domain(contact_info: str) -> str:
    if pd.isna(contact_info):
        return ""
    s = str(contact_info).lower()
    m = re.search(r"@([a-z0-9\.-]+\.[a-z]{2,})", s)
    return m.group(1) if m else ""

def bucket_type(org_type: str) -> str:
    t = clean_text_basic(org_type)
    if not t:
        return "Organization"
    if any(x in t for x in ["university", "college", "school"]):
        return "School"
    if any(x in t for x in ["church", "chapel"]):
        return "Church"
    if any(x in t for x in ["business", "company", "inc", "ltd"]):
        return "Business"
    return "Organization"

def strip_legal_suffix(name: str) -> str:
    """Remove legal suffix tokens but KEEP SPACES"""
    name = clean_text_basic(name)
    if not name:
        return ""
    words = [w for w in name.split() if w not in LEGAL_SUFFIXES]
    return " ".join(words).strip()

def load_optional_map(path: str, key_col: str, val_col: str) -> dict:
    """Load mapping csv; cleans keys/values with clean_text_basic()."""
    try:
        dfm = pd.read_csv(path, encoding="utf-8-sig")
        if dfm.shape[1] ==1:
            dfm = pd.read_csv(path, encoding="utf-8-sig", sep=";")
            if dfm.shape[1]==1:
                dfm = pd.read_csv(path, encoding="utf-8-sig", sep="\t")

        dfm.columns = (
            dfm.columns
            .astype(str)
            .str.replace("\ufeff", "",regex=False)
            .str.strip()
            .str.lower()
        )

        key_col_clean = key_col.strip().lower()
        val_col_clean = val_col.strip().lower()

        print (f"\nLoaded {path}columns:", list(dfm.columns))

        if key_col_clean not in dfm.columns or val_col_clean not in dfm.columns:
            print(f"WARNING: {path} missing columns {key_col}/{val_col}. Skipping")
            return {}

        dfm[key_col_clean]= dfm[key_col_clean].apply(clean_text_basic)
        dfm[val_col_clean]= dfm[val_col_clean].apply(clean_text_basic)
        dfm = dfm[dfm[key_col_clean] != ""]

        return dict(zip(dfm[key_col_clean], dfm[val_col_clean]))

    except FileNotFoundError:
        print (f"WARNING: {path} not fnd. Skipping.")
        return{}
    except Exception as e:
        print (f"WARNING: cOULD NIT read {path} ({e}). Skipping.")
        return{}

def format_date_dmy(dt):
    if pd.isna(dt):
        return ""
    return f"{int(dt.day)}/{int(dt.month)}/{int(dt.year)}"

# LOAD RAW DATA

raw = pd.read_csv(INPUT_FILE, encoding="latin1")
raw.columns = [c.strip() for c in raw.columns]

use_cols = [c for c in DESIRED_COLS if c in raw.columns]
df = raw[use_cols].copy()

cleaned_df = df.map(lambda x: x.clean_text_basic() if isinstance(x, str) else x) 
print(cleaned_df)
input()

# Stable Row ID for audit and tie breaks
df.insert(0, "Row ID", range(1, len(df) + 1))

# DATE PARSING

df["Date_dt"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
df["Date Display"] = df["Date_dt"].apply(format_date_dmy)
df["Year"] = df["Date_dt"].dt.year
df["YearMonth"] = df["Date_dt"].dt.to_period("M").astype(str)  # YYYY-MM


# CITY CLEAN + ALIAS

df["City Clean"] = df["City"].apply(clean_city)
df["City Canonical"] = df["City Clean"].map(CITY_ALIAS_MAP).fillna(df["City Clean"])


# ORG CLEAN + ALIASES + GROUPING

df["Org Clean"] = df["Organization Name"].apply(clean_text_basic)

# Apply typo corrections: alias_clean -> canonical_clean
alias_map = load_optional_map(ORG_ALIASES_FILE, "alias_clean", "canonical_clean")
df["Canonical Org Clean"] = df["Org Clean"].map(alias_map).fillna(df["Org Clean"])
df["Canonical Org Clean"] = df["Canonical Org Clean"].apply(strip_legal_suffix)

# Apply grouped names: canonical_clean -> reporting_org
group_map = load_optional_map(GROUPED_NAMES_FILE, "canonical_clean", "reporting_org")

df["Reporting Org Name"] = df["Canonical Org Clean"]
if group_map:
    df["Reporting Org Name"] = df["Canonical Org Clean"].map(group_map).fillna(df["Reporting Org Name"])

df["Reporting Org Name"] = df["Reporting Org Name"].apply(strip_legal_suffix)
df["Reporting Org Clean"] = df["Reporting Org Name"].apply(clean_text_basic)


# TYPE BUCKET

df["Organization Type"] = df["Organization Type"].apply(bucket_type)
df["Type Bucket"] = df["Organization Type"]


# CONTACT FIELDS

df["Contact Clean"] = df["Contact Name"].apply(clean_text_basic)
df["Email Domain"] = df["Contact Info"].apply(extract_email_domain)


# EXCLUSIONS (DO NOT DROP)

df["Excluded"] = 0
df["Exclusion Reason"] = ""

missing_org = df["Org Clean"].eq("")
missing_date = df["Date_dt"].isna()
mask_missing = missing_org | missing_date
df.loc[mask_missing, "Excluded"] = 1
df.loc[mask_missing, "Exclusion Reason"] = "Missing Org Name or Date"

not_2025 = df["Year"].ne(2025)
df.loc[not_2025, "Excluded"] = 1
df.loc[not_2025 & (df["Exclusion Reason"] == ""), "Exclusion Reason"] = "Not in 2025"

# Raw org type exactly "TMS Internal" (use raw values)
org_type_raw = raw["Organization Type"].astype(str).str.strip().str.lower()
tms_internal_type = org_type_raw.eq("tms internal")
df.loc[tms_internal_type, "Excluded"] = 1
df.loc[tms_internal_type & (df["Exclusion Reason"] == ""), "Exclusion Reason"] = "TMS Internal (Org Type)"

kw_pattern = r"\b(" + "|".join(map(re.escape, EXCLUDE_KEYWORDS_IN_ORGNAME)) + r")\b"
has_kw = df["Org Clean"].str.contains(kw_pattern, na=False, regex=True)
df.loc[has_kw, "Excluded"] = 1
df.loc[has_kw & (df["Exclusion Reason"] == ""), "Exclusion Reason"] = "Internal keyword in Org Name"


# ORG KEY + FIRST CONTACT

df["Org Key"] = df["Reporting Org Clean"] + " | " + df["City Canonical"]

df["First Contact"] = 0
valid = df[df["Excluded"] == 0].copy().sort_values(["Date_dt", "Row ID"])
first_idx = valid.drop_duplicates(subset=["Org Key"], keep="first").index
df.loc[first_idx, "First Contact"] = 1

df["Countable"] = ((df["Excluded"] == 0) & (df["First Contact"] == 1)).astype(int)


'''# MONTHLY + YTD (MERGED INTO MASTER)

fc = df[df["Countable"] == 1].copy()

monthly = (
    fc.pivot_table(
        index=["YearMonth", "City Canonical"],
        columns="Type Bucket",
        values="Org Key",
        aggfunc=pd.Series.nunique,
        fill_value=0
    )
    .reset_index()
)

for c in TYPE_BUCKETS:
    if c not in monthly.columns:
        monthly[c] = 0

monthly["Monthly Total"] = monthly[TYPE_BUCKETS].sum(axis=1)

monthly["YearMonth_dt"] = pd.to_datetime(monthly["YearMonth"] + "-01", errors="coerce")
monthly["Year"] = monthly["YearMonth_dt"].dt.year
monthly = monthly.sort_values(["Year", "City Canonical", "YearMonth_dt"])

for c in TYPE_BUCKETS:
    monthly[f"{c} YTD"] = monthly.groupby(["Year", "City Canonical"])[c].cumsum()
monthly["Monthly Total YTD"] = monthly.groupby(["Year", "City Canonical"])["Monthly Total"].cumsum()

merge_cols = ["YearMonth", "City Canonical"] + TYPE_BUCKETS + ["Monthly Total"] + \
             [f"{c} YTD" for c in TYPE_BUCKETS] + ["Monthly Total YTD"]

df = df.merge(monthly[merge_cols], on=["YearMonth", "City Canonical"], how="left")

fill_cols = TYPE_BUCKETS + ["Monthly Total"] + [f"{c} YTD" for c in TYPE_BUCKETS] + ["Monthly Total YTD"]
for c in fill_cols:
    df[c] = df[c].fillna(0).astype(int)'''


# RAPIDFUZZ SUGGESTIONS 

try:
    from rapidfuzz import process, fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

if RAPIDFUZZ_AVAILABLE:
    suggestions = []
    seen_pairs = set()

    cand = df[df["Excluded"] == 0][["City Canonical", "Canonical Org Clean"]].dropna().copy()

    for city, g in cand.groupby("City Canonical"):
        names = sorted([x for x in g["Canonical Org Clean"].unique().tolist() if x])
        if len(names) < 2:
            continue

        for name in names:
            matches = process.extract(
                query=name,
                choices=names,
                scorer=fuzz.token_sort_ratio,
                limit=FUZZY_LIMIT
            )
            for match_name, score, _ in matches:
                if match_name == name or score < LOW_CONF:
                    continue

                if len(name) < 5 or len(match_name) < 5:
                    continue

                pair = (city,) + tuple(sorted([name, match_name]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                confidence = "HIGH" if score >= HIGH_CONF else "LOW"

                suggestions.append({
                    "City": city,
                    "alias_clean": pair[1],
                    "canonical_suggestion": pair[2],
                    "score": score,
                    "confidence": confidence,
                    "Action": ""  # fill: APPROVE / IGNORE 
                })

    pd.DataFrame(suggestions).sort_values(["City", "score"], ascending=[True, False]).to_csv(
        SUGGESTED_ALIASES_FILE, index=False
    )


# MASTER OUTPUT 

# build master without the triple-quoted literal inside the dict
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

MASTER_COL_ORDER = [
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

master = master[MASTER_COL_ORDER]
master.to_csv(MASTER_OUT, index=False)

# CITY + YEAR TOTALS 

city_year = (
    df[df["Countable"] == 1]
      .groupby(["Year", "City Canonical"], as_index=False)["Org Key"]
      .nunique()
      .rename(columns={"City Canonical": "City", "Org Key": "Year Total (All Types)"})
)

# MONTH per ORG per CITY 
# Uses ONLY Countable rows (Excluded=0 AND First Contact=1)

MONTH_ORG_CITY_OUT = "community_engagement_v2_MONTH_ORG_CITY.csv"

month_org_city = (
    df[df["Countable"] == 1]
      .groupby(["YearMonth", "City Canonical", "Reporting Org Name"], as_index=False)
      .agg(
          First_Contacts=("Org Key", "nunique"),
          First_Contact_Date=("Date_dt", "min"),
      )
)

# Format date for easy viewing (D/M/YYYY)
month_org_city["First Contact Date"] = month_org_city["First_Contact_Date"].apply(format_date_dmy)
month_org_city = month_org_city.drop(columns=["First_Contact_Date"])

# Column order exactly
month_org_city = month_org_city.rename(columns={
    "YearMonth": "Month",
    "City Canonical": "City",
    "Reporting Org Name": "Organization",
    "First_Contacts": "First Contacts (Count)"
})

month_org_city = month_org_city[
    ["Month", "City", "Organization", "First Contacts (Count)", "First Contact Date"]
].sort_values(["Month", "City", "Organization"])


'''#MONTH per Org Typer per City
type_total = (
    df[df["Countable"] == 1]
      .groupby(["YearMonth", "City Canonical", "Type Bucket"], as_index=False)
      .agg(
          First_Contacts=("Org Key", "nunique"),
      )
)'''



month_org_city.to_csv(MONTH_ORG_CITY_OUT, index=False)
print("Saved:", MONTH_ORG_CITY_OUT)
year_total = city_year.groupby("Year", as_index=False)["Year Total (All Types)"].sum()
year_total["City"] = "Total"

city_year = pd.concat([city_year, year_total], ignore_index=True)
city_year.to_csv(CITY_YEAR_OUT, index=False)

print("Saved:", MASTER_OUT)
print("Saved:", CITY_YEAR_OUT)
if RAPIDFUZZ_AVAILABLE:
    print("Saved:", SUGGESTED_ALIASES_FILE)
else:
    print("RapidFuzz not installed -> skipped suggestions.")

print("RUNNING FROM:", os.getcwd())
print("org_aliases path:", os.path.abspath("org_aliases.csv"))
print("grouped_names path:", os.path.abspath("grouped_names.csv"))