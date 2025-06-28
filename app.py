import streamlit as st
import pandas as pd
from rapidfuzz import fuzz, process
import hashlib
import io
from pathlib import Path


def load_sample_files():
    sample_folder = Path("input/")
    dataframes = []
    file_headers = {}

    for file in sample_folder.glob("*"):
        if file.suffix.lower() not in [".csv", ".xlsx"]:
            continue
        try:
            if file.suffix.lower() == ".xlsx":
                df = pd.read_excel(file, engine="openpyxl")
            else:
                df = pd.read_csv(file)

            df.columns = [col.strip().title() for col in df.columns]
            df["__source_file__"] = file.name
            df["row_number"] = df.index + 2
            dataframes.append(df)
            file_headers[file.name] = set(df.columns)
        except Exception as e:
            st.error(f"❌ Could not read sample file {file.name}: {e}")

    return dataframes, file_headers

def preview_sample_files():
    sample_folder = Path("input/")
    previews = []

    for file in sample_folder.glob("*"):
        if file.suffix.lower() not in [".csv", ".xlsx"]:
            continue
        try:
            if file.suffix.lower() == ".xlsx":
                df = pd.read_excel(file, engine="openpyxl", nrows=5)
            else:
                df = pd.read_csv(file, nrows=5)

            previews.append((file.name, df))
        except Exception as e:
            st.warning(f"Could not preview {file.name}: {e}")
    
    return previews



# ---- Page Config ----
st.set_page_config(
    layout="wide",
    page_title="FuzzyMerge: Smart Data Cleaner",
    page_icon="🐢"
)

# ---- Custom Styles ----
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    font-size: 16px;
}
.stApp {
    background-color: #f9fafb;
}
    .stFileUploader {
        cursor: pointer !important;
    }
h1 {
    font-size: 2.8rem;
    font-weight: 700;
    color: #ffffff;
}
h2 {
    font-size: 1.6rem;
    font-weight: 600;
    color: #1f2937;
}
.stButton>button {
    border-radius: 8px;
    padding: 0.5em 1em;
    background: linear-gradient(to right, #6366F1, #3B82F6);
    color: white;
    font-weight: 500;
    border: none;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
}
.stButton>button:hover {
    background: linear-gradient(to right, #4338CA, #2563EB);
}
.st-expanderHeader {
    font-size: 1.2rem !important;
    font-weight: 600 !important;
}
.stRadio > div {
    gap: 0.5rem;
}
section[data-testid="stFileUploader"] {
    border: 2px dashed #3B82F6 !important;
    background-color: #f0f4ff;
    cursor: pointer !important;
    transition: background-color 0.3s ease;
}
section[data-testid="stFileUploader"]:hover {
    background-color: #e3eaff;
}
@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
.emoji-spin {
  display: inline-block;
  animation: spin 1.5s linear infinite;
  font-size: 2.5rem;
  vertical-align: middle;
}
</style>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown("""
<div style="text-align: center; padding: 1.5rem 0; background: linear-gradient(to right, #6366F1, #3B82F6); color: white; border-radius: 12px; margin-bottom: 2rem;">
    <h1 style="margin-bottom: 0.3rem;">
        <span class="emoji-spin">🐢</span> FuzzyMerge
    </h1>
    <p style="font-size: 1.1rem; font-weight: 500;">Clean your messy data... with a little help from Tuco the Clean Turtle</p>
</div>
""", unsafe_allow_html=True)


# ---- Helpers ----
def make_safe_key(col, val1, val2):
    return hashlib.md5(f"{col}_{val1}_{val2}".encode()).hexdigest()

def infer_dtype(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    elif pd.api.types.is_numeric_dtype(series):
        return "numeric"
    else:
        return "string"

if "step" not in st.session_state:
    st.session_state.step = 0

# ---- Step 0: Upload ----
# ---- Step 0: Upload ----
if st.session_state.step == 0:
    st.markdown("🐢 **Tuco says:** Drop those messy files in here, friend. I’ll take good care of 'em.")
    uploaded_files = st.file_uploader("📁 Upload CSV or Excel files", type=["csv", "xlsx"], accept_multiple_files=True)

    with st.expander("📋 Or preview and test Tuco with these problematic sample files", expanded=False):
        st.markdown("Here’s a quick look at the sample files Tuco has ready for you:")
    
        previews = preview_sample_files()
        for fname, df in previews:
            st.markdown(f"**📄 {fname}**")
            st.dataframe(df, use_container_width=True)

        st.markdown("---")
        if st.button("🚀 Use These Sample Files"):
            dataframes, file_headers = load_sample_files()
            st.session_state.dataframes = dataframes
            st.session_state.file_headers = file_headers
            st.session_state.step = 0.5
            st.rerun()

    if uploaded_files:
        dataframes = []
        file_headers = {}

        for file in uploaded_files:
            df = pd.read_excel(file) if file.name.endswith(".xlsx") else pd.read_csv(file)
            df.columns = [col.strip().title() for col in df.columns]
            df["__source_file__"] = file.name
            df["row_number"] = df.index + 2
            dataframes.append(df)
            file_headers[file.name] = set(df.columns)

        st.session_state.dataframes = dataframes
        st.session_state.file_headers = file_headers
        st.session_state.step = 0.5
        st.rerun()



# ---- Step 0.5: Rename Unique Columns ----
# ---- Step 0.5: Rename Unique Columns ----
elif st.session_state.step == 0.5:
    dataframes = st.session_state.dataframes
    file_headers = st.session_state.file_headers

    # First pass: get common/unique columns (excluding helper cols)
    header_sets = list(file_headers.values())
    common_cols = set.intersection(*header_sets) - {"__source_file__", "row_number"}
    all_cols = set.union(*header_sets)

    unique_cols_per_file = {
        fname: cols - common_cols - {"__source_file__", "row_number"}
        for fname, cols in file_headers.items()
    }

    unique_cols_exist = any(uniques for uniques in unique_cols_per_file.values())

    if unique_cols_exist:
        st.markdown("🐢 **Tuco says:** Found some columns that don’t match across files. Wanna rename them to line up?")
    else:
        st.markdown("🐢 **Tuco says:** All columns already match across your files. Nothing to rename!")
        if st.button("➡️ Continue to Common Column Confirmation"):
            st.session_state.step = 1
            st.rerun()
        st.stop()

    rename_map = {}

    st.markdown("### 🧩 Unique Columns Per File")
    for fname, uniques in unique_cols_per_file.items():
        if not uniques:
            continue

        st.subheader(f"📄 {fname}")
        df = next(df for df in dataframes if df['__source_file__'].iloc[0] == fname)

        for col in sorted(uniques):
            new_col = st.text_input(
                f"Rename '{col}' in {fname}",
                value=col,
                key=f"{fname}_{col}"
            )
            if new_col != col and new_col.strip():
                rename_map[(fname, col)] = new_col.strip()

    if st.button("✅ Apply Renames"):
        # Apply renames
        for (fname, old_col), new_col in rename_map.items():
            df = next(df for df in dataframes if df["__source_file__"].iloc[0] == fname)
            df.rename(columns={old_col: new_col}, inplace=True)
            file_headers[fname].remove(old_col)
            file_headers[fname].add(new_col)

        # Recompute common columns
        header_sets = list(file_headers.values())
        common_cols = set.intersection(*header_sets) - {"__source_file__", "row_number"}

        if not common_cols:
            st.error("❌ Still no common columns found after renaming. Please review your changes.")
        else:
            st.success("✅ Renames applied! Proceeding with updated common columns.")
            st.session_state.dataframes = dataframes
            st.session_state.file_headers = file_headers
            st.session_state.step = 1
            st.rerun()



# ---- Step 1: Confirm Columns ----
elif st.session_state.step == 1:
    st.markdown("🐢 **Tuco says:** I sniffed out the common columns. These ones can work together like good turtle friends.")
    dataframes = st.session_state.dataframes
    file_headers = st.session_state.file_headers
    common_cols = set.intersection(*list(file_headers.values())) - {"__source_file__", "row_number"}

    st.success("✅ Common Columns Found")
    st.write(sorted(common_cols))

    st.warning("🧩 Unique Columns Per File")
    for fname, cols in file_headers.items():
        unique = sorted(cols - common_cols - {"__source_file__", "row_number"})
        if unique:
            st.markdown(f"**{fname}**: {', '.join(unique)}")

    if st.button("➡️ Continue with Common Columns"):
        cols = list(common_cols) + ["__source_file__", "row_number"]
        merged = pd.concat([df[cols] for df in dataframes], ignore_index=True)
        st.session_state.merged = merged
        st.session_state.step = 2
        st.rerun()

# ---- Step 2: Fix Types ----
elif st.session_state.step == 2:
    st.markdown("🐢 **Tuco says:** Time to tell me what kind of data we’re dealing with. I don’t judge — strings, numbers, dates — all are welcome.")
    merged = st.session_state.merged

    if "dtype_map" not in st.session_state:
        st.session_state.dtype_map = {
            col: infer_dtype(merged[col])
            for col in merged.columns
            if col not in ["__source_file__", "row_number"]
        }

    dtype_map = st.session_state.dtype_map
    for col in dtype_map:
        dtype_map[col] = st.selectbox(f"{col} type", ["string", "numeric", "datetime"],
                                      index=["string", "numeric", "datetime"].index(dtype_map[col]))

    if st.button("✅ Apply & Check for Missing"):
        for col, dtype in dtype_map.items():
            try:
                if dtype == "numeric":
                    merged[col] = pd.to_numeric(merged[col], errors="coerce")
                elif dtype == "datetime":
                    merged[col] = pd.to_datetime(merged[col], errors="coerce")
                else:
                    merged[col] = merged[col].astype(str)
            except Exception as e:
                st.warning(f"⚠️ Could not convert {col}: {e}")
        st.session_state.merged = merged
        st.session_state.step = 2.5
        st.rerun()

# ---- Step 2.5: Show NaNs ----
# ---- Step 2.5: Show NaNs ----
elif st.session_state.step == 2.5:
    st.markdown("🐢 **Tuco says:** Uh-oh, found a few blanks! Nothing we can't handle, but better to know.")

    merged = st.session_state.merged
    check_cols = [col for col in merged.columns if col not in ["__source_file__", "row_number"]]
    nan_rows = merged[merged[check_cols].isna().any(axis=1)]

    col_missing_counts = merged[check_cols].isna().sum()
    col_missing_counts = col_missing_counts[col_missing_counts > 0]

    if nan_rows.empty:
        st.success("✅ No missing values found.")
        if st.button("➡️ Continue to Outlier Detection"):
            st.session_state.step = 2.6
            st.rerun()
    else:
        st.warning(f"⚠️ Found {len(nan_rows)} rows with missing values.")
        st.dataframe(nan_rows, use_container_width=True)

        st.markdown("#### Missing Value Count Per Column")
        st.markdown("#### 🚨 Missing Value Breakdown")
        for col, count in col_missing_counts.items():
            st.markdown(f"⚠️ **{col}**: `{count}` missing")


        st.markdown("### What would you like to do?")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("✅ Merge Anyway"):
                st.session_state.nan_rows = nan_rows  # store for logs
                st.session_state.step = 2.6
                st.rerun()

        with col2:
            if st.button("📝 Add to Log & Remove from Data"):
                st.session_state.nan_rows = nan_rows  # store for logs
                st.session_state.merged = merged.drop(nan_rows.index).reset_index(drop=True)
                st.success("🔍 Rows with missing values have been removed from the dataset.")
                st.session_state.step = 2.6
                st.rerun()


# ---- Step 2.6: Detect Numeric Outliers ----
elif st.session_state.step == 2.6:
    st.markdown("🐢 **Tuco says:** I sniffed out some numbers that just don’t fit in. Let’s highlight the oddballs.")

    merged = st.session_state.merged.copy()
    numeric_cols = merged.select_dtypes(include='number').columns.tolist()

    def detect_outlier_mask(data, iqr_multiplier=1.5):
        mask = pd.DataFrame(False, index=data.index, columns=data.columns)
        for col in numeric_cols:
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr
            mask[col] = (data[col] < lower) | (data[col] > upper)
        return mask

    # Detect mask and filtered outliers
    outlier_mask = detect_outlier_mask(merged)
    has_outliers = outlier_mask.any(axis=1)
    outlier_df = merged[has_outliers]
    outlier_style_mask = outlier_mask[has_outliers]

    def highlight_outliers(val, is_outlier):
        return 'background-color: #fca5a5; font-weight: bold;' if is_outlier else ''

    if outlier_df.empty:
        st.success("✅ No numeric outliers detected.")
    else:
        st.warning(f"⚠️ Detected {len(outlier_df)} rows with numeric outliers.")

        styled = outlier_df.style.apply(
            lambda df: outlier_style_mask.applymap(lambda x: 'background-color: #fca5a5; font-weight: bold;' if x else ''),
            axis=None
        )

        st.dataframe(styled, use_container_width=True)

    if st.button("➡️ Continue to Fuzzy Matching"):
        st.session_state.step = 3
        st.rerun()


# ---- Step 3: Fuzzy Matching ----
elif st.session_state.step == 3:
    st.markdown("🐢 **Tuco says:** These values look suspiciously similar... Want me to tidy them up? 🧐")
    df = st.session_state.merged.copy()
    dtype_map = st.session_state.dtype_map

    for col in dtype_map:
        if dtype_map[col] != "string":
            continue

        st.subheader(f"🔍 Fuzzy Matching — `{col}`")

        if f"fixed_{col}" not in st.session_state:
            st.session_state[f"fixed_{col}"] = {}

        fixed_values = st.session_state[f"fixed_{col}"]
        seen_pairs = set()
        unique_vals = df[col].dropna().unique().tolist()
        unique_vals = sorted(set(unique_vals) - set(fixed_values.keys()), key=str)

        for val in unique_vals:
            matches = process.extract(val, unique_vals, scorer=fuzz.ratio, limit=None)
            for other, score, _ in matches:
                if val == other or score < 30 or (other, val) in seen_pairs:
                    continue
                seen_pairs.add((val, other))
                key = make_safe_key(col, val, other)

                val_source = df[df[col] == val][["__source_file__", "row_number"]].iloc[0]
                other_source = df[df[col] == other][["__source_file__", "row_number"]].iloc[0]

                with st.expander(f"🧐 `{val}` ↔ `{other}` ({int(score)}%)"):
                    st.markdown(f"📄 `{val}` → **{val_source['__source_file__']}**, row {val_source['row_number']}")
                    st.markdown(f"📄 `{other}` → **{other_source['__source_file__']}**, row {other_source['row_number']}")

                    choice = st.radio("Same value?", ["Skip", "Yes", "No"], key=f"{key}_choice")
                    if choice == "Yes":
                        replacement = st.text_input("Replace both with:", value=val, key=f"{key}_yes")
                        if st.button("Confirm", key=f"{key}_yes_btn"):
                            df[col] = df[col].replace({val: replacement, other: replacement})
                            fixed_values[val] = replacement
                            fixed_values[other] = replacement
                            st.session_state.merged = df
                            st.rerun()
                    elif choice == "No":
                        val_new = st.text_input(f"Replace `{val}` with:", key=f"{key}_val")
                        other_new = st.text_input(f"Replace `{other}` with:", key=f"{key}_other")
                        if st.button("Confirm", key=f"{key}_no_btn"):
                            if val_new:
                                df[col] = df[col].replace(val, val_new)
                                fixed_values[val] = val_new
                            if other_new:
                                df[col] = df[col].replace(other, other_new)
                                fixed_values[other] = other_new
                            st.session_state.merged = df
                            st.rerun()
                break

    if st.button("➡️ Continue to Deduplication"):
        st.session_state.fuzzy_cleaned = df
        st.session_state.step = 4
        st.rerun()

# ---- Step 4: Deduplication ----
elif st.session_state.step == 4:
    st.markdown("🐢 **Tuco says:** Sometimes data likes to repeat itself. I’ll help you clean it gently by either keeping all or keeping just one record in each dupe group.")
    df = st.session_state.fuzzy_cleaned
    cols_to_check = [col for col in df.columns if col != "row_number"]
    dupes = df[df.duplicated(subset=cols_to_check, keep=False)]

    if dupes.empty:
        st.success("✅ No duplicates found.")
        st.session_state.deduped = df
    else:
        df["group_num"] = df.groupby(cols_to_check).ngroup()
        unique_groups = df[df.duplicated(subset=cols_to_check, keep=False)]["group_num"].unique()
        cleaned = []

        for group in unique_groups:
            group_df = df[df["group_num"] == group]
            st.subheader("🔁 Duplicate Group")
            st.dataframe(group_df.drop(columns="group_num"))
            action = st.radio("Keep all or remove duplicates?", ["Keep All", "Remove Duplicates"], key=f"group_{group}")
            if action == "Keep All":
                cleaned.append(group_df)
            else:
                cleaned.append(group_df.iloc[[0]])

        deduped = pd.concat(cleaned + [df[~df["group_num"].isin(unique_groups)]], ignore_index=True)
        st.session_state.deduped = deduped.drop(columns="group_num")

    if st.button("📥 Proceed to Download"):
        st.session_state.step = 5
        st.rerun()

# ---- Step 5: Download ----
elif st.session_state.step == 5:
    st.markdown("🐢 **Tuco says:** All done! Here's your fresh, squeaky-clean dataset — with a little 🐢 love.")
    result_df = st.session_state.deduped
    st.dataframe(result_df)

    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", csv, file_name="cleaned_data.csv", mime="text/csv")

    fuzzy_logs = []
    for key in st.session_state.keys():
        if key.startswith("fixed_"):
            col = key.replace("fixed_", "")
            mapping = st.session_state[key]
            for original, replacement in mapping.items():
                fuzzy_logs.append({"column": col, "original": original, "replacement": replacement})
    fuzzy_df = pd.DataFrame(fuzzy_logs)

    deduped_rows = st.session_state.fuzzy_cleaned.merge(
        st.session_state.deduped,
        how="outer",
        indicator=True
    ).query('_merge == "left_only"').drop(columns=["_merge"])

    nan_rows = st.session_state.get("nan_rows", pd.DataFrame())

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        result_df.to_excel(writer, sheet_name="Cleaned Data", index=False)
        if not fuzzy_df.empty:
            fuzzy_df.to_excel(writer, sheet_name="Fuzzy Mappings", index=False)
        if not deduped_rows.empty:
            deduped_rows.to_excel(writer, sheet_name="Duplicates Removed", index=False)
        if not nan_rows.empty:
            nan_rows.to_excel(writer, sheet_name="Missing Values", index=False)

    excel_data = output.getvalue()
    st.download_button(
        label="📘 Download Excel with Logs",
        data=excel_data,
        file_name="cleaned_data_with_logs.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
