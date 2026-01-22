import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# 0) Mac absolute path (your requested style)
# ============================================================
BASE_DIR = "/Users/xujiahe/Downloads/"
OUT_XLSX = BASE_DIR + "demographics_analysis_report.xlsx"

# Optional: if you want to ensure minus sign displays correctly
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 1) Read all demographics CSV files
# ============================================================
def read_all_demographics_files():
    """Read all demographics CSV files (Task 1–Task 5) from Mac Downloads folder."""
    files = {
        "Task 1": BASE_DIR + "demographics_task1.csv",
        "Task 2": BASE_DIR + "demographics_task2.csv",
        "Task 3": BASE_DIR + "demographics_task3.csv",
        "Task 4": BASE_DIR + "demographics_task4.csv",
        "Task 5": BASE_DIR + "demographics_task5.csv",
    }

    dataframes = {}
    for task_name, file_name in files.items():
        try:
            df = pd.read_csv(file_name)
            df["Task"] = task_name
            dataframes[task_name] = df
            print(f"Loaded {task_name}: {len(df)} records")
        except FileNotFoundError:
            print(f"WARNING: File not found: {file_name}")
            continue

    return dataframes


# ============================================================
# 2) Overview analysis
# ============================================================
def analyze_data_overview(dataframes):
    """Summarize status counts and approval rate per task."""
    print("\n" + "=" * 60)
    print("Data Overview")
    print("=" * 60)

    overview_stats = []
    for task_name, df in dataframes.items():
        total_records = len(df)
        approved = int((df["Status"] == "APPROVED").sum()) if "Status" in df.columns else 0
        returned = int((df["Status"] == "RETURNED").sum()) if "Status" in df.columns else 0
        timed_out = int((df["Status"] == "TIMED-OUT").sum()) if "Status" in df.columns else 0
        consent_revoked = int((df["Status"] == "CONSENT_REVOKED").sum()) if "Status" in df.columns else 0

        # DATA_EXPIRED may appear outside Status column
        data_expired = int(df.apply(lambda x: "DATA_EXPIRED" in str(x).upper(), axis=1).sum())

        approved_rate = (approved / total_records * 100) if total_records > 0 else 0

        overview_stats.append(
            {
                "Task": task_name,
                "Total records": total_records,
                "APPROVED": approved,
                "RETURNED": returned,
                "TIMED-OUT": timed_out,
                "CONSENT_REVOKED": consent_revoked,
                "DATA_EXPIRED": data_expired,
                "Approved rate (%)": round(approved_rate, 1),
            }
        )

    overview_df = pd.DataFrame(overview_stats)
    print(overview_df.to_string(index=False))

    totals = overview_df.drop(columns=["Task", "Approved rate (%)"]).sum(numeric_only=True)
    total_all = int(totals.get("Total records", 0))
    total_approved = int(totals.get("APPROVED", 0))
    overall_rate = (total_approved / total_all * 100) if total_all > 0 else 0

    print("\nTotals:")
    print(f"Total records: {total_all}")
    print(f"Total APPROVED: {total_approved}")
    print(f"Overall approval rate: {overall_rate:.1f}%")

    return overview_df


# ============================================================
# 3) Demographics analysis (APPROVED only)
# ============================================================
def analyze_demographics(dataframes):
    """Analyze participant demographics using APPROVED records only."""
    print("\n" + "=" * 60)
    print("Participant Demographics (APPROVED only)")
    print("=" * 60)

    approved_dfs = []
    for task_name, df in dataframes.items():
        if "Status" not in df.columns:
            continue
        approved_df = df[df["Status"] == "APPROVED"].copy()
        approved_dfs.append(approved_df)

    if not approved_dfs:
        print("No APPROVED records found.")
        return None, None

    combined_df = pd.concat(approved_dfs, ignore_index=True)
    total_participants = len(combined_df)
    print(f"Total APPROVED participants: {total_participants}")

    analyses = {}

    # Sex
    if "Sex" in combined_df.columns:
        counts = combined_df["Sex"].value_counts()
        perc = (counts / total_participants * 100).round(1)
        analyses["Sex distribution"] = pd.DataFrame({"Count": counts, "Percent": perc})

    # Age
    combined_df["Age_numeric"] = pd.to_numeric(combined_df.get("Age"), errors="coerce")
    age_mean = combined_df["Age_numeric"].mean()

    age_bins = [0, 20, 30, 40, 50, 60, 100]
    age_labels = ["<20", "20-29", "30-39", "40-49", "50-59", "60+"]
    combined_df["Age_group"] = pd.cut(combined_df["Age_numeric"], bins=age_bins, labels=age_labels, right=False)

    age_counts = combined_df["Age_group"].value_counts().reindex(age_labels)
    age_percent = (age_counts / total_participants * 100).round(1)
    analyses["Age distribution"] = pd.DataFrame({"Count": age_counts, "Percent": age_percent})

    # Ethnicity
    if "Ethnicity simplified" in combined_df.columns:
        counts = combined_df["Ethnicity simplified"].value_counts()
        perc = (counts / total_participants * 100).round(1)
        analyses["Ethnicity distribution"] = pd.DataFrame({"Count": counts, "Percent": perc})

    # Country of birth
    if "Country of birth" in combined_df.columns:
        counts = combined_df["Country of birth"].value_counts()
        perc = (counts / total_participants * 100).round(1)
        analyses["Country of birth distribution"] = pd.DataFrame({"Count": counts, "Percent": perc})

    # Country of residence
    if "Country of residence" in combined_df.columns:
        counts = combined_df["Country of residence"].value_counts()
        perc = (counts / total_participants * 100).round(1)
        analyses["Country of residence distribution"] = pd.DataFrame({"Count": counts, "Percent": perc})

    # Language
    if "Language" in combined_df.columns:
        counts = combined_df["Language"].value_counts()
        perc = (counts / total_participants * 100).round(1)
        analyses["Language distribution"] = pd.DataFrame({"Count": counts, "Percent": perc})

    # Student status
    if "Student status" in combined_df.columns:
        counts = combined_df["Student status"].value_counts()
        perc = (counts / total_participants * 100).round(1)
        analyses["Student status distribution"] = pd.DataFrame({"Count": counts, "Percent": perc})

    # Employment status
    if "Employment status" in combined_df.columns:
        counts = combined_df["Employment status"].value_counts()
        perc = (counts / total_participants * 100).round(1)
        analyses["Employment status distribution"] = pd.DataFrame({"Count": counts, "Percent": perc})

    for name, table in analyses.items():
        print(f"\n{name}:")
        print(table.to_string())

    print(f"\nMean age: {age_mean:.1f}")

    return combined_df, analyses


# ============================================================
# 4) Completion time analysis (APPROVED only)
# ============================================================
def analyze_task_completion_time(dataframes):
    """Analyze completion time for APPROVED records."""
    print("\n" + "=" * 60)
    print("Task Completion Time (APPROVED only)")
    print("=" * 60)

    rows = []
    all_times = []

    for task_name, df in dataframes.items():
        if "Status" not in df.columns:
            continue
        approved_df = df[df["Status"] == "APPROVED"].copy()
        if len(approved_df) == 0:
            continue

        t = pd.to_numeric(approved_df.get("Time taken"), errors="coerce").dropna()
        if len(t) == 0:
            continue

        rows.append(
            {
                "Task": task_name,
                "N": len(t),
                "Mean (sec)": round(t.mean(), 1),
                "Min (sec)": round(t.min(), 1),
                "Max (sec)": round(t.max(), 1),
                "Std (sec)": round(t.std(), 1),
                "Median (sec)": round(t.median(), 1),
            }
        )
        all_times.extend(t.tolist())

    time_df = pd.DataFrame(rows)
    if len(time_df) > 0:
        print(time_df.to_string(index=False))

    if all_times:
        s = pd.Series(all_times)
        print("\nOverall:")
        print(f"Overall mean: {s.mean():.1f} sec")
        print(f"Overall min:  {s.min():.1f} sec")
        print(f"Overall max:  {s.max():.1f} sec")
        print(f"Overall std:  {s.std():.1f} sec")
        print(f"Overall median: {s.median():.1f} sec")

    return time_df


# ============================================================
# 5) Participation time distribution (APPROVED only)
# ============================================================
def analyze_participation_time(dataframes):
    """Analyze when participants started tasks (hour/period) using APPROVED records."""
    print("\n" + "=" * 60)
    print("Participation Time Distribution (APPROVED only)")
    print("=" * 60)

    records = []
    for task_name, df in dataframes.items():
        if "Status" not in df.columns:
            continue
        approved_df = df[df["Status"] == "APPROVED"].copy()
        if len(approved_df) == 0:
            continue

        for _, row in approved_df.iterrows():
            start_time_str = row.get("Started at", None)
            if pd.isna(start_time_str):
                continue
            try:
                dt = pd.to_datetime(start_time_str)
                records.append(
                    {
                        "Task": task_name,
                        "Participant": row.get("Participant id", "Unknown"),
                        "Hour": dt.hour,
                        "Date": dt.date(),
                        "Time": dt.time(),
                    }
                )
            except Exception:
                continue

    if not records:
        print("No usable start-time data found.")
        return None, None

    time_df = pd.DataFrame(records)

    def hour_to_period(hour):
        if 6 <= hour < 9:
            return "06:00-09:00"
        elif 9 <= hour < 12:
            return "09:00-12:00"
        elif 12 <= hour < 15:
            return "12:00-15:00"
        elif 15 <= hour < 18:
            return "15:00-18:00"
        elif 18 <= hour < 21:
            return "18:00-21:00"
        else:
            return "Other hours"

    time_df["Period"] = time_df["Hour"].apply(hour_to_period)
    period_counts = time_df["Period"].value_counts()
    period_percent = (period_counts / len(time_df) * 100).round(1)

    period_df = pd.DataFrame({"Starts": period_counts, "Percent": period_percent})

    print("Start-time period distribution:")
    print(period_df.to_string())

    most_active = period_counts.idxmax()
    most_active_count = int(period_counts.max())
    most_active_percent = float(period_percent.loc[most_active])

    print(f"\nMost active period: {most_active} ({most_active_count} starts, {most_active_percent}%)")

    return time_df, period_df


# ============================================================
# 6) Participation level analysis (Total approvals)
# ============================================================
def analyze_participation_level(dataframes):
    """Analyze participation history based on 'Total approvals' for APPROVED records."""
    print("\n" + "=" * 60)
    print("Participation Level (based on Total approvals)")
    print("=" * 60)

    rows = []
    for task_name, df in dataframes.items():
        if "Status" not in df.columns:
            continue
        approved_df = df[df["Status"] == "APPROVED"].copy()
        if len(approved_df) == 0:
            continue

        approvals = pd.to_numeric(approved_df.get("Total approvals"), errors="coerce")

        # safer row-wise iteration
        tmp = approved_df.copy()
        tmp["_approvals"] = approvals

        for _, row in tmp.iterrows():
            val = row.get("_approvals", np.nan)
            if pd.isna(val):
                continue
            rows.append(
                {
                    "Task": task_name,
                    "Participant": row.get("Participant id", "Unknown"),
                    "Total_approvals": float(val),
                }
            )

    if not rows:
        print("No usable participation-history data found.")
        return None

    approvals_df = pd.DataFrame(rows)

    avg_approvals = approvals_df["Total_approvals"].mean()
    max_approvals = approvals_df["Total_approvals"].max()
    min_approvals = approvals_df["Total_approvals"].min()
    median_approvals = approvals_df["Total_approvals"].median()
    std_approvals = approvals_df["Total_approvals"].std()

    print(f"Mean approvals history: {avg_approvals:.0f}")
    print(f"Max approvals history:  {max_approvals:.0f}")
    print(f"Min approvals history:  {min_approvals:.0f}")
    print(f"Median approvals history: {median_approvals:.0f}")
    print(f"Std: {std_approvals:.1f}")

    max_participant = approvals_df.loc[approvals_df["Total_approvals"].idxmax()]
    min_participant = approvals_df.loc[approvals_df["Total_approvals"].idxmin()]

    print("\nParticipant with the highest approvals history:")
    print(f"  Participant ID: {max_participant['Participant']}")
    print(f"  Task: {max_participant['Task']}")
    print(f"  Total approvals: {max_participant['Total_approvals']:.0f}")

    print("\nParticipant with the lowest approvals history:")
    print(f"  Participant ID: {min_participant['Participant']}")
    print(f"  Task: {min_participant['Task']}")
    print(f"  Total approvals: {min_participant['Total_approvals']:.0f}")

    return approvals_df


# ============================================================
# 7) Data quality issue analysis
# ============================================================
def analyze_data_quality(dataframes):
    """Analyze invalid records by task."""
    print("\n" + "=" * 60)
    print("Data Quality Issues")
    print("=" * 60)

    issues = []

    for task_name, df in dataframes.items():
        total_records = len(df)

        returned_count = int((df["Status"] == "RETURNED").sum()) if "Status" in df.columns else 0
        timed_out_count = int((df["Status"] == "TIMED-OUT").sum()) if "Status" in df.columns else 0

        consent_revoked_count = 0
        data_expired_count = 0
        for col in df.columns:
            s = df[col].astype(str)
            consent_revoked_count += int(s.str.contains("CONSENT_REVOKED", case=False, na=False).sum())
            data_expired_count += int(s.str.contains("DATA_EXPIRED", case=False, na=False).sum())

        invalid_total = returned_count + timed_out_count + consent_revoked_count + data_expired_count
        invalid_rate = (invalid_total / total_records * 100) if total_records > 0 else 0

        issues.append(
            {
                "Task": task_name,
                "Total records": total_records,
                "RETURNED": returned_count,
                "TIMED-OUT": timed_out_count,
                "CONSENT_REVOKED": consent_revoked_count,
                "DATA_EXPIRED": data_expired_count,
                "Total invalid": invalid_total,
                "Invalid rate (%)": round(invalid_rate, 1),
            }
        )

    quality_df = pd.DataFrame(issues)
    print(quality_df.to_string(index=False))

    total_invalid = int(quality_df["Total invalid"].sum()) if len(quality_df) > 0 else 0
    total_records = int(quality_df["Total records"].sum()) if len(quality_df) > 0 else 0
    overall_invalid_rate = (total_invalid / total_records * 100) if total_records > 0 else 0

    print("\nOverall data quality:")
    print(f"Total records: {total_records}")
    print(f"Total invalid records: {total_invalid}")
    print(f"Overall invalid rate: {overall_invalid_rate:.1f}%")

    if len(quality_df) > 0:
        worst_task = quality_df.loc[quality_df["Invalid rate (%)"].idxmax()]
        print(f"\nWorst task: {worst_task['Task']} (invalid rate: {worst_task['Invalid rate (%)']}%)")

    return quality_df


# ============================================================
# 8) Generate comprehensive report
# ============================================================
def generate_comprehensive_report(dataframes):
    """Generate a comprehensive report across the five demographics files."""
    print("\n" + "=" * 80)
    print("Comprehensive Report: Five Demographics Files")
    print("=" * 80)

    overview_df = analyze_data_overview(dataframes)
    combined_df, demographics_analyses = analyze_demographics(dataframes)
    time_df = analyze_task_completion_time(dataframes)
    participation_time_df, period_df = analyze_participation_time(dataframes)
    approvals_df = analyze_participation_level(dataframes)
    quality_df = analyze_data_quality(dataframes)

    print("\n" + "=" * 80)
    print("Key Findings and Suggestions")
    print("=" * 80)

    # Safe extraction for key indicators
    female_percent = 0.0
    if demographics_analyses and "Sex distribution" in demographics_analyses:
        sex_tbl = demographics_analyses["Sex distribution"]
        if "Female" in sex_tbl.index:
            female_percent = float(sex_tbl.loc["Female", "Percent"])

    avg_age = float(combined_df["Age_numeric"].mean()) if combined_df is not None else 0.0

    south_africa_percent = 0.0
    if demographics_analyses and "Country of birth distribution" in demographics_analyses:
        birth_tbl = demographics_analyses["Country of birth distribution"]
        if "South Africa" in birth_tbl.index:
            south_africa_percent = float(birth_tbl.loc["South Africa", "Percent"])

    black_percent = 0.0
    if demographics_analyses and "Ethnicity distribution" in demographics_analyses:
        eth_tbl = demographics_analyses["Ethnicity distribution"]
        if "Black" in eth_tbl.index:
            black_percent = float(eth_tbl.loc["Black", "Percent"])

    avg_completion_time = float(time_df["Mean (sec)"].mean()) if time_df is not None and len(time_df) > 0 else 0.0
    hardest_task = time_df.loc[time_df["Mean (sec)"].idxmax(), "Task"] if time_df is not None and len(time_df) > 0 else "Unknown"
    hardest_task_time = float(time_df["Mean (sec)"].max()) if time_df is not None and len(time_df) > 0 else 0.0

    worst_quality_task = quality_df.loc[quality_df["Invalid rate (%)"].idxmax(), "Task"] if quality_df is not None and len(quality_df) > 0 else "Unknown"
    worst_quality_rate = float(quality_df["Invalid rate (%)"].max()) if quality_df is not None and len(quality_df) > 0 else 0.0

    overall_valid_rate = (100 - float(quality_df["Invalid rate (%)"].mean())) if quality_df is not None and len(quality_df) > 0 else 0.0

    print("1. Sample representativeness:")
    print(f"   - Female participants: {female_percent:.1f}% | Male (approx.): {100 - female_percent:.1f}%")
    print(f"   - Mean age: {avg_age:.1f}")
    print(f"   - Country-of-birth concentration (South Africa): {south_africa_percent:.1f}%")
    print(f"   - Ethnicity concentration (Black): {black_percent:.1f}%")

    print("\n2. Task difficulty (proxy by completion time):")
    print(f"   - Mean completion time (across tasks): {avg_completion_time/60:.1f} minutes")
    print(f"   - Slowest task: {hardest_task} ({hardest_task_time/60:.1f} minutes)")

    print("\n3. Data quality:")
    print(f"   - Approx. overall valid rate: {overall_valid_rate:.1f}%")
    print(f"   - Worst task: {worst_quality_task} (invalid rate: {worst_quality_rate:.1f}%)")

    print("\n4. Suggestions:")
    print(f"   - Improve recruitment/flow for {worst_quality_task} to reduce withdrawals/timeouts.")
    print("   - Balance sample demographics by recruiting more diverse participants.")
    print(f"   - Streamline {hardest_task} to reduce completion time.")

    results = {
        "overview": overview_df,
        "combined_data": combined_df,
        "demographics": demographics_analyses,
        "completion_time": time_df,
        "participation_time": participation_time_df,
        "period_distribution": period_df,
        "participation_level": approvals_df,
        "data_quality": quality_df,
    }
    return results


# ============================================================
# 9) Visualization
# ============================================================
def create_visualizations(results):
    """Create visualization charts."""
    if results is None:
        print("No results available for visualization.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Demographics Analysis Visualizations", fontsize=16)

    # Sex distribution pie
    if results.get("demographics") and "Sex distribution" in results["demographics"]:
        sex_df = results["demographics"]["Sex distribution"]
        axes[0, 0].pie(sex_df["Count"], labels=sex_df.index, autopct="%1.1f%%", startangle=90)
        axes[0, 0].set_title("Sex distribution")

    # Age distribution bar
    if results.get("demographics") and "Age distribution" in results["demographics"]:
        age_df = results["demographics"]["Age distribution"]
        axes[0, 1].bar(age_df.index.astype(str), age_df["Count"])
        axes[0, 1].set_title("Age distribution")
        axes[0, 1].set_xlabel("Age group")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].tick_params(axis="x", rotation=45)

    # Mean completion time per task
    if results.get("completion_time") is not None and len(results["completion_time"]) > 0:
        time_df = results["completion_time"]
        axes[0, 2].bar(time_df["Task"], time_df["Mean (sec)"])
        axes[0, 2].set_title("Mean completion time by task")
        axes[0, 2].set_xlabel("Task")
        axes[0, 2].set_ylabel("Mean time (sec)")
        axes[0, 2].tick_params(axis="x", rotation=45)

    # Ethnicity distribution pie
    if results.get("demographics") and "Ethnicity distribution" in results["demographics"]:
        eth_df = results["demographics"]["Ethnicity distribution"]
        axes[1, 0].pie(eth_df["Count"], labels=eth_df.index, autopct="%1.1f%%", startangle=90)
        axes[1, 0].set_title("Ethnicity distribution")

    # Invalid rate by task
    if results.get("data_quality") is not None and len(results["data_quality"]) > 0:
        qdf = results["data_quality"]
        axes[1, 1].bar(qdf["Task"], qdf["Invalid rate (%)"])
        axes[1, 1].set_title("Invalid rate by task")
        axes[1, 1].set_xlabel("Task")
        axes[1, 1].set_ylabel("Invalid rate (%)")
        axes[1, 1].tick_params(axis="x", rotation=45)
        for i, v in enumerate(qdf["Invalid rate (%)"].tolist()):
            axes[1, 1].text(i, v + 0.5, f"{v}%", ha="center")

    # Start-time period distribution
    if results.get("period_distribution") is not None and len(results["period_distribution"]) > 0:
        pdf = results["period_distribution"]
        axes[1, 2].bar(pdf.index.astype(str), pdf["Starts"])
        axes[1, 2].set_title("Start-time period distribution")
        axes[1, 2].set_xlabel("Period")
        axes[1, 2].set_ylabel("Starts")
        axes[1, 2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


# ============================================================
# 10) Save results to Excel
# ============================================================
def save_results_to_excel(results, out_path):
    """Save all outputs to a single Excel file."""
    try:
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            for sheet_name, data in results.items():
                if isinstance(data, pd.DataFrame):
                    data.to_excel(writer, sheet_name=sheet_name[:31], index=False)
                elif isinstance(data, dict):
                    for sub_sheet, sub_data in data.items():
                        if isinstance(sub_data, pd.DataFrame):
                            safe_name = (sub_sheet[:28]).replace("/", "_")
                            sub_data.to_excel(writer, sheet_name=safe_name, index=True)

        print(f"Saved report to: {out_path}")
    except Exception as e:
        print(f"ERROR: Failed to save Excel file: {e}")


# ============================================================
# Main
# ============================================================
def main():
    print("Starting analysis of five demographics tables...")

    dataframes = read_all_demographics_files()
    if not dataframes:
        print("No data files were found in your Downloads folder.")
        return

    results = generate_comprehensive_report(dataframes)

    print("\nGenerating visualizations...")
    create_visualizations(results)

    print("\nSaving results to Excel...")
    save_results_to_excel(results, OUT_XLSX)

    print("\nDone.")


if __name__ == "__main__":
    main()
