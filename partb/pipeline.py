"""Chicago Airbnb reporting pipeline (Inside Airbnb scrape, Sept 2025).

raw listings + reviews -> cleaned tables -> summary tables -> charts + insights.md
every step is sent to marquez as an OpenLineage job, so the lineage graph builds itself.

run from the repo root (marquez api on localhost:9000):
    uv run python partb/pipeline.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from lineage import count_events, reset_log, run_step

RAW = Path("data/raw/airbnb")
OUT = Path("partb/output")
CLEAN, REPORTS, CHARTS = OUT / "clean", OUT / "reports", OUT / "charts"

RAW_LISTINGS = RAW / "listings.csv"
RAW_REVIEWS = RAW / "reviews.csv"
LISTINGS = CLEAN / "listings_clean.csv"
REVIEWS_MONTHLY = CLEAN / "reviews_monthly.csv"
NEIGHBOURHOODS = REPORTS / "neighbourhood_summary.csv"
ROOM_TYPES = REPORTS / "room_type_summary.csv"
HOSTS = REPORTS / "host_summary.csv"
STAY_LICENSE = REPORTS / "stay_length_vs_license.csv"
REVIEW_TREND = REPORTS / "review_trend_yearly.csv"
INSIGHTS = OUT / "insights.md"

KEEP = ["id", "host_id", "host_is_superhost", "calculated_host_listings_count",
        "neighbourhood_cleansed", "room_type", "property_type", "accommodates", "bedrooms",
        "price", "minimum_nights", "availability_365", "number_of_reviews",
        "number_of_reviews_ltm", "review_scores_rating", "estimated_occupancy_l365d",
        "estimated_revenue_l365d", "license", "instant_bookable"]
MIN_PRICE, MAX_PRICE = 10, 2000

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def clean_listings():
    """keep the useful columns, fix types, drop bad prices, add a few groupings"""
    df = pd.read_csv(RAW_LISTINGS, usecols=KEEP)
    df = df.rename(columns={"neighbourhood_cleansed": "neighbourhood",
                            "calculated_host_listings_count": "host_listings"})
    df["price"] = df["price"].str.replace(r"[$,]", "", regex=True).astype(float)
    # missing prices are mostly inactive listings; >$2000 are placeholder values like $50,000
    df = df[df["price"].between(MIN_PRICE, MAX_PRICE)].copy()
    for col in ["host_is_superhost", "instant_bookable"]:
        df[col] = df[col].map({"t": 1, "f": 0})

    df["stay_type"] = (df["minimum_nights"] >= 30).map({True: "30+ nights", False: "short-term"})
    lic = df["license"].fillna("").str.strip()
    df["license_status"] = "missing"
    df.loc[lic.str.match(r"^R?\d"), "license_status"] = "registered"
    df.loc[lic.str.contains("pending", case=False), "license_status"] = "pending"
    df["host_type"] = pd.cut(df["host_listings"], [0, 1, 9, float("inf")],
                             labels=["single listing", "2-9 listings", "10+ listings"])
    df = df.drop(columns=["license", "host_id"])
    CLEAN.mkdir(parents=True, exist_ok=True)
    df.to_csv(LISTINGS, index=False)


def monthly_reviews():
    """count reviews per month (review count = rough proxy for bookings)"""
    reviews = pd.read_csv(RAW_REVIEWS, usecols=["listing_id", "date"], parse_dates=["date"])
    room = pd.read_csv(LISTINGS, usecols=["id", "room_type"])
    reviews = reviews.merge(room, left_on="listing_id", right_on="id", how="left")
    reviews["room_type"] = reviews["room_type"].fillna("not in clean set")
    reviews["month"] = reviews["date"].dt.to_period("M").astype(str)
    out = (reviews.groupby("month")
           .agg(reviews=("listing_id", "size"), listings_reviewed=("listing_id", "nunique"))
           .reset_index())
    out.to_csv(REVIEWS_MONTHLY, index=False)


def summarise(df, by):
    """same set of metrics for any grouping"""
    return (df.groupby(by, observed=True)
            .agg(listings=("id", "size"),
                 median_price=("price", "median"),
                 median_rating=("review_scores_rating", "median"),
                 median_occupancy_days=("estimated_occupancy_l365d", "median"),
                 total_est_revenue=("estimated_revenue_l365d", "sum"),
                 superhost_share=("host_is_superhost", "mean"))
            .assign(share_of_listings=lambda t: t["listings"] / t["listings"].sum())
            .round(3)
            .sort_values("listings", ascending=False)
            .reset_index())


def report_neighbourhoods():
    """listing counts, prices, ratings by neighbourhood"""
    REPORTS.mkdir(parents=True, exist_ok=True)
    summarise(pd.read_csv(LISTINGS), "neighbourhood").to_csv(NEIGHBOURHOODS, index=False)


def report_room_types():
    """same metrics by room type"""
    summarise(pd.read_csv(LISTINGS), "room_type").to_csv(ROOM_TYPES, index=False)


def report_hosts():
    """single-listing hosts vs bigger operators"""
    summarise(pd.read_csv(LISTINGS), "host_type").to_csv(HOSTS, index=False)


def report_stay_license():
    """how licensing differs for short-term vs monthly rentals"""
    df = pd.read_csv(LISTINGS)
    table = pd.crosstab(df["stay_type"], df["license_status"], normalize="index").round(3)
    table["listings"] = df["stay_type"].value_counts()
    table.reset_index().to_csv(STAY_LICENSE, index=False)


def report_review_trend():
    """roll monthly reviews up to years (2025 is only Jan-Sep)"""
    monthly = pd.read_csv(REVIEWS_MONTHLY)
    monthly["year"] = monthly["month"].str[:4].astype(int)
    yearly = monthly.groupby("year")["reviews"].sum().reset_index()
    yearly["yoy_change"] = yearly["reviews"].pct_change().round(3)
    yearly.to_csv(REVIEW_TREND, index=False)


def _style(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=10)
    ax.set_xlabel(xlabel or "", color=INK_2)
    ax.set_ylabel(ylabel or "", color=INK_2)
    ax.tick_params(colors=INK_2, length=0)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_axisbelow(True)


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(CHARTS / name, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def _hbar(table, label_col, value_col, title, fmt, name):  # pylint: disable=too-many-arguments,too-many-positional-arguments
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(table) + 1))
    table = table.iloc[::-1]  # biggest on top
    bars = ax.barh(table[label_col].astype(str), table[value_col], color=BLUE, height=0.6)
    ax.bar_label(bars, labels=[fmt(v) for v in table[value_col]], padding=4, color=INK_2,
                 fontsize=9)
    ax.xaxis.set_visible(False)
    _style(ax, title)
    _save(fig, name)


def build_report():
    """charts + a short markdown write-up with the actual numbers"""
    CHARTS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.facecolor": "#fcfcfb"})
    listings = pd.read_csv(LISTINGS)
    hoods = pd.read_csv(NEIGHBOURHOODS)
    rooms = pd.read_csv(ROOM_TYPES)
    hosts = pd.read_csv(HOSTS)
    stay = pd.read_csv(STAY_LICENSE).set_index("stay_type")
    yearly = pd.read_csv(REVIEW_TREND)
    monthly = pd.read_csv(REVIEWS_MONTHLY)

    _hbar(hoods.head(10), "neighbourhood", "listings", "Top 10 neighbourhoods by listings",
          lambda v: f"{v:,.0f}", "1_top_neighbourhoods.png")
    big = hoods[hoods["listings"] >= 50].sort_values("median_price", ascending=False).head(10)
    _hbar(big, "neighbourhood", "median_price",
          "Priciest neighbourhoods (median nightly price, 50+ listings)",
          lambda v: f"${v:,.0f}", "2_priciest_neighbourhoods.png")
    _hbar(rooms, "room_type", "median_price", "Median nightly price by room type",
          lambda v: f"${v:,.0f}", "3_price_by_room_type.png")
    _hbar(hosts, "host_type", "share_of_listings", "Share of listings by host size",
          lambda v: f"{v:.0%}", "4_host_size.png")

    lic = stay["registered"].rename({"short-term": "Short-term (<30 nights)",
                                     "30+ nights": "Monthly (30+ nights)"})
    _hbar(lic.reset_index(), "stay_type", "registered", "Share of listings with a city license",
          lambda v: f"{v:.1%}", "5_license_by_stay_type.png")

    # scrape was 22 Sep, so Sep 2025 is a partial month - leave it off the chart
    m = monthly[monthly["month"].between("2015-01", "2025-08")].copy()
    m["month"] = pd.to_datetime(m["month"])
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(m["month"], m["reviews"], color=BLUE, linewidth=2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    _style(ax, "Reviews per month, Jan 2015 to Aug 2025 (proxy for bookings)")
    _save(fig, "6_reviews_per_month.png")

    # pull the numbers used in the write-up
    n = len(listings)
    top3 = hoods.head(3)
    top3_share = top3["share_of_listings"].sum()
    entire = rooms.set_index("room_type").loc["Entire home/apt"]
    private = rooms.set_index("room_type").loc["Private room"]
    multi = hosts.loc[hosts["host_type"] != "single listing", "share_of_listings"].sum()
    big_ops = hosts.set_index("host_type").loc["10+ listings"]
    single = hosts.set_index("host_type").loc["single listing"]
    y = yearly.set_index("year")["reviews"]
    # average each calendar month over 2022-2025 to see the seasonal pattern
    by_month = (monthly[monthly["month"] >= "2022-01"]
                .assign(m=lambda t: pd.to_datetime(t["month"]).dt.month_name())
                .groupby("m")["reviews"].mean())
    sh = listings.groupby("host_is_superhost")["review_scores_rating"].median()

    lines = [
        "# Chicago Airbnb, Sept 2025: reporting insights",
        "",
        (f"Data: Inside Airbnb scrape of Chicago (2025-09-22), {n:,} active listings after "
         f"cleaning (price ${MIN_PRICE}-${MAX_PRICE}), "
         f"{int(monthly['reviews'].sum()):,} reviews since 2009."),
        "",
        "## Where",
        (f"- Supply is concentrated: {', '.join(top3['neighbourhood'])} hold "
        f"{top3_share:.0%} of all listings (out of {len(hoods)} neighbourhoods)."),
        (f"- Priciest areas with 50+ listings: {', '.join(big['neighbourhood'].head(3))} "
        f"(median ${big['median_price'].iloc[0]:,.0f}/night vs ${listings['price'].median():,.0f} "
        "citywide)."),
        "",
        "## What",
        (f"- {entire['share_of_listings']:.0%} of listings are entire homes/apts, at a median "
        f"${entire['median_price']:,.0f}/night vs ${private['median_price']:,.0f} "
        "for a private room."),
        ("- Ratings are high almost everywhere: median "
         f"{listings['review_scores_rating'].median():.2f}/5; "
         f"superhosts {sh.get(1, float('nan')):.2f} vs others {sh.get(0, float('nan')):.2f}."),
        "",
        "## Who",
        (f"- {multi:.0%} of listings belong to hosts with 2+ listings; operators with 10+ listings "
        f"alone run {big_ops['share_of_listings']:.0%} of the market."),
        (f"- Single-listing hosts rate higher (median {single['median_rating']:.2f} vs "
        f"{big_ops['median_rating']:.2f}) and are superhosts more often "
        f"({single['superhost_share']:.0%} vs {big_ops['superhost_share']:.0%})."),
        "",
        "## Rules",
        (f"- {stay.loc['30+ nights', 'listings'] / n:.0%} of listings require 30+ night stays. "
        f"Only {stay.loc['30+ nights', 'registered']:.0%} of those show a city license, vs "
        f"{stay.loc['short-term', 'registered']:.1%} of short-term listings. Chicago doesn't "
        "require registration for 30+ day rentals, so many hosts likely set the minimum stay "
        "to avoid it."),
        "",
        "## When",
        (f"- Reviews fell {1 - y[2020] / y[2019]:.0%} in 2020 (COVID), recovered past 2019 by "
        f"{int(y[(y.index > 2020) & (y > y[2019])].index.min())}, and peaked at "
        f"{y.drop(2025).max():,} in {int(y.drop(2025).idxmax())}."),
        (f"- Strong seasonality (2022-2025 average): {by_month.idxmax()} is the busiest month "
        f"({by_month.max():,.0f} reviews), {by_month.idxmin()} the quietest "
        f"({by_month.min():,.0f}), about {by_month.max() / by_month.min():.1f}x apart."),
    ]
    INSIGHTS.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    """main"""
    reset_log()
    clean_note = "cleaned listings: typed price, $10-2000, flags + groupings, no host ids"
    run_step("clean_listings", "Select columns, parse price, drop bad prices, add stay_type / "
             "license_status / host_type", {RAW_LISTINGS: "Inside Airbnb listings (raw)"},
             {LISTINGS: clean_note}, clean_listings)
    run_step("aggregate_reviews_monthly", "Join reviews to listings and count reviews per month",
             {RAW_REVIEWS: "Inside Airbnb reviews (raw)", LISTINGS: clean_note},
             {REVIEWS_MONTHLY: "reviews per month"}, monthly_reviews)
    run_step("report_neighbourhoods", "Listings, price, rating, revenue by neighbourhood",
             {LISTINGS: clean_note}, {NEIGHBOURHOODS: "neighbourhood summary"},
             report_neighbourhoods)
    run_step("report_room_types", "Same metrics by room type", {LISTINGS: clean_note},
             {ROOM_TYPES: "room type summary"}, report_room_types)
    run_step("report_hosts", "Single vs multi-listing hosts", {LISTINGS: clean_note},
             {HOSTS: "host size summary"}, report_hosts)
    run_step("report_stay_license", "License status by minimum-stay type",
             {LISTINGS: clean_note}, {STAY_LICENSE: "stay type x license share"},
             report_stay_license)
    run_step("report_review_trend", "Yearly review totals + YoY change",
             {REVIEWS_MONTHLY: "reviews per month"}, {REVIEW_TREND: "yearly reviews"},
             report_review_trend)
    run_step("build_report", "Charts + insights.md for the slide",
             {NEIGHBOURHOODS: "", ROOM_TYPES: "", HOSTS: "", STAY_LICENSE: "", REVIEW_TREND: "",
              REVIEWS_MONTHLY: "", LISTINGS: clean_note},
             {INSIGHTS: "written insights", CHARTS: "PNG charts"}, build_report)
    print(f"\n{count_events()} lineage events sent, log in partb/output/lineage_events.jsonl")
    print(INSIGHTS.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
