from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

APP_TITLE = "River Explorer — Guided USGS Hydrology"
USGS_IV_URL = "https://waterservices.usgs.gov/nwis/iv/"
USGS_SITE_URL = "https://waterservices.usgs.gov/nwis/site/"
USER_AGENT = "We are dougalien River Explorer/1.0"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
REQUEST_TIMEOUT = 25

# Full-name lookup helps when users type "Charles River Massachusetts"
STATE_NAME_TO_ABBR = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC",
}


@dataclass
class StationCandidate:
    site_no: str
    station_nm: str
    state_cd: str = ""
    site_tp_cd: str = ""
    dec_lat_va: float | None = None
    dec_long_va: float | None = None
    drain_area_va: float | None = None
    score: float = 0.0

    @property
    def label(self) -> str:
        pieces = [self.station_nm, f"Site {self.site_no}"]
        if self.state_cd:
            pieces.append(self.state_cd)
        if self.drain_area_va is not None and not np.isnan(self.drain_area_va):
            pieces.append(f"Drainage area {self.drain_area_va:.1f} mi²")
        return " | ".join(pieces)


def get_openai_client() -> OpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def extract_state_from_query(query: str) -> tuple[str, str | None]:
    """Return cleaned river phrase and optional state abbreviation."""
    cleaned = query.strip()
    lowered = cleaned.lower()

    # full state names first
    for full_name, abbr in sorted(STATE_NAME_TO_ABBR.items(), key=lambda x: len(x[0]), reverse=True):
        if re.search(rf"\b{re.escape(full_name)}\b", lowered):
            cleaned = re.sub(rf"\b{re.escape(full_name)}\b", "", cleaned, flags=re.IGNORECASE).strip(" ,-")
            return cleaned, abbr

    # 2-letter abbreviation next
    tokens = re.split(r"[\s,]+", cleaned)
    if tokens:
        last = tokens[-1].upper()
        if last in set(STATE_NAME_TO_ABBR.values()):
            cleaned = re.sub(rf"\b{re.escape(tokens[-1])}\b$", "", cleaned, flags=re.IGNORECASE).strip(" ,-")
            return cleaned, last

    return cleaned, None


def safe_float(value: Any) -> float | None:
    try:
        if value in (None, "", "NaN"):
            return None
        return float(value)
    except Exception:
        return None


def parse_usgs_rdb(text: str) -> pd.DataFrame:
    """Parse USGS RDB text into a DataFrame.

    RDB has comment lines starting with #, then a header row, then a field-width row.
    """
    lines = [line for line in text.splitlines() if not line.startswith("#") and line.strip()]
    if len(lines) < 3:
        return pd.DataFrame()

    header = lines[0].split("\t")
    data_lines = lines[2:]  # skip field-width row
    buffer = io.StringIO("\n".join(["\t".join(header)] + data_lines))
    df = pd.read_csv(buffer, sep="\t", dtype=str)
    return df


def rank_station_candidates(df: pd.DataFrame, river_phrase: str, state_abbr: str | None) -> list[StationCandidate]:
    river_norm = normalize_text(river_phrase)
    results: list[StationCandidate] = []

    for _, row in df.iterrows():
        site_no = str(row.get("site_no", "")).strip()
        station_nm = str(row.get("station_nm", "")).strip()
        if not site_no.isdigit() or not station_nm:
            continue

        state_cd = str(row.get("state_cd", "")).strip().upper()
        station_norm = normalize_text(station_nm)
        score = 0.0

        if river_norm and river_norm in station_norm:
            score += 60
        if river_norm:
            river_words = set(river_norm.split())
            station_words = set(station_norm.split())
            overlap = len(river_words & station_words)
            score += 8 * overlap
        if state_abbr and state_cd == state_abbr:
            score += 25
        if station_norm.startswith(river_norm):
            score += 15
        if " river " in f" {station_norm} ":
            score += 5
        if any(token in station_norm for token in ["near", "at", "below", "above"]):
            score += 2

        results.append(
            StationCandidate(
                site_no=site_no,
                station_nm=station_nm,
                state_cd=state_cd,
                site_tp_cd=str(row.get("site_tp_cd", "")).strip(),
                dec_lat_va=safe_float(row.get("dec_lat_va")),
                dec_long_va=safe_float(row.get("dec_long_va")),
                drain_area_va=safe_float(row.get("drain_area_va")),
                score=score,
            )
        )

    # De-duplicate by site_no and sort by score
    deduped: dict[str, StationCandidate] = {}
    for item in sorted(results, key=lambda x: x.score, reverse=True):
        deduped.setdefault(item.site_no, item)
    return list(deduped.values())[:15]


@st.cache_data(show_spinner=False, ttl=3600)
def search_usgs_stations(user_query: str) -> list[dict[str, Any]]:
    river_phrase, state_abbr = extract_state_from_query(user_query)

    params = {
        "format": "rdb",
        "siteStatus": "active",
        "siteType": "ST",
        "hasDataTypeCd": "iv",
        "parameterCd": "00060",
        "siteName": river_phrase,
        "siteNameMatchOperator": "any",
        "siteOutput": "expanded",
    }
    if state_abbr:
        params["stateCd"] = state_abbr

    response = requests.get(
        USGS_SITE_URL,
        params=params,
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    df = parse_usgs_rdb(response.text)

    if df.empty and not state_abbr:
        # second pass: try the first two words only for broad matching
        short_phrase = " ".join(river_phrase.split()[:2]).strip()
        if short_phrase and short_phrase.lower() != river_phrase.lower():
            params["siteName"] = short_phrase
            response = requests.get(
                USGS_SITE_URL,
                params=params,
                timeout=REQUEST_TIMEOUT,
                headers={"User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            df = parse_usgs_rdb(response.text)

    candidates = rank_station_candidates(df, river_phrase, state_abbr)
    return [candidate.__dict__ for candidate in candidates]


@st.cache_data(show_spinner=False, ttl=900)
def fetch_iv_data(site_no: str, days_back: int) -> dict[str, Any]:
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days_back)
    params = {
        "format": "json",
        "sites": site_no,
        "parameterCd": "00060,00065",
        "startDT": start_dt.strftime("%Y-%m-%d"),
        "endDT": end_dt.strftime("%Y-%m-%d"),
    }
    response = requests.get(
        USGS_IV_URL,
        params=params,
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    payload = response.json()

    series_list = payload.get("value", {}).get("timeSeries", [])
    parsed: dict[str, pd.DataFrame] = {}
    station_name = None

    for series in series_list:
        source = series.get("sourceInfo", {})
        variable = series.get("variable", {})
        station_name = station_name or source.get("siteName")
        parameter_code = variable.get("variableCode", [{}])[0].get("value", "")
        unit = variable.get("unit", {}).get("unitCode", "")
        values = series.get("values", [{}])[0].get("value", [])

        rows = []
        for item in values:
            value = item.get("value")
            try:
                numeric_value = float(value)
            except Exception:
                numeric_value = np.nan
            rows.append(
                {
                    "datetime": pd.to_datetime(item.get("dateTime"), errors="coerce"),
                    "value": numeric_value,
                }
            )

        df = pd.DataFrame(rows).dropna(subset=["datetime"]).sort_values("datetime")
        df["unit"] = unit
        parsed[parameter_code] = df

    discharge = parsed.get("00060", pd.DataFrame(columns=["datetime", "value", "unit"]))
    stage = parsed.get("00065", pd.DataFrame(columns=["datetime", "value", "unit"]))

    return {
        "station_name": station_name or f"USGS site {site_no}",
        "discharge": discharge,
        "stage": stage,
    }


def compute_hydrology_summary(discharge_df: pd.DataFrame) -> dict[str, Any]:
    if discharge_df.empty:
        return {}

    df = discharge_df.dropna(subset=["value"]).copy()
    if df.empty:
        return {}

    latest = df.iloc[-1]
    first = df.iloc[0]
    mean_val = df["value"].mean()
    min_val = df["value"].min()
    max_val = df["value"].max()
    median_val = df["value"].median()

    if len(df) >= 2:
        delta = latest["value"] - df.iloc[-2]["value"]
    else:
        delta = np.nan

    net_change = latest["value"] - first["value"]
    trend = "rising" if net_change > 0 else "falling" if net_change < 0 else "steady"

    # A simple recent variability metric for student interpretation
    coeff_var = (df["value"].std() / mean_val * 100) if mean_val else np.nan

    return {
        "latest_flow": float(latest["value"]),
        "latest_time": latest["datetime"],
        "window_mean": float(mean_val),
        "window_median": float(median_val),
        "window_min": float(min_val),
        "window_max": float(max_val),
        "last_step_change": float(delta) if not pd.isna(delta) else None,
        "net_change": float(net_change),
        "trend": trend,
        "cv_percent": float(coeff_var) if not pd.isna(coeff_var) else None,
        "n_points": int(len(df)),
        "unit": str(df["unit"].iloc[0]) if "unit" in df.columns and not df.empty else "ft3/s",
    }


def build_hydrograph(discharge_df: pd.DataFrame, station_name: str) -> go.Figure:
    fig = go.Figure()
    if discharge_df.empty:
        fig.update_layout(title=f"No discharge data available for {station_name}")
        return fig

    df = discharge_df.dropna(subset=["value"]).copy()
    if df.empty:
        fig.update_layout(title=f"No discharge data available for {station_name}")
        return fig

    df = df.sort_values("datetime")
    df = df.set_index("datetime")
    df["rolling_24h"] = df["value"].rolling("24H", min_periods=2).mean()
    window_mean = df["value"].mean()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["value"],
            mode="lines",
            name="Discharge",
            line=dict(width=2),
        )
    )

    if df["rolling_24h"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["rolling_24h"],
                mode="lines",
                name="24-hour rolling mean",
                line=dict(width=2, dash="dash"),
            )
        )

    fig.add_hline(
        y=window_mean,
        line_dash="dot",
        annotation_text=f"Selected-window mean: {window_mean:.1f}",
        annotation_position="top left",
    )

    fig.update_layout(
        title=f"Discharge hydrograph — {station_name}",
        xaxis_title="Date",
        yaxis_title="Discharge (cfs)",
        legend_title="Series",
        height=520,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def station_context_text(station: dict[str, Any], summary: dict[str, Any], days_back: int) -> str:
    unit = summary.get("unit", "ft3/s")
    latest_time = summary.get("latest_time")
    latest_time_str = latest_time.strftime("%Y-%m-%d %H:%M") if isinstance(latest_time, pd.Timestamp) else str(latest_time)
    lines = [
        f"Station name: {station.get('station_nm', '')}",
        f"USGS site number: {station.get('site_no', '')}",
        f"State: {station.get('state_cd', '')}",
        f"Hydrograph look-back window: {days_back} days",
        f"Latest discharge: {summary.get('latest_flow', 'NA'):.2f} {unit}",
        f"Latest timestamp: {latest_time_str}",
        f"Selected-window mean discharge: {summary.get('window_mean', 'NA'):.2f} {unit}",
        f"Selected-window median discharge: {summary.get('window_median', 'NA'):.2f} {unit}",
        f"Selected-window minimum discharge: {summary.get('window_min', 'NA'):.2f} {unit}",
        f"Selected-window maximum discharge: {summary.get('window_max', 'NA'):.2f} {unit}",
        f"Net change across window: {summary.get('net_change', 'NA'):.2f} {unit}",
        f"Overall trend across window: {summary.get('trend', 'NA')}",
        f"Coefficient of variation across window: {summary.get('cv_percent', 'NA'):.1f}%" if summary.get('cv_percent') is not None else "Coefficient of variation across window: NA",
        "Important note: the plotted mean is the mean of the selected time window, not a long-term historical normal.",
    ]
    return "\n".join(lines)


def tutor_instructions() -> str:
    return (
        "You are Dr. River Explorer, a supportive hydrology tutor. "
        "Use the Socratic method. Ask one question at a time during guided mode. "
        "Build on what the student already noticed. Do not repeat a question they already answered. "
        "Keep the tone warm, crisp, and educational. Focus on hydrograph interpretation: trend, peaks, timing, variability, runoff response, baseflow, and scale. "
        "If the guided portion is complete, answer follow-up questions directly and clearly. "
        "Do not pretend the selected-window mean is a long-term normal; say exactly what it is."
    )


def call_tutor(
    client: OpenAI,
    model_name: str,
    context_text: str,
    history: list[dict[str, str]],
    mode: str,
) -> str:
    transcript = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)

    if mode == "initial":
        prompt = (
            f"Hydrograph context:\n{context_text}\n\n"
            "Ask the first guided question. Start broad. Ask the student what they notice in the hydrograph: patterns, rises, falls, peaks, or unusual behavior."
        )
    elif mode == "guided":
        prompt = (
            f"Hydrograph context:\n{context_text}\n\n"
            f"Conversation so far:\n{transcript}\n\n"
            "Respond to the student's latest answer. First acknowledge something specific they said. Then ask exactly one follow-up question that moves the interpretation forward."
        )
    elif mode == "wrap_up":
        prompt = (
            f"Hydrograph context:\n{context_text}\n\n"
            f"Conversation so far:\n{transcript}\n\n"
            "The guided portion should now conclude. Give a concise, warm summary of what the student figured out, highlight the key hydrologic patterns, say the guided portion is complete, and invite follow-up questions."
        )
    else:
        prompt = (
            f"Hydrograph context:\n{context_text}\n\n"
            f"Conversation so far:\n{transcript}\n\n"
            "Answer the student's follow-up question clearly and helpfully. Tie your answer back to the hydrograph whenever possible."
        )

    response = client.responses.create(
        model=model_name,
        instructions=tutor_instructions(),
        input=prompt,
    )
    return response.output_text.strip()


def reset_guided_state() -> None:
    st.session_state.chat_history = []
    st.session_state.student_turns = 0
    st.session_state.guided_complete = False


def init_state() -> None:
    st.session_state.setdefault("station_candidates", [])
    st.session_state.setdefault("selected_station_index", 0)
    st.session_state.setdefault("selected_station", None)
    st.session_state.setdefault("hydro_loaded", False)
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("student_turns", 0)
    st.session_state.setdefault("guided_complete", False)
    st.session_state.setdefault("days_back", 7)
    st.session_state.setdefault("last_query", "")


def render_chat() -> None:
    for item in st.session_state.chat_history:
        role = item["role"]
        with st.chat_message("assistant" if role == "assistant" else "user"):
            st.markdown(item["content"])


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🌊", layout="wide")
    init_state()

    st.title("🌊 River Explorer — Guided USGS Hydrology")
    st.caption("Search a real USGS stream gage, plot live discharge, and guide students through hydrograph interpretation.")

    with st.expander("What this version does"):
        st.markdown(
            "- Searches real USGS stream gaging stations instead of a fixed station list.\n"
            "- Plots the discharge hydrograph on the page.\n"
            "- Labels the mean correctly as the mean of the selected time window.\n"
            "- Uses OpenAI only for tutoring, not for guessing the station from a hard-coded list.\n"
            "- Lets the student continue with follow-up questions after the guided portion ends."
        )

    with st.sidebar:
        st.subheader("Settings")
        model_name = st.text_input("OpenAI model name", value=DEFAULT_MODEL)
        client = get_openai_client()
        if client is None:
            st.warning("OPENAI_API_KEY not found. The hydrograph will still work, but guided tutoring will be disabled until you add the key.")
        else:
            st.success("OpenAI key detected.")
        st.info("Best search results usually come from a river name plus a state, such as 'Charles River MA' or 'Colorado River Arizona'.")

    with st.form("station_search"):
        col1, col2 = st.columns([3, 1])
        river_query = col1.text_input(
            "River or stream name",
            value=st.session_state.last_query,
            placeholder="Example: Charles River MA",
        )
        days_back = col2.number_input("Days back", min_value=1, max_value=60, value=int(st.session_state.days_back), step=1)
        submitted = st.form_submit_button("Find stations", use_container_width=True)

    if submitted:
        st.session_state.last_query = river_query.strip()
        st.session_state.days_back = int(days_back)
        reset_guided_state()
        st.session_state.hydro_loaded = False
        st.session_state.selected_station = None

        if not river_query.strip():
            st.error("Please type a river or stream name.")
        else:
            with st.spinner("Searching USGS stream gages..."):
                try:
                    candidates = search_usgs_stations(river_query)
                    st.session_state.station_candidates = candidates
                    st.session_state.selected_station_index = 0
                except Exception as exc:
                    st.session_state.station_candidates = []
                    st.error(f"USGS station search failed: {exc}")

    candidates = st.session_state.station_candidates
    if candidates:
        st.subheader("Candidate gaging stations")
        labels = [
            f"{item['station_nm']} | Site {item['site_no']} | {item.get('state_cd', '')}"
            for item in candidates
        ]
        selected_label = st.selectbox(
            "Choose the station to analyze",
            options=range(len(labels)),
            index=min(st.session_state.selected_station_index, len(labels) - 1),
            format_func=lambda i: labels[i],
        )
        st.session_state.selected_station_index = int(selected_label)

        if st.button("Load hydrograph", type="primary", use_container_width=True):
            chosen = candidates[st.session_state.selected_station_index]
            st.session_state.selected_station = chosen
            reset_guided_state()
            with st.spinner("Fetching USGS discharge data..."):
                try:
                    data = fetch_iv_data(chosen["site_no"], int(st.session_state.days_back))
                    st.session_state.hydro_data = data
                    st.session_state.hydro_loaded = True
                except Exception as exc:
                    st.session_state.hydro_loaded = False
                    st.error(f"USGS data request failed: {exc}")
    elif st.session_state.last_query:
        st.info("No candidate stations found. Try adding a state abbreviation or making the river name more specific.")

    if st.session_state.get("hydro_loaded"):
        station = st.session_state.selected_station
        data = st.session_state.hydro_data
        discharge_df = data["discharge"]
        stage_df = data["stage"]
        station_name = data["station_name"]
        summary = compute_hydrology_summary(discharge_df)

        if not summary:
            st.warning("This station did not return usable discharge data for the selected window.")
            return

        c1, c2, c3, c4 = st.columns(4)
        unit = summary.get("unit", "cfs")
        c1.metric("Latest discharge", f"{summary['latest_flow']:.1f} {unit}")
        c2.metric("Window mean", f"{summary['window_mean']:.1f} {unit}")
        c3.metric("Window minimum", f"{summary['window_min']:.1f} {unit}")
        c4.metric("Window maximum", f"{summary['window_max']:.1f} {unit}")

        fig = build_hydrograph(discharge_df, station_name)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Station and interpretation notes", expanded=True):
            st.markdown(
                f"**Station:** {station_name}\n\n"
                f"**USGS site:** {station['site_no']}\n\n"
                f"**State:** {station.get('state_cd', '')}\n\n"
                f"**Trend across selected window:** {summary['trend']}\n\n"
                f"**Net change across selected window:** {summary['net_change']:.1f} {unit}"
            )
            if not stage_df.empty and stage_df['value'].notna().any():
                latest_stage = stage_df.dropna(subset=['value']).iloc[-1]
                stage_unit = stage_df['unit'].iloc[0] if 'unit' in stage_df.columns and not stage_df.empty else 'ft'
                st.markdown(f"**Latest gage height:** {latest_stage['value']:.2f} {stage_unit}")
            st.caption("The horizontal mean line is the mean for the selected look-back window only. It is not a long-term normal discharge.")

        csv_df = discharge_df.copy()
        csv_df["datetime"] = csv_df["datetime"].astype(str)
        st.download_button(
            "Download discharge data as CSV",
            data=csv_df.to_csv(index=False),
            file_name=f"usgs_{station['site_no']}_discharge.csv",
            mime="text/csv",
        )

        st.subheader("Guided interpretation")
        if client is None:
            st.info("Add OPENAI_API_KEY to enable the guided tutor.")
            return

        context_text = station_context_text(station, summary, int(st.session_state.days_back))

        col_a, col_b = st.columns([1, 1])
        if col_a.button("Start guided exploration", use_container_width=True):
            reset_guided_state()
            with st.spinner("Dr. River Explorer is thinking..."):
                try:
                    reply = call_tutor(client, model_name, context_text, [], mode="initial")
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                except Exception as exc:
                    st.error(f"OpenAI error: {exc}")
        if col_b.button("Reset conversation", use_container_width=True):
            reset_guided_state()

        render_chat()

        student_text = st.chat_input("Type your answer or follow-up question")
        if student_text:
            st.session_state.chat_history.append({"role": "user", "content": student_text})

            mode = "follow_up" if st.session_state.guided_complete else "guided"
            st.session_state.student_turns += 1
            if not st.session_state.guided_complete and st.session_state.student_turns >= 6:
                mode = "wrap_up"

            with st.spinner("Dr. River Explorer is responding..."):
                try:
                    reply = call_tutor(client, model_name, context_text, st.session_state.chat_history, mode=mode)
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                    if mode == "wrap_up":
                        st.session_state.guided_complete = True
                except Exception as exc:
                    st.error(f"OpenAI error: {exc}")

            st.rerun()


if __name__ == "__main__":
    main()
