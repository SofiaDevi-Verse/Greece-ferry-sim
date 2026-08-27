"""
Core computation logic for the Greece Ferry Traffic Simulator.

Turns raw schedule data into:
- distance between ports (haversine)
- hourly crowd score per route
- estimated CO2 per sailing
- a simple 1-10 "best time to visit" score, blending crowd density with
  real weather (Open-Meteo, free, no API key required)
"""

import json
import math
import re
import time
from datetime import datetime, date as date_cls
from pathlib import Path

import pandas as pd
import requests

try:
    from zoneinfo import ZoneInfo
    ATHENS_TZ = ZoneInfo("Europe/Athens")
except Exception:
    ATHENS_TZ = None  # falls back to naive local time if zoneinfo unavailable

DATA_DIR = Path(__file__).parent.parent / "data"

# Simple in-memory cache so we don't hit the weather API on every single
# request. Keyed by rounded (lat, lon); refreshed every 30 minutes.
_WEATHER_CACHE = {}
_WEATHER_CACHE_TTL_SECONDS = 1800


_WEATHER_DETAILS_CACHE = {}


def get_weather_details(lat, lon):
    """Fetches today's REAL forecast for a port: actual temperature (°C),
    rain chance, and wind — plus a 1-10 travel-friendliness score derived
    from them. Falls back to neutral values if offline, so the app never
    breaks because of a network hiccup."""
    key = (round(lat, 3), round(lon, 3))
    now = time.time()
    if key in _WEATHER_DETAILS_CACHE:
        ts, cached = _WEATHER_DETAILS_CACHE[key]
        if now - ts < _WEATHER_CACHE_TTL_SECONDS:
            return cached

    details = {"temp_c": None, "precip_chance_pct": None, "wind_kmh": None, "score": 5.0}
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max,precipitation_probability_max,windspeed_10m_max",
                "timezone": "auto",
                "forecast_days": 1,
            },
            timeout=5,
        )
        resp.raise_for_status()
        daily = resp.json()["daily"]
        temp = daily["temperature_2m_max"][0]
        precip = daily["precipitation_probability_max"][0] or 0
        wind = daily["windspeed_10m_max"][0] or 0

        score = 10.0
        if temp is not None and (temp < 15 or temp > 35):
            score -= 3
        score -= precip / 10  # rainier day -> lower score
        if wind and wind > 30:
            score -= min(3, (wind - 30) / 5)  # high wind matters a lot for ferries
        score = max(1.0, min(10.0, score))

        details = {
            "temp_c": round(temp, 1) if temp is not None else None,
            "precip_chance_pct": round(precip),
            "wind_kmh": round(wind),
            "score": round(score, 1),
        }
    except Exception:
        pass  # keep neutral fallback — no internet, API down, etc.

    _WEATHER_DETAILS_CACHE[key] = (now, details)
    return details


def get_weather_score(lat, lon):
    """Backward-compatible wrapper — just the 1-10 score, for any code that
    only needs the number (best-time blending, etc.)."""
    return get_weather_details(lat, lon)["score"]


def best_time_label(score):
    """Turns the abstract 1-10 'best time' number into a plain sentence —
    no one intuitively knows what '4.7/10' means, but 'Okay time to go' or
    'Busy — worth checking other times' is immediately understandable."""
    if score >= 7:
        return "Great time to go"
    elif score >= 4.5:
        return "Okay time to go"
    else:
        return "Busy — consider another time"


_FORECAST_CACHE = {}


def get_weather_forecast(lat, lon, days=3):
    """Returns a real multi-day forecast (today + next `days`-1 days) for a
    port: date, max temp, rain chance, and a 1-10 travel-friendliness score
    for each day — using the same real Open-Meteo data as get_weather_score,
    just extended. Falls back to an empty list if offline."""
    key = (round(lat, 3), round(lon, 3), days)
    now = time.time()
    if key in _FORECAST_CACHE:
        ts, cached = _FORECAST_CACHE[key]
        if now - ts < _WEATHER_CACHE_TTL_SECONDS:
            return cached

    forecast = []
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max,precipitation_probability_max,windspeed_10m_max",
                "timezone": "auto",
                "forecast_days": days,
            },
            timeout=5,
        )
        resp.raise_for_status()
        daily = resp.json()["daily"]
        dates = daily.get("time", [])
        for i in range(len(dates)):
            temp = daily["temperature_2m_max"][i]
            precip = daily["precipitation_probability_max"][i] or 0
            wind = daily["windspeed_10m_max"][i] or 0
            score = 10.0
            if temp is not None and (temp < 15 or temp > 35):
                score -= 3
            score -= precip / 10
            if wind and wind > 30:
                score -= min(3, (wind - 30) / 5)
            score = max(1.0, min(10.0, round(score, 1)))
            forecast.append({
                "date": dates[i],
                "temp_max_c": temp,
                "precip_chance_pct": precip,
                "score": score,
            })
    except Exception:
        pass  # offline or API down — return whatever we got (possibly empty)

    _FORECAST_CACHE[key] = (now, forecast)
    return forecast


# Published-style reference values (grams CO2 per passenger-km)
EMISSION_FACTORS = {
    "conventional": 170,
    "highspeed": 275,
}

# Typical passenger capacity by vessel type
CAPACITY = {
    "conventional": 1800,
    "highspeed": 800,
}

# Real-world vessel type labels (from Ferryhopper etc.) mapped down to our
# two internal categories. Add more aliases here as you find them.
VESSEL_TYPE_ALIASES = {
    "conventional": "conventional",
    "open deck": "conventional",
    "ferry": "conventional",
    "highspeed": "highspeed",
    "high-speed": "highspeed",
    "high speed": "highspeed",
    "catamaran": "highspeed",
    "hydrofoil": "highspeed",
    "speedboat": "highspeed",
}


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two lat/lon points, in km."""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.asin(math.sqrt(a))


def load_ports():
    with open(DATA_DIR / "ports.json") as f:
        data = json.load(f)
    ports_by_id = {p["id"]: p for p in data["ports"]}
    return ports_by_id, data["routes"]


def route_distance_km(route, ports_by_id):
    a = ports_by_id[route["from"]]
    b = ports_by_id[route["to"]]
    return haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])


def load_schedule(csv_path=None):
    csv_path = csv_path or (DATA_DIR / "schedule_sample.csv")
    return pd.read_csv(csv_path)


# ============================================================================
# LIVE DATA: Ferryhopper MCP server — real, current sailings, replacing the
# need to manually re-collect a CSV every week. Falls back to the manual CSV
# automatically if the live service is unreachable, so the app never breaks.
# ============================================================================

FERRYHOPPER_MCP_URL = "https://mcp.ferryhopper.com/mcp"
_LIVE_SCHEDULE_CACHE = {}
_LIVE_SCHEDULE_CACHE_TTL_SECONDS = 1200  # 20 minutes


def _extract_hhmm(value):
    """Pulls HH:MM out of either a plain time string or a full ISO
    datetime like '2026-08-26T07:25:00+03:00'."""
    s = str(value)
    m = re.search(r"T(\d{2}):(\d{2})", s)  # ISO datetime
    if m:
        return f"{m.group(1)}:{m.group(2)}"
    m = re.match(r"^(\d{1,2}):(\d{2})", s)  # plain HH:MM
    if m:
        return f"{m.group(1)}:{m.group(2)}"
    return s[:5]


def _normalize_trip_record(raw, route_id, trip_date):
    """Ferryhopper's real response wraps each trip's actual departure/
    arrival inside a 'segments' array (confirmed live) — not flat fields on
    the trip itself, since a sailing can have multiple legs/stops. This
    reads the first segment's departure and the last segment's arrival."""
    def first_present(d, keys):
        for k in keys:
            if isinstance(d, dict) and k in d and d[k] not in (None, ""):
                return d[k]
        return None

    if not isinstance(raw, dict):
        return None

    segments = raw.get("segments")
    if not isinstance(segments, list) or len(segments) == 0:
        print(f"[Ferryhopper MCP] no 'segments' array for {route_id} — top-level keys: {list(raw.keys())}", flush=True)
        return None

    first_seg, last_seg = segments[0], segments[-1]

    dep = first_present(first_seg, [
        "departureTime", "departure_time", "depTime", "departure", "departureDateTime", "startTime"
    ])
    arr = first_present(last_seg, [
        "arrivalTime", "arrival_time", "arrTime", "arrival", "arrivalDateTime", "endTime"
    ])
    if not dep or not arr:
        print(f"[Ferryhopper MCP] couldn't find departure/arrival inside a segment for {route_id} — "
              f"first segment keys: {list(first_seg.keys()) if isinstance(first_seg, dict) else type(first_seg)}", flush=True)
        return None

    def find_nested(d, outer_keys, inner_keys):
        """Checks for a value nested one level deep, e.g. seg['vessel']['operator']."""
        for ok in outer_keys:
            if isinstance(d, dict) and ok in d and isinstance(d[ok], dict):
                val = first_present(d[ok], inner_keys)
                if val:
                    return val
        return None

    operator_keys = ["operator", "company", "carrier", "operatorName", "companyName",
                      "carrierName", "operatingCompany", "shipCompany", "operator_name"]
    operator = (
        first_present(first_seg, operator_keys)
        or first_present(raw, operator_keys)
        or find_nested(first_seg, ["vessel", "ship", "carrier", "operator"], ["name", "operator", "company"])
        or find_nested(raw, ["vessel", "ship", "carrier", "operator"], ["name", "operator", "company"])
    )
    if not operator:
        print(f"[Ferryhopper MCP] operator not found for {route_id} — "
              f"first segment full keys: {list(first_seg.keys()) if isinstance(first_seg, dict) else type(first_seg)}, "
              f"top-level trip keys: {list(raw.keys())}", flush=True)
        operator = "Unknown operator"
    vessel_type = first_present(first_seg, ["vesselType", "vessel_type", "shipType", "type"]) or ""

    # Intermediate stops: departure port of every segment after the first
    via_stops = []
    if len(segments) > 1:
        for seg in segments[1:]:
            stop_name = first_present(seg, [
                "departurePort", "departureLocation", "fromPort", "origin", "departure_port"
            ])
            if stop_name:
                via_stops.append(str(stop_name))
    via = ";".join(via_stops)

    return {
        "route_id": route_id,
        "operator": str(operator),
        "departure_time": _extract_hhmm(dep),
        "arrival_time": _extract_hhmm(arr),
        "vessel_type": str(vessel_type),
        "date": trip_date,
        "via": via,
    }


def _unwrap_exception_group(e):
    """Recursively pulls the real underlying exception(s) out of a Python
    ExceptionGroup — 'except Exception as e: print(e)' alone just shows an
    unhelpful 'unhandled errors in a TaskGroup' summary otherwise."""
    if hasattr(e, "exceptions"):
        parts = []
        for sub in e.exceptions:
            parts.extend(_unwrap_exception_group(sub))
        return parts
    return [f"{type(e).__name__}: {e}"]


async def _fetch_live_trips(departure_name, arrival_name, trip_date):
    """Calls the REAL Ferryhopper MCP server for one route/date. Returns a
    list of raw trip records, or None on any failure (network, protocol,
    empty result) — logged clearly so we can debug together if needed."""
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        async with streamable_http_client(FERRYHOPPER_MCP_URL) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print(f"[Ferryhopper MCP] querying: '{departure_name}' -> '{arrival_name}' on {trip_date}...", flush=True)
                result = await session.call_tool("search_trips", {
                    "departureLocation": departure_name,
                    "arrivalLocation": arrival_name,
                    "date": trip_date,
                })

                if result.is_error:
                    print(f"[Ferryhopper MCP] tool returned an error for {departure_name} -> {arrival_name}", flush=True)
                    return None

                if result.structured_content:
                    data = result.structured_content
                    # Might be a list directly, or wrapped under a key.
                    # "foundDirectItinerariesForTrip" is the REAL key name,
                    # confirmed live from Ferryhopper's actual response.
                    if isinstance(data, list):
                        print(f"[Ferryhopper MCP] {departure_name} -> {arrival_name}: got {len(data)} trip(s) directly", flush=True)
                        return data
                    if isinstance(data, dict):
                        for key in ["foundDirectItinerariesForTrip", "trips", "results", "data", "itineraries"]:
                            if key in data:
                                val = data[key]
                                if isinstance(val, list):
                                    print(f"[Ferryhopper MCP] {departure_name} -> {arrival_name}: got {len(val)} trip(s) under '{key}'", flush=True)
                                    if len(val) == 0:
                                        print(f"[Ferryhopper MCP] ^ that's ZERO trips — Ferryhopper found this route but no sailings for {trip_date}", flush=True)
                                    return val
                                if isinstance(val, dict):
                                    # one more level deep, in case it's nested further
                                    for inner_key in ["trips", "results", "data", "itineraries"]:
                                        if inner_key in val and isinstance(val[inner_key], list):
                                            print(f"[Ferryhopper MCP] {departure_name} -> {arrival_name}: got {len(val[inner_key])} trip(s) under '{key}.{inner_key}'", flush=True)
                                            return val[inner_key]
                                    print(f"[Ferryhopper MCP] '{key}' for {departure_name} -> {arrival_name} "
                                          f"is a dict, not a list — inner keys: {list(val.keys())}", flush=True)
                                    return None
                                print(f"[Ferryhopper MCP] '{key}' for {departure_name} -> {arrival_name} "
                                      f"is type {type(val).__name__}, value: {val!r}", flush=True)
                                return None
                    print(f"[Ferryhopper MCP] structured_content shape unrecognized for "
                          f"{departure_name} -> {arrival_name}: keys={list(data.keys()) if isinstance(data, dict) else type(data)}", flush=True)
                    return None

                # Fallback: look for JSON inside plain text content blocks
                for block in result.content:
                    text = getattr(block, "text", None)
                    if text:
                        try:
                            parsed = json.loads(text)
                            return parsed if isinstance(parsed, list) else parsed.get("trips", [])
                        except Exception:
                            continue

                print(f"[Ferryhopper MCP] no usable content for {departure_name} -> {arrival_name}", flush=True)
                return None
    except ImportError:
        print("[Ferryhopper MCP] the 'mcp' package isn't installed — run: pip3 install mcp", flush=True)
        return None
    except Exception as e:
        real_errors = " | ".join(_unwrap_exception_group(e))
        print(f"[Ferryhopper MCP] live fetch failed for {departure_name} -> {arrival_name}: {real_errors}", flush=True)
        return None


async def get_live_schedule_dataframe():
    """Queries Ferryhopper's live MCP server for every route we track, for
    today's real date. Returns a real pandas DataFrame if at least one route
    returned usable data, otherwise None (signals: fall back to the CSV)."""
    import asyncio

    ports_by_id, routes = load_ports()
    today_str = date_cls.today().isoformat()

    async def fetch_one(route):
        # Strip parenthetical qualifiers like "(New Port)" or "(Parikia)" —
        # Ferryhopper's location lookup appears to want the plain island
        # name, confirmed by some routes erroring with the fuller name
        # while plain names (Piraeus, Naxos) succeeded.
        raw_from = ports_by_id[route["from"]]["name"]
        raw_to = ports_by_id[route["to"]]["name"]
        from_name = re.sub(r"\s*\(.*?\)", "", raw_from).strip()
        to_name = re.sub(r"\s*\(.*?\)", "", raw_to).strip()
        try:
            raw_trips = await asyncio.wait_for(
                _fetch_live_trips(from_name, to_name, today_str), timeout=10
            )
        except asyncio.TimeoutError:
            print(f"[Ferryhopper MCP] timed out for {from_name} -> {to_name}", flush=True)
            return []
        if not raw_trips:
            print(f"[Ferryhopper MCP] {from_name} -> {to_name}: nothing usable came back (see above for why)", flush=True)
            return []
        rows = []
        for raw in raw_trips:
            normalized = _normalize_trip_record(raw, route["id"], today_str)
            if normalized:
                rows.append(normalized)
        return rows

    # Sequential, not concurrent — firing all 9 requests at once appeared to
    # trigger "Server returned an error response" on most of them, while a
    # single request succeeded. One at a time, with a small pause, is much
    # gentler on their server.
    all_rows = []
    for route in routes:
        route_rows = await fetch_one(route)
        all_rows.extend(route_rows)
        await asyncio.sleep(0.3)

    if not all_rows:
        print("[Ferryhopper MCP] no live data came back for any route — falling back to your saved CSV", flush=True)
        return None

    routes_with_data = set(row["route_id"] for row in all_rows)
    print(f"[Ferryhopper MCP] got {len(all_rows)} real live sailings across "
          f"{len(routes_with_data)} routes", flush=True)
    return pd.DataFrame(all_rows)


async def get_current_schedule():
    """Returns (dataframe, source) where source is 'live' or 'csv_fallback'.
    Caches the live result for 20 minutes so we're not hitting Ferryhopper's
    server on every single page load."""
    cache_key = "schedule"
    now = time.time()
    if cache_key in _LIVE_SCHEDULE_CACHE:
        ts, df, source = _LIVE_SCHEDULE_CACHE[cache_key]
        if now - ts < _LIVE_SCHEDULE_CACHE_TTL_SECONDS:
            return df, source

    live_df = await get_live_schedule_dataframe()
    if live_df is not None and len(live_df) > 0:
        _LIVE_SCHEDULE_CACHE[cache_key] = (now, live_df, "live")
        return live_df, "live"

    csv_df = load_schedule()
    _LIVE_SCHEDULE_CACHE[cache_key] = (now, csv_df, "csv_fallback")
    return csv_df, "csv_fallback"


def parse_hour_minute(time_str):
    """Robustly parse a time string like '7:25', '07:25', '07:25:00',
    or '03:45 (+1 day)' into (hour, minute). Any extra text (like a
    '+1 day' note) is ignored — we detect overnight crossings ourselves."""
    s = str(time_str).strip()
    match = re.match(r"^(\d{1,2}):(\d{2})", s)
    if match:
        return int(match.group(1)), int(match.group(2))
    parts = s.split(":")
    hour = int(parts[0])
    minute = int(parts[1]) if len(parts) > 1 else 0
    return hour, minute


def compute_all(schedule=None, data_source="manual"):
    """Returns a dict keyed by route_id with hourly crowd/CO2 breakdown + a best-time score."""
    ports_by_id, routes = load_ports()
    schedule = schedule if schedule is not None else load_schedule()

    schedule["passengers"] = (
        schedule["vessel_type"]
        .str.strip()
        .str.lower()
        .map(VESSEL_TYPE_ALIASES)
        .fillna("conventional")  # unrecognized labels default to conventional (the safer/larger estimate)
        .map(CAPACITY)
    )
    schedule["vessel_type_norm"] = (
        schedule["vessel_type"].str.strip().str.lower().map(VESSEL_TYPE_ALIASES).fillna("conventional")
    )
    # Make route matching case/whitespace-insensitive, since spreadsheet edits
    # often introduce inconsistent capitalization (e.g. "Mykonos_santorini")
    schedule["route_id"] = schedule["route_id"].str.strip().str.lower()
    schedule["hour"] = schedule["departure_time"].apply(lambda t: parse_hour_minute(t)[0])

    # Duration in minutes, only computed if an arrival_time column exists
    if "arrival_time" in schedule.columns:
        def duration_minutes(row):
            dep_h, dep_m = parse_hour_minute(row["departure_time"])
            arr_h, arr_m = parse_hour_minute(row["arrival_time"])
            dep_total = dep_h * 60 + dep_m
            arr_total = arr_h * 60 + arr_m
            if arr_total < dep_total:  # sailing crosses midnight
                arr_total += 24 * 60
            return arr_total - dep_total

        schedule["duration_min"] = schedule.apply(duration_minutes, axis=1)
    else:
        schedule["duration_min"] = None

    results = {}
    for route in routes:
        rid = route["id"]
        dist_km = route_distance_km(route, ports_by_id)

        route_sched = schedule[schedule["route_id"] == rid].copy()
        route_sched["co2_kg"] = (
            dist_km
            * route_sched["vessel_type_norm"].map(EMISSION_FACTORS)
            * route_sched["passengers"]
            / 1000
        )

        hourly_crowd = (
            route_sched.groupby("hour")["passengers"].sum().reindex(range(24), fill_value=0)
        )

        total_co2 = round(route_sched["co2_kg"].sum(), 1)
        total_passengers = int(route_sched["passengers"].sum())
        sailings = len(route_sched)

        # Average trip duration, if we have arrival times
        if route_sched["duration_min"].notna().any():
            avg_duration_min = round(route_sched["duration_min"].mean(), 0)
        else:
            avg_duration_min = None

        # Real per-sailing details — same as the on-demand search feature,
        # so the "Popular routes" cards are just as informative, not a
        # second-class aggregate-only experience.
        sailings_detail = []
        if route_sched["duration_min"].notna().any():
            for _, srow in route_sched.dropna(subset=["duration_min"]).sort_values("departure_time").iterrows():
                via_raw = srow.get("via", "")
                via_list = [v for v in str(via_raw).split(";") if v and v.lower() != "nan"] if pd.notna(via_raw) else []
                sailings_detail.append({
                    "operator": srow.get("operator", "Unknown"),
                    "vessel_type": srow.get("vessel_type_norm", ""),
                    "departure_time": srow["departure_time"],
                    "arrival_time": srow.get("arrival_time"),
                    "duration_min": int(srow["duration_min"]),
                    "co2_kg_this_sailing": round(srow["co2_kg"], 1),
                    "via": via_list,
                })

        # crowd score 0-10 (relative to a "busy day" ceiling of ~6000 passengers/route)
        crowd_score = min(10, round((total_passengers / 6000) * 10, 1))

        # Most common operator on this route, so the frontend can show the
        # right ferry icon (SeaJets, Blue Star, etc.) instead of a generic dot
        if sailings > 0 and "operator" in route_sched.columns:
            dominant_operator = route_sched["operator"].mode().iloc[0]
        else:
            dominant_operator = None

        # Real weather at the destination port, blended with crowd density.
        # crowd contributes 60% (inverted — busier = worse), weather 40%.
        dest_port = ports_by_id[route["to"]]
        weather_details = get_weather_details(dest_port["lat"], dest_port["lon"])
        weather_score = weather_details["score"]
        weather_forecast = get_weather_forecast(dest_port["lat"], dest_port["lon"], days=3)
        best_time_score = round(
            max(1, min(10, (10 - crowd_score * 0.8) * 0.6 + weather_score * 0.4)), 1
        )

        # CO2 comparison vs. flying the same real distance. Flight factor is
        # a standard published short-haul figure (~250g CO2/passenger-km),
        # applied to the SAME real distance as the ferry for a fair compare.
        FLIGHT_EMISSION_FACTOR_G_PER_KM = 250
        flight_co2_kg_est = round(dist_km * FLIGHT_EMISSION_FACTOR_G_PER_KM * total_passengers / 1000, 1)
        co2_savings_pct = (
            round((1 - total_co2 / flight_co2_kg_est) * 100, 1)
            if flight_co2_kg_est > 0 else None
        )

        # One real sample sailing for this route (earliest departure in the
        # data), used to power the "add to calendar" feature with genuine
        # times rather than an invented placeholder.
        if sailings > 0:
            sample_row = route_sched.sort_values("departure_time").iloc[0]
            sample_sailing = {
                "operator": sample_row.get("operator"),
                "departure_time": sample_row.get("departure_time"),
                "arrival_time": sample_row.get("arrival_time") if pd.notna(sample_row.get("arrival_time")) else None,
                "date": sample_row.get("date") if "date" in sample_row and pd.notna(sample_row.get("date")) else None,
            }
        else:
            sample_sailing = None

        # Real Ferryhopper route pages — only using URLs actually confirmed
        # live during research (their slug pattern isn't fully consistent,
        # e.g. some routes need "-to-" or a reordered prefix). Anything not
        # personally verified falls back to Ferryhopper's general route
        # browser rather than guessing a URL that might 404.
        BOOKING_URL_OVERRIDES = {
            "piraeus_mykonos": "https://www.ferryhopper.com/en/ferry-routes/direct/piraeus-athens-to-mykonos",
            "piraeus_santorini": "https://www.ferryhopper.com/en/ferry-routes/direct/athens-piraeus-to-santorini",
            "piraeus_naxos": "https://www.ferryhopper.com/en/ferry-routes/direct/piraeus-naxos",
            "naxos_santorini": "https://www.ferryhopper.com/en/ferry-routes/direct/santorini-to-naxos",
            "mykonos_naxos": "https://www.ferryhopper.com/en/ferry-routes/direct/mykonos-to-naxos",
        }
        booking_url = BOOKING_URL_OVERRIDES.get(rid, "https://www.ferryhopper.com/en/ferry-routes")

        results[rid] = {
            "route_id": rid,
            "from": ports_by_id[route["from"]]["name"],
            "to": ports_by_id[route["to"]]["name"],
            "from_id": route["from"],
            "to_id": route["to"],
            "distance_km": round(dist_km, 1),
            "avg_duration_min": avg_duration_min,
            "sailings": sailings,
            "sailings_detail": sailings_detail,
            "total_passengers_est": total_passengers,
            "total_co2_kg_est": total_co2,
            "flight_co2_kg_est": flight_co2_kg_est,
            "co2_savings_pct": co2_savings_pct,
            "crowd_score": crowd_score,
            "weather_score": weather_score,
            "weather_temp_c": weather_details["temp_c"],
            "weather_precip_pct": weather_details["precip_chance_pct"],
            "weather_forecast": weather_forecast,
            "best_time_score": best_time_score,
            "best_time_label": best_time_label(best_time_score),
            "dominant_operator": dominant_operator,
            "sample_sailing": sample_sailing,
            "booking_url": booking_url,
            "hourly_crowd": hourly_crowd.to_dict(),
            "data_source": data_source,
        }

    return results


def get_live_ferries(now=None, schedule=None):
    """Checks the REAL schedule against the current time-of-day and returns
    only sailings that are genuinely in progress right now — with how far
    along (0-1) they are, based on real elapsed time between their real
    departure and arrival. No sailing in progress on a route = nothing
    returned for it. This replaces any made-up animation with something
    tied to your actual timetable."""
    if now is None:
        now = datetime.now(ATHENS_TZ) if ATHENS_TZ else datetime.now()

    now_minutes = now.hour * 60 + now.minute

    schedule = schedule if schedule is not None else load_schedule()
    schedule = schedule.copy()
    schedule["route_id"] = schedule["route_id"].str.strip().str.lower()

    active = []
    for _, row in schedule.iterrows():
        dep_h, dep_m = parse_hour_minute(row["departure_time"])
        dep_total = dep_h * 60 + dep_m

        if "arrival_time" not in row or pd.isna(row.get("arrival_time")):
            continue
        arr_h, arr_m = parse_hour_minute(row["arrival_time"])
        arr_total = arr_h * 60 + arr_m
        if arr_total < dep_total:
            arr_total += 24 * 60

        # Compare against "now", also checking the wrapped (+1 day) window
        # so a sailing that started yesterday but is still going is caught.
        now_candidates = [now_minutes, now_minutes + 24 * 60]
        for now_t in now_candidates:
            if dep_total <= now_t <= arr_total:
                progress = (now_t - dep_total) / max(1, (arr_total - dep_total))
                via_raw = row.get("via", "")
                via_list = (
                    [v.strip() for v in str(via_raw).split(";") if v.strip()]
                    if pd.notna(via_raw) and str(via_raw).strip()
                    else []
                )
                active.append({
                    "route_id": row["route_id"],
                    "operator": row.get("operator", None),
                    "vessel_type": row.get("vessel_type", None),
                    "departure_time": row["departure_time"],
                    "arrival_time": row["arrival_time"],
                    "via": via_list,
                    "progress": round(progress, 4),
                })
                break

    return active


async def search_route_live(from_id, to_id):
    """On-demand lookup for ANY two ports (not just the 9 precomputed
    routes) — queries Ferryhopper live for exactly this pair, right now.
    Returns a dict with real stats if sailings were found, or a 'found':
    False result if not (still a valid, honest response, not an error)."""
    ports_by_id, _ = load_ports()
    if from_id not in ports_by_id or to_id not in ports_by_id:
        return {"found": False, "error": "unknown port"}

    from_port = ports_by_id[from_id]
    to_port = ports_by_id[to_id]
    from_name = re.sub(r"\s*\(.*?\)", "", from_port["name"]).strip()
    to_name = re.sub(r"\s*\(.*?\)", "", to_port["name"]).strip()
    today_str = date_cls.today().isoformat()
    route_id = f"{from_id}_{to_id}"

    raw_trips = await _fetch_live_trips(from_name, to_name, today_str)
    if not raw_trips:
        return {
            "found": False,
            "from": from_port["name"],
            "to": to_port["name"],
            "from_id": from_id,
            "to_id": to_id,
        }

    rows = [r for r in (_normalize_trip_record(t, route_id, today_str) for t in raw_trips) if r]
    if not rows:
        return {
            "found": False,
            "from": from_port["name"],
            "to": to_port["name"],
            "from_id": from_id,
            "to_id": to_id,
        }

    dist_km = haversine_km(from_port["lat"], from_port["lon"], to_port["lat"], to_port["lon"])
    schedule_df = pd.DataFrame(rows)
    schedule_df["passengers"] = (
        schedule_df["vessel_type"].str.strip().str.lower().map(VESSEL_TYPE_ALIASES).fillna("conventional").map(CAPACITY)
    )
    schedule_df["vessel_type_norm"] = (
        schedule_df["vessel_type"].str.strip().str.lower().map(VESSEL_TYPE_ALIASES).fillna("conventional")
    )
    schedule_df["co2_kg"] = (
        dist_km * schedule_df["vessel_type_norm"].map(EMISSION_FACTORS) * schedule_df["passengers"] / 1000
    )

    def duration_minutes(row):
        dep_h, dep_m = parse_hour_minute(row["departure_time"])
        arr_h, arr_m = parse_hour_minute(row["arrival_time"])
        dep_total, arr_total = dep_h * 60 + dep_m, arr_h * 60 + arr_m
        if arr_total < dep_total:
            arr_total += 24 * 60
        return arr_total - dep_total

    schedule_df["duration_min"] = schedule_df.apply(duration_minutes, axis=1)

    total_passengers = int(schedule_df["passengers"].sum())
    total_co2 = round(schedule_df["co2_kg"].sum(), 1)
    avg_duration_min = round(schedule_df["duration_min"].mean(), 0)
    crowd_score = min(10, round((total_passengers / 6000) * 10, 1))
    weather_details = get_weather_details(to_port["lat"], to_port["lon"])
    weather_score = weather_details["score"]
    weather_forecast = get_weather_forecast(to_port["lat"], to_port["lon"], days=3)
    best_time_score = round(max(1, min(10, (10 - crowd_score * 0.8) * 0.6 + weather_score * 0.4)), 1)
    FLIGHT_EMISSION_FACTOR_G_PER_KM = 250
    flight_co2 = round(dist_km * FLIGHT_EMISSION_FACTOR_G_PER_KM * total_passengers / 1000, 1)
    co2_savings_pct = round((1 - total_co2 / flight_co2) * 100, 1) if flight_co2 > 0 else None
    dominant_operator = schedule_df["operator"].mode().iloc[0] if len(schedule_df) else None

    # Real per-sailing details — this is what a traveler actually needs:
    # exact departure/arrival times, per-sailing CO2 (not a confusing total
    # across every sailing's full capacity), and any real stopover islands.
    sailings_detail = []
    for _, row in schedule_df.sort_values("departure_time").iterrows():
        via_raw = row.get("via", "")
        via_list = [v for v in str(via_raw).split(";") if v and v.lower() != "nan"] if pd.notna(via_raw) else []
        sailings_detail.append({
            "operator": row["operator"],
            "vessel_type": row["vessel_type_norm"],
            "departure_time": row["departure_time"],
            "arrival_time": row["arrival_time"],
            "duration_min": int(row["duration_min"]),
            "co2_kg_this_sailing": round(row["co2_kg"], 1),
            "via": via_list,
        })

    return {
        "found": True,
        "route_id": route_id,
        "from": from_port["name"],
        "to": to_port["name"],
        "from_id": from_id,
        "to_id": to_id,
        "distance_km": round(dist_km, 1),
        "avg_duration_min": avg_duration_min,
        "sailings": len(rows),
        "sailings_detail": sailings_detail,
        "total_passengers_est": total_passengers,
        "total_co2_kg_est": total_co2,
        "flight_co2_kg_est": flight_co2,
        "co2_savings_pct": co2_savings_pct,
        "crowd_score": crowd_score,
        "weather_score": weather_score,
        "weather_temp_c": weather_details["temp_c"],
        "weather_precip_pct": weather_details["precip_chance_pct"],
        "weather_forecast": weather_forecast,
        "best_time_score": best_time_score,
        "best_time_label": best_time_label(best_time_score),
        "dominant_operator": dominant_operator,
        "data_source": "live",
    }


def get_next_sailings(now=None, schedule=None):
    """For every route, finds the REAL next upcoming departure from the
    actual schedule (today, or wrapping to the next day if everything today
    has already left), and how many minutes until it departs. Used for a
    live countdown — built from real timetable data, not invented."""
    if now is None:
        now = datetime.now(ATHENS_TZ) if ATHENS_TZ else datetime.now()
    now_minutes = now.hour * 60 + now.minute

    schedule = schedule if schedule is not None else load_schedule()
    schedule = schedule.copy()
    schedule["route_id"] = schedule["route_id"].str.strip().str.lower()

    next_by_route = {}
    for _, row in schedule.iterrows():
        rid = row["route_id"]
        dep_h, dep_m = parse_hour_minute(row["departure_time"])
        dep_total = dep_h * 60 + dep_m

        minutes_until = dep_total - now_minutes
        if minutes_until < 0:
            minutes_until += 24 * 60  # already departed today — next one is tomorrow same time

        candidate = {
            "route_id": rid,
            "operator": row.get("operator", None),
            "departure_time": row["departure_time"],
            "minutes_until": int(minutes_until),
        }
        if rid not in next_by_route or minutes_until < next_by_route[rid]["minutes_until"]:
            next_by_route[rid] = candidate

    return next_by_route


# Real, publicly documented facts about each port (gates, transport links)
# — not invented. Only included where we have a specific, checkable fact;
# ports without an entry here just won't show extra amenities info.
PORT_AMENITIES = {
    "piraeus": {
        "gates": "Gates E4, E6, E7, E9, E10 (Cyclades ferries)",
        "transport": "Athens Metro Lines 1 & 3 serve the port directly",
    },
    "naxos": {
        "transport": "Short walk into Chora (the island's capital) from the port",
    },
    "santorini": {
        "note": "Athinios Port — 8km from Fira, 20km from Oia",
    },
}


if __name__ == "__main__":
    import pprint

    pprint.pprint(compute_all())

"""
Core computation logic for the Greece Ferry Traffic Simulator.

Turns raw schedule data into:
- distance between ports (haversine)
- hourly crowd score per route
- estimated CO2 per sailing
- a simple 1-10 "best time to visit" score, blending crowd density with
  real weather (Open-Meteo, free, no API key required)
"""

import json
import math
import re
import time
from datetime import datetime, date as date_cls
from pathlib import Path

import pandas as pd
import requests

try:
    from zoneinfo import ZoneInfo
    ATHENS_TZ = ZoneInfo("Europe/Athens")
except Exception:
    ATHENS_TZ = None  # falls back to naive local time if zoneinfo unavailable


def today_in_athens():
    """Real Greek local date — NOT the server's own timezone. Servers
    (e.g. Render) run on UTC, so late at night in Greece, a naive
    date.today() would return the WRONG date (still 'yesterday' in UTC),
    causing live ferry searches to silently return zero results."""
    if ATHENS_TZ:
        return datetime.now(ATHENS_TZ).date().isoformat()
    return date_cls.today().isoformat()


DATA_DIR = Path(__file__).parent.parent / "data"

# Simple in-memory cache so we don't hit the weather API on every single
# request. Keyed by rounded (lat, lon); refreshed every 30 minutes.
_WEATHER_CACHE = {}
_WEATHER_CACHE_TTL_SECONDS = 1800


_WEATHER_DETAILS_CACHE = {}


def get_weather_details(lat, lon):
    """Fetches today's REAL forecast for a port: actual temperature (°C),
    rain chance, and wind — plus a 1-10 travel-friendliness score derived
    from them. Falls back to neutral values if offline, so the app never
    breaks because of a network hiccup."""
    key = (round(lat, 3), round(lon, 3))
    now = time.time()
    if key in _WEATHER_DETAILS_CACHE:
        ts, cached = _WEATHER_DETAILS_CACHE[key]
        if now - ts < _WEATHER_CACHE_TTL_SECONDS:
            return cached

    details = {"temp_c": None, "precip_chance_pct": None, "wind_kmh": None, "score": 5.0}
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max,precipitation_probability_max,windspeed_10m_max",
                "timezone": "auto",
                "forecast_days": 1,
            },
            timeout=5,
        )
        resp.raise_for_status()
        daily = resp.json()["daily"]
        temp = daily["temperature_2m_max"][0]
        precip = daily["precipitation_probability_max"][0] or 0
        wind = daily["windspeed_10m_max"][0] or 0

        score = 10.0
        if temp is not None and (temp < 15 or temp > 35):
            score -= 3
        score -= precip / 10  # rainier day -> lower score
        if wind and wind > 30:
            score -= min(3, (wind - 30) / 5)  # high wind matters a lot for ferries
        score = max(1.0, min(10.0, score))

        details = {
            "temp_c": round(temp, 1) if temp is not None else None,
            "precip_chance_pct": round(precip),
            "wind_kmh": round(wind),
            "score": round(score, 1),
        }
    except Exception:
        pass  # keep neutral fallback — no internet, API down, etc.

    _WEATHER_DETAILS_CACHE[key] = (now, details)
    return details


def get_weather_score(lat, lon):
    """Backward-compatible wrapper — just the 1-10 score, for any code that
    only needs the number (best-time blending, etc.)."""
    return get_weather_details(lat, lon)["score"]


def best_time_label(score):
    """Turns the abstract 1-10 'best time' number into a plain sentence —
    no one intuitively knows what '4.7/10' means, but 'Okay time to go' or
    'Busy — worth checking other times' is immediately understandable."""
    if score >= 7:
        return "Great time to go"
    elif score >= 4.5:
        return "Okay time to go"
    else:
        return "Busy — consider another time"


_FORECAST_CACHE = {}


def get_weather_forecast(lat, lon, days=3):
    """Returns a real multi-day forecast (today + next `days`-1 days) for a
    port: date, max temp, rain chance, and a 1-10 travel-friendliness score
    for each day — using the same real Open-Meteo data as get_weather_score,
    just extended. Falls back to an empty list if offline."""
    key = (round(lat, 3), round(lon, 3), days)
    now = time.time()
    if key in _FORECAST_CACHE:
        ts, cached = _FORECAST_CACHE[key]
        if now - ts < _WEATHER_CACHE_TTL_SECONDS:
            return cached

    forecast = []
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max,precipitation_probability_max,windspeed_10m_max",
                "timezone": "auto",
                "forecast_days": days,
            },
            timeout=5,
        )
        resp.raise_for_status()
        daily = resp.json()["daily"]
        dates = daily.get("time", [])
        for i in range(len(dates)):
            temp = daily["temperature_2m_max"][i]
            precip = daily["precipitation_probability_max"][i] or 0
            wind = daily["windspeed_10m_max"][i] or 0
            score = 10.0
            if temp is not None and (temp < 15 or temp > 35):
                score -= 3
            score -= precip / 10
            if wind and wind > 30:
                score -= min(3, (wind - 30) / 5)
            score = max(1.0, min(10.0, round(score, 1)))
            forecast.append({
                "date": dates[i],
                "temp_max_c": temp,
                "precip_chance_pct": precip,
                "score": score,
            })
    except Exception:
        pass  # offline or API down — return whatever we got (possibly empty)

    _FORECAST_CACHE[key] = (now, forecast)
    return forecast


# Published-style reference values (grams CO2 per passenger-km)
EMISSION_FACTORS = {
    "conventional": 170,
    "highspeed": 275,
}

# Typical passenger capacity by vessel type
CAPACITY = {
    "conventional": 1800,
    "highspeed": 800,
}

# Real-world vessel type labels (from Ferryhopper etc.) mapped down to our
# two internal categories. Add more aliases here as you find them.
VESSEL_TYPE_ALIASES = {
    "conventional": "conventional",
    "open deck": "conventional",
    "ferry": "conventional",
    "highspeed": "highspeed",
    "high-speed": "highspeed",
    "high speed": "highspeed",
    "catamaran": "highspeed",
    "hydrofoil": "highspeed",
    "speedboat": "highspeed",
}


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two lat/lon points, in km."""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.asin(math.sqrt(a))


def load_ports():
    with open(DATA_DIR / "ports.json") as f:
        data = json.load(f)
    ports_by_id = {p["id"]: p for p in data["ports"]}
    return ports_by_id, data["routes"]


def route_distance_km(route, ports_by_id):
    a = ports_by_id[route["from"]]
    b = ports_by_id[route["to"]]
    return haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])


def load_schedule(csv_path=None):
    csv_path = csv_path or (DATA_DIR / "schedule_sample.csv")
    return pd.read_csv(csv_path)


# ============================================================================
# LIVE DATA: Ferryhopper MCP server — real, current sailings, replacing the
# need to manually re-collect a CSV every week. Falls back to the manual CSV
# automatically if the live service is unreachable, so the app never breaks.
# ============================================================================

FERRYHOPPER_MCP_URL = "https://mcp.ferryhopper.com/mcp"
_LIVE_SCHEDULE_CACHE = {}
_LIVE_SCHEDULE_CACHE_TTL_SECONDS = 1200  # 20 minutes


def _extract_hhmm(value):
    """Pulls HH:MM out of either a plain time string or a full ISO
    datetime like '2026-08-26T07:25:00+03:00'."""
    s = str(value)
    m = re.search(r"T(\d{2}):(\d{2})", s)  # ISO datetime
    if m:
        return f"{m.group(1)}:{m.group(2)}"
    m = re.match(r"^(\d{1,2}):(\d{2})", s)  # plain HH:MM
    if m:
        return f"{m.group(1)}:{m.group(2)}"
    return s[:5]


def _normalize_trip_record(raw, route_id, trip_date):
    """Ferryhopper's real response wraps each trip's actual departure/
    arrival inside a 'segments' array (confirmed live) — not flat fields on
    the trip itself, since a sailing can have multiple legs/stops. This
    reads the first segment's departure and the last segment's arrival."""
    def first_present(d, keys):
        for k in keys:
            if isinstance(d, dict) and k in d and d[k] not in (None, ""):
                return d[k]
        return None

    if not isinstance(raw, dict):
        return None

    segments = raw.get("segments")
    if not isinstance(segments, list) or len(segments) == 0:
        print(f"[Ferryhopper MCP] no 'segments' array for {route_id} — top-level keys: {list(raw.keys())}", flush=True)
        return None

    first_seg, last_seg = segments[0], segments[-1]

    dep = first_present(first_seg, [
        "departureTime", "departure_time", "depTime", "departure", "departureDateTime", "startTime"
    ])
    arr = first_present(last_seg, [
        "arrivalTime", "arrival_time", "arrTime", "arrival", "arrivalDateTime", "endTime"
    ])
    if not dep or not arr:
        print(f"[Ferryhopper MCP] couldn't find departure/arrival inside a segment for {route_id} — "
              f"first segment keys: {list(first_seg.keys()) if isinstance(first_seg, dict) else type(first_seg)}", flush=True)
        return None

    def find_nested(d, outer_keys, inner_keys):
        """Checks for a value nested one level deep, e.g. seg['vessel']['operator']."""
        for ok in outer_keys:
            if isinstance(d, dict) and ok in d and isinstance(d[ok], dict):
                val = first_present(d[ok], inner_keys)
                if val:
                    return val
        return None

    operator_keys = ["operator", "company", "carrier", "operatorName", "companyName",
                      "carrierName", "operatingCompany", "shipCompany", "operator_name"]
    operator = (
        first_present(first_seg, operator_keys)
        or first_present(raw, operator_keys)
        or find_nested(first_seg, ["vessel", "ship", "carrier", "operator"], ["name", "operator", "company"])
        or find_nested(raw, ["vessel", "ship", "carrier", "operator"], ["name", "operator", "company"])
    )
    if not operator:
        print(f"[Ferryhopper MCP] operator not found for {route_id} — "
              f"first segment full keys: {list(first_seg.keys()) if isinstance(first_seg, dict) else type(first_seg)}, "
              f"top-level trip keys: {list(raw.keys())}", flush=True)
        operator = "Unknown operator"
    vessel_type = first_present(first_seg, ["vesselType", "vessel_type", "shipType", "type"]) or ""

    # Intermediate stops: departure port of every segment after the first
    via_stops = []
    if len(segments) > 1:
        for seg in segments[1:]:
            stop_name = first_present(seg, [
                "departurePort", "departureLocation", "fromPort", "origin", "departure_port"
            ])
            if stop_name:
                via_stops.append(str(stop_name))
    via = ";".join(via_stops)

    return {
        "route_id": route_id,
        "operator": str(operator),
        "departure_time": _extract_hhmm(dep),
        "arrival_time": _extract_hhmm(arr),
        "vessel_type": str(vessel_type),
        "date": trip_date,
        "via": via,
    }


def _unwrap_exception_group(e):
    """Recursively pulls the real underlying exception(s) out of a Python
    ExceptionGroup — 'except Exception as e: print(e)' alone just shows an
    unhelpful 'unhandled errors in a TaskGroup' summary otherwise."""
    if hasattr(e, "exceptions"):
        parts = []
        for sub in e.exceptions:
            parts.extend(_unwrap_exception_group(sub))
        return parts
    return [f"{type(e).__name__}: {e}"]


async def _fetch_live_trips(departure_name, arrival_name, trip_date):
    """Calls the REAL Ferryhopper MCP server for one route/date. Returns a
    list of raw trip records, or None on any failure (network, protocol,
    empty result) — logged clearly so we can debug together if needed."""
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        async with streamable_http_client(FERRYHOPPER_MCP_URL) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print(f"[Ferryhopper MCP] querying: '{departure_name}' -> '{arrival_name}' on {trip_date}...", flush=True)
                result = await session.call_tool("search_trips", {
                    "departureLocation": departure_name,
                    "arrivalLocation": arrival_name,
                    "date": trip_date,
                })

                if result.is_error:
                    print(f"[Ferryhopper MCP] tool returned an error for {departure_name} -> {arrival_name}", flush=True)
                    return None

                if result.structured_content:
                    data = result.structured_content
                    # Might be a list directly, or wrapped under a key.
                    # "foundDirectItinerariesForTrip" is the REAL key name,
                    # confirmed live from Ferryhopper's actual response.
                    if isinstance(data, list):
                        print(f"[Ferryhopper MCP] {departure_name} -> {arrival_name}: got {len(data)} trip(s) directly", flush=True)
                        return data
                    if isinstance(data, dict):
                        for key in ["foundDirectItinerariesForTrip", "trips", "results", "data", "itineraries"]:
                            if key in data:
                                val = data[key]
                                if isinstance(val, list):
                                    print(f"[Ferryhopper MCP] {departure_name} -> {arrival_name}: got {len(val)} trip(s) under '{key}'", flush=True)
                                    if len(val) == 0:
                                        print(f"[Ferryhopper MCP] ^ that's ZERO trips — Ferryhopper found this route but no sailings for {trip_date}", flush=True)
                                    return val
                                if isinstance(val, dict):
                                    # one more level deep, in case it's nested further
                                    for inner_key in ["trips", "results", "data", "itineraries"]:
                                        if inner_key in val and isinstance(val[inner_key], list):
                                            print(f"[Ferryhopper MCP] {departure_name} -> {arrival_name}: got {len(val[inner_key])} trip(s) under '{key}.{inner_key}'", flush=True)
                                            return val[inner_key]
                                    print(f"[Ferryhopper MCP] '{key}' for {departure_name} -> {arrival_name} "
                                          f"is a dict, not a list — inner keys: {list(val.keys())}", flush=True)
                                    return None
                                print(f"[Ferryhopper MCP] '{key}' for {departure_name} -> {arrival_name} "
                                      f"is type {type(val).__name__}, value: {val!r}", flush=True)
                                return None
                    print(f"[Ferryhopper MCP] structured_content shape unrecognized for "
                          f"{departure_name} -> {arrival_name}: keys={list(data.keys()) if isinstance(data, dict) else type(data)}", flush=True)
                    return None

                # Fallback: look for JSON inside plain text content blocks
                for block in result.content:
                    text = getattr(block, "text", None)
                    if text:
                        try:
                            parsed = json.loads(text)
                            return parsed if isinstance(parsed, list) else parsed.get("trips", [])
                        except Exception:
                            continue

                print(f"[Ferryhopper MCP] no usable content for {departure_name} -> {arrival_name}", flush=True)
                return None
    except ImportError:
        print("[Ferryhopper MCP] the 'mcp' package isn't installed — run: pip3 install mcp", flush=True)
        return None
    except Exception as e:
        real_errors = " | ".join(_unwrap_exception_group(e))
        print(f"[Ferryhopper MCP] live fetch failed for {departure_name} -> {arrival_name}: {real_errors}", flush=True)
        return None


async def get_live_schedule_dataframe():
    """Queries Ferryhopper's live MCP server for every route we track, for
    today's real date. Returns a real pandas DataFrame if at least one route
    returned usable data, otherwise None (signals: fall back to the CSV)."""
    import asyncio

    ports_by_id, routes = load_ports()
    today_str = today_in_athens()

    async def fetch_one(route):
        # Strip parenthetical qualifiers like "(New Port)" or "(Parikia)" —
        # Ferryhopper's location lookup appears to want the plain island
        # name, confirmed by some routes erroring with the fuller name
        # while plain names (Piraeus, Naxos) succeeded.
        raw_from = ports_by_id[route["from"]]["name"]
        raw_to = ports_by_id[route["to"]]["name"]
        from_name = re.sub(r"\s*\(.*?\)", "", raw_from).strip()
        to_name = re.sub(r"\s*\(.*?\)", "", raw_to).strip()
        try:
            raw_trips = await asyncio.wait_for(
                _fetch_live_trips(from_name, to_name, today_str), timeout=10
            )
        except asyncio.TimeoutError:
            print(f"[Ferryhopper MCP] timed out for {from_name} -> {to_name}", flush=True)
            return []
        if not raw_trips:
            print(f"[Ferryhopper MCP] {from_name} -> {to_name}: nothing usable came back (see above for why)", flush=True)
            return []
        rows = []
        for raw in raw_trips:
            normalized = _normalize_trip_record(raw, route["id"], today_str)
            if normalized:
                rows.append(normalized)
        return rows

    # Sequential, not concurrent — firing all 9 requests at once appeared to
    # trigger "Server returned an error response" on most of them, while a
    # single request succeeded. One at a time, with a small pause, is much
    # gentler on their server.
    all_rows = []
    for route in routes:
        route_rows = await fetch_one(route)
        all_rows.extend(route_rows)
        await asyncio.sleep(0.3)

    if not all_rows:
        print("[Ferryhopper MCP] no live data came back for any route — falling back to your saved CSV", flush=True)
        return None

    routes_with_data = set(row["route_id"] for row in all_rows)
    print(f"[Ferryhopper MCP] got {len(all_rows)} real live sailings across "
          f"{len(routes_with_data)} routes", flush=True)
    return pd.DataFrame(all_rows)


async def get_current_schedule():
    """Returns (dataframe, source) where source is 'live' or 'csv_fallback'.
    Caches the live result for 20 minutes so we're not hitting Ferryhopper's
    server on every single page load."""
    cache_key = "schedule"
    now = time.time()
    if cache_key in _LIVE_SCHEDULE_CACHE:
        ts, df, source = _LIVE_SCHEDULE_CACHE[cache_key]
        if now - ts < _LIVE_SCHEDULE_CACHE_TTL_SECONDS:
            return df, source

    live_df = await get_live_schedule_dataframe()
    if live_df is not None and len(live_df) > 0:
        _LIVE_SCHEDULE_CACHE[cache_key] = (now, live_df, "live")
        return live_df, "live"

    csv_df = load_schedule()
    _LIVE_SCHEDULE_CACHE[cache_key] = (now, csv_df, "csv_fallback")
    return csv_df, "csv_fallback"


def parse_hour_minute(time_str):
    """Robustly parse a time string like '7:25', '07:25', '07:25:00',
    or '03:45 (+1 day)' into (hour, minute). Any extra text (like a
    '+1 day' note) is ignored — we detect overnight crossings ourselves."""
    s = str(time_str).strip()
    match = re.match(r"^(\d{1,2}):(\d{2})", s)
    if match:
        return int(match.group(1)), int(match.group(2))
    parts = s.split(":")
    hour = int(parts[0])
    minute = int(parts[1]) if len(parts) > 1 else 0
    return hour, minute


def compute_all(schedule=None, data_source="manual"):
    """Returns a dict keyed by route_id with hourly crowd/CO2 breakdown + a best-time score."""
    ports_by_id, routes = load_ports()
    schedule = schedule if schedule is not None else load_schedule()

    schedule["passengers"] = (
        schedule["vessel_type"]
        .str.strip()
        .str.lower()
        .map(VESSEL_TYPE_ALIASES)
        .fillna("conventional")  # unrecognized labels default to conventional (the safer/larger estimate)
        .map(CAPACITY)
    )
    schedule["vessel_type_norm"] = (
        schedule["vessel_type"].str.strip().str.lower().map(VESSEL_TYPE_ALIASES).fillna("conventional")
    )
    # Make route matching case/whitespace-insensitive, since spreadsheet edits
    # often introduce inconsistent capitalization (e.g. "Mykonos_santorini")
    schedule["route_id"] = schedule["route_id"].str.strip().str.lower()
    schedule["hour"] = schedule["departure_time"].apply(lambda t: parse_hour_minute(t)[0])

    # Duration in minutes, only computed if an arrival_time column exists
    if "arrival_time" in schedule.columns:
        def duration_minutes(row):
            dep_h, dep_m = parse_hour_minute(row["departure_time"])
            arr_h, arr_m = parse_hour_minute(row["arrival_time"])
            dep_total = dep_h * 60 + dep_m
            arr_total = arr_h * 60 + arr_m
            if arr_total < dep_total:  # sailing crosses midnight
                arr_total += 24 * 60
            return arr_total - dep_total

        schedule["duration_min"] = schedule.apply(duration_minutes, axis=1)
    else:
        schedule["duration_min"] = None

    results = {}
    for route in routes:
        rid = route["id"]
        dist_km = route_distance_km(route, ports_by_id)

        route_sched = schedule[schedule["route_id"] == rid].copy()
        route_sched["co2_kg"] = (
            dist_km
            * route_sched["vessel_type_norm"].map(EMISSION_FACTORS)
            * route_sched["passengers"]
            / 1000
        )

        hourly_crowd = (
            route_sched.groupby("hour")["passengers"].sum().reindex(range(24), fill_value=0)
        )

        total_co2 = round(route_sched["co2_kg"].sum(), 1)
        total_passengers = int(route_sched["passengers"].sum())
        sailings = len(route_sched)

        # Average trip duration, if we have arrival times
        if route_sched["duration_min"].notna().any():
            avg_duration_min = round(route_sched["duration_min"].mean(), 0)
        else:
            avg_duration_min = None

        # Real per-sailing details — same as the on-demand search feature,
        # so the "Popular routes" cards are just as informative, not a
        # second-class aggregate-only experience.
        sailings_detail = []
        if route_sched["duration_min"].notna().any():
            for _, srow in route_sched.dropna(subset=["duration_min"]).sort_values("departure_time").iterrows():
                via_raw = srow.get("via", "")
                via_list = [v for v in str(via_raw).split(";") if v and v.lower() != "nan"] if pd.notna(via_raw) else []
                sailings_detail.append({
                    "operator": srow.get("operator", "Unknown"),
                    "vessel_type": srow.get("vessel_type_norm", ""),
                    "departure_time": srow["departure_time"],
                    "arrival_time": srow.get("arrival_time"),
                    "duration_min": int(srow["duration_min"]),
                    "co2_kg_this_sailing": round(srow["co2_kg"], 1),
                    "via": via_list,
                })

        # crowd score 0-10 (relative to a "busy day" ceiling of ~6000 passengers/route)
        crowd_score = min(10, round((total_passengers / 6000) * 10, 1))

        # Most common operator on this route, so the frontend can show the
        # right ferry icon (SeaJets, Blue Star, etc.) instead of a generic dot
        if sailings > 0 and "operator" in route_sched.columns:
            dominant_operator = route_sched["operator"].mode().iloc[0]
        else:
            dominant_operator = None

        # Real weather at the destination port, blended with crowd density.
        # crowd contributes 60% (inverted — busier = worse), weather 40%.
        dest_port = ports_by_id[route["to"]]
        weather_details = get_weather_details(dest_port["lat"], dest_port["lon"])
        weather_score = weather_details["score"]
        weather_forecast = get_weather_forecast(dest_port["lat"], dest_port["lon"], days=3)
        best_time_score = round(
            max(1, min(10, (10 - crowd_score * 0.8) * 0.6 + weather_score * 0.4)), 1
        )

        # CO2 comparison vs. flying the same real distance. Flight factor is
        # a standard published short-haul figure (~250g CO2/passenger-km),
        # applied to the SAME real distance as the ferry for a fair compare.
        FLIGHT_EMISSION_FACTOR_G_PER_KM = 250
        flight_co2_kg_est = round(dist_km * FLIGHT_EMISSION_FACTOR_G_PER_KM * total_passengers / 1000, 1)
        co2_savings_pct = (
            round((1 - total_co2 / flight_co2_kg_est) * 100, 1)
            if flight_co2_kg_est > 0 else None
        )

        # One real sample sailing for this route (earliest departure in the
        # data), used to power the "add to calendar" feature with genuine
        # times rather than an invented placeholder.
        if sailings > 0:
            sample_row = route_sched.sort_values("departure_time").iloc[0]
            sample_sailing = {
                "operator": sample_row.get("operator"),
                "departure_time": sample_row.get("departure_time"),
                "arrival_time": sample_row.get("arrival_time") if pd.notna(sample_row.get("arrival_time")) else None,
                "date": sample_row.get("date") if "date" in sample_row and pd.notna(sample_row.get("date")) else None,
            }
        else:
            sample_sailing = None

        # Real Ferryhopper route pages — only using URLs actually confirmed
        # live during research (their slug pattern isn't fully consistent,
        # e.g. some routes need "-to-" or a reordered prefix). Anything not
        # personally verified falls back to Ferryhopper's general route
        # browser rather than guessing a URL that might 404.
        BOOKING_URL_OVERRIDES = {
            "piraeus_mykonos": "https://www.ferryhopper.com/en/ferry-routes/direct/piraeus-athens-to-mykonos",
            "piraeus_santorini": "https://www.ferryhopper.com/en/ferry-routes/direct/athens-piraeus-to-santorini",
            "piraeus_naxos": "https://www.ferryhopper.com/en/ferry-routes/direct/piraeus-naxos",
            "naxos_santorini": "https://www.ferryhopper.com/en/ferry-routes/direct/santorini-to-naxos",
            "mykonos_naxos": "https://www.ferryhopper.com/en/ferry-routes/direct/mykonos-to-naxos",
        }
        booking_url = BOOKING_URL_OVERRIDES.get(rid, "https://www.ferryhopper.com/en/ferry-routes")

        results[rid] = {
            "route_id": rid,
            "from": ports_by_id[route["from"]]["name"],
            "to": ports_by_id[route["to"]]["name"],
            "from_id": route["from"],
            "to_id": route["to"],
            "distance_km": round(dist_km, 1),
            "avg_duration_min": avg_duration_min,
            "sailings": sailings,
            "sailings_detail": sailings_detail,
            "total_passengers_est": total_passengers,
            "total_co2_kg_est": total_co2,
            "flight_co2_kg_est": flight_co2_kg_est,
            "co2_savings_pct": co2_savings_pct,
            "crowd_score": crowd_score,
            "weather_score": weather_score,
            "weather_temp_c": weather_details["temp_c"],
            "weather_precip_pct": weather_details["precip_chance_pct"],
            "weather_forecast": weather_forecast,
            "best_time_score": best_time_score,
            "best_time_label": best_time_label(best_time_score),
            "dominant_operator": dominant_operator,
            "sample_sailing": sample_sailing,
            "booking_url": booking_url,
            "hourly_crowd": hourly_crowd.to_dict(),
            "data_source": data_source,
        }

    return results


def get_live_ferries(now=None, schedule=None):
    """Checks the REAL schedule against the current time-of-day and returns
    only sailings that are genuinely in progress right now — with how far
    along (0-1) they are, based on real elapsed time between their real
    departure and arrival. No sailing in progress on a route = nothing
    returned for it. This replaces any made-up animation with something
    tied to your actual timetable."""
    if now is None:
        now = datetime.now(ATHENS_TZ) if ATHENS_TZ else datetime.now()

    now_minutes = now.hour * 60 + now.minute

    schedule = schedule if schedule is not None else load_schedule()
    schedule = schedule.copy()
    schedule["route_id"] = schedule["route_id"].str.strip().str.lower()

    active = []
    for _, row in schedule.iterrows():
        dep_h, dep_m = parse_hour_minute(row["departure_time"])
        dep_total = dep_h * 60 + dep_m

        if "arrival_time" not in row or pd.isna(row.get("arrival_time")):
            continue
        arr_h, arr_m = parse_hour_minute(row["arrival_time"])
        arr_total = arr_h * 60 + arr_m
        if arr_total < dep_total:
            arr_total += 24 * 60

        # Compare against "now", also checking the wrapped (+1 day) window
        # so a sailing that started yesterday but is still going is caught.
        now_candidates = [now_minutes, now_minutes + 24 * 60]
        for now_t in now_candidates:
            if dep_total <= now_t <= arr_total:
                progress = (now_t - dep_total) / max(1, (arr_total - dep_total))
                via_raw = row.get("via", "")
                via_list = (
                    [v.strip() for v in str(via_raw).split(";") if v.strip()]
                    if pd.notna(via_raw) and str(via_raw).strip()
                    else []
                )
                active.append({
                    "route_id": row["route_id"],
                    "operator": row.get("operator", None),
                    "vessel_type": row.get("vessel_type", None),
                    "departure_time": row["departure_time"],
                    "arrival_time": row["arrival_time"],
                    "via": via_list,
                    "progress": round(progress, 4),
                })
                break

    return active


async def search_route_live(from_id, to_id):
    """On-demand lookup for ANY two ports (not just the 9 precomputed
    routes) — queries Ferryhopper live for exactly this pair, right now.
    Returns a dict with real stats if sailings were found, or a 'found':
    False result if not (still a valid, honest response, not an error)."""
    ports_by_id, _ = load_ports()
    if from_id not in ports_by_id or to_id not in ports_by_id:
        return {"found": False, "error": "unknown port"}

    from_port = ports_by_id[from_id]
    to_port = ports_by_id[to_id]
    from_name = re.sub(r"\s*\(.*?\)", "", from_port["name"]).strip()
    to_name = re.sub(r"\s*\(.*?\)", "", to_port["name"]).strip()
    today_str = today_in_athens()
    route_id = f"{from_id}_{to_id}"

    raw_trips = await _fetch_live_trips(from_name, to_name, today_str)
    if not raw_trips:
        return {
            "found": False,
            "from": from_port["name"],
            "to": to_port["name"],
            "from_id": from_id,
            "to_id": to_id,
        }

    rows = [r for r in (_normalize_trip_record(t, route_id, today_str) for t in raw_trips) if r]
    if not rows:
        return {
            "found": False,
            "from": from_port["name"],
            "to": to_port["name"],
            "from_id": from_id,
            "to_id": to_id,
        }

    dist_km = haversine_km(from_port["lat"], from_port["lon"], to_port["lat"], to_port["lon"])
    schedule_df = pd.DataFrame(rows)
    schedule_df["passengers"] = (
        schedule_df["vessel_type"].str.strip().str.lower().map(VESSEL_TYPE_ALIASES).fillna("conventional").map(CAPACITY)
    )
    schedule_df["vessel_type_norm"] = (
        schedule_df["vessel_type"].str.strip().str.lower().map(VESSEL_TYPE_ALIASES).fillna("conventional")
    )
    schedule_df["co2_kg"] = (
        dist_km * schedule_df["vessel_type_norm"].map(EMISSION_FACTORS) * schedule_df["passengers"] / 1000
    )

    def duration_minutes(row):
        dep_h, dep_m = parse_hour_minute(row["departure_time"])
        arr_h, arr_m = parse_hour_minute(row["arrival_time"])
        dep_total, arr_total = dep_h * 60 + dep_m, arr_h * 60 + arr_m
        if arr_total < dep_total:
            arr_total += 24 * 60
        return arr_total - dep_total

    schedule_df["duration_min"] = schedule_df.apply(duration_minutes, axis=1)

    total_passengers = int(schedule_df["passengers"].sum())
    total_co2 = round(schedule_df["co2_kg"].sum(), 1)
    avg_duration_min = round(schedule_df["duration_min"].mean(), 0)
    crowd_score = min(10, round((total_passengers / 6000) * 10, 1))
    weather_details = get_weather_details(to_port["lat"], to_port["lon"])
    weather_score = weather_details["score"]
    weather_forecast = get_weather_forecast(to_port["lat"], to_port["lon"], days=3)
    best_time_score = round(max(1, min(10, (10 - crowd_score * 0.8) * 0.6 + weather_score * 0.4)), 1)
    FLIGHT_EMISSION_FACTOR_G_PER_KM = 250
    flight_co2 = round(dist_km * FLIGHT_EMISSION_FACTOR_G_PER_KM * total_passengers / 1000, 1)
    co2_savings_pct = round((1 - total_co2 / flight_co2) * 100, 1) if flight_co2 > 0 else None
    dominant_operator = schedule_df["operator"].mode().iloc[0] if len(schedule_df) else None

    # Real per-sailing details — this is what a traveler actually needs:
    # exact departure/arrival times, per-sailing CO2 (not a confusing total
    # across every sailing's full capacity), and any real stopover islands.
    sailings_detail = []
    for _, row in schedule_df.sort_values("departure_time").iterrows():
        via_raw = row.get("via", "")
        via_list = [v for v in str(via_raw).split(";") if v and v.lower() != "nan"] if pd.notna(via_raw) else []
        sailings_detail.append({
            "operator": row["operator"],
            "vessel_type": row["vessel_type_norm"],
            "departure_time": row["departure_time"],
            "arrival_time": row["arrival_time"],
            "duration_min": int(row["duration_min"]),
            "co2_kg_this_sailing": round(row["co2_kg"], 1),
            "via": via_list,
        })

    return {
        "found": True,
        "route_id": route_id,
        "from": from_port["name"],
        "to": to_port["name"],
        "from_id": from_id,
        "to_id": to_id,
        "distance_km": round(dist_km, 1),
        "avg_duration_min": avg_duration_min,
        "sailings": len(rows),
        "sailings_detail": sailings_detail,
        "total_passengers_est": total_passengers,
        "total_co2_kg_est": total_co2,
        "flight_co2_kg_est": flight_co2,
        "co2_savings_pct": co2_savings_pct,
        "crowd_score": crowd_score,
        "weather_score": weather_score,
        "weather_temp_c": weather_details["temp_c"],
        "weather_precip_pct": weather_details["precip_chance_pct"],
        "weather_forecast": weather_forecast,
        "best_time_score": best_time_score,
        "best_time_label": best_time_label(best_time_score),
        "dominant_operator": dominant_operator,
        "data_source": "live",
    }


def get_next_sailings(now=None, schedule=None):
    """For every route, finds the REAL next upcoming departure from the
    actual schedule (today, or wrapping to the next day if everything today
    has already left), and how many minutes until it departs. Used for a
    live countdown — built from real timetable data, not invented."""
    if now is None:
        now = datetime.now(ATHENS_TZ) if ATHENS_TZ else datetime.now()
    now_minutes = now.hour * 60 + now.minute

    schedule = schedule if schedule is not None else load_schedule()
    schedule = schedule.copy()
    schedule["route_id"] = schedule["route_id"].str.strip().str.lower()

    next_by_route = {}
    for _, row in schedule.iterrows():
        rid = row["route_id"]
        dep_h, dep_m = parse_hour_minute(row["departure_time"])
        dep_total = dep_h * 60 + dep_m

        minutes_until = dep_total - now_minutes
        if minutes_until < 0:
            minutes_until += 24 * 60  # already departed today — next one is tomorrow same time

        candidate = {
            "route_id": rid,
            "operator": row.get("operator", None),
            "departure_time": row["departure_time"],
            "minutes_until": int(minutes_until),
        }
        if rid not in next_by_route or minutes_until < next_by_route[rid]["minutes_until"]:
            next_by_route[rid] = candidate

    return next_by_route


# Real, publicly documented facts about each port (gates, transport links)
# — not invented. Only included where we have a specific, checkable fact;
# ports without an entry here just won't show extra amenities info.
PORT_AMENITIES = {
    "piraeus": {
        "gates": "Gates E4, E6, E7, E9, E10 (Cyclades ferries)",
        "transport": "Athens Metro Lines 1 & 3 serve the port directly",
    },
    "naxos": {
        "transport": "Short walk into Chora (the island's capital) from the port",
    },
    "santorini": {
        "note": "Athinios Port — 8km from Fira, 20km from Oia",
    },
}


if __name__ == "__main__":
    import pprint

    pprint.pprint(compute_all())
