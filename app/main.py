from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from compute import (
    compute_all,
    get_current_schedule,
    get_live_ferries,
    get_next_sailings,
    search_route_live,
    PORT_AMENITIES,
    load_ports,
)

app = FastAPI(title="Greece Ferry Traffic Simulator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/ports")
def get_ports():
    ports_by_id, routes = load_ports()
    return {"ports": list(ports_by_id.values()), "routes": routes}


@app.get("/api/routes")
async def get_routes():
    schedule, source = await get_current_schedule()
    return compute_all(schedule=schedule, data_source=source)


@app.get("/api/live_ferries")
async def get_live():
    """Sailings genuinely in progress right now, based on the real schedule
    (live from Ferryhopper when available, else your saved CSV) and the
    real current time — not a simulated loop."""
    schedule, _ = await get_current_schedule()
    return get_live_ferries(schedule=schedule)


@app.get("/api/next_sailings")
async def next_sailings():
    """Real next departure per route, from the real schedule."""
    schedule, _ = await get_current_schedule()
    return get_next_sailings(schedule=schedule)


@app.get("/api/schedule_status")
async def schedule_status():
    """Tells you whether the app is currently running on LIVE Ferryhopper
    data or has fallen back to your saved CSV — check this first if numbers
    look stale. Check your terminal for detailed [Ferryhopper MCP] logs."""
    _, source = await get_current_schedule()
    return {"source": source}


@app.get("/api/search_route")
async def search_route(from_id: str, to_id: str):
    """On-demand real ferry lookup for ANY two ports you pick, not just the
    9 precomputed 'popular' routes — queries Ferryhopper live right now."""
    return await search_route_live(from_id, to_id)


@app.get("/api/port_amenities")
def port_amenities():
    """Real, documented facts about each port (gates, transport links)."""
    return PORT_AMENITIES


@app.get("/health")
def health():
    return {"status": "ok"}


# Serve the frontend (index.html, etc.) from the same server as the API,
# so the browser never has to make a cross-origin request. This mount is
# registered last so it acts as a catch-all for anything not matched above.
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
