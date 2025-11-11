import os
import logging
import time
import tomllib
from typing import Any, Dict, Mapping, Optional
from pathlib import Path
from base64 import b64encode

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy import create_engine, select, MetaData, Table
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode, DataReturnMode, ColumnsAutoSizeMode

# Page configuration
st.set_page_config(page_title="TAILORED_OFFERS", layout="wide")

# Setup logging
logger = logging.getLogger(__name__)

# Constants
LOCAL_SECRETS_FILENAME = "local_secrets.toml"
LOCAL_SECRETS_PATH = Path(__file__).with_name(LOCAL_SECRETS_FILENAME)
DATABASE_URL_ENV_KEYS = ["DATABASE_URL", "DB_URL", "POSTGRES_URL", "POSTGRESQL_URL", "NEON_DATABASE_URL"]


# Utility function for color contrast
def get_contrast_color(hex_color: str) -> str:
    """Determine if text should be black or white based on background color luminance."""
    # Remove # if present
    hex_color = hex_color.lstrip('#')
    
    # Convert to RGB
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
    except (ValueError, IndexError):
        return '#ffffff'  # Default to white for invalid colors
    
    # Calculate relative luminance using WCAG formula
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    
    # Return black for light backgrounds, white for dark backgrounds
    return '#000000' if luminance > 0.5 else '#ffffff'


# Database helper functions
def _extract_database_url_from_mapping(mapping: Mapping[str, Any]) -> Optional[str]:
    for key in DATABASE_URL_ENV_KEYS:
        if key in mapping:
            return str(mapping[key]).strip()
    return None


def _load_local_database_url() -> Optional[str]:
    if not LOCAL_SECRETS_PATH.exists():
        return None
    try:
        with LOCAL_SECRETS_PATH.open("rb") as handle:
            contents = tomllib.load(handle)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse {LOCAL_SECRETS_FILENAME}: {exc}") from exc
    
    if not isinstance(contents, Mapping):
        raise RuntimeError(f"{LOCAL_SECRETS_FILENAME} must contain TOML key/value pairs.")
    
    url = _extract_database_url_from_mapping(contents)
    if not url:
        raise RuntimeError(f"{LOCAL_SECRETS_FILENAME} exists but is missing a database URL.")
    return url


def _normalize_database_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        raise ValueError("Database URL is empty.")
    lowered = raw.lower()
    mappings = {
        "postgres://": "postgresql+psycopg://",
        "postgresql://": "postgresql+psycopg://",
        "postgresql+asyncpg://": "postgresql+psycopg://",
    }
    for legacy, target in mappings.items():
        if lowered.startswith(legacy):
            raw = target + raw[len(legacy):]
            break
    if "sslmode=" not in raw.lower():
        raw = f"{raw}{'&' if '?' in raw else '?'}sslmode=require"
    return raw


def _get_database_url() -> str:
    cache_key = "_database_url"
    cached = st.session_state.get(cache_key)
    if cached:
        return cached
    
    local_url = _load_local_database_url()
    if local_url:
        normalized = _normalize_database_url(local_url)
        st.session_state[cache_key] = normalized
        return normalized
    
    for key in DATABASE_URL_ENV_KEYS:
        env_val = os.getenv(key)
        if env_val:
            normalized = _normalize_database_url(env_val)
            st.session_state[cache_key] = normalized
            return normalized
    
    raise RuntimeError("Database URL not configured.")


@st.cache_resource(show_spinner=False)
def get_db_engine() -> Engine:
    url = _get_database_url()
    return create_engine(
        url, 
        poolclass=NullPool, 
        future=True,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 10},
        isolation_level="AUTOCOMMIT"
    )


def fetch_client_tags_dataframe() -> pd.DataFrame:
    """Fetch all client tags with related tag_config information."""
    engine = get_db_engine()
    
    query = """
        SELECT 
            ct.client_id,
            ct.ont_id,
            ct.tag_id,
            ct.assigned_at,
            ct.assigned_by,
            ct.reason,
            tc.system_name,
            tc.display_name,
            tc.tag_type,
            tc.color,
            tc.description,
            tc.is_active
        FROM client_tag ct
        LEFT JOIN tag_config tc ON ct.tag_id = tc.id
        ORDER BY ct.client_id
    """
    
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            df = pd.DataFrame(columns=[
                "client_id", "ont_id", "tag_id", "assigned_at", "assigned_by", 
                "reason", "system_name", "display_name", "tag_type", "color", 
                "description", "is_active"
            ])
        
        # Format datetime columns
        if "assigned_at" in df.columns:
            df["assigned_at"] = pd.to_datetime(df["assigned_at"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        
        # Fill NaN values
        for col in df.columns:
            if col not in ["client_id", "tag_id", "is_active"]:
                df[col] = df[col].fillna("")
        
        if "is_active" in df.columns:
            df["is_active"] = df["is_active"].fillna(True)
        
        return df
    except Exception as exc:
        logger.exception("Failed to fetch client tags: %s", exc)
        st.error(f"Database error: {exc}")
        return pd.DataFrame()


def fetch_auto_tag_statistics() -> pd.DataFrame:
    """Fetch auto tag statistics with tag information for chart."""
    engine = get_db_engine()
    
    query = """
        SELECT 
            ats.run_finished_at::date as date,
            ats.assigned_count,
            tc.display_name,
            tc.color
        FROM auto_tag_statistic ats
        LEFT JOIN tag_config tc ON ats.tag_id = tc.id
        WHERE tc.tag_type = 'A'
        ORDER BY ats.run_finished_at ASC
    """
    
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            df = pd.DataFrame(columns=["date", "assigned_count", "display_name", "color"])
        return df
    except Exception as exc:
        logger.exception("Failed to fetch auto tag statistics: %s", exc)
        st.error(f"Database error: {exc}")
        return pd.DataFrame()


def fetch_manual_tag_counts() -> pd.DataFrame:
    """Fetch manual tag counts grouped by tag."""
    engine = get_db_engine()
    
    query = """
        SELECT 
            tc.display_name,
            tc.color,
            COUNT(ct.tag_id) as tag_count
        FROM client_tag ct
        LEFT JOIN tag_config tc ON ct.tag_id = tc.id
        WHERE tc.tag_type = 'M'
        GROUP BY tc.display_name, tc.color
        ORDER BY tag_count ASC
    """
    
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            df = pd.DataFrame(columns=["display_name", "color", "tag_count"])
        return df
    except Exception as exc:
        logger.exception("Failed to fetch manual tag counts: %s", exc)
        st.error(f"Database error: {exc}")
        return pd.DataFrame()


def fetch_tag_config(tag_type: str = None) -> pd.DataFrame:
    """Fetch tag configuration from tag_config table."""
    engine = get_db_engine()
    
    if tag_type:
        query = """
            SELECT 
                id,
                system_name,
                display_name,
                tag_type,
                color,
                description,
                is_active,
                created_at,
                updated_at
            FROM tag_config
            WHERE tag_type = %(tag_type)s
            ORDER BY id
        """
        try:
            df = pd.read_sql(query, engine, params={"tag_type": tag_type})
        except Exception as exc:
            logger.exception("Failed to fetch tag config: %s", exc)
            st.error(f"Database error: {exc}")
            return pd.DataFrame()
    else:
        query = """
            SELECT 
                id,
                system_name,
                display_name,
                tag_type,
                color,
                description,
                is_active,
                created_at,
                updated_at
            FROM tag_config
            ORDER BY tag_type, id
        """
        try:
            df = pd.read_sql(query, engine)
        except Exception as exc:
            logger.exception("Failed to fetch tag config: %s", exc)
            st.error(f"Database error: {exc}")
            return pd.DataFrame()
    
    return df


def update_tag_config(tag_id: int, display_name: str = None, color: str = None, is_active: bool = None, description: str = None) -> bool:
    """Update tag configuration in the database."""
    engine = get_db_engine()
    
    updates = []
    params = {"tag_id": tag_id}
    
    if display_name is not None:
        updates.append("display_name = %(display_name)s")
        params["display_name"] = display_name
    
    if color is not None:
        updates.append("color = %(color)s")
        params["color"] = color
    
    if is_active is not None:
        updates.append("is_active = %(is_active)s")
        params["is_active"] = is_active
    
    if description is not None:
        updates.append("description = %(description)s")
        params["description"] = description
    
    if not updates:
        return False
    
    updates.append("updated_at = CURRENT_TIMESTAMP")
    
    query = f"""
        UPDATE tag_config
        SET {', '.join(updates)}
        WHERE id = %(tag_id)s
    """
    
    try:
        with engine.begin() as conn:
            conn.exec_driver_sql(query, params)
        return True
    except Exception as exc:
        logger.exception("Failed to update tag config: %s", exc)
        st.error(f"Database error: {exc}")
        return False


def create_manual_tag(system_name: str, display_name: str, color: str, description: str = "", is_active: bool = True) -> bool:
    """Create a new manual tag in the database."""
    engine = get_db_engine()
    
    query = """
        INSERT INTO tag_config (system_name, display_name, tag_type, color, description, is_active, created_at, updated_at)
        VALUES (%(system_name)s, %(display_name)s, 'M', %(color)s, %(description)s, %(is_active)s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
    """
    
    params = {
        "system_name": system_name,
        "display_name": display_name,
        "color": color,
        "description": description,
        "is_active": is_active
    }
    
    try:
        with engine.begin() as conn:
            conn.exec_driver_sql(query, params)
        return True
    except Exception as exc:
        logger.exception("Failed to create manual tag: %s", exc)
        st.error(f"Database error: {exc}")
        return False


def fetch_client_tags(client_id: int) -> pd.DataFrame:
    """Fetch tags assigned to a specific client."""
    engine = get_db_engine()
    
    query = """
        SELECT 
            ct.tag_id,
            ct.assigned_at,
            ct.assigned_by,
            ct.reason,
            tc.system_name,
            tc.display_name,
            tc.tag_type,
            tc.color,
            tc.description
        FROM client_tag ct
        LEFT JOIN tag_config tc ON ct.tag_id = tc.id
        WHERE ct.client_id = %(client_id)s
        ORDER BY ct.assigned_at DESC
    """
    
    params = {"client_id": client_id}
    
    try:
        df = pd.read_sql(query, engine, params=params)
        return df
    except Exception as exc:
        logger.exception("Failed to fetch client tags: %s", exc)
        st.error(f"Database error: {exc}")
        return pd.DataFrame()


def add_client_tag(client_id: int, tag_id: int, ont_id: str = None, assigned_by: str = "System", reason: str = "") -> bool:
    """Add a tag to a client."""
    engine = get_db_engine()
    
    query = """
        INSERT INTO client_tag (client_id, ont_id, tag_id, assigned_at, assigned_by, reason)
        VALUES (%(client_id)s, %(ont_id)s, %(tag_id)s, CURRENT_TIMESTAMP, %(assigned_by)s, %(reason)s)
        ON CONFLICT (client_id, tag_id) DO NOTHING
    """
    
    params = {
        "client_id": client_id,
        "ont_id": ont_id,
        "tag_id": tag_id,
        "assigned_by": assigned_by,
        "reason": reason
    }
    
    try:
        with engine.begin() as conn:
            conn.exec_driver_sql(query, params)
        return True
    except Exception as exc:
        logger.exception("Failed to add client tag: %s", exc)
        st.error(f"Database error: {exc}")
        return False


def remove_client_tag(client_id: int, tag_id: int) -> bool:
    """Remove a tag from a client."""
    engine = get_db_engine()
    
    query = """
        DELETE FROM client_tag
        WHERE client_id = %(client_id)s AND tag_id = %(tag_id)s
    """
    
    params = {
        "client_id": client_id,
        "tag_id": tag_id
    }
    
    try:
        with engine.begin() as conn:
            conn.exec_driver_sql(query, params)
        return True
    except Exception as exc:
        logger.exception("Failed to remove client tag: %s", exc)
        st.error(f"Database error: {exc}")
        return False


def fetch_clients() -> pd.DataFrame:
    """Fetch all clients from the client table."""
    engine = get_db_engine()
    
    query = """
        SELECT 
            client_id,
            ont_id,
            name,
            phone,
            service_id,
            city,
            area,
            address,
            type,
            sip
        FROM client
        ORDER BY client_id
    """
    
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            df = pd.DataFrame(columns=[
                "client_id", "ont_id", "name", "phone", "service_id",
                "city", "area", "address", "type", "sip"
            ])
        
        # Fill NaN values
        for col in df.columns:
            if col not in ["client_id"]:
                df[col] = df[col].fillna("")
        
        return df
    except Exception as exc:
        logger.exception("Failed to fetch clients: %s", exc)
        st.error(f"Database error: {exc}")
        return pd.DataFrame()


# Load CSS
with open("tailored_offers_theme.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize session state
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Client"

# Honor query param for tab switching
if "tab" in st.query_params:
    requested_tab = st.query_params["tab"]
    if requested_tab in ["Home", "Requests", "Card", "Client", "CPE", "IVR", "Settings", "Dahi Nemutlu", "Exit"]:
        st.session_state.active_tab = requested_tab


def _get_brand_logo_data_uri() -> Optional[str]:
    logo_path = Path(__file__).with_name("assets").joinpath("fibercare.png")
    try:
        with logo_path.open("rb") as fh:
            encoded = b64encode(fh.read()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except Exception:
        return None


# Top-level tab icons
TAB_ICONS = {
    "Home": "home",
    "Requests": "article",
    "Card": "credit_score",
    "Client": "group",
    "CPE": "router",
    "IVR": "support_agent",
    "Settings": "settings",
    "Dahi Nemutlu": "notifications",
    "Exit": "exit_to_app",
}
TAB_NAMES = list(TAB_ICONS.keys())

# Build top bar
html = ['<div class="topbar">']

# Brand/Logo
logo_uri = _get_brand_logo_data_uri()
logo_markup = f'<img src="{logo_uri}" alt="FiberCare logo" />' if logo_uri else ""
html.append(f'<div class="brand">{logo_markup}<span class="brandStack">FiberCare</span></div>')

# Build tab items
tab_items = []
for name in TAB_NAMES:
    icon = f'<span class="material-icons">{TAB_ICONS[name]}</span>'
    if name == "Client":
        active_cls = " call-center-active" if st.session_state.active_tab == name else ""
        tab_items.append(
            f'<a href="?tab=Client" target="_self" class="tab{active_cls}">{icon} {name}</a>'
        )
    elif name == "Dahi Nemutlu":
        tab_items.append(
            '<span class="tab notifications-tab">'
            f'<span class="notifications-trigger" aria-label="Notifications">{icon}</span>'
            f'<span class="notifications-name">{name}</span>'
            '</span>'
        )
    elif name == "Exit":
        tab_items.append(f'<span class="tab-disabled" title="Exit">{icon}</span>')
    else:
        active_cls = " active" if st.session_state.active_tab == name else ""
        tab_items.append(f'<span class="tab-disabled{active_cls}">{icon} {name}</span>')

# Inline tabs row
html.append('<div class="tabs" id="topbar-tabs">')
html.extend(tab_items)
html.append('</div>')

# Burger toggle (CSS only) and overlay drawer menu
html.append('<input type="checkbox" id="topbar-burger-toggle" class="burger-toggle" />')
html.append('<label for="topbar-burger-toggle" class="burger" id="topbar-burger" aria-label="Menu"><span class="material-icons">menu</span></label>')
html.append('<div class="hamburger-overlay" id="topbar-overlay">')
html.append('<label for="topbar-burger-toggle" class="overlay-backdrop"></label>')
html.append('<div class="hamburger-drawer">')
html.append('<div class="menu-header"><div class="title">Menu</div><label for="topbar-burger-toggle" class="close-btn" aria-label="Close"><span class="material-icons">close</span></label></div>')
html.append('<div class="menu-items">')
html.extend(tab_items)
html.append('</div></div></div>')

# Close topbar
html.append('</div>')

# Render topbar
st.markdown("".join(html), unsafe_allow_html=True)

# Main content area
if st.session_state.active_tab == "Home":
    st.title("Tailored Offers")
    st.write("This is a prototype application.")
elif st.session_state.active_tab == "Client":
    # Check if a specific client is selected via query params
    selected_client_id = None
    if "client_id" in st.query_params:
        try:
            selected_client_id = int(st.query_params["client_id"])
        except (ValueError, TypeError):
            selected_client_id = None
    
    if selected_client_id is None:
        # Client page subtabs (Clients / Client Tags)
        CLIENT_PAGE_SUBTAB_ICONS = {"Clients": "group", "Client Tags": "label"}
        CLIENT_PAGE_SUBTAB_NAMES = list(CLIENT_PAGE_SUBTAB_ICONS.keys())
        
        # Initialize active subtab
        if "active_client_page_subtab" not in st.session_state:
            st.session_state.active_client_page_subtab = "Clients"
        
        # Allow switch via query param
        if "client_subtab" in st.query_params:
            q = st.query_params["client_subtab"]
            if q in CLIENT_PAGE_SUBTAB_NAMES:
                st.session_state.active_client_page_subtab = q
        
        # Build subtabs HTML
        base_tab_q = "?tab=Client"
        sub_html = ['<div class="subtabs">']
        for name in CLIENT_PAGE_SUBTAB_NAMES:
            icon = f'<span class="material-icons">{CLIENT_PAGE_SUBTAB_ICONS[name]}</span>'
            cls = " sub-active" if st.session_state.active_client_page_subtab == name else ""
            sub_html.append(
                f'<a href="{base_tab_q}&client_subtab={name.replace(" ", "%20")}" target="_self" class="subtab{cls}">{icon} {name}</a>'
            )
        sub_html.append('</div>')
        st.markdown("".join(sub_html), unsafe_allow_html=True)

        # Content based on active subtab
        if st.session_state.active_client_page_subtab == "Clients":
        
            # Show client list grid with action bar
            st.markdown("""
                <div style="background: white; padding: 16px; border-radius: 8px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); pointer-events: none;">
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <button style="display: flex; align-items: center; gap: 6px; padding: 8px 16px; background: #0066cc; color: white; border: none; border-radius: 4px; cursor: default; font-size: 14px; opacity: 0.7;">
                            <span class="material-icons" style="font-size: 18px;">add_circle</span>
                            Add Client
                        </button>
                        <button style="display: flex; align-items: center; gap: 6px; padding: 8px 16px; background: #0066cc; color: white; border: none; border-radius: 4px; cursor: default; font-size: 14px; opacity: 0.7;">
                            <span class="material-icons" style="font-size: 18px;">change_circle</span>
                            Un Delete Client
                        </button>
                        <input type="text" placeholder="Search" style="flex: 1; padding: 8px 12px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px; max-width: 300px; cursor: default; opacity: 0.7;" />
                        <button style="display: flex; align-items: center; gap: 6px; padding: 8px 16px; background: #0066cc; color: white; border: none; border-radius: 4px; cursor: default; font-size: 14px; opacity: 0.7;">
                            <span class="material-icons" style="font-size: 18px;">filter_alt</span>
                            Filter
                        </button>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
            # Fetch clients
            clients_df = fetch_clients()
        
            if clients_df.empty:
                st.info("No clients found in the database.")
            else:
                # Fetch all client tags to merge with clients
                all_tags_df = fetch_client_tags_dataframe()
            
                # Group tags by client_id to get all tags for each client
                if not all_tags_df.empty:
                    # Create a mapping of client_id to list of tag info (display_name, color)
                    client_tags_map = {}
                    for _, tag_row in all_tags_df.iterrows():
                        cid = tag_row.get('client_id')
                        if cid not in client_tags_map:
                            client_tags_map[cid] = []
                        client_tags_map[cid].append({
                            'display_name': tag_row.get('display_name', ''),
                            'color': tag_row.get('color', '#6b7280')
                        })
                else:
                    client_tags_map = {}
            
                # Prepare grid - add tag column
                clients_view = clients_df.copy()
            
                # Add tags column
                clients_view['tags'] = clients_view['client_id'].apply(
                    lambda cid: client_tags_map.get(cid, [])
                )
            
                # Remove ONT ID column
                if 'ont_id' in clients_view.columns:
                    clients_view = clients_view.drop(columns=['ont_id'])
            
                # Build AG Grid
                gb = GridOptionsBuilder.from_dataframe(clients_view, enableRowGroup=False, enableValue=False, enablePivot=False)
            
                # Configure default column settings - NO FILTER, NO MENU
                gb.configure_default_column(
                    editable=False,
                    resizable=True,
                    filter=False,
                    sortable=True,
                )
            
                # Disable all menu buttons globally
                gb.configure_grid_options(suppressMenuHide=True)
            
                # Disable selection completely - we'll use cell click events
                gb.configure_selection(selection_mode=None, use_checkbox=False)
                
                # Configure cell clicked event to only select on Name column
                gb.configure_grid_options(
                    onCellClicked=JsCode("""
                        function(params) {
                            if (params.column.colId === 'name') {
                                params.api.deselectAll();
                                params.node.setSelected(true);
                            }
                        }
                    """),
                    suppressRowClickSelection=True,
                    rowSelection='single'
                )
            
                # Configure column headers
                column_headers = {
                    "client_id": "#",
                    "name": "Name",
                    "phone": "Phone",
                    "service_id": "Service ID",
                    "city": "City",
                    "area": "Area",
                    "address": "Address",
                    "type": "Type",
                    "sip": "SIP",
                    "tags": "Tags",
                }
            
                for col, header in column_headers.items():
                    if col in clients_view.columns:
                        gb.configure_column(
                            col, 
                            headerName=header,
                            filter=False,
                        )
            
                # Style Name column to look like a link
                if "name" in clients_view.columns:
                    gb.configure_column(
                        "name", 
                        headerName="Name",
                        cellStyle={'color': '#0066cc', 'cursor': 'pointer', 'textDecoration': 'underline'},
                        filter=False,
                    )
            
                # Configure Tags column with badge rendering
                if "tags" in clients_view.columns:
                    tags_renderer = JsCode("""
                        class TagsBadgeRenderer {
                            init(params) {
                                const tags = params.value || [];
                            
                                // Function to determine if text should be black or white based on background
                                function getContrastColor(hexColor) {
                                    const hex = hexColor.replace('#', '');
                                    const r = parseInt(hex.substr(0, 2), 16);
                                    const g = parseInt(hex.substr(2, 2), 16);
                                    const b = parseInt(hex.substr(4, 2), 16);
                                    const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
                                    return luminance > 0.5 ? '#000000' : '#ffffff';
                                }
                            
                                this.eGui = document.createElement('div');
                                this.eGui.style.display = 'flex';
                                this.eGui.style.flexWrap = 'wrap';
                                this.eGui.style.gap = '4px';
                                this.eGui.style.alignItems = 'center';
                                this.eGui.style.height = '100%';
                            
                                if (tags.length === 0) {
                                    this.eGui.textContent = '—';
                                } else {
                                    tags.forEach(tag => {
                                        const badge = document.createElement('span');
                                        const color = tag.color || '#6b7280';
                                        const textColor = getContrastColor(color);
                                    
                                        badge.style.display = 'inline-block';
                                        badge.style.padding = '2px 8px';
                                        badge.style.borderRadius = '9999px';
                                        badge.style.backgroundColor = color;
                                        badge.style.color = textColor;
                                        badge.style.fontWeight = '600';
                                        badge.style.fontSize = '13px';
                                        badge.style.lineHeight = '1.3';
                                        badge.textContent = tag.display_name || '';
                                    
                                        this.eGui.appendChild(badge);
                                    });
                                }
                            }
                            getGui() { return this.eGui; }
                        }
                    """)
                    gb.configure_column(
                        "tags", 
                        headerName="Tags", 
                        cellRenderer=tags_renderer, 
                        autoHeight=True,
                        cellStyle={'display': 'flex', 'alignItems': 'center'}
                    )
            
                # Enable pagination
                gb.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=10)
                gb.configure_grid_options(
                    paginationPageSizeSelector=[10, 20, 50, 100],
                    pagination=True,
                    paginationNumberFormatter=JsCode("""function(params) { return params.value.toLocaleString(); }"""),
                )
            
                # Build grid options
                grid_options = gb.build()
            
                # Force disable menu on ALL columns after build and center client_id
                if 'columnDefs' in grid_options:
                    for col_def in grid_options['columnDefs']:
                        col_def['suppressMenu'] = True
                        col_def['suppressHeaderMenuButton'] = True
                        col_def['menuTabs'] = []
                        col_def['filter'] = False
                        # Center align the client_id column (# column)
                        if col_def.get('field') == 'client_id':
                            # Force center alignment for both cell and header
                            col_def['cellStyle'] = {'textAlign': 'center'}
                            col_def['headerComponentParams'] = {
                                'template': '<div class="ag-cell-label-container" role="presentation" style="justify-content: center;"><span ref="eText" class="ag-header-cell-text" style="text-align: center; width: 100%;">#</span></div>'
                            }
            
                # Add custom CSS for left-aligned pagination
                custom_css = {
                    ".ag-paging-panel": {"justify-content": "flex-start !important"}
                }
            
                # Use a dynamic key that changes when navigating to reset grid state
                grid_key = 'clients_grid' if st.session_state.get('last_selected_client') is None else f'clients_grid_{st.session_state.last_selected_client}'
                
                grid_response = AgGrid(
                    clients_view,
                    gridOptions=grid_options,
                    height=520,
                    fit_columns_on_grid_load=False,  # Don't fit columns
                    columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,  # Auto-adjust column widths
                    update_mode=GridUpdateMode.SELECTION_CHANGED,
                    allow_unsafe_jscode=True,
                    theme="balham",
                    reload_data=True,
                    custom_css=custom_css,
                    key=grid_key
                )
            
                # Check if a row was selected (only happens when Name column is clicked)
                selected_rows_df = grid_response['selected_rows']
                if selected_rows_df is not None and not selected_rows_df.empty:
                    selected_id = int(selected_rows_df.iloc[0]['client_id'])
                    # Only navigate if it's a different client
                    if "last_selected_client" not in st.session_state or st.session_state.last_selected_client != selected_id:
                        st.session_state.last_selected_client = selected_id
                        st.query_params["tab"] = "Client"
                        st.query_params["client_id"] = str(selected_id)
                        st.rerun()

        elif st.session_state.active_client_page_subtab == "Client Tags":
            # Second-level tabs for Client Tags subtab (Dashboard / Settings)
            CLIENT_TAGS_SUBTAB_ICONS = {"Dashboard": "dashboard", "Settings": "settings"}
            CLIENT_TAGS_SUBTAB_NAMES = list(CLIENT_TAGS_SUBTAB_ICONS.keys())

            # initialize active_subtab for Client Tags (default: Dashboard)
            if "active_client_tags_subtab" not in st.session_state:
                st.session_state.active_client_tags_subtab = "Dashboard"

            # allow switch via query param
            if "client_tags_subtab" in st.query_params:
                q = st.query_params["client_tags_subtab"]
                if q in CLIENT_TAGS_SUBTAB_NAMES:
                    st.session_state.active_client_tags_subtab = q

            # build subtabs HTML for Client Tags
            base_tab_q = "?tab=Client&client_subtab=Client%20Tags"
            sub_html = ['<div class="subtabs">']
            for name in CLIENT_TAGS_SUBTAB_NAMES:
                icon = f'<span class="material-icons">{CLIENT_TAGS_SUBTAB_ICONS[name]}</span>'
                cls = " sub-active" if st.session_state.active_client_tags_subtab == name else ""
                sub_html.append(
                    f'<a href="{base_tab_q}&client_tags_subtab={name.replace(" ", "%20")}" target="_self" class="subtab{cls}">{icon} {name}</a>'
                )
            sub_html.append('</div>')
            st.markdown("".join(sub_html), unsafe_allow_html=True)

            # Content based on active subtab
            if st.session_state.active_client_tags_subtab == "Dashboard":
                # Fetch auto tag statistics for chart
                stats_df = fetch_auto_tag_statistics()
        
                if not stats_df.empty:
                    # Convert date column to datetime for proper handling
                    stats_df['date'] = pd.to_datetime(stats_df['date'])
            
                    # Get min and max dates
                    min_date = stats_df['date'].min()
                    max_date = stats_df['date'].max()
            
                    # Calculate default date range (last 7 days)
                    default_start = max_date - pd.Timedelta(days=6)
                    if default_start < min_date:
                        default_start = min_date
            
                    # Add section header
                    st.subheader("Automatic Tag Assignment History")
            
                    with st.container(border=True):
                        # Add filters row with tag filter and date slider
                        col1, col2 = st.columns([1, 2])
                
                        with col1:
                            # Get unique tag names for filter
                            tag_names = sorted(stats_df['display_name'].unique().tolist())
                            tag_options = ["All"] + tag_names
                            selected_tag = st.selectbox(
                                "Tag",
                                options=tag_options,
                                index=0,
                                key="chart_tag_filter"
                            )
                
                        with col2:
                            date_range = st.slider(
                                "Date",
                                min_value=min_date.date(),
                                max_value=max_date.date(),
                                value=(default_start.date(), max_date.date()),
                                format="YYYY-MM-DD"
                            )
                
                        # Filter data based on selected tag
                        if selected_tag != "All":
                            filtered_stats = stats_df[stats_df['display_name'] == selected_tag]
                        else:
                            filtered_stats = stats_df.copy()
                
                        # Filter data based on selected date range
                        mask = (filtered_stats['date'].dt.date >= date_range[0]) & (filtered_stats['date'].dt.date <= date_range[1])
                        filtered_stats = filtered_stats[mask]
                
                        if not filtered_stats.empty:
                            # Get color mapping for each tag
                            color_map = {}
                            for _, row in filtered_stats[['display_name', 'color']].drop_duplicates().iterrows():
                                if pd.notna(row['display_name']) and pd.notna(row['color']):
                                    color_map[row['display_name']] = row['color']
                    
                            # Create Plotly figure
                            fig = go.Figure()
                    
                            # Add a line for each tag
                            for tag_name in filtered_stats['display_name'].unique():
                                tag_data = filtered_stats[filtered_stats['display_name'] == tag_name].sort_values('date')
                                fig.add_trace(go.Scatter(
                                    x=tag_data['date'],
                                    y=tag_data['assigned_count'],
                                    name=tag_name,
                                    mode='lines+markers',
                                    line=dict(color=color_map.get(tag_name, '#6b7280'), width=2),
                                    marker=dict(size=6)
                                ))
                    
                            # Update layout with legend on the right
                            fig.update_layout(
                                height=350,
                                margin=dict(l=0, r=0, t=20, b=0),
                                legend=dict(
                                    orientation="v",
                                    yanchor="top",
                                    y=1,
                                    xanchor="left",
                                    x=1.02,
                                    font=dict(size=14),
                                    itemwidth=30
                                ),
                                xaxis_title="Date",
                                yaxis_title="Assigned Count",
                                hovermode='x unified'
                            )
                    
                            # Display the chart with hidden modebar
                            st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
            
                    # Add two charts side by side below
                    chart_col1, chart_col2 = st.columns(2)
            
                    with chart_col1:
                        # Add section header
                        st.subheader("Automatic Tag Assignments")
                
                        with st.container(border=True):
                            # Vertical bar chart for last day's assignment counts
                            # Get the last date from stats_df
                            last_date = stats_df['date'].max()
                            last_day_data = stats_df[stats_df['date'] == last_date].copy()
                    
                            if not last_day_data.empty:
                                # Sort by assigned_count for better visualization
                                last_day_data = last_day_data.sort_values('assigned_count', ascending=True)
                        
                                # Create bar chart
                                bar_fig = go.Figure()
                        
                                bar_fig.add_trace(go.Bar(
                                    x=last_day_data['assigned_count'],
                                    y=last_day_data['display_name'],
                                    orientation='h',
                                    marker=dict(
                                        color=last_day_data['color'].tolist(),
                                    ),
                                    text=last_day_data['assigned_count'],
                                    textposition='auto',
                                ))
                        
                                bar_fig.update_layout(
                                    title=last_date.strftime('%Y-%m-%d'),
                                    height=400,
                                    margin=dict(l=0, r=0, t=40, b=0),
                                    xaxis_title="Tag Count",
                                    yaxis_title="Tag",
                                    showlegend=False
                                )
                        
                                st.plotly_chart(bar_fig, width='stretch', config={'displayModeBar': False})
            
                    with chart_col2:
                        # Add section header
                        st.subheader("Manual Tag Assignments")
                
                        with st.container(border=True):
                            # Fetch manual tag counts
                            manual_tags_df = fetch_manual_tag_counts()
                    
                            if not manual_tags_df.empty:
                                # Create bar chart
                                manual_bar_fig = go.Figure()
                        
                                manual_bar_fig.add_trace(go.Bar(
                                    x=manual_tags_df['tag_count'],
                                    y=manual_tags_df['display_name'],
                                    orientation='h',
                                    marker=dict(
                                        color=manual_tags_df['color'].tolist(),
                                    ),
                                    text=manual_tags_df['tag_count'],
                                    textposition='auto',
                                    width=0.4,
                                ))
                        
                                manual_bar_fig.update_layout(
                                    height=400,
                                    margin=dict(l=0, r=0, t=20, b=0),
                                    xaxis_title="Tag Count",
                                    yaxis_title="Tag",
                                    showlegend=False
                                )
                        
                                st.plotly_chart(manual_bar_fig, width='stretch', config={'displayModeBar': False})
                            else:
                                st.info("No manual tags found.")
        
                # Fetch data
                df = fetch_client_tags_dataframe()
        
                if df.empty:
                    st.info("No client tags found in the database.")
                else:
                    # Add section header
                    st.subheader("Client Tags")
            
                    # Placeholder for messages
                    message_container = st.container()
            
                    # Add filter dropdown for Tag Type and Remove button
                    col1, col2, col3 = st.columns([1, 1, 4])
                    with col1:
                        tag_type_filter = st.selectbox(
                            "Tag Type",
                            options=["All", "Automatic", "Manual"],
                            index=0,
                            key="tag_type_filter"
                        )
            
                    # Create grid view - remove unwanted columns
                    columns_to_remove = ["description", "color", "is_active", "tag_id", "system_name"]
                    df_view = df.drop(columns=[col for col in columns_to_remove if col in df.columns], errors='ignore')
            
                    # Add a selection column at the beginning
                    df_view.insert(0, "Select", "")
            
                    # Reorder columns - ont_id, client_id, display_name, tag_type
                    desired_order = ["Select", "ont_id", "client_id", "display_name", "tag_type"]
                    remaining_cols = [col for col in df_view.columns if col not in desired_order]
                    new_order = [col for col in desired_order if col in df_view.columns] + remaining_cols
                    df_view = df_view[new_order]
            
                    # Map tag_type from codes to full names
                    if "tag_type" in df_view.columns:
                        tag_type_mapping = {"A": "Automatic", "M": "Manual"}
                        df_view["tag_type"] = df_view["tag_type"].map(tag_type_mapping).fillna(df_view["tag_type"])
            
                    # Apply tag type filter
                    if tag_type_filter != "All":
                        df_view = df_view[df_view["tag_type"] == tag_type_filter]
            
                    # Build AG Grid
                    gb = GridOptionsBuilder.from_dataframe(df_view, enableRowGroup=False, enableValue=False, enablePivot=False)
            
                    # Configure default column settings
                    gb.configure_default_column(
                        editable=False,
                        resizable=True,
                        filter=False,
                        sortable=True,
                        suppressMenu=True,
                        menuTabs=[],
                        suppressHeaderMenuButton=True,
                    )
            
                    # Configure selection - disable row click selection
                    gb.configure_selection(
                        selection_mode="multiple",
                        use_checkbox=False,
                        rowMultiSelectWithClick=False,
                    )
            
                    # Configure the Select column with checkboxes
                    gb.configure_column(
                        "Select",
                        headerName="",
                        pinned="left",
                        checkboxSelection=True,
                        headerCheckboxSelection=True,
                        headerCheckboxSelectionFilteredOnly=False,
                        sortable=False,
                        filter=False,
                        suppressMenu=True,
                        menuTabs=[],
                        suppressHeaderMenuButton=True,
                        width=50,
                        maxWidth=50,
                    )
            
                    # Configure column headers
                    column_headers = {
                        "ont_id": "ONT ID",
                        "client_id": "Client ID",
                        "display_name": "Tag",
                        "assigned_at": "Assigned At",
                        "assigned_by": "Assigned By",
                        "reason": "Reason",
                        "tag_type": "Tag Type",
                    }
            
                    for col, header in column_headers.items():
                        if col in df_view.columns:
                            gb.configure_column(
                                col, 
                                headerName=header,
                                filter=False,
                                suppressMenu=True,
                                menuTabs=[],
                                suppressHeaderMenuButton=True,
                            )
            
                    # Configure display_name column with badge rendering using color from original df
                    if "display_name" in df_view.columns:
                        # Store color mapping in session state for the renderer
                        color_map = {}
                        if "display_name" in df.columns and "color" in df.columns:
                            for _, row in df.iterrows():
                                if pd.notna(row.get("display_name")) and pd.notna(row.get("color")):
                                    color_map[str(row["display_name"])] = str(row["color"])
                
                        # Create color lookup string for JS using proper JSON encoding
                        import json
                        color_json = json.dumps(color_map)
                
                        color_renderer = JsCode(f"""
                            class ColorBadgeRenderer {{
                                init(params) {{
                                    const colorMap = {color_json};
                                    const displayName = params.data.display_name || '';
                                    const color = colorMap[displayName] || '#6b7280';
                            
                                    // Function to determine if text should be black or white based on background
                                    function getContrastColor(hexColor) {{
                                        // Remove # if present
                                        const hex = hexColor.replace('#', '');
                                
                                        // Convert to RGB
                                        const r = parseInt(hex.substr(0, 2), 16);
                                        const g = parseInt(hex.substr(2, 2), 16);
                                        const b = parseInt(hex.substr(4, 2), 16);
                                
                                        // Calculate relative luminance using WCAG formula
                                        const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
                                
                                        // Return black for light backgrounds, white for dark backgrounds
                                        return luminance > 0.5 ? '#000000' : '#ffffff';
                                    }}
                            
                                    const textColor = getContrastColor(color);
                            
                                    this.eGui = document.createElement('span');
                                    this.eGui.style.display = 'inline-block';
                                    this.eGui.style.padding = '2px 8px';
                                    this.eGui.style.borderRadius = '9999px';
                                    this.eGui.style.backgroundColor = color;
                                    this.eGui.style.color = textColor;
                                    this.eGui.style.fontWeight = '600';
                                    this.eGui.style.fontSize = '13px';
                                    this.eGui.style.lineHeight = '1.3';
                                    this.eGui.textContent = displayName;
                                }}
                                getGui() {{ return this.eGui; }}
                            }}
                        """)
                        gb.configure_column("display_name", headerName="Tag", cellRenderer=color_renderer)
            
                    # Enable pagination
                    gb.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=10)
                    gb.configure_grid_options(
                        paginationPageSizeSelector=[10, 20, 50, 100],
                        rowSelection='multiple',
                        suppressRowDeselection=False,
                    )
            
                    # Add auto-size on grid load and resize
                    gb.configure_grid_options(
                        onFirstDataRendered=JsCode("""
                            function(params) {
                                if (!params || !params.columnApi || !params.columnApi.autoSizeAllColumns) {
                                    return;
                                }
                                const autoSize = () => {
                                    try { params.columnApi.autoSizeAllColumns(); } catch (e) {}
                                    try { params.api.resetRowHeights(); } catch (e) {}
                                };
                                window.requestAnimationFrame(autoSize);
                                [0, 120, 400].forEach((delay) => {
                                    window.setTimeout(autoSize, delay);
                                });
                            }
                        """),
                        onGridSizeChanged=JsCode("""
                            function(params) {
                                if (!params || !params.columnApi || !params.columnApi.autoSizeAllColumns) {
                                    return;
                                }
                                const run = () => {
                                    try { params.columnApi.autoSizeAllColumns(); } catch (e) {}
                                    try { params.api.resetRowHeights(); } catch (e) {}
                                };
                                window.requestAnimationFrame(run);
                                window.setTimeout(run, 150);
                            }
                        """),
                        suppressRowClickSelection=True,
                        suppressHeaderMenuButton=True,
                        suppressColumnMenu=True,
                        columnMenu="none",
                    )
            
                    # Build grid options
                    grid_options = gb.build()
            
                    # Ensure column menu is disabled for all columns
                    grid_options.setdefault("columnMenu", "none")
                    default_col_def = grid_options.setdefault("defaultColDef", {})
                    default_col_def["suppressMenu"] = True
                    default_col_def["menuTabs"] = []
                    default_col_def["suppressHeaderMenuButton"] = True
            
                    # Disable menu for all column definitions
                    col_defs = grid_options.get("columnDefs", [])
                    if isinstance(col_defs, list):
                        for col_def in col_defs:
                            if isinstance(col_def, dict):
                                col_def["suppressMenu"] = True
                                col_def["menuTabs"] = []
                                col_def["suppressHeaderMenuButton"] = True
                                # For the checkbox selection column - only select current page
                                if col_def.get("checkboxSelection") == True:
                                    col_def["lockPosition"] = "left"
                                    col_def["headerCheckboxSelection"] = True
                                    col_def["headerCheckboxSelectionCurrentPageOnly"] = True
            
                    # Add custom CSS for left-aligned pagination
                    custom_css = {
                        ".ag-paging-panel": {"justify-content": "flex-start !important"}
                    }
            
                    # Initialize deletion counter for grid key management
                    if 'dashboard_deletion_count' not in st.session_state:
                        st.session_state.dashboard_deletion_count = 0
                    
                    # Display grid with dynamic key that changes after deletions
                    grid_key = f'client_tags_dashboard_grid_{st.session_state.dashboard_deletion_count}'
                    grid_response = AgGrid(
                        df_view,
                        gridOptions=grid_options,
                        height=520,
                        fit_columns_on_grid_load=False,
                        update_mode=GridUpdateMode.SELECTION_CHANGED,
                        data_return_mode=DataReturnMode.AS_INPUT,
                        allow_unsafe_jscode=True,
                        theme="balham",
                        custom_css=custom_css,
                        key=grid_key,
                        reload_data=False
                    )
            
                    # Show remove button only if rows are selected
                    selected_rows = grid_response.get("selected_rows", [])
                    if selected_rows is not None and len(selected_rows) > 0:
                        selected_count = len(selected_rows)
                        with col2:
                            st.markdown('<div style="padding-top: 26px;"></div>', unsafe_allow_html=True)
                            if st.button(f"🗑️ Remove Selected ({selected_count})", key="remove_tags_button", type="primary"):
                                # Get the original indices to find client_id and tag_id
                                removed_count = 0
                        
                                # Convert selected_rows to DataFrame if it's not already
                                if isinstance(selected_rows, pd.DataFrame):
                                    selected_df = selected_rows
                                else:
                                    selected_df = pd.DataFrame(selected_rows)
                        
                                for idx, selected_row in selected_df.iterrows():
                                    # Find the matching row in the original dataframe
                                    client_id = selected_row.get("client_id")
                                    # Get tag_id from display_name by looking up in original df
                                    display_name = selected_row.get("display_name")
                            
                                    if client_id and display_name:
                                        # Find tag_id from original df
                                        matching_rows = df[(df["client_id"] == client_id) & (df["display_name"] == display_name)]
                                        if not matching_rows.empty:
                                            tag_id = matching_rows.iloc[0]["tag_id"]
                                            if remove_client_tag(client_id, tag_id):
                                                removed_count += 1
                        
                                if removed_count > 0:
                                    # Increment deletion counter to reset grid selection
                                    st.session_state.dashboard_deletion_count += 1
                                    with message_container:
                                        st.success(f"Removed {removed_count} tag(s)")
                                    time.sleep(2)
                                    st.rerun()
            
            elif st.session_state.active_client_tags_subtab == "Settings":
                # Show auto tag edit modal
                if 'edit_auto_tag' in st.session_state and st.session_state.edit_auto_tag is not None:
                    tag_data = st.session_state.edit_auto_tag
                    
                    @st.dialog(f"Editing: {tag_data['display_name']}", width="small")
                    def edit_auto_tag_modal():
                        # First thing: check if dialog should close
                        if st.session_state.get('close_auto_dialog', False):
                            st.session_state.edit_auto_tag = None
                            st.session_state.close_auto_dialog = False
                            st.rerun()
                        
                        new_display_name = st.text_input("Tag", value=tag_data['display_name'], key="auto_tag_name")
                        
                        # Description (read-only for automatic tags)
                        st.text_area("Description", value=tag_data['description'] if pd.notna(tag_data['description']) else "", disabled=True, height=80, key="auto_tag_desc")
                        
                        new_color = st.color_picker("Color", value=tag_data['color'], key="auto_tag_color")
                        new_is_active = st.checkbox("Active", value=bool(tag_data['is_active']), key="auto_tag_active")
                        
                        # Add spacing
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # Placeholder for success message (full width)
                        message_placeholder = st.empty()
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Save", use_container_width=True, type="primary", key="save_auto_tag"):
                                success = update_tag_config(
                                    tag_id=int(tag_data['id']),
                                    display_name=new_display_name,
                                    color=new_color,
                                    is_active=new_is_active
                                )
                                if success:
                                    message_placeholder.success(f"Updated {new_display_name}")
                                    time.sleep(1)
                                st.session_state.close_auto_dialog = True
                                st.rerun()
                        with col2:
                            if st.button("Cancel", use_container_width=True, key="cancel_auto_tag"):
                                st.session_state.close_auto_dialog = True
                                st.rerun()
                    
                    edit_auto_tag_modal()
                
                # Show manual tag edit modal
                elif 'edit_manual_tag' in st.session_state and st.session_state.edit_manual_tag is not None:
                    tag_data = st.session_state.edit_manual_tag
                    
                    @st.dialog(f"Editing: {tag_data['display_name']}", width="small")
                    def edit_manual_tag_modal():
                        # First thing: check if dialog should close
                        if st.session_state.get('close_manual_dialog', False):
                            st.session_state.edit_manual_tag = None
                            st.session_state.close_manual_dialog = False
                            st.rerun()
                        
                        new_display_name = st.text_input("Tag", value=tag_data['display_name'], key="manual_tag_name")
                        new_description = st.text_area("Description", value=tag_data['description'] if pd.notna(tag_data['description']) else "", height=80, key="manual_tag_desc")
                        new_color = st.color_picker("Color", value=tag_data['color'], key="manual_tag_color")
                        new_is_active = st.checkbox("Active", value=bool(tag_data['is_active']), key="manual_tag_active")
                        
                        # Add spacing
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # Placeholder for success message (full width)
                        message_placeholder = st.empty()
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Save", use_container_width=True, type="primary", key="save_manual_tag"):
                                success = update_tag_config(
                                    tag_id=int(tag_data['id']),
                                    display_name=new_display_name,
                                    color=new_color,
                                    is_active=new_is_active,
                                    description=new_description
                                )
                                if success:
                                    message_placeholder.success(f"Updated {new_display_name}")
                                    time.sleep(1)
                                st.session_state.close_manual_dialog = True
                                st.rerun()
                        with col2:
                            if st.button("Cancel", use_container_width=True, key="cancel_manual_tag"):
                                st.session_state.close_manual_dialog = True
                                st.rerun()
                    
                    edit_manual_tag_modal()
                
                # Automatic Tags Section
                st.subheader("Automatic Tags")
                auto_tags_df = fetch_tag_config(tag_type='A')
        
                if not auto_tags_df.empty:
                    # Prepare grid data - add action column as FIRST column, keep color but hide it
                    auto_grid_df = auto_tags_df[['id', 'display_name', 'color', 'is_active', 'description']].copy()
                    auto_grid_df.insert(0, 'edit', '✏️')  # Edit emoji as first column
                    
                    # Build AG Grid
                    gb = GridOptionsBuilder.from_dataframe(auto_grid_df)
                    
                    gb.configure_default_column(
                        editable=False,
                        resizable=True,
                        filter=False,
                        sortable=False,
                        suppressMenu=True,
                    )
                    
                    # Configure edit button column (first column)
                    gb.configure_column(
                        "edit",
                        headerName="",
                        width=60,
                        cellStyle={'textAlign': 'center', 'cursor': 'pointer', 'fontSize': '18px'},
                        editable=False,
                        pinned='left',
                        suppressSizeToFit=True
                    )
                    
                    # Configure Tag column with color background using cellStyle function
                    gb.configure_column(
                        "display_name", 
                        headerName="Tag", 
                        autoHeaderHeight=True, 
                        wrapHeaderText=True,
                        cellStyle=JsCode("""
                            function(params) {
                                const color = params.data.color;
                                // Calculate contrasting text color
                                const hex = color.replace('#', '');
                                const r = parseInt(hex.substr(0, 2), 16);
                                const g = parseInt(hex.substr(2, 2), 16);
                                const b = parseInt(hex.substr(4, 2), 16);
                                const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
                                const textColor = luminance > 0.5 ? '#000000' : '#ffffff';
                                
                                return {
                                    'backgroundColor': color,
                                    'color': textColor,
                                    'fontWeight': '500',
                                    'padding': '8px 12px',
                                    'display': 'flex',
                                    'alignItems': 'center',
                                    'height': '100%'
                                };
                            }
                        """)
                    )
                    
                    # Hide color column (but keep it in data for styling)
                    gb.configure_column("color", hide=True)
                    
                    # Configure other data columns
                    gb.configure_column("is_active", headerName="Active", autoHeaderHeight=True, wrapHeaderText=True)
                    gb.configure_column("description", headerName="Description", flex=1, autoHeaderHeight=True, wrapHeaderText=True)
                    
                    # Hide ID column but keep it in data
                    gb.configure_column("id", hide=True)
                    
                    # Disable selection completely
                    gb.configure_selection(selection_mode=None, use_checkbox=False)
                    
                    # Configure cell clicked event to only select on edit column
                    gb.configure_grid_options(
                        onCellClicked=JsCode("""
                            function(params) {
                                if (params.column.colId === 'edit') {
                                    params.api.deselectAll();
                                    params.node.setSelected(true);
                                }
                            }
                        """),
                        suppressRowClickSelection=True,
                        rowSelection='single'
                    )
                    
                    grid_options = gb.build()
                    
                    # Calculate dynamic height based on row count
                    row_height = 42
                    header_height = 48
                    min_height = 150
                    dynamic_height = header_height + (len(auto_grid_df) * row_height) + 10
                    auto_grid_height = max(min_height, min(dynamic_height, 600))
                    
                    # Use a dynamic key that changes when modal closes to reset grid state
                    grid_key = 'auto_tags_grid' if st.session_state.get('edit_auto_tag') is not None else 'auto_tags_grid_reset'
                    
                    grid_response = AgGrid(
                        auto_grid_df,
                        gridOptions=grid_options,
                        height=auto_grid_height,
                        update_mode=GridUpdateMode.SELECTION_CHANGED,
                        allow_unsafe_jscode=True,
                        theme="balham",
                        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
                        key=grid_key,
                        reload_data=True
                    )
                    
                    # Check if a row was selected (only happens when edit column is clicked)
                    if 'edit_auto_tag' not in st.session_state or st.session_state.edit_auto_tag is None:
                        selected_rows = grid_response.get('selected_rows')
                        if selected_rows is not None and not selected_rows.empty:
                            selected_row = selected_rows.iloc[0].to_dict()
                            st.session_state.edit_auto_tag = selected_row
                            st.rerun()
                
                # Show message if no tags
                if auto_tags_df.empty:
                    st.info("No automatic tags found.")
        
                # Manual Tags Section  
                st.subheader("Manual Tags")
                
                # Add new tag button
                if st.button("➕ Create New Tag", key="add_manual_tag_btn"):
                    st.session_state.show_add_manual_tag_dialog = True
        
                # Show add tag modal
                if st.session_state.get("show_add_manual_tag_dialog", False):
                    @st.dialog("Create New Tag", width="small")
                    def add_manual_tag_modal():
                        # Check if dialog should close
                        if st.session_state.get('close_add_dialog', False):
                            st.session_state.show_add_manual_tag_dialog = False
                            st.session_state.close_add_dialog = False
                            st.rerun()
                        
                        new_tag_name = st.text_input("Tag Name", placeholder="Enter tag name...", key="new_tag_name")
                        new_tag_description = st.text_area("Description", placeholder="Enter tag description...", height=80, key="new_tag_desc")
                        new_tag_color = st.color_picker("Color", value="#6b7280", key="new_tag_color")
                        new_tag_active = st.checkbox("Active", value=True, key="new_tag_active")
                        
                        # Add spacing
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # Placeholder for success message
                        message_placeholder = st.empty()
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Create", use_container_width=True, type="primary", key="create_new_tag"):
                                if new_tag_name:
                                    # Convert tag name to system_name (lowercase with underscores)
                                    new_tag_system_name = new_tag_name.lower().replace(" ", "_")
                                    
                                    # Insert new tag into database
                                    success = create_manual_tag(
                                        system_name=new_tag_system_name,
                                        display_name=new_tag_name,
                                        color=new_tag_color,
                                        description=new_tag_description,
                                        is_active=new_tag_active
                                    )
                                    if success:
                                        message_placeholder.success(f"Created new tag: {new_tag_name}")
                                        time.sleep(1)
                                    st.session_state.close_add_dialog = True
                                    st.rerun()
                                else:
                                    message_placeholder.error("Tag Name is required!")
                        with col2:
                            if st.button("Cancel", use_container_width=True, key="cancel_new_tag"):
                                st.session_state.close_add_dialog = True
                                st.rerun()
                    
                    add_manual_tag_modal()
        
                manual_tags_df = fetch_tag_config(tag_type='M')
        
                if not manual_tags_df.empty:
                    # Prepare grid data - add action column as FIRST column, keep color but hide it
                    manual_grid_df = manual_tags_df[['id', 'display_name', 'color', 'is_active', 'description']].copy()
                    manual_grid_df.insert(0, 'edit', '✏️')  # Edit emoji as first column
                    
                    # Build AG Grid
                    gb = GridOptionsBuilder.from_dataframe(manual_grid_df)
                    
                    gb.configure_default_column(
                        editable=False,
                        resizable=True,
                        filter=False,
                        sortable=False,
                        suppressMenu=True,
                    )
                    
                    # Configure edit button column (first column)
                    gb.configure_column(
                        "edit",
                        headerName="",
                        width=60,
                        cellStyle={'textAlign': 'center', 'cursor': 'pointer', 'fontSize': '18px'},
                        editable=False,
                        pinned='left',
                        suppressSizeToFit=True
                    )
                    
                    # Configure Tag column with color background using cellStyle function
                    gb.configure_column(
                        "display_name", 
                        headerName="Tag", 
                        autoHeaderHeight=True, 
                        wrapHeaderText=True,
                        cellStyle=JsCode("""
                            function(params) {
                                const color = params.data.color;
                                // Calculate contrasting text color
                                const hex = color.replace('#', '');
                                const r = parseInt(hex.substr(0, 2), 16);
                                const g = parseInt(hex.substr(2, 2), 16);
                                const b = parseInt(hex.substr(4, 2), 16);
                                const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
                                const textColor = luminance > 0.5 ? '#000000' : '#ffffff';
                                
                                return {
                                    'backgroundColor': color,
                                    'color': textColor,
                                    'fontWeight': '500',
                                    'padding': '8px 12px',
                                    'display': 'flex',
                                    'alignItems': 'center',
                                    'height': '100%'
                                };
                            }
                        """)
                    )
                    
                    # Hide color column (but keep it in data for styling)
                    gb.configure_column("color", hide=True)
                    
                    # Configure other data columns
                    gb.configure_column("is_active", headerName="Active", autoHeaderHeight=True, wrapHeaderText=True)
                    gb.configure_column("description", headerName="Description", flex=1, autoHeaderHeight=True, wrapHeaderText=True)
                    
                    # Hide ID column but keep it in data
                    gb.configure_column("id", hide=True)
                    
                    # Disable selection completely
                    gb.configure_selection(selection_mode=None, use_checkbox=False)
                    
                    # Configure cell clicked event to only select on edit column
                    gb.configure_grid_options(
                        onCellClicked=JsCode("""
                            function(params) {
                                if (params.column.colId === 'edit') {
                                    params.api.deselectAll();
                                    params.node.setSelected(true);
                                }
                            }
                        """),
                        suppressRowClickSelection=True,
                        rowSelection='single'
                    )
                    
                    grid_options = gb.build()
                    
                    # Calculate dynamic height based on row count
                    row_height = 42
                    header_height = 48
                    min_height = 150
                    dynamic_height = header_height + (len(manual_grid_df) * row_height) + 10
                    manual_grid_height = max(min_height, min(dynamic_height, 600))
                    
                    # Use a dynamic key that changes when modal closes to reset grid state
                    grid_key = 'manual_tags_grid' if st.session_state.get('edit_manual_tag') is not None else 'manual_tags_grid_reset'
                    
                    grid_response = AgGrid(
                        manual_grid_df,
                        gridOptions=grid_options,
                        height=manual_grid_height,
                        update_mode=GridUpdateMode.SELECTION_CHANGED,
                        allow_unsafe_jscode=True,
                        theme="balham",
                        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
                        key=grid_key,
                        reload_data=True
                    )
                    
                    # Check if a row was selected (only happens when edit column is clicked)
                    if 'edit_manual_tag' not in st.session_state or st.session_state.edit_manual_tag is None:
                        selected_rows = grid_response.get('selected_rows')
                        if selected_rows is not None and not selected_rows.empty:
                            selected_row = selected_rows.iloc[0].to_dict()
                            st.session_state.edit_manual_tag = selected_row
                            st.rerun()
                
                # Show message if no tags
                if manual_tags_df.empty:
                    st.info("No manual tags found.")

    else:
        # Show client detail page
        # Client page layout: two columns
        st.markdown('<div id="client-layout">', unsafe_allow_html=True)
        
        # Add back button
        if st.button("← Back to Client List"):
            st.query_params.clear()
            st.query_params["tab"] = "Client"
            st.rerun()
        
        left, right = st.columns([2, 5], gap="large")
        with left:
            # Fetch client data from database
            client_id = selected_client_id
            clients_df = fetch_clients()
            client_data = clients_df[clients_df['client_id'] == client_id]
            
            if client_data.empty:
                st.error(f"Client #{client_id} not found")
                st.stop()
            
            client_row = client_data.iloc[0]
            
            # Display client name as header
            client_name = client_row.get('name', 'Unknown Client')
            st.subheader(client_name)
            
            # Build client rows with dynamic data
            client_type = client_row.get('type', '')
            phone = client_row.get('phone', '')
            sip_value = client_row.get('sip', '')
            sip_display = sip_value if sip_value else "This Client has no Sip."
            
            client_rows = [
                ("ID", str(client_id)),
                ("Client Type", client_type),
                ("Account Number", phone),
                ("Service ID", ""),
                ("Sip", sip_display),
            ]

            # Render first set of rows (before Tags)
            for idx, (lbl, val) in enumerate(client_rows):
                lcol, vcol = st.columns([1, 2], gap="small")
                with lcol:
                    st.caption(lbl)
                with vcol:
                    st.write(val or "—")
                st.markdown('<hr style="margin:0;border:0;border-top:1px solid rgba(0,0,0,0.10);" />', unsafe_allow_html=True)
            
            # Tags section
            lcol, vcol = st.columns([1, 2], gap="small")
            with lcol:
                st.caption("Tags")
            with vcol:
                # Fetch tags for this client
                client_tags_df = fetch_client_tags(client_id)
                
                # Track if we need to show a message
                tag_message = None
                tag_message_type = None
                
                if not client_tags_df.empty:
                    # Display existing tags as colored badges without remove buttons
                    for idx, tag_row in client_tags_df.iterrows():
                        tag_color = tag_row.get('color', '#6b7280')
                        tag_name = tag_row.get('display_name', 'Unknown')
                        text_color = get_contrast_color(tag_color)
                        
                        st.markdown(
                            f'<span style="display:inline-block;background:{tag_color};color:{text_color};'
                            f'padding:4px 10px;border-radius:12px;font-size:14px;margin-bottom:4px;">{tag_name}</span>',
                            unsafe_allow_html=True
                        )
                else:
                    st.write("—")
            
            st.markdown('<hr style="margin:0;border:0;border-top:1px solid rgba(0,0,0,0.10);" />', unsafe_allow_html=True)
            
            # Remaining rows (after Tags) - use dynamic data
            city = client_row.get('city', '')
            address = client_row.get('address', '')
            
            remaining_rows = [
                ("City", city),
                ("Address", address),
                ("Comment", ""),
                ("Created At", ""),
                ("Last Change", ""),
            ]
            
            for idx, (lbl, val) in enumerate(remaining_rows):
                lcol, vcol = st.columns([1, 2], gap="small")
                with lcol:
                    st.caption(lbl)
                with vcol:
                    st.write(val or "—")
                # Compact separator between rows (skip after the last row)
                if idx < len(remaining_rows) - 1:
                    st.markdown('<hr style="margin:0;border:0;border-top:1px solid rgba(0,0,0,0.10);" />', unsafe_allow_html=True)
        
        with right:
                # Replace Streamlit tabs with HTML subtabs similar to "Call Tickets"
                CLIENT_SUBTAB_ICONS = {
                    "CPEs": "router",
                    "Tags": "label",
                    "Assign": "assignment_turned_in",
                    "SIP": "dialer_sip",
                    "Client Operations": "assignment",
                    "Client Attachments": "attachment",
                    "Client Attachments KYC": "fingerprint",
                }
                CLIENT_SUBTAB_NAMES = list(CLIENT_SUBTAB_ICONS.keys())

                # initialize active client subtab
                if "active_client_subtab" not in st.session_state:
                    st.session_state.active_client_subtab = "CPEs"
                # allow switch via query param only when on Client tab
                if st.session_state.active_tab == "Client" and "client_subtab" in st.query_params:
                    cq = st.query_params["client_subtab"]
                    if cq in CLIENT_SUBTAB_NAMES:
                        st.session_state.active_client_subtab = cq

                base_q = f"?tab=Client&client_id={client_id}"
                sub_html = ['<div class="subtabs" style="margin-left:12px;">']
                for i, name in enumerate(CLIENT_SUBTAB_NAMES):
                    icon = f'<span class="material-icons">{CLIENT_SUBTAB_ICONS[name]}</span>'
                    cls = " sub-active" if st.session_state.active_client_subtab == name else ""
                    if name in ["CPEs", "Tags"]:
                        # CPEs and Tags tabs are clickable
                        extra_cls = " tags-subtab" if name == "Tags" else ""
                        sub_html.append(
                            f'<a href="{base_q}&client_subtab={name.replace(" ", "%20")}" target="_self" class="subtab{cls}{extra_cls}">{icon} {name}</a>'
                        )
                    else:
                        # Render others as disabled/non-clickable
                        sub_html.append(
                            f'<span class="subtab subtab-disabled{cls}" title="Disabled">{icon} {name}</span>'
                        )
                sub_html.append('</div>')
                st.markdown("".join(sub_html), unsafe_allow_html=True)

                # Right column content based on active client subtab
                current = st.session_state.active_client_subtab
                if current == "CPEs":
                    ont_id = "7701234567"
                    st.markdown(
                        f"""
                        <div class="exp-card">
                            <details open>
                                <summary>#62000 - ccbe.5991.0000 - {ont_id} - Employee-1 - <span class="dt-green">2026-12-31 23:59:50</span></summary>
                                <div class="cpe-actions">
                                    <span class="cpe-btn cpe-btn-disabled" title="Disabled"><span class="material-icons">settings_remote</span> Remote Access</span>
                                    <span class="cpe-btn cpe-btn-disabled" title="Disabled"><span class="material-icons">restart_alt</span> Restart Session</span>
                                    <span class="cpe-btn cpe-btn-disabled" title="Disabled"><span class="material-icons">remove_circle</span> Unblock</span>
                                    <span class="cpe-btn cpe-btn-disabled" title="Disabled"><span class="material-icons">description</span> Request</span>
                                    <span class="cpe-btn cpe-btn-disabled" title="Disabled"><span class="material-icons">public</span> Public Ip</span>
                                    <span class="cpe-btn cpe-btn-disabled" title="Disabled"><span class="material-icons">credit_card</span> Recharge</span>
                                    <span class="cpe-btn cpe-btn-disabled" title="Disabled"><span class="material-icons">undo</span> Undo Recharge</span>
                                    <span class="cpe-btn cpe-btn-disabled" title="Disabled"><span class="material-icons">near_me</span> Transfer</span>
                                    <span class="cpe-btn cpe-btn-disabled" title="Disabled"><span class="material-icons">find_replace</span> Replace</span>
                                    <span class="cpe-btn cpe-btn-disabled" title="Disabled"><span class="material-icons">highlight_off</span> Un-Assign</span>
                                </div>
                                <div class="exp-content">
                                    <div class="kv-cols">
                                        <div class="kv-list">
                                            <div class="kv-row"><div class="kv-label">ID</div><div class="kv-value">62756</div></div>
                                            <div class="kv-sep"></div>
                                            <div class="kv-row"><div class="kv-label">Phone</div><div class="kv-value">7701234567</div></div>
                                            <div class="kv-sep"></div>
                                            <div class="kv-row"><div class="kv-label">ONT Model</div><div class="kv-value">844G</div></div>
                                            <div class="kv-sep"></div>
                                            <div class="kv-row"><div class="kv-label">Package</div><div class="kv-value">Employee-1</div></div>
                                            <div class="kv-sep"></div>
                                            <div class="kv-row"><div class="kv-label">Expiration</div><div class="kv-value">2026-12-31 23:59:50</div></div>
                                        </div>
                                        <div class="kv-list">
                                            <div class="kv-row"><div class="kv-label">OLT</div><div class="kv-value">NTWK-Sul-Pasha-OLT-00</div></div>
                                            <div class="kv-sep"></div>
                                            <div class="kv-row"><div class="kv-label">ONT ID</div><div class="kv-value">{ont_id}</div></div>
                                            <div class="kv-sep"></div>
                                            <div class="kv-row"><div class="kv-label">Serial</div><div class="kv-value">CXNK00000000</div></div>
                                            <div class="kv-sep"></div>
                                            <div class="kv-row"><div class="kv-label">MAC</div><div class="kv-value">ccbe.5991.0000</div></div>
                                            <div class="kv-sep"></div>
                                            <div class="kv-row"><div class="kv-label">Line Card</div><div class="kv-value">Cisco NCS 5500</div></div>
                                        </div>
                                        <div class="kv-list">
                                            <div class="kv-row"><div class="kv-label">Operational Status</div><div class="kv-value">enable</div></div>
                                            <div class="kv-sep"></div>
                                            <div class="kv-row"><div class="kv-label">Status</div><div class="kv-value">Online</div></div>
                                            <div class="kv-sep"></div>
                                            <div class="kv-row"><div class="kv-label">IP</div><div class="kv-value">10.49.72.000</div></div>
                                            <div class="kv-sep"></div>
                                            <div class="kv-row"><div class="kv-label">VLAN</div><div class="kv-value">3021</div></div>
                                            <div class="kv-sep"></div>
                                            <div class="kv-row"><div class="kv-label">GPON</div><div class="kv-value">2.5G/1.25G</div></div>
                                        </div>
                                    </div>
                                </div>
                            </details>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                elif current == "Tags":
                    # Fetch tags for this specific client
                    all_tags_df = fetch_client_tags_dataframe()
                    
                    # Placeholder for messages (used by both empty and non-empty states)
                    message_container = st.container()
                    
                    if all_tags_df.empty:
                        st.info("No tags found in the database.")
                        client_tags_df = pd.DataFrame()  # Empty dataframe for consistency
                    else:
                        # Filter for current client only
                        client_tags_df = all_tags_df[all_tags_df['client_id'] == client_id].copy()
                        
                        if client_tags_df.empty:
                            st.info("No tags assigned to this client.")
                        else:
                            # Grid content only shows when there are tags
                            # Select columns to display, keep tag_id for deletion
                            columns_to_show = ['display_name', 'tag_type', 'assigned_at', 'assigned_by', 'reason']
                            df_view = client_tags_df[[col for col in columns_to_show if col in client_tags_df.columns]].copy()
                            
                            # Add a selection column at the beginning
                            df_view.insert(0, "Select", "")
                            
                            # Map tag_type from codes to full names
                            if "tag_type" in df_view.columns:
                                tag_type_mapping = {"A": "Automatic", "M": "Manual"}
                                df_view["tag_type"] = df_view["tag_type"].map(tag_type_mapping).fillna(df_view["tag_type"])
                            
                            # Build AG Grid
                            gb = GridOptionsBuilder.from_dataframe(df_view, enableRowGroup=False, enableValue=False, enablePivot=False)
                            
                            # Configure selection - disable row click selection
                            gb.configure_selection(
                                selection_mode="multiple",
                                use_checkbox=False,
                                rowMultiSelectWithClick=False,
                            )
                            
                            # Configure grid options to prevent row click selection
                            gb.configure_grid_options(
                                suppressRowClickSelection=True,
                                rowSelection='multiple'
                            )
                            
                            # Configure the Select column with checkboxes
                            gb.configure_column(
                                "Select",
                                headerName="",
                                pinned="left",
                                checkboxSelection=True,
                                headerCheckboxSelection=True,
                                headerCheckboxSelectionFilteredOnly=False,
                                sortable=False,
                                filter=False,
                                suppressMenu=True,
                                menuTabs=[],
                                suppressHeaderMenuButton=True,
                                width=50,
                                maxWidth=50,
                            )
                            
                            # Configure default column settings
                            gb.configure_default_column(
                                editable=False,
                                resizable=True,
                                filter=False,
                                sortable=True,
                                suppressMenu=True,
                                menuTabs=[],
                                suppressHeaderMenuButton=True,
                            )
                            
                            # Configure column headers
                            column_headers = {
                                "display_name": "Tag",
                                "tag_type": "Tag Type",
                                "assigned_at": "Assigned At",
                                "assigned_by": "Assigned By",
                                "reason": "Reason",
                            }
                            
                            for col, header in column_headers.items():
                                if col in df_view.columns:
                                    gb.configure_column(
                                        col, 
                                        headerName=header,
                                        filter=False,
                                        suppressMenu=True,
                                        menuTabs=[],
                                        suppressHeaderMenuButton=True,
                                    )
                            
                            # Configure display_name column with badge rendering
                            if "display_name" in df_view.columns:
                                # Store color mapping for the renderer
                                color_map = {}
                                if "display_name" in client_tags_df.columns and "color" in client_tags_df.columns:
                                    for _, row in client_tags_df.iterrows():
                                        if pd.notna(row.get("display_name")) and pd.notna(row.get("color")):
                                            color_map[str(row["display_name"])] = str(row["color"])
                                
                                # Create color lookup string for JS using proper JSON encoding
                                import json
                                color_json = json.dumps(color_map)
                                
                                color_renderer = JsCode(f"""
                                    class ColorBadgeRenderer {{
                                        init(params) {{
                                            const colorMap = {color_json};
                                            const displayName = params.data.display_name || '';
                                            const color = colorMap[displayName] || '#6b7280';
                                            
                                            function getContrastColor(hexColor) {{
                                                const hex = hexColor.replace('#', '');
                                                const r = parseInt(hex.substr(0, 2), 16);
                                                const g = parseInt(hex.substr(2, 2), 16);
                                                const b = parseInt(hex.substr(4, 2), 16);
                                                const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
                                                return luminance > 0.5 ? '#000000' : '#ffffff';
                                            }}
                                            
                                            const textColor = getContrastColor(color);
                                            
                                            this.eGui = document.createElement('span');
                                            this.eGui.style.display = 'inline-block';
                                            this.eGui.style.padding = '2px 8px';
                                            this.eGui.style.borderRadius = '9999px';
                                            this.eGui.style.backgroundColor = color;
                                            this.eGui.style.color = textColor;
                                            this.eGui.style.fontWeight = '600';
                                            this.eGui.style.fontSize = '13px';
                                            this.eGui.style.lineHeight = '1.3';
                                            this.eGui.textContent = displayName;
                                        }}
                                        getGui() {{ return this.eGui; }}
                                    }}
                                """)
                                gb.configure_column("display_name", headerName="Tag", cellRenderer=color_renderer)
                            
                            # No pagination - show all rows
                            gb.configure_pagination(enabled=False)
                            
                            # Build grid options BEFORE displaying
                            grid_options = gb.build()
                            
                            # Post-build: force suppressMenu on all columns
                            if 'columnDefs' in grid_options:
                                for col_def in grid_options['columnDefs']:
                                    col_def['suppressMenu'] = True
                                    col_def['suppressHeaderMenuButton'] = True
                                    col_def['menuTabs'] = []
                                    col_def['filter'] = False
                            
                            # Calculate dynamic height based on row count (header + rows + padding)
                            row_height = 42  # approximate height per row
                            header_height = 48  # header height
                            min_height = 150  # minimum height
                            dynamic_height = header_height + (len(df_view) * row_height) + 10
                            grid_height = max(min_height, min(dynamic_height, 800))  # cap at 800px
                            
                            # Initialize deletion counter for grid key management
                            if 'client_detail_deletion_count' not in st.session_state:
                                st.session_state.client_detail_deletion_count = 0
                            
                            # Display the grid with dynamic key that changes after deletions
                            grid_key = f'client_tags_grid_{st.session_state.client_detail_deletion_count}'
                            grid_response = AgGrid(
                                df_view,
                                gridOptions=grid_options,
                                height=grid_height,
                                fit_columns_on_grid_load=False,
                                update_mode=GridUpdateMode.SELECTION_CHANGED,
                                data_return_mode=DataReturnMode.AS_INPUT,
                                allow_unsafe_jscode=True,
                                theme="balham",
                                key=grid_key,
                            )
                            
                            # Show remove button if rows are selected (below the grid)
                            selected_rows = grid_response.get("selected_rows", [])
                            if selected_rows is not None and len(selected_rows) > 0:
                                selected_count = len(selected_rows)
                                if st.button(f"🗑️ Remove Selected ({selected_count})", key="remove_client_tags_button", type="primary"):
                                    removed_count = 0
                                    
                                    # Convert selected_rows to DataFrame if it's not already
                                    if isinstance(selected_rows, pd.DataFrame):
                                        selected_df = selected_rows
                                    else:
                                        selected_df = pd.DataFrame(selected_rows)
                                    
                                    for idx, selected_row in selected_df.iterrows():
                                        # Get tag info from selected row
                                        display_name = selected_row.get("display_name")
                                        
                                        if display_name:
                                            # Find tag_id from original client_tags_df
                                            matching_rows = client_tags_df[client_tags_df["display_name"] == display_name]
                                            if not matching_rows.empty:
                                                tag_id = matching_rows.iloc[0]["tag_id"]
                                                if remove_client_tag(client_id, tag_id):
                                                    removed_count += 1
                                    
                                    if removed_count > 0:
                                        # Increment deletion counter to reset grid selection
                                        st.session_state.client_detail_deletion_count += 1
                                        with message_container:
                                            st.success(f"Successfully removed {removed_count} tag(s).")
                                        st.rerun()
                    
                    # Add Tag button that opens a dialog (always show, even if no tags)
                    if st.button("➕ Add Tag", key="open_add_tag_dialog"):
                        st.session_state.show_add_tag_dialog = True
                    
                    # Add Tag Dialog (always available)
                    if st.session_state.get("show_add_tag_dialog", False):
                        @st.dialog("Add Tag")
                        def add_tag_dialog():
                                    manual_tags_df = fetch_tag_config(tag_type='M')
                                    if not manual_tags_df.empty:
                                        # Filter out already assigned tags
                                        assigned_tag_ids = client_tags_df['tag_id'].tolist() if not client_tags_df.empty else []
                                        available_tags = manual_tags_df[~manual_tags_df['id'].isin(assigned_tag_ids)]
                                        
                                        if not available_tags.empty:
                                            tag_options = ["Select a tag..."] + available_tags['display_name'].tolist()
                                            selected_tag = st.selectbox(
                                                "Tag",
                                                options=tag_options,
                                                index=0,
                                                key="dialog_tag_selector"
                                            )
                                            
                                            reason_input = st.text_input(
                                                "Reason (optional)",
                                                key="dialog_reason_input",
                                                placeholder="Enter reason for adding this tag..."
                                            )
                                            
                                            # Buttons in two columns
                                            col1, col2 = st.columns(2)
                                            add_clicked = False
                                            with col1:
                                                if st.button("Add", key="confirm_add_tag", type="primary", disabled=(selected_tag == "Select a tag..."), use_container_width=True):
                                                    add_clicked = True
                                            with col2:
                                                if st.button("Cancel", key="cancel_add_tag", use_container_width=True):
                                                    st.session_state.show_add_tag_dialog = False
                                                    st.rerun()
                                            
                                            # Handle add action outside columns
                                            if add_clicked:
                                                tag_id = available_tags[available_tags['display_name'] == selected_tag]['id'].values[0]
                                                if add_client_tag(client_id, tag_id, ont_id="7701234567", assigned_by="Dahi Nemutlu", reason=reason_input):
                                                    st.session_state.show_add_tag_dialog = False
                                                    st.success(f"Added tag: {selected_tag}")
                                                    time.sleep(1)
                                                    st.rerun()
                                        else:
                                            st.info("All available manual tags are already assigned to this client.")
                                            if st.button("Close", key="close_no_tags"):
                                                st.session_state.show_add_tag_dialog = False
                                                st.rerun()
                                    else:
                                        st.info("No manual tags available.")
                                        if st.button("Close", key="close_no_manual_tags"):
                                            st.session_state.show_add_tag_dialog = False
                                            st.rerun()
                        
                        add_tag_dialog()
                elif current == "Assign":
                    st.write("")
                elif current == "SIP":
                    st.write("")
                elif current == "Client Operations":
                    st.write("")
                elif current == "Client Attachments":
                    st.write("")
                elif current == "Client Attachments KYC":
                    st.write("")
                elif current == "Dicigare Tickets":
                    st.write("")
                elif current == "Call Tickets":
                    st.write("")
        
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.title("Welcome to Tailored Offers")
    st.write("This is a prototype application.")
