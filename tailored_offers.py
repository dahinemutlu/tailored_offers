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
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode, DataReturnMode

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
    return create_engine(url, pool_pre_ping=True, pool_recycle=3600, future=True)


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


# Load CSS
with open("tailored_offers_theme.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize session state
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Client Tags"

# Honor query param for tab switching
if "tab" in st.query_params:
    requested_tab = st.query_params["tab"]
    if requested_tab in ["Home", "Client Tags", "Requests", "Card", "Client", "CPE", "IVR", "Settings", "Dahi Nemutlu", "Exit"]:
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
    "Client Tags": "label",
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
    if name == "Client Tags":
        active_cls = " call-center-active" if st.session_state.active_tab == name else ""
        tab_items.append(
            f'<a href="?tab=Client%20Tags" target="_self" class="tab{active_cls}">{icon} {name}</a>'
        )
    elif name == "Client":
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

# -------- Second-level tabs for Client Tags (Client Tags / Settings) --------
CLIENT_TAGS_SUBTAB_ICONS = {"Client Tags": "label", "Settings": "settings"}
CLIENT_TAGS_SUBTAB_NAMES = list(CLIENT_TAGS_SUBTAB_ICONS.keys())

# initialize active_subtab for Client Tags (default: Client Tags)
if "active_client_tags_subtab" not in st.session_state:
    st.session_state.active_client_tags_subtab = "Client Tags"

# allow switch via query param
if st.session_state.active_tab == "Client Tags" and "subtab" in st.query_params:
    q = st.query_params["subtab"]
    if q in CLIENT_TAGS_SUBTAB_NAMES:
        st.session_state.active_client_tags_subtab = q

# build subtabs HTML only for Client Tags
if st.session_state.active_tab == "Client Tags":
    base_tab_q = f"?tab={st.session_state.active_tab.replace(' ', '%20')}"
    sub_html = ['<div class="subtabs">']
    for name in CLIENT_TAGS_SUBTAB_NAMES:
        icon = f'<span class="material-icons">{CLIENT_TAGS_SUBTAB_ICONS[name]}</span>'
        cls = " sub-active" if st.session_state.active_client_tags_subtab == name else ""
        sub_html.append(
            f'<a href="{base_tab_q}&subtab={name.replace(" ", "%20")}" target="_self" class="subtab{cls}">{icon} {name}</a>'
        )
    sub_html.append('</div>')
    st.markdown("".join(sub_html), unsafe_allow_html=True)

# Main content area
if st.session_state.active_tab == "Client Tags":
    # Content based on active subtab
    if st.session_state.active_client_tags_subtab == "Client Tags":
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
            
            # Configure selection
            gb.configure_selection(
                selection_mode="multiple",
                use_checkbox=False,
                rowMultiSelectWithClick=True,
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
                
                # Create color lookup string for JS
                color_json = str(color_map).replace("'", '"')
                
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
            
            # Display grid
            grid_response = AgGrid(
                df_view,
                gridOptions=grid_options,
                height=520,
                fit_columns_on_grid_load=False,
                update_mode=GridUpdateMode.MODEL_CHANGED,
                data_return_mode=DataReturnMode.AS_INPUT,
                allow_unsafe_jscode=True,
                theme="balham",
            )
            
            # Show remove button if rows are selected
            selected_rows = grid_response.get("selected_rows", [])
            if selected_rows is not None and len(selected_rows) > 0:
                with col2:
                    st.markdown('<div style="padding-top: 26px;"></div>', unsafe_allow_html=True)
                    if st.button("🗑️ Remove Selected", key="remove_tags_button", type="primary"):
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
                            with message_container:
                                st.success(f"Removed {removed_count} tag(s)")
                            time.sleep(2)
                            st.rerun()
            
    elif st.session_state.active_client_tags_subtab == "Settings":
        # Add custom CSS for compact tag rows and expander styling
        st.markdown("""
            <style>
            [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
                gap: 0.5rem;
            }
            div[data-testid="stHorizontalBlock"] {
                align-items: center;
                gap: 0.5rem;
            }
            /* Style expander headers like subheaders */
            [data-testid="stExpander"] summary {
                font-size: 1.5rem !important;
                font-weight: 600 !important;
                color: rgb(49, 51, 63) !important;
            }
            [data-testid="stExpander"] summary p {
                font-size: 1.5rem !important;
                font-weight: 600 !important;
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Automatic Tags Section
        with st.expander("Automatic Tags", expanded=True):
            # Placeholder for success/info messages
            message_container = st.container()
            
            auto_tags_df = fetch_tag_config(tag_type='A')
            
            if not auto_tags_df.empty:
                for idx, row in auto_tags_df.iterrows():
                    col1, col2, col3, col4 = st.columns([4, 0.7, 0.7, 0.7], gap="small")
                    
                    with col1:
                        new_display_name = st.text_input(
                            "Display Name",
                            value=row['display_name'],
                            key=f"auto_name_{row['id']}",
                            label_visibility="collapsed"
                        )
                    
                    with col2:
                        new_color = st.color_picker(
                            "Color",
                            value=row['color'] if pd.notna(row['color']) else "#6b7280",
                            key=f"auto_color_{row['id']}",
                            label_visibility="collapsed"
                        )
                    
                    with col3:
                        new_is_active = st.checkbox(
                            "Active",
                            value=bool(row['is_active']) if pd.notna(row['is_active']) else True,
                            key=f"auto_active_{row['id']}"
                        )
                    
                    # Description field (read-only, same width as display name)
                    desc_col1, desc_col2 = st.columns([4, 2.1])
                    with desc_col1:
                        st.text_area(
                            "Description",
                            value=row['description'] if pd.notna(row['description']) else "",
                            key=f"auto_desc_{row['id']}",
                            label_visibility="collapsed",
                            height=60,
                            disabled=True,
                            placeholder="No description"
                        )
                    
                    # Check if anything changed
                    changed = False
                    if new_display_name != row['display_name']:
                        changed = True
                    if new_color != row['color']:
                        changed = True
                    if new_is_active != bool(row['is_active']):
                        changed = True
                    
                    with col4:
                        if changed:
                            if st.button("Save", key=f"auto_save_{row['id']}", width='stretch'):
                                success = update_tag_config(
                                    tag_id=row['id'],
                                    display_name=new_display_name,
                                    color=new_color,
                                    is_active=new_is_active
                                )
                                if success:
                                    with message_container:
                                        st.success(f"Updated {new_display_name}")
                                    time.sleep(2)  # Show message for 2 seconds
                                    st.rerun()
                        else:
                            st.write("")  # Empty placeholder to maintain layout
                    
                    st.markdown("<hr style='margin: 0.5rem 0; border: 0; border-top: 1px solid rgba(0,0,0,0.1);' />", unsafe_allow_html=True)
            else:
                st.info("No automatic tags found.")
        
        # Manual Tags Section
        with st.expander("Manual Tags", expanded=True):
            # Placeholder for success/info messages
            manual_message_container = st.container()
            
            # Add new tag button and form
            if st.button("➕ Add New Manual Tag", key="add_manual_tag_btn"):
                st.session_state.show_add_manual_tag_form = True
            
            # Show add tag form if button was clicked
            if st.session_state.get("show_add_manual_tag_form", False):
                with st.form("add_manual_tag_form"):
                    st.write("**Create New Manual Tag**")
                    
                    form_col1, form_col2, form_col3 = st.columns([3, 0.7, 0.7])
                    with form_col1:
                        new_tag_name = st.text_input("Tag Name", placeholder="Enter tag name...")
                    
                    with form_col2:
                        new_tag_color = st.color_picker("Color", value="#6b7280")
                    
                    with form_col3:
                        new_tag_active = st.checkbox("Active", value=True)
                    
                    # Description below (same width as tag name)
                    desc_form_col1, desc_form_col2 = st.columns([3, 1.4])
                    with desc_form_col1:
                        new_tag_description = st.text_area("Description", placeholder="Enter tag description...", height=60)
                    
                    form_col_submit, form_col_cancel, form_col_spacer = st.columns([1, 1, 2])
                    with form_col_submit:
                        submit_btn = st.form_submit_button("Create Tag", width='stretch')
                    with form_col_cancel:
                        cancel_btn = st.form_submit_button("Cancel", width='stretch')
                    
                    if submit_btn:
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
                                st.session_state.show_add_manual_tag_form = False
                                with manual_message_container:
                                    st.success(f"Created new tag: {new_tag_name}")
                                time.sleep(2)
                                st.rerun()
                        else:
                            st.error("Tag Name is required!")
                    
                    if cancel_btn:
                        st.session_state.show_add_manual_tag_form = False
                        st.rerun()
                
                st.markdown("<hr style='margin: 1rem 0; border: 0; border-top: 2px solid rgba(0,0,0,0.2);' />", unsafe_allow_html=True)
            
            manual_tags_df = fetch_tag_config(tag_type='M')
            
            if not manual_tags_df.empty:
                for idx, row in manual_tags_df.iterrows():
                    col1, col2, col3, col4 = st.columns([4, 0.7, 0.7, 0.7], gap="small")
                    
                    with col1:
                        new_display_name = st.text_input(
                            "Display Name",
                            value=row['display_name'],
                            key=f"manual_name_{row['id']}",
                            label_visibility="collapsed"
                        )
                    
                    with col2:
                        new_color = st.color_picker(
                            "Color",
                            value=row['color'] if pd.notna(row['color']) else "#6b7280",
                            key=f"manual_color_{row['id']}",
                            label_visibility="collapsed"
                        )
                    
                    with col3:
                        new_is_active = st.checkbox(
                            "Active",
                            value=bool(row['is_active']) if pd.notna(row['is_active']) else True,
                            key=f"manual_active_{row['id']}"
                        )
                    
                    # Description field (editable, same width as display name)
                    desc_col1, desc_col2 = st.columns([4, 2.1])
                    with desc_col1:
                        new_description = st.text_area(
                            "Description",
                            value=row['description'] if pd.notna(row['description']) else "",
                            key=f"manual_desc_{row['id']}",
                            label_visibility="collapsed",
                            height=60,
                            placeholder="Enter tag description..."
                        )
                    
                    # Check if anything changed
                    changed = False
                    if new_display_name != row['display_name']:
                        changed = True
                    if new_color != row['color']:
                        changed = True
                    if new_is_active != bool(row['is_active']):
                        changed = True
                    if new_description != (row['description'] if pd.notna(row['description']) else ""):
                        changed = True
                    
                    with col4:
                        if changed:
                            if st.button("Save", key=f"manual_save_{row['id']}", width='stretch'):
                                success = update_tag_config(
                                    tag_id=row['id'],
                                    display_name=new_display_name,
                                    color=new_color,
                                    is_active=new_is_active,
                                    description=new_description
                                )
                                if success:
                                    with manual_message_container:
                                        st.success(f"Updated {new_display_name}")
                                    time.sleep(2)  # Show message for 2 seconds
                                    st.rerun()
                        else:
                            st.write("")  # Empty placeholder to maintain layout
                    
                    st.markdown("<hr style='margin: 0.5rem 0; border: 0; border-top: 1px solid rgba(0,0,0,0.1);' />", unsafe_allow_html=True)
            else:
                st.info("No manual tags found.")
elif st.session_state.active_tab == "Client":
    # Client page layout: two columns
    st.markdown('<div id="client-layout">', unsafe_allow_html=True)
    left, right = st.columns([2, 5], gap=None)
    with left:
        # Native Streamlit version of the client card (no custom HTML)
        st.subheader("Dahi Nemutlu")

        client_id = 22000
        
        client_rows = [
            ("ID", "22000"),
            ("Client Type", "Employee"),
            ("Account Number", "7701234567"),
            ("Service ID", "91000000"),
            ("Sip", "This Client has no Sip."),
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
                # Display existing tags as colored badges with remove button for all tags
                for idx, tag_row in client_tags_df.iterrows():
                    tag_color = tag_row.get('color', '#6b7280')
                    tag_name = tag_row.get('display_name', 'Unknown')
                    tag_type = tag_row.get('tag_type', '')
                    tag_id = tag_row.get('tag_id')
                    text_color = get_contrast_color(tag_color)
                    
                    # Create columns for tag badge and remove button (for all tags)
                    tag_col, btn_col = st.columns([4, 1], gap="small")
                    with tag_col:
                        st.markdown(
                            f'<span style="display:inline-block;background:{tag_color};color:{text_color};'
                            f'padding:4px 10px;border-radius:12px;font-size:14px;">{tag_name}</span>',
                            unsafe_allow_html=True
                        )
                    with btn_col:
                        if st.button("×", key=f"remove_tag_{tag_id}", help=f"Remove {tag_name}"):
                            if remove_client_tag(client_id, tag_id):
                                tag_message = f"Removed tag: {tag_name}"
                                tag_message_type = "success"
            else:
                st.write("—")
            
            # Add manual tag selector
            manual_tags_df = fetch_tag_config(tag_type='M')
            if not manual_tags_df.empty:
                # Filter out already assigned tags
                assigned_tag_ids = client_tags_df['tag_id'].tolist() if not client_tags_df.empty else []
                available_tags = manual_tags_df[~manual_tags_df['id'].isin(assigned_tag_ids)]
                
                if not available_tags.empty:
                    tag_options = ["Select to add..."] + available_tags['display_name'].tolist()
                    selected_tag = st.selectbox(
                        "Add manual tag",
                        options=tag_options,
                        index=0,
                        key="add_manual_tag_selector",
                        label_visibility="collapsed"
                    )
                    
                    # Show reason input if a tag is selected
                    if selected_tag != "Select to add...":
                        reason_input = st.text_input(
                            "Reason (optional)",
                            key="tag_reason_input",
                            placeholder="Enter reason for adding this tag...",
                            label_visibility="collapsed"
                        )
                        
                        if st.button("Add Tag", key="add_tag_button"):
                            # Get the tag_id for the selected tag
                            tag_id = available_tags[available_tags['display_name'] == selected_tag]['id'].values[0]
                            if add_client_tag(client_id, tag_id, ont_id="7701234567", assigned_by="Dahi Nemutlu", reason=reason_input):
                                tag_message = f"Added tag: {selected_tag}"
                                tag_message_type = "success"
            
            # Message container below the dropdown
            message_container = st.container()
            with message_container:
                if tag_message:
                    if tag_message_type == "success":
                        st.success(tag_message)
                        time.sleep(2)
                        st.rerun()
        
        st.markdown('<hr style="margin:0;border:0;border-top:1px solid rgba(0,0,0,0.10);" />', unsafe_allow_html=True)
        
        # Remaining rows (after Tags)
        remaining_rows = [
            ("City", ""),
            ("Address", ""),
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
            "Assign": "assignment_turned_in",
            "SIP": "dialer_sip",
            "Client Operations": "assignment",
            "Client Attachments": "attachment",
            "Client Attachments KYC": "fingerprint",
            "Dicigare Tickets": "report_problem",
            "Call Tickets": "phone",
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

        base_q = "?tab=Client"
        sub_html = ['<div class="subtabs" style="margin-left:12px;">']
        for i, name in enumerate(CLIENT_SUBTAB_NAMES):
            icon = f'<span class="material-icons">{CLIENT_SUBTAB_ICONS[name]}</span>'
            cls = " sub-active" if st.session_state.active_client_subtab == name else ""
            if i == 0:
                # Only the first subtab is clickable
                sub_html.append(
                    f'<a href="{base_q}&client_subtab={name.replace(" ", "%20")}" target="_self" class="subtab{cls}">{icon} {name}</a>'
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
