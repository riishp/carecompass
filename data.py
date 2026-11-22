"""Reads the CareCompass Gold tables from the team's Databricks SQL warehouse.

Locally: export DATABRICKS_CONFIG_PROFILE=datathon (uses your CLI login).
In Databricks Apps: the app's service principal is picked up from the environment automatically,
and DATABRICKS_WAREHOUSE_ID comes from the app's `sql-warehouse` resource (see app.yaml).
"""
import os

import certifi

import pandas as pd
import streamlit as st
from databricks import sql
from databricks.sdk.core import Config

SCHEMA = "workspace.datathon_2026"
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID", "8f7ab3060bb98c92")


@st.cache_resource
def _cfg() -> Config:
    return Config()


def query(statement: str, params: dict | None = None) -> pd.DataFrame:
    cfg = _cfg()
    with sql.connect(server_hostname=cfg.host, http_path=f"/sql/1.0/warehouses/{WAREHOUSE_ID}",
                     credentials_provider=lambda: cfg.authenticate,
                     _tls_trusted_ca_file=certifi.where()) as conn:
        with conn.cursor() as cur:
            cur.execute(statement, params or {})
            return cur.fetchall_arrow().to_pandas()


def _listify(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        df[c] = df[c].map(lambda v: list(v) if v is not None and not isinstance(v, str) else [])
    return df


@st.cache_data(ttl=3600, show_spinner="Loading facilities…")
def facilities() -> pd.DataFrame:
    df = query(f"""
        SELECT unique_id, name, facility_type, operator_type, city, state, district, pincode,
               latitude, longitude, location_confidence, location_vote, location_note,
               specialty_codes, specialty_labels, procedures, equipment, capabilities,
               phone, website, capacity_beds, number_doctors, search_text
        FROM {SCHEMA}.gold_facility_search""")
    return _listify(df, ["specialty_codes", "specialty_labels", "procedures", "equipment", "capabilities"])


@st.cache_data(ttl=3600, show_spinner="Loading districts…")
def districts() -> pd.DataFrame:
    """District need (NFHS) vs supply (facility counts), with a map point for each district."""
    return query(f"""
        SELECT g.*, coalesce(d1.lat, d2.lat) AS lat, coalesce(d1.lon, d2.lon) AS lon
        FROM {SCHEMA}.gold_district_summary g
        LEFT JOIN {SCHEMA}.nfhs_district_crosswalk x ON x.state = g.state AND x.district = g.district
        LEFT JOIN (SELECT * FROM {SCHEMA}.geo_lookup WHERE type = 'district') d1
               ON d1.place = lower(x.pin_district) AND d1.state = g.state
        LEFT JOIN (SELECT * FROM {SCHEMA}.geo_lookup WHERE type = 'district') d2
               ON d2.place = lower(g.district) AND d2.state = g.state""")
