"""
=============================================================================
⚡ ENERGY PORTFOLIO BACKTESTING & OPTIMIZATION SYSTEM
=============================================================================
Module 1: Data Architecture · Upload Engine · Validation · Shaping · Explorer
=============================================================================
Institutional-Grade Decision Support for Energy Procurement
Harvard / MIT Quantitative Finance Level
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from datetime import datetime, timedelta, date
import io
import json
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# 1. KONFIGURATION & KONSTANTEN
# ============================================================================

class Config:
    """Zentrale Konfiguration – Single Source of Truth."""

    APP_TITLE = "⚡ Energy Portfolio Backtesting System"
    APP_SUBTITLE = "Institutional-Grade Decision Support · Quantitative Procurement Analytics"
    VERSION = "1.0.0 – Module 1: Data Foundation"

    # Zeitliche Parameter
    BASE_HOURS_PER_DAY = 24
    PEAK_START = 8
    PEAK_END = 20
    PEAK_HOURS_PER_DAY = 12
    OFFPEAK_HOURS_PER_DAY = 12
    WEEKEND_DAYS = [5, 6]

    # Validierungsgrenzen
    SPOT_PRICE_MIN = -500.0
    SPOT_PRICE_MAX = 15000.0
    SETTLEMENT_PRICE_MIN = 0.0
    SETTLEMENT_PRICE_MAX = 5000.0
    MAX_VOLUME_MW = 10000.0

    # Farbschema (Plotly-kompatibel)
    COLORS = {
        "spot": "#e74c3c",
        "forward": "#3498db",
        "actual": "#2ecc71",
        "hedge": "#9b59b6",
        "unhedged": "#e67e22",
        "danger": "#e74c3c",
        "warning": "#f39c12",
        "success": "#27ae60",
        "info": "#2980b9",
        "neutral": "#95a5a6",
        "bg_card": "rgba(30, 39, 73, 0.6)",
    }

    # Standard-Seitenkonfiguration
    PAGE_CONFIG = dict(
        page_title="Energy Portfolio Backtesting",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )


# ============================================================================
# 2. CUSTOM CSS & STYLING
# ============================================================================

CUSTOM_CSS = """
<style>
    /* ---- Hauptüberschrift ---- */
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #3498db 0%, #e74c3c 50%, #f39c12 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
        margin-bottom: 0;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #7f8c8d;
        margin-top: -8px;
        margin-bottom: 25px;
    }
    /* ---- Metrikkarten ---- */
    .metric-row {
        display: flex; gap: 12px; flex-wrap: wrap; margin: 10px 0;
    }
    .metric-card {
        flex: 1; min-width: 160px;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2c3e6b; border-radius: 12px;
        padding: 18px 16px; text-align: center;
    }
    .metric-card .label { color: #7f8c8d; font-size: 0.8rem; text-transform: uppercase; }
    .metric-card .value { color: #ecf0f1; font-size: 1.6rem; font-weight: 700; }
    .metric-card .delta-pos { color: #2ecc71; font-size: 0.85rem; }
    .metric-card .delta-neg { color: #e74c3c; font-size: 0.85rem; }
    /* ---- Status-Badges ---- */
    .badge-ok { 
        display: inline-block; padding: 3px 10px; border-radius: 12px;
        background: rgba(39,174,96,0.2); color: #2ecc71; font-weight: 600; font-size: 0.82rem;
    }
    .badge-miss {
        display: inline-block; padding: 3px 10px; border-radius: 12px;
        background: rgba(231,76,60,0.2); color: #e74c3c; font-weight: 600; font-size: 0.82rem;
    }
    .badge-warn {
        display: inline-block; padding: 3px 10px; border-radius: 12px;
        background: rgba(243,156,18,0.2); color: #f39c12; font-weight: 600; font-size: 0.82rem;
    }
    /* ---- Validation Messages ---- */
    .val-error { background: #2c0b0e; border-left: 4px solid #e74c3c; padding: 10px 14px; margin: 4px 0; border-radius: 0 6px 6px 0; }
    .val-warn  { background: #2c2200; border-left: 4px solid #f39c12; padding: 10px 14px; margin: 4px 0; border-radius: 0 6px 6px 0; }
    .val-info  { background: #0a1628; border-left: 4px solid #3498db; padding: 10px 14px; margin: 4px 0; border-radius: 0 6px 6px 0; }
    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0e1117 0%, #1a1a2e 100%); }
    /* ---- Divider ---- */
    .section-divider { border-top: 1px solid #2c3e50; margin: 30px 0; }
</style>
"""


# ============================================================================
# 3. SAMPLE DATA GENERATOR  –  Realistische synthetische Marktdaten
# ============================================================================

class SampleDataGenerator:
    """
    Erzeugt realistische Energiemarktdaten mit:
    - Ornstein-Uhlenbeck Mean Reversion (Spotpreis)
    - Saisonalität (Jahreszeit, Wochentag, Tageszeit)
    - Energiekrise 2022/23 Simulation
    - Jump-Diffusion (Preisspitzen)
    - Korrelierte Forward-Kurven
    """

    @staticmethod
    @st.cache_data(show_spinner=False)
    def generate_spot_prices(
        start: str = "2022-01-01",
        end: str = "2024-12-31",
        granularity: str = "hourly",
        seed: int = 42,
    ) -> pd.DataFrame:
        np.random.seed(seed)

        freq_map = {"15min": "15min", "hourly": "h", "daily": "D"}
        timestamps = pd.date_range(start=start, end=end, freq=freq_map[granularity])
        n = len(timestamps)

        # --- Ornstein-Uhlenbeck Prozess ---
        theta = 0.15          # Mean-Reversion-Geschwindigkeit
        mu_base = 65.0        # Langfrist-Mittelwert (EUR/MWh)
        sigma = 12.0          # Volatilität
        dt = {"15min": 1/96, "hourly": 1/24, "daily": 1.0}[granularity]

        prices = np.zeros(n)
        prices[0] = mu_base

        for i in range(1, n):
            dW = np.random.normal(0, np.sqrt(dt))
            prices[i] = prices[i - 1] + theta * (mu_base - prices[i - 1]) * dt + sigma * dW

        # --- Saisonalität ---
        hour = timestamps.hour if granularity != "daily" else np.zeros(n)
        month = timestamps.month
        dow = timestamps.dayofweek

        # Tagesgang
        hourly_shape = np.where(
            (hour >= 8) & (hour < 20), 18.0,
            np.where((hour >= 6) & (hour < 8), 8.0, -8.0),
        )
        # Monatssaisonalität (Winter teurer)
        monthly_shape = 22.0 * np.cos(2 * np.pi * (month - 1) / 12)
        # Wochenende billiger
        weekend_shape = np.where(dow >= 5, -18.0, 0.0)

        # --- Energiekrise 2022/23 ---
        crisis = np.zeros(n)
        for i, ts in enumerate(timestamps):
            ts_dt = ts.to_pydatetime()
            if datetime(2022, 6, 1) <= ts_dt <= datetime(2023, 4, 30):
                progress = (ts_dt - datetime(2022, 6, 1)).days / 335.0
                crisis[i] = 180.0 * np.sin(np.pi * progress) + 60.0

        # --- Jump-Diffusion (Preisspitzen) ---
        jump_prob = 0.002 if granularity == "hourly" else 0.02
        jumps = (np.random.random(n) < jump_prob) * np.random.exponential(80, n)
        neg_jumps = (np.random.random(n) < jump_prob * 0.3) * np.random.exponential(40, n)

        # Zusammenführung
        prices = prices + hourly_shape + monthly_shape + weekend_shape + crisis + jumps - neg_jumps
        prices = np.maximum(prices, -100.0)  # Negative Preise möglich, aber mit Floor

        return pd.DataFrame({
            "timestamp": timestamps,
            "price_eur_mwh": np.round(prices, 2),
        })

    @staticmethod
    @st.cache_data(show_spinner=False)
    def generate_settlement_prices(
        start: str = "2022-01-01",
        end: str = "2024-12-31",
        seed: int = 123,
    ) -> pd.DataFrame:
        np.random.seed(seed)
        trade_dates = pd.date_range(start=start, end=end, freq="B")
        records = []

        base_level = 68.0

        for td in trade_dates:
            td_dt = td.to_pydatetime()
            # Langsamer Random Walk des Basisniveaus
            base_level += np.random.normal(0, 0.3)
            base_level = max(base_level, 20)

            # Krise
            crisis_add = 0.0
            if datetime(2022, 6, 1) <= td_dt <= datetime(2023, 4, 30):
                progress = (td_dt - datetime(2022, 6, 1)).days / 335.0
                crisis_add = 150.0 * np.sin(np.pi * progress) + 50.0

            # Monatsprodukte (nächste 6 Monate)
            for m_off in range(1, 7):
                del_start = (td + pd.DateOffset(months=m_off)).replace(day=1)
                del_end = (del_start + pd.DateOffset(months=1)) - timedelta(days=1)

                seasonal = 18.0 * np.cos(2 * np.pi * (del_start.month - 1) / 12)
                time_premium = m_off * 0.8
                noise = np.random.normal(0, 2.5)

                price_base = max(base_level + seasonal + time_premium + crisis_add + noise, 8.0)
                price_peak = price_base * (1.25 + np.random.normal(0, 0.03))

                product_name = f"M-{del_start.strftime('%Y-%m')}"

                for ptype, price in [("Base", price_base), ("Peak", price_peak)]:
                    records.append({
                        "trade_date": td,
                        "product": product_name,
                        "product_type": ptype,
                        "delivery_start": del_start,
                        "delivery_end": del_end,
                        "settlement_price": round(price, 2),
                    })

            # Quartalsprodukte (nächste 4 Quartale)
            for q_off in range(1, 5):
                current_q = (td_dt.month - 1) // 3
                target_q = current_q + q_off
                target_year = td_dt.year + target_q // 4
                target_q_mod = target_q % 4
                q_start_month = target_q_mod * 3 + 1

                del_start = datetime(target_year, q_start_month, 1)
                del_end = (del_start + pd.DateOffset(months=3)) - timedelta(days=1)

                seasonal = 15.0 * np.cos(2 * np.pi * (q_start_month - 1) / 12)
                price_base = max(base_level + seasonal + crisis_add + np.random.normal(0, 2), 8.0)

                records.append({
                    "trade_date": td,
                    "product": f"Q-{target_year}-Q{target_q_mod + 1}",
                    "product_type": "Base",
                    "delivery_start": del_start,
                    "delivery_end": del_end,
                    "settlement_price": round(price_base, 2),
                })

        return pd.DataFrame(records)

    @staticmethod
    @st.cache_data(show_spinner=False)
    def generate_purchases(
        start: str = "2022-01-01",
        end: str = "2024-12-31",
        seed: int = 456,
    ) -> pd.DataFrame:
        np.random.seed(seed)
        records = []

        months = pd.date_range(start=start, end=end, freq="MS")

        for m_start in months:
            m_dt = m_start.to_pydatetime()

            # 1-3 Forward-Tranchen pro Monat
            n_tranches = np.random.randint(1, 4)
            for _ in range(n_tranches):
                del_offset = np.random.randint(1, 7)
                del_start = (m_start + pd.DateOffset(months=del_offset)).replace(day=1)
                del_end = (del_start + pd.DateOffset(months=1)) - timedelta(days=1)

                volume = float(np.random.choice([5, 10, 15, 20, 25, 30]))
                base_price = 62.0 + np.random.normal(0, 8)

                crisis = 0.0
                if datetime(2022, 6, 1) <= m_dt <= datetime(2023, 4, 30):
                    progress = (m_dt - datetime(2022, 6, 1)).days / 335.0
                    crisis = 140.0 * np.sin(np.pi * progress) + 40.0

                price = max(base_price + crisis, 12.0)

                records.append({
                    "purchase_date": m_start + timedelta(days=np.random.randint(0, 25)),
                    "product": f"M-{del_start.strftime('%Y-%m')}",
                    "volume_mw": volume,
                    "price_eur_mwh": round(price, 2),
                    "delivery_start": del_start,
                    "delivery_end": del_end,
                    "purchase_type": "forward",
                })

        return pd.DataFrame(records)

    @staticmethod
    @st.cache_data(show_spinner=False)
    def generate_load_profile(
        start: str = "2022-01-01",
        end: str = "2024-12-31",
        seed: int = 789,
    ) -> pd.DataFrame:
        np.random.seed(seed)
        timestamps = pd.date_range(start=start, end=end, freq="h")
        n = len(timestamps)

        base_load = 50.0
        hour = timestamps.hour
        month = timestamps.month
        dow = timestamps.dayofweek

        hourly = np.where((hour >= 6) & (hour < 22), 20.0, 0.0) + \
                 np.where((hour >= 9) & (hour < 17), 12.0, 0.0)
        seasonal = 10.0 * np.cos(2 * np.pi * (month - 1) / 12)
        weekend = np.where(dow >= 5, -18.0, 0.0)
        noise = np.random.normal(0, 2.5, n)

        volume = np.maximum(base_load + hourly + seasonal + weekend + noise, 8.0)

        return pd.DataFrame({
            "timestamp": timestamps,
            "volume_mwh": np.round(volume, 2),
        })


# ============================================================================
# 4. DATA VALIDATION ENGINE  –  Mehrstufige Prüfung mit detailliertem Report
# ============================================================================

@dataclass
class ValidationIssue:
    severity: str          # ERROR, WARNING, INFO
    message: str
    affected_rows: int
    details: str = ""

    @property
    def icon(self) -> str:
        return {"ERROR": "❌", "WARNING": "⚠️", "INFO": "ℹ️"}.get(self.severity, "❓")

    @property
    def css_class(self) -> str:
        return {"ERROR": "val-error", "WARNING": "val-warn", "INFO": "val-info"}.get(
            self.severity, "val-info"
        )


class DataValidator:
    """Institutionelle Datenvalidierung mit detailliertem Reporting."""

    @staticmethod
    def validate_spot_prices(df: pd.DataFrame) -> Tuple[bool, List[ValidationIssue], pd.DataFrame]:
        issues: List[ValidationIssue] = []
        df_c = df.copy()

        # --- Spaltenprüfung ---
        required = ["timestamp", "price_eur_mwh"]
        missing = [c for c in required if c not in df_c.columns]
        if missing:
            issues.append(ValidationIssue("ERROR", f"Fehlende Spalten: {missing}", len(df_c)))
            return False, issues, df_c

        # --- Timestamp-Parsing ---
        try:
            df_c["timestamp"] = pd.to_datetime(df_c["timestamp"], dayfirst=False)
        except Exception as e:
            issues.append(ValidationIssue("ERROR", f"Timestamp-Parsing fehlgeschlagen: {e}", len(df_c)))
            return False, issues, df_c

        # --- Numerische Konvertierung ---
        df_c["price_eur_mwh"] = pd.to_numeric(df_c["price_eur_mwh"], errors="coerce")

        # --- Duplikate ---
        dupes = df_c.duplicated(subset=["timestamp"], keep="first")
        if dupes.any():
            issues.append(ValidationIssue("WARNING", f"Doppelte Timestamps entfernt", int(dupes.sum())))
            df_c = df_c[~dupes].copy()

        # --- Fehlende Werte ---
        nulls = df_c["price_eur_mwh"].isna().sum()
        if nulls > 0:
            issues.append(ValidationIssue("WARNING", f"Fehlende Preise interpoliert", int(nulls)))
            df_c = df_c.sort_values("timestamp").reset_index(drop=True)
            df_c["price_eur_mwh"] = df_c["price_eur_mwh"].interpolate(method="linear")

        # --- Ausreißer ---
        below = (df_c["price_eur_mwh"] < Config.SPOT_PRICE_MIN).sum()
        above = (df_c["price_eur_mwh"] > Config.SPOT_PRICE_MAX).sum()
        if below + above > 0:
            issues.append(
                ValidationIssue(
                    "WARNING",
                    f"{below + above} Preis-Ausreißer erkannt "
                    f"(< {Config.SPOT_PRICE_MIN} oder > {Config.SPOT_PRICE_MAX} €/MWh)",
                    int(below + above),
                )
            )

        # --- Zeitliche Kontinuität ---
        df_c = df_c.sort_values("timestamp").reset_index(drop=True)
        if len(df_c) > 1:
            diffs = df_c["timestamp"].diff().dropna()
            mode_freq = diffs.mode().iloc[0] if len(diffs.mode()) > 0 else pd.Timedelta("1h")
            gaps = diffs[diffs > mode_freq * 2]
            if len(gaps) > 0:
                issues.append(
                    ValidationIssue(
                        "INFO",
                        f"{len(gaps)} Zeitlücken erkannt (erwartete Frequenz: {mode_freq})",
                        len(gaps),
                    )
                )

            # Granularität ableiten
            granularity = "hourly"
            if mode_freq <= pd.Timedelta("20min"):
                granularity = "15min"
            elif mode_freq >= pd.Timedelta("20h"):
                granularity = "daily"
            issues.append(ValidationIssue("INFO", f"Erkannte Granularität: {granularity}", 0))

        is_valid = not any(i.severity == "ERROR" for i in issues)
        return is_valid, issues, df_c

    @staticmethod
    def validate_settlement_prices(df: pd.DataFrame) -> Tuple[bool, List[ValidationIssue], pd.DataFrame]:
        issues: List[ValidationIssue] = []
        df_c = df.copy()

        required = ["trade_date", "product", "product_type", "delivery_start", "delivery_end", "settlement_price"]
        missing = [c for c in required if c not in df_c.columns]
        if missing:
            issues.append(ValidationIssue("ERROR", f"Fehlende Spalten: {missing}", len(df_c)))
            return False, issues, df_c

        for col in ["trade_date", "delivery_start", "delivery_end"]:
            try:
                df_c[col] = pd.to_datetime(df_c[col])
            except Exception as e:
                issues.append(ValidationIssue("ERROR", f"{col}-Parsing fehlgeschlagen: {e}", len(df_c)))
                return False, issues, df_c

        df_c["settlement_price"] = pd.to_numeric(df_c["settlement_price"], errors="coerce")

        bad_dates = (df_c["trade_date"] >= df_c["delivery_start"]).sum()
        if bad_dates > 0:
            issues.append(
                ValidationIssue("WARNING", f"{bad_dates} Einträge: Handelsdatum ≥ Lieferbeginn", int(bad_dates))
            )

        n_products = df_c["product"].nunique()
        n_trade_days = df_c["trade_date"].nunique()
        issues.append(ValidationIssue("INFO", f"{n_products} Produkte über {n_trade_days} Handelstage", 0))

        is_valid = not any(i.severity == "ERROR" for i in issues)
        return is_valid, issues, df_c

    @staticmethod
    def validate_purchases(df: pd.DataFrame) -> Tuple[bool, List[ValidationIssue], pd.DataFrame]:
        issues: List[ValidationIssue] = []
        df_c = df.copy()

        required = ["purchase_date", "product", "volume_mw", "price_eur_mwh",
                     "delivery_start", "delivery_end", "purchase_type"]
        missing = [c for c in required if c not in df_c.columns]
        if missing:
            issues.append(ValidationIssue("ERROR", f"Fehlende Spalten: {missing}", len(df_c)))
            return False, issues, df_c

        for col in ["purchase_date", "delivery_start", "delivery_end"]:
            try:
                df_c[col] = pd.to_datetime(df_c[col])
            except Exception as e:
                issues.append(ValidationIssue("ERROR", f"{col}-Parsing fehlgeschlagen: {e}", len(df_c)))
                return False, issues, df_c

        df_c["volume_mw"] = pd.to_numeric(df_c["volume_mw"], errors="coerce")
        df_c["price_eur_mwh"] = pd.to_numeric(df_c["price_eur_mwh"], errors="coerce")

        neg = (df_c["volume_mw"] <= 0).sum()
        if neg > 0:
            issues.append(ValidationIssue("WARNING", f"{neg} Einträge mit Volumen ≤ 0", int(neg)))

        valid_types = {"forward", "spot"}
        bad_types = ~df_c["purchase_type"].str.lower().isin(valid_types)
        if bad_types.any():
            issues.append(ValidationIssue("WARNING", f"{bad_types.sum()} ungültige Kauftypen", int(bad_types.sum())))

        fwd_count = (df_c["purchase_type"].str.lower() == "forward").sum()
        spot_count = (df_c["purchase_type"].str.lower() == "spot").sum()
        issues.append(ValidationIssue("INFO", f"{fwd_count} Forward- und {spot_count} Spot-Transaktionen", 0))

        is_valid = not any(i.severity == "ERROR" for i in issues)
        return is_valid, issues, df_c

    @staticmethod
    def validate_load_profile(df: pd.DataFrame) -> Tuple[bool, List[ValidationIssue], pd.DataFrame]:
        issues: List[ValidationIssue] = []
        df_c = df.copy()

        required = ["timestamp", "volume_mwh"]
        missing = [c for c in required if c not in df_c.columns]
        if missing:
            issues.append(ValidationIssue("ERROR", f"Fehlende Spalten: {missing}", len(df_c)))
            return False, issues, df_c

        try:
            df_c["timestamp"] = pd.to_datetime(df_c["timestamp"])
        except Exception as e:
            issues.append(ValidationIssue("ERROR", f"Timestamp-Parsing: {e}", len(df_c)))
            return False, issues, df_c

        df_c["volume_mwh"] = pd.to_numeric(df_c["volume_mwh"], errors="coerce")

        total_mwh = df_c["volume_mwh"].sum()
        avg_mw = df_c["volume_mwh"].mean()
        issues.append(
            ValidationIssue("INFO", f"Gesamtverbrauch: {total_mwh:,.0f} MWh | Ø Last: {avg_mw:.1f} MW", 0)
        )

        is_valid = not any(i.severity == "ERROR" for i in issues)
        return is_valid, issues, df_c


# ============================================================================
# 5. SHAPING ENGINE  –  Terminmarkt → Stündliche Profile
# ============================================================================

class ShapingEngine:
    """
    Zerlegt Terminmarkt-Standardprodukte (Base/Peak/OffPeak) in
    stündliche Lieferprofile für mathematisches Matching mit Spot & Last.

    Berücksichtigt:
    - Wochentage vs. Wochenende
    - Peak-/OffPeak-Definition (konfigurierbar)
    - Feiertage (optional erweiterbar)
    """

    @staticmethod
    def shape_product(
        product_type: str,
        delivery_start: datetime,
        delivery_end: datetime,
        price: float,
        volume_mw: float,
        peak_start: int = Config.PEAK_START,
        peak_end: int = Config.PEAK_END,
    ) -> pd.DataFrame:
        """Einzelnes Produkt → stündliches Profil."""

        hours = pd.date_range(
            start=delivery_start,
            end=pd.Timestamp(delivery_end) + timedelta(days=1),
            freq="h",
            inclusive="left",
        )

        df = pd.DataFrame({"timestamp": hours})
        df["hour"] = df["timestamp"].dt.hour
        df["dow"] = df["timestamp"].dt.dayofweek

        is_weekday = df["dow"] < 5
        is_peak_hour = (df["hour"] >= peak_start) & (df["hour"] < peak_end)

        ptype = product_type.strip().lower()
        if ptype == "base":
            df["is_delivery"] = True
        elif ptype == "peak":
            df["is_delivery"] = is_weekday & is_peak_hour
        elif ptype in ("offpeak", "off-peak"):
            df["is_delivery"] = ~(is_weekday & is_peak_hour)
        else:
            df["is_delivery"] = True  # Fallback

        df["price_eur_mwh"] = np.where(df["is_delivery"], price, 0.0)
        df["volume_mw"] = np.where(df["is_delivery"], volume_mw, 0.0)
        df["volume_mwh"] = df["volume_mw"]  # 1 Stunde → MW = MWh
        df["cost_eur"] = df["price_eur_mwh"] * df["volume_mwh"]

        return df[["timestamp", "is_delivery", "price_eur_mwh", "volume_mw", "volume_mwh", "cost_eur"]]

    @staticmethod
    def shape_all_purchases(
        purchases: pd.DataFrame,
        peak_start: int = Config.PEAK_START,
        peak_end: int = Config.PEAK_END,
    ) -> pd.DataFrame:
        """Alle Forward-Käufe → aggregiertes stündliches Profil."""

        fwd = purchases[purchases["purchase_type"].str.lower() == "forward"].copy()
        if fwd.empty:
            return pd.DataFrame(columns=["timestamp", "volume_mw", "volume_mwh", "cost_eur", "avg_price_eur_mwh"])

        all_shaped = []

        for idx, row in fwd.iterrows():
            shaped = ShapingEngine.shape_product(
                product_type="Base",
                delivery_start=row["delivery_start"],
                delivery_end=row["delivery_end"],
                price=row["price_eur_mwh"],
                volume_mw=row["volume_mw"],
                peak_start=peak_start,
                peak_end=peak_end,
            )
            shaped["purchase_idx"] = idx
            all_shaped.append(shaped)

        combined = pd.concat(all_shaped, ignore_index=True)

        agg = (
            combined.groupby("timestamp")
            .agg(volume_mw=("volume_mw", "sum"),
                 volume_mwh=("volume_mwh", "sum"),
                 cost_eur=("cost_eur", "sum"))
            .reset_index()
        )
        agg["avg_price_eur_mwh"] = (agg["cost_eur"] / agg["volume_mwh"].replace(0, np.nan)).round(2)

        return agg


# ============================================================================
# 6. SESSION STATE MANAGER  –  Bitemporale Datenhaltung
# ============================================================================

class DataStore:
    """Verwaltet alle Daten in Streamlit Session State."""

    KEYS = [
        "spot_prices", "settlement_prices", "purchases",
        "load_profile", "shaped_purchases",
        "data_loaded", "using_sample_data", "validation_log",
    ]

    @classmethod
    def init(cls):
        for key in cls.KEYS:
            if key not in st.session_state:
                st.session_state[key] = None if key not in ("data_loaded", "using_sample_data") else False
        if "validation_log" not in st.session_state:
            st.session_state["validation_log"] = {}

    @staticmethod
    def store(key: str, df: pd.DataFrame):
        st.session_state[key] = df

    @staticmethod
    def get(key: str) -> Optional[pd.DataFrame]:
        return st.session_state.get(key)

    @staticmethod
    def status() -> Dict[str, bool]:
        return {
            "Spotpreise": st.session_state.get("spot_prices") is not None,
            "Settlements": st.session_state.get("settlement_prices") is not None,
            "Beschaffungen": st.session_state.get("purchases") is not None,
            "Lastprofil": st.session_state.get("load_profile") is not None,
            "Shaped Fwd": st.session_state.get("shaped_purchases") is not None,
        }

    @staticmethod
    def clear_all():
        for key in DataStore.KEYS:
            st.session_state[key] = None if key not in ("data_loaded", "using_sample_data") else False


# ============================================================================
# 7. HELPER: Datei einlesen (CSV / Excel, Auto-Separator-Erkennung)
# ============================================================================

def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Liest CSV oder Excel mit automatischer Separator-Erkennung."""
    name = uploaded_file.name.lower()

    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)

    # CSV: Separator erkennen
    raw = uploaded_file.read()
    uploaded_file.seek(0)

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
        uploaded_file.seek(0)

    # Heuristik: Welcher Separator kommt häufiger vor?
    semicolons = text[:5000].count(";")
    commas = text[:5000].count(",")
    tabs = text[:5000].count("\t")
    sep = ";" if semicolons > commas and semicolons > tabs else ("\t" if tabs > commas else ",")

    # Dezimaltrennzeichen
    decimal = "," if sep == ";" else "."

    try:
        return pd.read_csv(uploaded_file, sep=sep, decimal=decimal)
    except Exception:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file)


# ============================================================================
# 8. RENDERING FUNKTIONEN
# ============================================================================

def render_validation_issues(issues: List[ValidationIssue]):
    """Zeigt Validierungsergebnisse formatiert an."""
    for issue in issues:
        css = issue.css_class
        st.markdown(
            f'<div class="{css}">{issue.icon} <b>{issue.severity}</b>: '
            f'{issue.message} '
            f'{"(" + str(issue.affected_rows) + " Zeilen)" if issue.affected_rows > 0 else ""}'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_data_quality_score(issues: List[ValidationIssue], total_rows: int) -> float:
    """Berechnet und zeigt einen Datenqualitätsscore."""
    if total_rows == 0:
        return 0.0

    error_rows = sum(i.affected_rows for i in issues if i.severity == "ERROR")
    warning_rows = sum(i.affected_rows for i in issues if i.severity == "WARNING")

    # Score: 100% - Fehleranteil - halber Warnanteil
    score = max(0, 100 - (error_rows / total_rows * 100) - (warning_rows / total_rows * 50))

    color = Config.COLORS["success"] if score >= 80 else Config.COLORS["warning"] if score >= 50 else Config.COLORS["danger"]

    st.markdown(
        f"""
        <div style="text-align:center; padding: 15px; background: {Config.COLORS['bg_card']}; border-radius: 12px; border: 1px solid {color};">
            <div style="font-size: 2.5rem; font-weight: 800; color: {color};">{score:.0f}%</div>
            <div style="color: #95a5a6; font-size: 0.9rem;">Datenqualitäts-Score</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return score


# ============================================================================
# 9. SEITEN-RENDERER
# ============================================================================

# -------- 9A. DATEN-MANAGEMENT --------

def page_data_management():
    st.header("📊 Daten-Management")
    st.markdown(
        "Laden Sie Ihre Marktdaten hoch oder generieren Sie realistische Testdaten "
        "mit simulierter Energiekrise 2022/23."
    )

    # === SAMPLE DATA ===
    with st.expander(
        "🎲 **Beispieldaten generieren** (sofort loslegen)",
        expanded=not st.session_state.get("data_loaded", False),
    ):
        st.info(
            "🔬 Generiert Ornstein-Uhlenbeck Spotpreise, korrelierte Forward-Kurven, "
            "Beschaffungstranchen und ein Industrielastprofil – inklusive Energiekrise 2022/23."
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            s_start = st.date_input("Start", value=date(2022, 1, 1), key="sg_s")
        with c2:
            s_end = st.date_input("Ende", value=date(2024, 12, 31), key="sg_e")
        with c3:
            s_gran = st.selectbox("Spot-Granularität", ["hourly", "daily"], key="sg_g")

        if st.button("🚀 Beispieldaten generieren", type="primary", use_container_width=True):
            progress = st.progress(0, text="Initialisiere...")

            gen = SampleDataGenerator

            progress.progress(10, text="Generiere Spotpreise (OU-Prozess + Saisonalität)...")
            spot = gen.generate_spot_prices(str(s_start), str(s_end), s_gran)
            DataStore.store("spot_prices", spot)

            progress.progress(35, text="Generiere Settlement-Preise (Forward-Kurven)...")
            settlements = gen.generate_settlement_prices(str(s_start), str(s_end))
            DataStore.store("settlement_prices", settlements)

            progress.progress(60, text="Generiere Beschaffungstranchen...")
            purchases = gen.generate_purchases(str(s_start), str(s_end))
            DataStore.store("purchases", purchases)

            progress.progress(80, text="Generiere Lastprofil...")
            load = gen.generate_load_profile(str(s_start), str(s_end))
            DataStore.store("load_profile", load)

            st.session_state["data_loaded"] = True
            st.session_state["using_sample_data"] = True

            progress.progress(100, text="Fertig!")

            st.success(
                f"✅ **Daten erfolgreich generiert!**\n\n"
                f"- ⚡ Spotpreise: **{len(spot):,}** Datenpunkte\n"
                f"- 📋 Settlements: **{len(settlements):,}** Datenpunkte "
                f"({settlements['product'].nunique()} Produkte)\n"
                f"- 🛒 Beschaffungen: **{len(purchases)}** Transaktionen "
                f"({purchases['volume_mw'].sum():.0f} MW)\n"
                f"- 📊 Lastprofil: **{len(load):,}** Datenpunkte"
            )
            st.rerun()

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # === FILE UPLOADS ===
    st.subheader("📁 Eigene Daten hochladen")

    tab_spot, tab_sett, tab_purch, tab_load = st.tabs([
        "⚡ Spotpreise",
        "📋 Settlement-Preise",
        "🛒 Beschaffungen",
        "📊 Lastprofil",
    ])

    # --- SPOTPREISE ---
    with tab_spot:
        st.markdown("**Erwartete Spalten:** `timestamp` · `price_eur_mwh`")
        st.caption("Unterstützt: CSV (Komma/Semikolon/Tab) und Excel. Timestamps werden automatisch erkannt.")

        spot_file = st.file_uploader("CSV / Excel", type=["csv", "xlsx", "xls"], key="up_spot")

        if spot_file is not None:
            try:
                df_raw = read_uploaded_file(spot_file)
                st.dataframe(df_raw.head(8), use_container_width=True)

                c1, c2 = st.columns(2)
                with c1:
                    col_ts = st.selectbox("Timestamp-Spalte", df_raw.columns.tolist(), key="map_spot_ts")
                with c2:
                    col_price = st.selectbox("Preis-Spalte (EUR/MWh)", df_raw.columns.tolist(),
                                             index=min(1, len(df_raw.columns) - 1), key="map_spot_p")

                if st.button("✅ Validieren & Speichern", key="save_spot", type="primary"):
                    df_mapped = df_raw.rename(columns={col_ts: "timestamp", col_price: "price_eur_mwh"})
                    is_valid, issues, df_clean = DataValidator.validate_spot_prices(df_mapped)
                    render_validation_issues(issues)
                    if is_valid:
                        DataStore.store("spot_prices", df_clean)
                        st.session_state["data_loaded"] = True
                        render_data_quality_score(issues, len(df_raw))
                        st.success(f"✅ {len(df_clean):,} Spotpreise geladen!")
            except Exception as e:
                st.error(f"Fehler: {e}")

    # --- SETTLEMENTS ---
    with tab_sett:
        st.markdown(
            "**Erwartete Spalten:** `trade_date` · `product` · `product_type` · "
            "`delivery_start` · `delivery_end` · `settlement_price`"
        )

        sett_file = st.file_uploader("CSV / Excel", type=["csv", "xlsx", "xls"], key="up_sett")

        if sett_file is not None:
            try:
                df_raw = read_uploaded_file(sett_file)
                st.dataframe(df_raw.head(8), use_container_width=True)

                if st.button("✅ Validieren & Speichern", key="save_sett", type="primary"):
                    is_valid, issues, df_clean = DataValidator.validate_settlement_prices(df_raw)
                    render_validation_issues(issues)
                    if is_valid:
                        DataStore.store("settlement_prices", df_clean)
                        st.success(f"✅ {len(df_clean):,} Settlement-Preise geladen!")
            except Exception as e:
                st.error(f"Fehler: {e}")

    # --- BESCHAFFUNGEN ---
    with tab_purch:
        st.markdown(
            "**Erwartete Spalten:** `purchase_date` · `product` · `volume_mw` · "
            "`price_eur_mwh` · `delivery_start` · `delivery_end` · `purchase_type`"
        )

        purch_file = st.file_uploader("CSV / Excel", type=["csv", "xlsx", "xls"], key="up_purch")

        if purch_file is not None:
            try:
                df_raw = read_uploaded_file(purch_file)
                st.dataframe(df_raw.head(8), use_container_width=True)

                if st.button("✅ Validieren & Speichern", key="save_purch", type="primary"):
                    is_valid, issues, df_clean = DataValidator.validate_purchases(df_raw)
                    render_validation_issues(issues)
                    if is_valid:
                        DataStore.store("purchases", df_clean)
                        st.success(f"✅ {len(df_clean)} Beschaffungen geladen!")
            except Exception as e:
                st.error(f"Fehler: {e}")

    # --- LASTPROFIL ---
    with tab_load:
        st.markdown("**Erwartete Spalten:** `timestamp` · `volume_mwh`")

        load_file = st.file_uploader("CSV / Excel", type=["csv", "xlsx", "xls"], key="up_load")

        if load_file is not None:
            try:
                df_raw = read_uploaded_file(load_file)
                st.dataframe(df_raw.head(8), use_container_width=True)

                c1, c2 = st.columns(2)
                with c1:
                    col_ts = st.selectbox("Timestamp-Spalte", df_raw.columns.tolist(), key="map_load_ts")
                with c2:
                    col_vol = st.selectbox("Volumen-Spalte (MWh)", df_raw.columns.tolist(),
                                           index=min(1, len(df_raw.columns) - 1), key="map_load_v")

                if st.button("✅ Validieren & Speichern", key="save_load", type="primary"):
                    df_mapped = df_raw.rename(columns={col_ts: "timestamp", col_vol: "volume_mwh"})
                    is_valid, issues, df_clean = DataValidator.validate_load_profile(df_mapped)
                    render_validation_issues(issues)
                    if is_valid:
                        DataStore.store("load_profile", df_clean)
                        st.success(f"✅ {len(df_clean):,} Lastprofil-Einträge geladen!")
            except Exception as e:
                st.error(f"Fehler: {e}")


# -------- 9B. DATEN-EXPLORER --------

def page_data_explorer():
    st.header("🔬 Interaktiver Daten-Explorer")

    if not st.session_state.get("data_loaded"):
        st.warning("⚠️ Bitte laden Sie zuerst Daten unter **Daten-Management**.")
        return

    tab_spot, tab_sett, tab_purch, tab_load = st.tabs([
        "⚡ Spotpreise", "📋 Settlements", "🛒 Beschaffungen", "📊 Lastprofil"
    ])

    # ===================== SPOTPREISE =====================
    with tab_spot:
        spot = DataStore.get("spot_prices")
        if spot is None:
            st.info("Keine Spotpreise geladen.")
            return

        st.subheader("Spotpreis-Zeitreihe")

        # Filter
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            d_start = st.date_input("Von", value=spot["timestamp"].min().date(), key="ex_sp_s")
        with c2:
            d_end = st.date_input("Bis", value=spot["timestamp"].max().date(), key="ex_sp_e")
        with c3:
            agg = st.selectbox("Aggregation", ["Roh", "Täglich", "Wöchentlich", "Monatlich"], key="ex_sp_a")

        mask = (spot["timestamp"].dt.date >= d_start) & (spot["timestamp"].dt.date <= d_end)
        sf = spot[mask].copy()

        resample_map = {"Täglich": "D", "Wöchentlich": "W", "Monatlich": "ME"}
        if agg in resample_map:
            plot_df = sf.set_index("timestamp").resample(resample_map[agg])["price_eur_mwh"].mean().reset_index()
        else:
            if len(sf) > 80000:
                st.caption("⚡ Über 80k Punkte – automatische Tagesaggregation für Performance.")
                plot_df = sf.set_index("timestamp").resample("D")["price_eur_mwh"].mean().reset_index()
            else:
                plot_df = sf

        # Hauptchart
        fig = go.Figure()
        fig.add_trace(go.Scattergl(
            x=plot_df["timestamp"], y=plot_df["price_eur_mwh"],
            mode="lines", name="Spotpreis",
            line=dict(color=Config.COLORS["spot"], width=1),
        ))

        # Moving Averages
        if len(plot_df) > 30:
            ma30 = plot_df["price_eur_mwh"].rolling(30, min_periods=1).mean()
            ma90 = plot_df["price_eur_mwh"].rolling(90, min_periods=1).mean()
            fig.add_trace(go.Scattergl(
                x=plot_df["timestamp"], y=ma30,
                mode="lines", name="MA-30",
                line=dict(color=Config.COLORS["forward"], width=2, dash="dash"),
            ))
            fig.add_trace(go.Scattergl(
                x=plot_df["timestamp"], y=ma90,
                mode="lines", name="MA-90",
                line=dict(color=Config.COLORS["actual"], width=2, dash="dot"),
            ))

        fig.update_layout(
            title="Spotpreis-Entwicklung (EUR/MWh)",
            xaxis_title="", yaxis_title="EUR/MWh",
            template="plotly_dark", height=480,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        # KPIs
        st.subheader("📊 Deskriptive Statistik")
        cols = st.columns(6)
        stats = [
            ("Mittelwert", f"{sf['price_eur_mwh'].mean():.2f} €"),
            ("Median", f"{sf['price_eur_mwh'].median():.2f} €"),
            ("Std.Abw.", f"{sf['price_eur_mwh'].std():.2f} €"),
            ("Skewness", f"{sf['price_eur_mwh'].skew():.2f}"),
            ("Min", f"{sf['price_eur_mwh'].min():.2f} €"),
            ("Max", f"{sf['price_eur_mwh'].max():.2f} €"),
        ]
        for col, (label, val) in zip(cols, stats):
            col.metric(label, val)

        # Verteilung + Boxplot
        c1, c2 = st.columns(2)
        with c1:
            fig_h = px.histogram(
                sf, x="price_eur_mwh", nbins=120,
                title="Preisverteilung", labels={"price_eur_mwh": "EUR/MWh"},
                template="plotly_dark", color_discrete_sequence=[Config.COLORS["spot"]],
            )
            fig_h.update_layout(height=360)
            st.plotly_chart(fig_h, use_container_width=True)

        with c2:
            sf_box = sf.copy()
            sf_box["monat"] = sf_box["timestamp"].dt.to_period("M").astype(str)
            months = sorted(sf_box["monat"].unique())
            if len(months) > 36:
                months = months[-36:]
                sf_box = sf_box[sf_box["monat"].isin(months)]

            fig_b = px.box(
                sf_box, x="monat", y="price_eur_mwh",
                title="Monatliche Preisverteilung",
                labels={"price_eur_mwh": "EUR/MWh", "monat": ""},
                template="plotly_dark",
            )
            fig_b.update_layout(height=360, xaxis_tickangle=-45)
            st.plotly_chart(fig_b, use_container_width=True)

        # Heatmap: Stunde × Monat
        st.subheader("🗺️ Preis-Heatmap: Stunde × Monat")
        sh = sf.copy()
        sh["hour"] = sh["timestamp"].dt.hour
        sh["monat"] = sh["timestamp"].dt.to_period("M").astype(str)
        pivot = sh.pivot_table(values="price_eur_mwh", index="hour", columns="monat", aggfunc="mean")

        fig_hm = go.Figure(go.Heatmap(
            z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
            colorscale="RdYlGn_r", colorbar_title="€/MWh",
        ))
        fig_hm.update_layout(
            xaxis_title="Monat", yaxis_title="Stunde",
            template="plotly_dark", height=420, xaxis_tickangle=-45,
        )
        st.plotly_chart(fig_hm, use_container_width=True)

        # Volatilitätsanalyse
        st.subheader("📉 Rollende Volatilität")
        if agg != "Roh":
            vol_df = plot_df.copy()
        else:
            vol_df = sf.set_index("timestamp").resample("D")["price_eur_mwh"].mean().reset_index()

        vol_df["returns"] = vol_df["price_eur_mwh"].pct_change()
        vol_df["vol_30d"] = vol_df["returns"].rolling(30).std() * np.sqrt(252) * 100

        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=vol_df["timestamp"], y=vol_df["vol_30d"],
            mode="lines", name="30-Tage Volatilität (%)",
            line=dict(color=Config.COLORS["warning"], width=1.5),
            fill="tozeroy", fillcolor="rgba(243,156,18,0.15)",
        ))
        fig_vol.update_layout(
            title="Annualisierte 30-Tage Volatilität (%)",
            xaxis_title="", yaxis_title="Volatilität (%)",
            template="plotly_dark", height=320,
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    # ===================== SETTLEMENTS =====================
    with tab_sett:
        sett = DataStore.get("settlement_prices")
        if sett is None:
            st.info("Keine Settlement-Preise geladen.")
            return

        st.subheader("Settlement-Preis-Explorer")

        c1, c2 = st.columns(2)
        with c1:
            sel_types = st.multiselect(
                "Produkttyp", sett["product_type"].unique().tolist(),
                default=sett["product_type"].unique().tolist(), key="ex_st_t",
            )
        with c2:
            groups = ["Alle"] + sorted(
                set(p.split("-")[0] for p in sett["product"].unique())
            )
            sel_group = st.selectbox("Produktgruppe", groups, key="ex_st_g")

        sf2 = sett[sett["product_type"].isin(sel_types)].copy()
        if sel_group != "Alle":
            sf2 = sf2[sf2["product"].str.startswith(sel_group)]

        prods = sorted(sf2["product"].unique())
        if prods:
            sel_prods = st.multiselect(
                "Produkte (max. 8 empfohlen)", prods,
                default=prods[:min(5, len(prods))], key="ex_st_p",
            )

            if sel_prods:
                plot_s = sf2[sf2["product"].isin(sel_prods)]

                fig_s = go.Figure()
                for prod in sel_prods:
                    d = plot_s[plot_s["product"] == prod]
                    fig_s.add_trace(go.Scattergl(
                        x=d["trade_date"], y=d["settlement_price"],
                        mode="lines", name=prod, line=dict(width=1.5),
                    ))
                fig_s.update_layout(
                    title="Settlement-Preis-Entwicklung",
                    yaxis_title="EUR/MWh", template="plotly_dark", height=480,
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig_s, use_container_width=True)

        # Forward Curve Snapshot
        st.subheader("📐 Forward-Kurve (Stichtag)")
        avail_dates = sorted(sett["trade_date"].dt.date.unique())
        fc_date = st.select_slider("Stichtag", options=avail_dates, value=avail_dates[-1], key="fc_slider")

        fc = sett[
            (sett["trade_date"].dt.date == fc_date) & (sett["product_type"] == "Base")
        ].sort_values("delivery_start")

        if not fc.empty:
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Bar(
                x=fc["product"], y=fc["settlement_price"],
                marker_color=Config.COLORS["forward"],
                text=fc["settlement_price"].round(1),
                textposition="outside",
            ))
            fig_fc.update_layout(
                title=f"Forward-Kurve (Base) am {fc_date}",
                yaxis_title="EUR/MWh", template="plotly_dark", height=400,
                xaxis_tickangle=-45,
            )
            st.plotly_chart(fig_fc, use_container_width=True)
        else:
            st.info(f"Keine Daten für {fc_date}.")

        # Forward Curve Animation über Zeit
        st.subheader("🎬 Forward-Kurven-Animation")
        if st.checkbox("Forward-Kurven über Zeit animieren", key="fc_animate"):
            # Monatliche Snapshots
            sett_base = sett[sett["product_type"] == "Base"].copy()
            sett_base["trade_month"] = sett_base["trade_date"].dt.to_period("M").astype(str)

            trade_months = sorted(sett_base["trade_month"].unique())
            sel_month = st.select_slider(
                "Handelszeitraum", options=trade_months,
                value=trade_months[0], key="fc_anim_m",
            )

            snap = sett_base[sett_base["trade_month"] == sel_month]
            # Letzter Handelstag des Monats
            last_day = snap["trade_date"].max()
            snap_last = snap[snap["trade_date"] == last_day].sort_values("delivery_start")

            if not snap_last.empty:
                fig_anim = go.Figure()
                fig_anim.add_trace(go.Bar(
                    x=snap_last["product"], y=snap_last["settlement_price"],
                    marker_color=Config.COLORS["forward"],
                ))
                fig_anim.update_layout(
                    title=f"Forward-Kurve Ende {sel_month}",
                    yaxis_title="EUR/MWh", template="plotly_dark", height=380,
                    xaxis_tickangle=-45,
                )
                st.plotly_chart(fig_anim, use_container_width=True)

    # ===================== BESCHAFFUNGEN =====================
    with tab_purch:
        purch = DataStore.get("purchases")
        if purch is None:
            st.info("Keine Beschaffungen geladen.")
            return

        st.subheader("Beschaffungs-Dashboard")

        # KPIs
        total_vol = purch["volume_mw"].sum()
        total_cost = (purch["price_eur_mwh"] * purch["volume_mw"]).sum()
        avg_p = total_cost / total_vol if total_vol > 0 else 0
        fwd_share = purch[purch["purchase_type"].str.lower() == "forward"]["volume_mw"].sum() / total_vol * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Transaktionen", f"{len(purch)}")
        c2.metric("Gesamt-Volumen", f"{total_vol:,.0f} MW")
        c3.metric("Ø Preis (vol.-gew.)", f"{avg_p:.2f} €/MWh")
        c4.metric("Forward-Anteil", f"{fwd_share:.1f}%")

        # Bubble Chart: Kaufzeitpunkt × Preis × Volumen
        fig_bubble = go.Figure()
        for ptype in purch["purchase_type"].str.lower().unique():
            m = purch["purchase_type"].str.lower() == ptype
            d = purch[m]
            fig_bubble.add_trace(go.Scatter(
                x=d["purchase_date"], y=d["price_eur_mwh"],
                mode="markers", name=ptype.capitalize(),
                marker=dict(
                    size=d["volume_mw"] / purch["volume_mw"].max() * 35 + 6,
                    color=Config.COLORS.get(ptype, Config.COLORS["neutral"]),
                    opacity=0.75, line=dict(width=1, color="white"),
                ),
                customdata=np.stack([d["product"], d["volume_mw"], d["delivery_start"].dt.strftime("%Y-%m-%d")], axis=-1),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Kaufdatum: %{x|%Y-%m-%d}<br>"
                    "Preis: %{y:.2f} €/MWh<br>"
                    "Volumen: %{customdata[1]} MW<br>"
                    "Lieferung ab: %{customdata[2]}<extra></extra>"
                ),
            ))
        fig_bubble.update_layout(
            title="Beschaffungszeitpunkte (Größe = Volumen)",
            xaxis_title="Kaufdatum", yaxis_title="EUR/MWh",
            template="plotly_dark", height=460,
        )
        st.plotly_chart(fig_bubble, use_container_width=True)

        # Monatliche Beschaffungsvolumina
        purch_monthly = purch.copy()
        purch_monthly["month"] = purch_monthly["purchase_date"].dt.to_period("M").astype(str)
        monthly_agg = purch_monthly.groupby("month").agg(
            total_vol=("volume_mw", "sum"),
            avg_price=("price_eur_mwh", "mean"),
            n_trades=("volume_mw", "count"),
        ).reset_index()

        fig_bar = make_subplots(specs=[[{"secondary_y": True}]])
        fig_bar.add_trace(
            go.Bar(x=monthly_agg["month"], y=monthly_agg["total_vol"],
                   name="Volumen (MW)", marker_color=Config.COLORS["forward"], opacity=0.7),
            secondary_y=False,
        )
        fig_bar.add_trace(
            go.Scatter(x=monthly_agg["month"], y=monthly_agg["avg_price"],
                       name="Ø Preis", mode="lines+markers",
                       line=dict(color=Config.COLORS["spot"], width=2)),
            secondary_y=True,
        )
        fig_bar.update_layout(
            title="Monatliche Beschaffungsaktivität",
            template="plotly_dark", height=380,
            xaxis_tickangle=-45,
        )
        fig_bar.update_yaxes(title_text="MW", secondary_y=False)
        fig_bar.update_yaxes(title_text="€/MWh", secondary_y=True)
        st.plotly_chart(fig_bar, use_container_width=True)

        # Datentabelle
        with st.expander("📋 Detaillierte Transaktionsliste"):
            st.dataframe(
                purch.sort_values("purchase_date", ascending=False)
                .style.format({"price_eur_mwh": "{:.2f}", "volume_mw": "{:.0f}"}),
                use_container_width=True, height=400,
            )

    # ===================== LASTPROFIL =====================
    with tab_load:
        load = DataStore.get("load_profile")
        if load is None:
            st.info("Kein Lastprofil geladen.")
            return

        st.subheader("Lastprofil-Analyse")

        # Tägliche Aggregation
        daily = load.set_index("timestamp").resample("D")["volume_mwh"].agg(
            ["mean", "max", "min", "sum"]
        ).reset_index()

        fig_load = go.Figure()
        fig_load.add_trace(go.Scatter(
            x=daily["timestamp"], y=daily["max"],
            mode="lines", name="Max", line=dict(color=Config.COLORS["danger"], width=0.5, dash="dot"),
        ))
        fig_load.add_trace(go.Scatter(
            x=daily["timestamp"], y=daily["min"],
            mode="lines", name="Min", line=dict(color=Config.COLORS["info"], width=0.5, dash="dot"),
            fill="tonexty", fillcolor="rgba(46,204,113,0.08)",
        ))
        fig_load.add_trace(go.Scatter(
            x=daily["timestamp"], y=daily["mean"],
            mode="lines", name="Durchschnitt", line=dict(color=Config.COLORS["actual"], width=1.8),
        ))
        fig_load.update_layout(
            title="Tägliche Last (MWh)", yaxis_title="MWh",
            template="plotly_dark", height=420, hovermode="x unified",
        )
        st.plotly_chart(fig_load, use_container_width=True)

        # Typische Woche Heatmap
        st.subheader("📅 Typische Woche")
        lc = load.copy()
        lc["hour"] = lc["timestamp"].dt.hour
        day_map = {0: "Mo", 1: "Di", 2: "Mi", 3: "Do", 4: "Fr", 5: "Sa", 6: "So"}
        lc["day"] = lc["timestamp"].dt.dayofweek.map(day_map)
        day_order = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]

        piv = lc.pivot_table(values="volume_mwh", index="hour", columns="day", aggfunc="mean")
        piv = piv.reindex(columns=day_order)

        fig_week = go.Figure(go.Heatmap(
            z=piv.values, x=piv.columns.tolist(), y=piv.index.tolist(),
            colorscale="YlOrRd", colorbar_title="MWh",
        ))
        fig_week.update_layout(
            xaxis_title="Wochentag", yaxis_title="Stunde",
            template="plotly_dark", height=400,
        )
        st.plotly_chart(fig_week, use_container_width=True)

        # Dauerlinie (Load Duration Curve)
        st.subheader("📈 Dauerlinie (Load Duration Curve)")
        sorted_load = np.sort(load["volume_mwh"].values)[::-1]
        x_pct = np.linspace(0, 100, len(sorted_load))

        fig_ldc = go.Figure()
        fig_ldc.add_trace(go.Scatter(
            x=x_pct, y=sorted_load,
            mode="lines", name="Dauerlinie",
            line=dict(color=Config.COLORS["actual"], width=2),
            fill="tozeroy", fillcolor="rgba(46,204,113,0.15)",
        ))
        fig_ldc.add_hline(y=np.mean(sorted_load), line_dash="dash",
                          annotation_text=f"Ø {np.mean(sorted_load):.1f} MW")
        fig_ldc.update_layout(
            title="Geordnete Jahresdauerlinie",
            xaxis_title="% der Stunden", yaxis_title="MW",
            template="plotly_dark", height=380,
        )
        st.plotly_chart(fig_ldc, use_container_width=True)


# -------- 9C. SHAPING ENGINE --------

def page_shaping_engine():
    st.header("⚙️ Shaping Engine")
    st.markdown(
        """
        Terminmarkt-Produkte liefern in **Standardblöcken** (Base 24h, Peak Mo-Fr 08-20h).
        Die Shaping Engine zerlegt diese in **stündliche Lieferprofile** –
        die Voraussetzung für mathematisch exaktes Matching mit Spotmarkt und Lastprofil.
        """
    )

    purch = DataStore.get("purchases")
    if purch is None:
        st.warning("⚠️ Bitte laden Sie zuerst Beschaffungsdaten.")
        return

    fwd = purch[purch["purchase_type"].str.lower() == "forward"]

    if fwd.empty:
        st.info("Keine Forward-Beschaffungen vorhanden.")
        return

    st.success(f"📦 **{len(fwd)} Forward-Beschaffungen** bereit zum Shapen ({fwd['volume_mw'].sum():.0f} MW)")

    # Konfiguration
    st.subheader("🔧 Shaping-Konfiguration")
    c1, c2, c3 = st.columns(3)
    with c1:
        profile_type = st.selectbox("Profiltyp", ["Base (24h)", "Peak (Mo-Fr)", "OffPeak"], key="sh_pt")
    with c2:
        peak_s = st.number_input("Peak-Start (Stunde)", value=8, min_value=0, max_value=23, key="sh_ps")
    with c3:
        peak_e = st.number_input("Peak-Ende (Stunde)", value=20, min_value=1, max_value=24, key="sh_pe")

    if st.button("🔄 Shaping durchführen", type="primary", use_container_width=True):
        with st.spinner("Zerlege Terminmarkt-Produkte in stündliche Profile..."):
            shaped = ShapingEngine.shape_all_purchases(fwd, peak_start=peak_s, peak_end=peak_e)

        if shaped.empty:
            st.error("Shaping hat keine Ergebnisse produziert.")
            return

        DataStore.store("shaped_purchases", shaped)

        st.success(f"✅ **Shaping abgeschlossen!** {len(shaped):,} stündliche Datenpunkte generiert.")

        # --- Visualisierung ---
        st.subheader("📊 Gehedgtes Volumen (aus Forward-Käufen)")

        daily_s = shaped.set_index("timestamp").resample("D").agg(
            volume_mwh=("volume_mwh", "sum"),
            cost_eur=("cost_eur", "sum"),
        ).reset_index()
        daily_s["avg_price"] = (daily_s["cost_eur"] / daily_s["volume_mwh"].replace(0, np.nan)).round(2)

        fig_shaped = make_subplots(specs=[[{"secondary_y": True}]])
        fig_shaped.add_trace(
            go.Bar(x=daily_s["timestamp"], y=daily_s["volume_mwh"],
                   name="Volumen (MWh)", marker_color=Config.COLORS["forward"], opacity=0.6),
            secondary_y=False,
        )
        fig_shaped.add_trace(
            go.Scatter(x=daily_s["timestamp"], y=daily_s["avg_price"],
                       name="Ø Preis (€/MWh)", mode="lines",
                       line=dict(color=Config.COLORS["spot"], width=2)),
            secondary_y=True,
        )
        fig_shaped.update_layout(
            title="Tägliches gehedgtes Volumen & Durchschnittspreis",
            template="plotly_dark", height=420,
        )
        fig_shaped.update_yaxes(title_text="MWh", secondary_y=False)
        fig_shaped.update_yaxes(title_text="€/MWh", secondary_y=True)
        st.plotly_chart(fig_shaped, use_container_width=True)

        # --- Hedge vs. Last ---
        load = DataStore.get("load_profile")
        if load is not None:
            st.subheader("🔀 Hedge-Abdeckung vs. Verbrauch")

            daily_load = load.set_index("timestamp").resample("D")["volume_mwh"].sum().reset_index()
            daily_load.columns = ["timestamp", "load_mwh"]

            merged = pd.merge(daily_load, daily_s[["timestamp", "volume_mwh"]], on="timestamp", how="left")
            merged = merged.rename(columns={"volume_mwh": "hedged_mwh"}).fillna(0)
            merged["open_mwh"] = (merged["load_mwh"] - merged["hedged_mwh"]).clip(lower=0)
            merged["over_hedge_mwh"] = (merged["hedged_mwh"] - merged["load_mwh"]).clip(lower=0)
            merged["hedge_pct"] = (merged["hedged_mwh"] / merged["load_mwh"].replace(0, np.nan) * 100).fillna(0)

            # Stacked Area
            fig_cov = go.Figure()
            fig_cov.add_trace(go.Scatter(
                x=merged["timestamp"], y=merged["hedged_mwh"],
                name="Gehedgt (Forward)", fill="tozeroy",
                line=dict(color=Config.COLORS["forward"], width=0),
                fillcolor="rgba(52,152,219,0.5)",
            ))
            fig_cov.add_trace(go.Scatter(
                x=merged["timestamp"], y=merged["hedged_mwh"] + merged["open_mwh"],
                name="Offen (→ Spot)", fill="tonexty",
                line=dict(color=Config.COLORS["spot"], width=0),
                fillcolor="rgba(231,76,60,0.4)",
            ))
            fig_cov.add_trace(go.Scatter(
                x=merged["timestamp"], y=merged["load_mwh"],
                name="Gesamtlast", mode="lines",
                line=dict(color="white", width=1.5, dash="dot"),
            ))
            fig_cov.update_layout(
                title="Tägliche Hedge-Abdeckung vs. Verbrauch",
                yaxis_title="MWh", template="plotly_dark", height=450,
                hovermode="x unified",
            )
            st.plotly_chart(fig_cov, use_container_width=True)

            # Monatliche Hedge-Quote
            monthly = merged.set_index("timestamp").resample("ME").agg(
                load_mwh=("load_mwh", "sum"),
                hedged_mwh=("hedged_mwh", "sum"),
            ).reset_index()
            monthly["hedge_pct"] = (monthly["hedged_mwh"] / monthly["load_mwh"] * 100).clip(0, 250)

            colors_bar = [
                Config.COLORS["danger"] if v < 50 else
                Config.COLORS["warning"] if v < 80 else
                Config.COLORS["success"] if v <= 110 else
                Config.COLORS["info"]
                for v in monthly["hedge_pct"]
            ]

            fig_ratio = go.Figure()
            fig_ratio.add_trace(go.Bar(
                x=monthly["timestamp"].dt.strftime("%Y-%m"),
                y=monthly["hedge_pct"],
                marker_color=colors_bar,
                text=monthly["hedge_pct"].round(0).astype(int).astype(str) + "%",
                textposition="outside",
            ))
            fig_ratio.add_hline(y=100, line_dash="dash", line_color="white",
                                annotation_text="100% Abdeckung")
            fig_ratio.update_layout(
                title="Monatliche Hedge-Quote (%)",
                yaxis_title="Hedge-Quote (%)", template="plotly_dark", height=380,
                xaxis_tickangle=-45,
            )
            st.plotly_chart(fig_ratio, use_container_width=True)

            # Zusammenfassungstabelle
            st.subheader("📋 Monatliche Zusammenfassung")
            monthly_display = monthly.copy()
            monthly_display["monat"] = monthly_display["timestamp"].dt.strftime("%Y-%m")
            monthly_display["open_mwh"] = monthly_display["load_mwh"] - monthly_display["hedged_mwh"]
            monthly_display = monthly_display[["monat", "load_mwh", "hedged_mwh", "open_mwh", "hedge_pct"]]
            monthly_display.columns = ["Monat", "Last (MWh)", "Gehedgt (MWh)", "Offen (MWh)", "Hedge-Quote (%)"]

            st.dataframe(
                monthly_display.style.format({
                    "Last (MWh)": "{:,.0f}",
                    "Gehedgt (MWh)": "{:,.0f}",
                    "Offen (MWh)": "{:,.0f}",
                    "Hedge-Quote (%)": "{:.1f}%",
                }).background_gradient(subset=["Hedge-Quote (%)"], cmap="RdYlGn", vmin=0, vmax=120),
                use_container_width=True,
            )
        else:
            st.info("💡 Laden Sie ein Lastprofil, um die Hedge-Abdeckung zu analysieren.")

# ============================================================================
# ============================================================================
#
#  MODULE 2 :  BACKTESTING ENGINE
#  Strategiesimulation · Kontrafaktische Analyse · Optimierung
#
# ============================================================================
# ============================================================================

from enum import Enum
from itertools import product as itertools_product

# ============================================================================
# 12. STRATEGIE-DEFINITIONEN & DATENKLASSEN
# ============================================================================

class StrategyType(Enum):
    """Verfügbare Beschaffungsstrategien."""
    DCA = "DCA (Gleichmäßige Tranchen)"
    FRONT_LOADED = "Front-Loaded (Mehr am Anfang)"
    BACK_LOADED = "Back-Loaded (Mehr am Ende)"
    LIMIT_BASED = "Limit-basiert (Preisschwelle)"
    RSI_TRIGGERED = "RSI-getriggert (Technisch)"
    SEASONAL = "Saisonal (Sommer kaufen)"
    BENCHMARK_SPOT = "Benchmark: 100% Spot"
    BENCHMARK_FORWARD = "Benchmark: 100% Forward (DCA)"
    ACTUAL = "Actual (Echte Beschaffung)"


@dataclass
class StrategyConfig:
    """Vollständige Parametrisierung einer Beschaffungsstrategie."""
    name: str
    strategy_type: StrategyType
    hedge_ratio: float = 0.7                  # 0.0 – 1.0
    n_tranches: int = 6                       # Anzahl Forward-Tranchen
    buying_window_months: int = 6             # Monate vor Lieferung
    # Limit-spezifisch
    limit_percentile: float = 40.0            # Kaufe wenn Preis < X. Perzentil
    # RSI-spezifisch
    rsi_period: int = 14
    rsi_threshold: float = 35.0               # Kaufe wenn RSI < Schwelle
    # Saisonal-spezifisch
    buy_months: List[int] = field(default_factory=lambda: [4, 5, 6, 7])
    # Transaktionskosten
    slippage_bps: float = 5.0                 # Basis Points Slippage
    broker_fee_eur_mwh: float = 0.05          # Brokergebühr

    @property
    def total_transaction_cost_pct(self) -> float:
        return self.slippage_bps / 10000.0


@dataclass
class TrancheExecution:
    """Eine einzelne ausgeführte Forward-Tranche."""
    execution_date: pd.Timestamp
    delivery_month: str
    volume_mwh: float
    price_eur_mwh: float
    signal_value: Optional[float] = None      # RSI, Perzentil etc.

    @property
    def cost_eur(self) -> float:
        return self.volume_mwh * self.price_eur_mwh


@dataclass
class MonthResult:
    """Backtesting-Ergebnis für einen einzelnen Liefermonat."""
    delivery_month: str
    delivery_start: pd.Timestamp
    total_demand_mwh: float
    hedged_volume_mwh: float
    open_volume_mwh: float
    hedge_ratio_actual: float
    forward_cost_eur: float
    spot_cost_eur: float
    total_cost_eur: float
    avg_forward_price: float
    avg_spot_price: float
    blended_price: float
    n_tranches_executed: int
    tranches: List[TrancheExecution]


@dataclass
class BacktestResult:
    """Vollständiges Backtesting-Ergebnis einer Strategie."""
    strategy: StrategyConfig
    monthly_results: List[MonthResult]
    execution_time_ms: float = 0.0

    @property
    def total_cost(self) -> float:
        return sum(m.total_cost_eur for m in self.monthly_results)

    @property
    def total_demand(self) -> float:
        return sum(m.total_demand_mwh for m in self.monthly_results)

    @property
    def avg_blended_price(self) -> float:
        return self.total_cost / self.total_demand if self.total_demand > 0 else 0

    @property
    def total_forward_cost(self) -> float:
        return sum(m.forward_cost_eur for m in self.monthly_results)

    @property
    def total_spot_cost(self) -> float:
        return sum(m.spot_cost_eur for m in self.monthly_results)

    @property
    def effective_hedge_ratio(self) -> float:
        hedged = sum(m.hedged_volume_mwh for m in self.monthly_results)
        return hedged / self.total_demand if self.total_demand > 0 else 0

    @property
    def all_tranches(self) -> List[TrancheExecution]:
        return [t for m in self.monthly_results for t in m.tranches]

    def to_dataframe(self) -> pd.DataFrame:
        records = []
        for m in self.monthly_results:
            records.append({
                "delivery_month": m.delivery_month,
                "delivery_start": m.delivery_start,
                "demand_mwh": m.total_demand_mwh,
                "hedged_mwh": m.hedged_volume_mwh,
                "open_mwh": m.open_volume_mwh,
                "hedge_ratio": m.hedge_ratio_actual,
                "fwd_cost_eur": m.forward_cost_eur,
                "spot_cost_eur": m.spot_cost_eur,
                "total_cost_eur": m.total_cost_eur,
                "avg_fwd_price": m.avg_forward_price,
                "avg_spot_price": m.avg_spot_price,
                "blended_price": m.blended_price,
                "n_tranches": m.n_tranches_executed,
            })
        return pd.DataFrame(records)

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "Strategie": self.strategy.name,
            "Typ": self.strategy.strategy_type.value,
            "Ziel-Hedge-Quote": f"{self.strategy.hedge_ratio:.0%}",
            "Effektive Hedge-Quote": f"{self.effective_hedge_ratio:.1%}",
            "Gesamtkosten (€)": f"{self.total_cost:,.0f}",
            "Ø Preis (€/MWh)": f"{self.avg_blended_price:.2f}",
            "Forward-Kosten (€)": f"{self.total_forward_cost:,.0f}",
            "Spot-Kosten (€)": f"{self.total_spot_cost:,.0f}",
            "Gesamtnachfrage (MWh)": f"{self.total_demand:,.0f}",
            "Ausgeführte Tranchen": len(self.all_tranches),
            "Liefermonate": len(self.monthly_results),
        }


# ============================================================================
# 13. TECHNISCHE INDIKATOREN
# ============================================================================

class TechnicalIndicators:
    """Berechnet technische Indikatoren für Forward-Preiszeitreihen."""

    @staticmethod
    def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index (Wilder's Smoothing)."""
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(50)

    @staticmethod
    def compute_rolling_percentile(prices: pd.Series, window: int = 60) -> pd.Series:
        """Wo steht der aktuelle Preis relativ zu den letzten N Tagen? (0-100)."""
        def pct_rank(x):
            if len(x) < 2:
                return 50.0
            current = x.iloc[-1]
            return (x < current).sum() / (len(x) - 1) * 100

        return prices.rolling(window, min_periods=max(5, window // 4)).apply(pct_rank, raw=False).fillna(50)

    @staticmethod
    def compute_sma(prices: pd.Series, window: int = 20) -> pd.Series:
        return prices.rolling(window, min_periods=1).mean()

    @staticmethod
    def compute_bollinger_bands(prices: pd.Series, window: int = 20, n_std: float = 2.0):
        sma = prices.rolling(window, min_periods=1).mean()
        std = prices.rolling(window, min_periods=1).std()
        return sma, sma + n_std * std, sma - n_std * std


# ============================================================================
# 14. BACKTESTING ENGINE
# ============================================================================

class BacktestEngine:
    """
    Institutionelle Backtesting-Engine für Energiebeschaffung.

    Kernprinzipien:
    1. Look-Ahead Bias Prevention: Nur Settlement-Preise verwenden,
       die am Kaufdatum bekannt waren
    2. Vektorisierte Kostenberechnung: Stündliche Spot-Kosten
    3. Realistische Transaktionskosten: Slippage & Gebühren
    4. Graceful Degradation: Fehlende Daten werden übersprungen
    """

    def __init__(
        self,
        spot_prices: pd.DataFrame,
        settlement_prices: pd.DataFrame,
        load_profile: pd.DataFrame,
        purchases: Optional[pd.DataFrame] = None,
    ):
        self.spot_raw = spot_prices.copy()
        self.settlements_raw = settlement_prices.copy()
        self.load_raw = load_profile.copy()
        self.purchases_raw = purchases.copy() if purchases is not None else None

        self._prepare_data()

    def _prepare_data(self):
        """Bereitet alle Lookup-Strukturen vor."""

        # --- Spot: stündlich, indiziert ---
        self.spot = self.spot_raw.copy()
        self.spot["timestamp"] = pd.to_datetime(self.spot["timestamp"])
        self.spot = self.spot.set_index("timestamp").sort_index()

        # Sicherstellen: Stündliche Auflösung
        if len(self.spot) > 0:
            freq = self.spot.index.to_series().diff().median()
            if freq and freq < pd.Timedelta("50min"):
                # Bereits stündlich oder feiner → auf stündlich resamplen
                self.spot_hourly = self.spot.resample("h")["price_eur_mwh"].mean()
            elif freq and freq > pd.Timedelta("2h"):
                # Täglich → auf stündlich interpolieren (Forward-Fill)
                self.spot_hourly = self.spot["price_eur_mwh"].resample("h").ffill()
            else:
                self.spot_hourly = self.spot["price_eur_mwh"]

        # --- Load: stündlich, indiziert ---
        self.load = self.load_raw.copy()
        self.load["timestamp"] = pd.to_datetime(self.load["timestamp"])
        self.load = self.load.set_index("timestamp").sort_index()
        self.load_hourly = self.load["volume_mwh"].resample("h").mean()

        # --- Monatliche Aggregation ---
        self.monthly_demand = self.load_hourly.resample("MS").sum()
        self.monthly_avg_spot = self.spot_hourly.resample("MS").mean()

        # --- Settlements: Lookup-Dictionary ---
        self.settlements = self.settlements_raw.copy()
        self.settlements["trade_date"] = pd.to_datetime(self.settlements["trade_date"])
        self.settlements["delivery_start"] = pd.to_datetime(self.settlements["delivery_start"])

        # Forward-Preis-Lookup: {product_name: DataFrame mit trade_date als Index}
        self.forward_curves: Dict[str, pd.DataFrame] = {}

        base_settlements = self.settlements[
            self.settlements["product_type"].str.lower() == "base"
        ]

        for product, grp in base_settlements.groupby("product"):
            curve = grp[["trade_date", "settlement_price"]].copy()
            curve = curve.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
            curve = curve.set_index("trade_date")
            self.forward_curves[product] = curve

        # --- Delivery Months: Gemeinsame Schnittmenge ---
        load_months = set(self.monthly_demand.index.strftime("%Y-%m"))
        spot_months = set(self.spot_hourly.resample("MS").mean().index.strftime("%Y-%m"))
        fwd_products = set(p.replace("M-", "") for p in self.forward_curves.keys() if p.startswith("M-"))

        self.available_months = sorted(load_months & spot_months & fwd_products)

        # --- Actual Purchases ---
        self.actual_purchases: Dict[str, List[Dict]] = {}
        if self.purchases_raw is not None:
            pr = self.purchases_raw.copy()
            pr["delivery_start"] = pd.to_datetime(pr["delivery_start"])
            pr["delivery_month"] = pr["delivery_start"].dt.strftime("%Y-%m")

            fwd_p = pr[pr["purchase_type"].str.lower() == "forward"]
            for dm, grp in fwd_p.groupby("delivery_month"):
                self.actual_purchases[dm] = grp.to_dict("records")

    # ----- Forward-Preis Lookup (ohne Look-Ahead Bias) -----

    def _get_forward_price_on_date(
        self, trade_date: pd.Timestamp, delivery_month: str
    ) -> Optional[float]:
        """
        Gibt den Settlement-Preis für delivery_month zurück, der am trade_date
        oder dem letzten vorangegangenen Handelstag verfügbar war.
        """
        product = f"M-{delivery_month}"
        curve = self.forward_curves.get(product)
        if curve is None or curve.empty:
            return None

        available = curve.loc[:trade_date]
        if available.empty:
            return None

        return float(available.iloc[-1]["settlement_price"])

    def _get_forward_curve_for_month(self, delivery_month: str) -> Optional[pd.DataFrame]:
        """Gibt die vollständige Forward-Kurve für einen Liefermonat zurück."""
        product = f"M-{delivery_month}"
        return self.forward_curves.get(product)

    # ----- Spot-Kosten Berechnung (vektorisiert) -----

    def _calculate_spot_cost_hourly(
        self, delivery_month: str, open_ratio: float
    ) -> Tuple[float, float]:
        """
        Berechnet stündliche Spotkosten für das offene Volumen.
        Returns: (total_spot_cost_eur, volume_weighted_avg_spot_price)
        """
        month_start = pd.Timestamp(f"{delivery_month}-01")
        month_end = month_start + pd.DateOffset(months=1)

        load_slice = self.load_hourly[month_start:month_end - pd.Timedelta("1h")]
        spot_slice = self.spot_hourly[month_start:month_end - pd.Timedelta("1h")]

        # Alignment: nur Stunden, wo beide vorhanden
        common_idx = load_slice.index.intersection(spot_slice.index)
        if len(common_idx) == 0:
            return 0.0, 0.0

        load_h = load_slice.loc[common_idx]
        spot_h = spot_slice.loc[common_idx]

        open_volume = load_h * open_ratio
        spot_cost = (open_volume * spot_h).sum()
        avg_spot = spot_cost / open_volume.sum() if open_volume.sum() > 0 else 0.0

        return float(spot_cost), float(avg_spot)

    # ----- Strategieausführung -----

    def _determine_buying_dates(
        self,
        delivery_month: str,
        config: StrategyConfig,
        forward_curve: pd.DataFrame,
    ) -> List[pd.Timestamp]:
        """Bestimmt die konkreten Kaufdaten basierend auf der Strategie."""

        delivery_start = pd.Timestamp(f"{delivery_month}-01")
        window_start = delivery_start - pd.DateOffset(months=config.buying_window_months)

        # Nur Handelstage im Fenster, an denen Forward-Preise existieren
        available_dates = forward_curve.loc[window_start:delivery_start - pd.Timedelta("1D")].index

        if len(available_dates) == 0:
            return []

        st_type = config.strategy_type

        if st_type in (StrategyType.DCA, StrategyType.BENCHMARK_FORWARD):
            # Gleichmäßig über das Kaufenster verteilt
            n = min(config.n_tranches, len(available_dates))
            if n <= 1:
                return [available_dates[-1]]
            indices = np.linspace(0, len(available_dates) - 1, n, dtype=int)
            return [available_dates[i] for i in indices]

        elif st_type == StrategyType.FRONT_LOADED:
            # Mehr Tranchen am Anfang des Fensters
            n = min(config.n_tranches, len(available_dates))
            # Gewichtung: erste Hälfte bekommt 2/3 der Tranchen
            first_half = int(n * 2 / 3)
            second_half = n - first_half
            first_dates = available_dates[:len(available_dates) // 2]
            second_dates = available_dates[len(available_dates) // 2:]
            dates = []
            if len(first_dates) > 0 and first_half > 0:
                idx1 = np.linspace(0, len(first_dates) - 1, first_half, dtype=int)
                dates.extend([first_dates[i] for i in idx1])
            if len(second_dates) > 0 and second_half > 0:
                idx2 = np.linspace(0, len(second_dates) - 1, second_half, dtype=int)
                dates.extend([second_dates[i] for i in idx2])
            return dates

        elif st_type == StrategyType.BACK_LOADED:
            # Mehr Tranchen am Ende
            n = min(config.n_tranches, len(available_dates))
            first_half = int(n * 1 / 3)
            second_half = n - first_half
            first_dates = available_dates[:len(available_dates) // 2]
            second_dates = available_dates[len(available_dates) // 2:]
            dates = []
            if len(first_dates) > 0 and first_half > 0:
                idx1 = np.linspace(0, len(first_dates) - 1, first_half, dtype=int)
                dates.extend([first_dates[i] for i in idx1])
            if len(second_dates) > 0 and second_half > 0:
                idx2 = np.linspace(0, len(second_dates) - 1, second_half, dtype=int)
                dates.extend([second_dates[i] for i in idx2])
            return dates

        elif st_type == StrategyType.LIMIT_BASED:
            # Kaufe an Tagen, wo Preis unter Perzentil-Schwelle
            prices = forward_curve.loc[available_dates, "settlement_price"]
            rolling_pctl = TechnicalIndicators.compute_rolling_percentile(
                prices, window=max(20, len(prices) // 3)
            )
            buy_signals = rolling_pctl < config.limit_percentile
            signal_dates = available_dates[buy_signals.values]

            if len(signal_dates) == 0:
                # Fallback: DCA
                n = min(config.n_tranches, len(available_dates))
                indices = np.linspace(0, len(available_dates) - 1, n, dtype=int)
                return [available_dates[i] for i in indices]

            # Maximal n_tranches auswählen
            if len(signal_dates) > config.n_tranches:
                idx = np.linspace(0, len(signal_dates) - 1, config.n_tranches, dtype=int)
                return [signal_dates[i] for i in idx]
            return list(signal_dates)

        elif st_type == StrategyType.RSI_TRIGGERED:
            prices = forward_curve.loc[available_dates, "settlement_price"]
            rsi = TechnicalIndicators.compute_rsi(prices, period=config.rsi_period)
            buy_signals = rsi < config.rsi_threshold
            signal_dates = available_dates[buy_signals.values]

            if len(signal_dates) == 0:
                n = min(config.n_tranches, len(available_dates))
                indices = np.linspace(0, len(available_dates) - 1, n, dtype=int)
                return [available_dates[i] for i in indices]

            if len(signal_dates) > config.n_tranches:
                idx = np.linspace(0, len(signal_dates) - 1, config.n_tranches, dtype=int)
                return [signal_dates[i] for i in idx]
            return list(signal_dates)

        elif st_type == StrategyType.SEASONAL:
            # Kaufe nur in bestimmten Monaten
            buy_months = config.buy_months
            seasonal_dates = [d for d in available_dates if d.month in buy_months]

            if len(seasonal_dates) == 0:
                seasonal_dates = list(available_dates)

            n = min(config.n_tranches, len(seasonal_dates))
            if n <= 1:
                return [seasonal_dates[0]] if seasonal_dates else []
            indices = np.linspace(0, len(seasonal_dates) - 1, n, dtype=int)
            return [seasonal_dates[i] for i in indices]

        return []

    def _execute_strategy_for_month(
        self, delivery_month: str, config: StrategyConfig
    ) -> MonthResult:
        """Führt eine Strategie für einen einzelnen Liefermonat aus."""

        month_start = pd.Timestamp(f"{delivery_month}-01")

        # Gesamtnachfrage
        demand = float(self.monthly_demand.get(month_start, 0))
        if demand <= 0:
            return MonthResult(
                delivery_month=delivery_month, delivery_start=month_start,
                total_demand_mwh=0, hedged_volume_mwh=0, open_volume_mwh=0,
                hedge_ratio_actual=0, forward_cost_eur=0, spot_cost_eur=0,
                total_cost_eur=0, avg_forward_price=0, avg_spot_price=0,
                blended_price=0, n_tranches_executed=0, tranches=[],
            )

        # --- Sonderfall: 100% Spot Benchmark ---
        if config.strategy_type == StrategyType.BENCHMARK_SPOT:
            spot_cost, avg_spot = self._calculate_spot_cost_hourly(delivery_month, 1.0)
            return MonthResult(
                delivery_month=delivery_month, delivery_start=month_start,
                total_demand_mwh=demand, hedged_volume_mwh=0, open_volume_mwh=demand,
                hedge_ratio_actual=0, forward_cost_eur=0, spot_cost_eur=spot_cost,
                total_cost_eur=spot_cost, avg_forward_price=0, avg_spot_price=avg_spot,
                blended_price=spot_cost / demand if demand > 0 else 0,
                n_tranches_executed=0, tranches=[],
            )

        # --- Sonderfall: Actual (echte Beschaffung) ---
        if config.strategy_type == StrategyType.ACTUAL:
            return self._execute_actual_for_month(delivery_month, demand, month_start)

        # --- Forward-Kurve laden ---
        curve = self._get_forward_curve_for_month(delivery_month)
        if curve is None or curve.empty:
            # Kein Forward verfügbar → alles Spot
            spot_cost, avg_spot = self._calculate_spot_cost_hourly(delivery_month, 1.0)
            return MonthResult(
                delivery_month=delivery_month, delivery_start=month_start,
                total_demand_mwh=demand, hedged_volume_mwh=0, open_volume_mwh=demand,
                hedge_ratio_actual=0, forward_cost_eur=0, spot_cost_eur=spot_cost,
                total_cost_eur=spot_cost, avg_forward_price=0, avg_spot_price=avg_spot,
                blended_price=spot_cost / demand if demand > 0 else 0,
                n_tranches_executed=0, tranches=[],
            )

        # --- Kaufdaten bestimmen ---
        buy_dates = self._determine_buying_dates(delivery_month, config, curve)

        if len(buy_dates) == 0:
            spot_cost, avg_spot = self._calculate_spot_cost_hourly(delivery_month, 1.0)
            return MonthResult(
                delivery_month=delivery_month, delivery_start=month_start,
                total_demand_mwh=demand, hedged_volume_mwh=0, open_volume_mwh=demand,
                hedge_ratio_actual=0, forward_cost_eur=0, spot_cost_eur=spot_cost,
                total_cost_eur=spot_cost, avg_forward_price=0, avg_spot_price=avg_spot,
                blended_price=spot_cost / demand if demand > 0 else 0,
                n_tranches_executed=0, tranches=[],
            )

        # --- Tranchen ausführen ---
        hedge_volume = demand * config.hedge_ratio
        volume_per_tranche = hedge_volume / len(buy_dates)

        tranches = []
        total_fwd_cost = 0.0

        for buy_date in buy_dates:
            price = self._get_forward_price_on_date(buy_date, delivery_month)
            if price is None:
                continue

            # Transaktionskosten
            price_with_costs = price * (1 + config.total_transaction_cost_pct) + config.broker_fee_eur_mwh

            tranche = TrancheExecution(
                execution_date=buy_date,
                delivery_month=delivery_month,
                volume_mwh=volume_per_tranche,
                price_eur_mwh=price_with_costs,
            )
            tranches.append(tranche)
            total_fwd_cost += tranche.cost_eur

        # --- Spot-Kosten für offenes Volumen ---
        hedged = sum(t.volume_mwh for t in tranches)
        open_ratio = max(0, 1.0 - hedged / demand) if demand > 0 else 1.0
        spot_cost, avg_spot = self._calculate_spot_cost_hourly(delivery_month, open_ratio)

        total_cost = total_fwd_cost + spot_cost
        avg_fwd = total_fwd_cost / hedged if hedged > 0 else 0

        return MonthResult(
            delivery_month=delivery_month,
            delivery_start=month_start,
            total_demand_mwh=demand,
            hedged_volume_mwh=hedged,
            open_volume_mwh=demand - hedged,
            hedge_ratio_actual=hedged / demand if demand > 0 else 0,
            forward_cost_eur=total_fwd_cost,
            spot_cost_eur=spot_cost,
            total_cost_eur=total_cost,
            avg_forward_price=avg_fwd,
            avg_spot_price=avg_spot,
            blended_price=total_cost / demand if demand > 0 else 0,
            n_tranches_executed=len(tranches),
            tranches=tranches,
        )

    def _execute_actual_for_month(
        self, delivery_month: str, demand: float, month_start: pd.Timestamp
    ) -> MonthResult:
        """Berechnet Kosten basierend auf echten Beschaffungsdaten."""

        actual = self.actual_purchases.get(delivery_month, [])

        tranches = []
        total_fwd_cost = 0.0
        hedged = 0.0

        for purchase in actual:
            vol = float(purchase.get("volume_mw", 0))
            price = float(purchase.get("price_eur_mwh", 0))
            # volume_mw ≈ MW für den ganzen Monat
            # Stunden im Monat berechnen
            month_end = month_start + pd.DateOffset(months=1)
            hours = (month_end - month_start).total_seconds() / 3600
            vol_mwh = vol * hours

            tranche = TrancheExecution(
                execution_date=pd.Timestamp(purchase.get("purchase_date", month_start)),
                delivery_month=delivery_month,
                volume_mwh=vol_mwh,
                price_eur_mwh=price,
            )
            tranches.append(tranche)
            total_fwd_cost += tranche.cost_eur
            hedged += vol_mwh

        # Begrenze auf Nachfrage
        if hedged > demand:
            scale = demand / hedged
            hedged = demand
            total_fwd_cost *= scale

        open_ratio = max(0, 1.0 - hedged / demand) if demand > 0 else 1.0
        spot_cost, avg_spot = self._calculate_spot_cost_hourly(delivery_month, open_ratio)

        total_cost = total_fwd_cost + spot_cost
        avg_fwd = total_fwd_cost / hedged if hedged > 0 else 0

        return MonthResult(
            delivery_month=delivery_month,
            delivery_start=month_start,
            total_demand_mwh=demand,
            hedged_volume_mwh=hedged,
            open_volume_mwh=demand - hedged,
            hedge_ratio_actual=hedged / demand if demand > 0 else 0,
            forward_cost_eur=total_fwd_cost,
            spot_cost_eur=spot_cost,
            total_cost_eur=total_cost,
            avg_forward_price=avg_fwd,
            avg_spot_price=avg_spot,
            blended_price=total_cost / demand if demand > 0 else 0,
            n_tranches_executed=len(tranches),
            tranches=tranches,
        )

    # ----- Hauptmethoden -----

    def run_single(self, config: StrategyConfig) -> BacktestResult:
        """Führt einen vollständigen Backtest für eine einzelne Strategie durch."""
        import time
        t0 = time.time()

        monthly_results = []
        for dm in self.available_months:
            result = self._execute_strategy_for_month(dm, config)
            monthly_results.append(result)

        elapsed = (time.time() - t0) * 1000

        return BacktestResult(
            strategy=config,
            monthly_results=monthly_results,
            execution_time_ms=elapsed,
        )

    def run_grid_search(
        self,
        strategy_type: StrategyType,
        hedge_ratios: Optional[List[float]] = None,
        n_tranches_options: Optional[List[int]] = None,
        base_config: Optional[StrategyConfig] = None,
    ) -> pd.DataFrame:
        """
        Grid Search: Testet systematisch verschiedene Hedge-Quoten
        (und optional Tranchenanzahlen).
        """
        if hedge_ratios is None:
            hedge_ratios = [i / 20 for i in range(21)]   # 0%, 5%, …, 100%

        if n_tranches_options is None:
            n_tranches_options = [6]

        results = []

        for hr, nt in itertools_product(hedge_ratios, n_tranches_options):
            cfg = StrategyConfig(
                name=f"{strategy_type.value} HR={hr:.0%} T={nt}",
                strategy_type=strategy_type,
                hedge_ratio=hr,
                n_tranches=nt,
                buying_window_months=base_config.buying_window_months if base_config else 6,
                limit_percentile=base_config.limit_percentile if base_config else 40,
                rsi_period=base_config.rsi_period if base_config else 14,
                rsi_threshold=base_config.rsi_threshold if base_config else 35,
                buy_months=base_config.buy_months if base_config else [4, 5, 6, 7],
                slippage_bps=base_config.slippage_bps if base_config else 5,
                broker_fee_eur_mwh=base_config.broker_fee_eur_mwh if base_config else 0.05,
            )
            bt_result = self.run_single(cfg)
            results.append({
                "hedge_ratio": hr,
                "n_tranches": nt,
                "strategy": strategy_type.value,
                "total_cost": bt_result.total_cost,
                "avg_price": bt_result.avg_blended_price,
                "total_demand": bt_result.total_demand,
                "fwd_cost": bt_result.total_forward_cost,
                "spot_cost": bt_result.total_spot_cost,
                "n_executed": len(bt_result.all_tranches),
            })

        return pd.DataFrame(results)

    def run_multi_strategy(
        self, configs: List[StrategyConfig]
    ) -> List[BacktestResult]:
        """Führt mehrere Strategien parallel aus."""
        return [self.run_single(cfg) for cfg in configs]

    def find_optimal(
        self,
        strategy_types: Optional[List[StrategyType]] = None,
        hedge_ratios: Optional[List[float]] = None,
        base_config: Optional[StrategyConfig] = None,
    ) -> pd.DataFrame:
        """
        Vollständige Optimierung: Findet die beste Kombination
        aus Strategie und Hedge-Quote.
        """
        if strategy_types is None:
            strategy_types = [
                StrategyType.DCA,
                StrategyType.FRONT_LOADED,
                StrategyType.BACK_LOADED,
                StrategyType.LIMIT_BASED,
                StrategyType.RSI_TRIGGERED,
                StrategyType.SEASONAL,
            ]

        if hedge_ratios is None:
            hedge_ratios = [i / 20 for i in range(21)]

        all_results = []
        for st in strategy_types:
            grid = self.run_grid_search(st, hedge_ratios, base_config=base_config)
            all_results.append(grid)

        return pd.concat(all_results, ignore_index=True)


# ============================================================================
# 15. BACKTESTING-VISUALISIERUNGEN
# ============================================================================

class BacktestVisualizer:
    """Erstellt alle Charts für die Backtesting-Analyse."""

    @staticmethod
    def monthly_cost_breakdown(result: BacktestResult) -> go.Figure:
        """Gestapeltes Balkendiagramm: Forward- vs. Spotkosten pro Monat."""
        df = result.to_dataframe()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df["delivery_month"], y=df["fwd_cost_eur"],
            name="Forward-Kosten", marker_color=Config.COLORS["forward"],
        ))
        fig.add_trace(go.Bar(
            x=df["delivery_month"], y=df["spot_cost_eur"],
            name="Spot-Kosten", marker_color=Config.COLORS["spot"],
        ))
        fig.update_layout(
            barmode="stack",
            title=f"Monatliche Kostenaufschlüsselung – {result.strategy.name}",
            xaxis_title="Liefermonat", yaxis_title="EUR",
            template="plotly_dark", height=440,
            xaxis_tickangle=-45,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        return fig

    @staticmethod
    def blended_price_timeline(result: BacktestResult) -> go.Figure:
        """Blended Price + Forward + Spot Price Timeline."""
        df = result.to_dataframe()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["delivery_month"], y=df["avg_spot_price"],
            name="Ø Spotpreis", mode="lines+markers",
            line=dict(color=Config.COLORS["spot"], width=1.5, dash="dot"),
        ))
        if (df["avg_fwd_price"] > 0).any():
            fig.add_trace(go.Scatter(
                x=df["delivery_month"], y=df["avg_fwd_price"],
                name="Ø Forward-Preis", mode="lines+markers",
                line=dict(color=Config.COLORS["forward"], width=1.5, dash="dot"),
            ))
        fig.add_trace(go.Scatter(
            x=df["delivery_month"], y=df["blended_price"],
            name="Blended Price", mode="lines+markers",
            line=dict(color=Config.COLORS["actual"], width=3),
            marker=dict(size=7),
        ))
        fig.update_layout(
            title=f"Preisentwicklung – {result.strategy.name}",
            yaxis_title="EUR/MWh", template="plotly_dark", height=420,
            xaxis_tickangle=-45, hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        return fig

    @staticmethod
    def cumulative_cost_comparison(results: List[BacktestResult]) -> go.Figure:
        """Kumulative Kostenkurven für mehrere Strategien."""
        fig = go.Figure()

        colors = [
            Config.COLORS["spot"], Config.COLORS["forward"],
            Config.COLORS["actual"], Config.COLORS["hedge"],
            Config.COLORS["warning"], Config.COLORS["info"],
            "#e74c3c", "#1abc9c",
        ]

        for i, res in enumerate(results):
            df = res.to_dataframe()
            df["cum_cost"] = df["total_cost_eur"].cumsum()
            color = colors[i % len(colors)]

            fig.add_trace(go.Scatter(
                x=df["delivery_month"], y=df["cum_cost"],
                mode="lines+markers", name=res.strategy.name,
                line=dict(color=color, width=2.5),
                marker=dict(size=4),
            ))

        fig.update_layout(
            title="Kumulative Beschaffungskosten – Strategievergleich",
            xaxis_title="Liefermonat", yaxis_title="Kumulative Kosten (EUR)",
            template="plotly_dark", height=480,
            xaxis_tickangle=-45, hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        return fig

    @staticmethod
    def tranche_timeline(result: BacktestResult, forward_curves: Dict) -> go.Figure:
        """Zeigt Tranchenkäufe auf der Forward-Kurve."""
        fig = go.Figure()

        # Forward-Kurven im Hintergrund (ausgewählte Delivery-Months)
        months = list(set(t.delivery_month for t in result.all_tranches))[:6]

        for dm in months:
            product = f"M-{dm}"
            curve = forward_curves.get(product)
            if curve is not None:
                fig.add_trace(go.Scatter(
                    x=curve.index, y=curve["settlement_price"],
                    mode="lines", name=f"Fwd {dm}",
                    line=dict(width=1, dash="dot"), opacity=0.5,
                ))

        # Tranchen als Punkte
        if result.all_tranches:
            t_dates = [t.execution_date for t in result.all_tranches]
            t_prices = [t.price_eur_mwh for t in result.all_tranches]
            t_volumes = [t.volume_mwh for t in result.all_tranches]
            t_months = [t.delivery_month for t in result.all_tranches]

            max_vol = max(t_volumes) if t_volumes else 1
            sizes = [v / max_vol * 25 + 6 for v in t_volumes]

            fig.add_trace(go.Scatter(
                x=t_dates, y=t_prices,
                mode="markers", name="Tranchenkäufe",
                marker=dict(
                    size=sizes, color=Config.COLORS["actual"],
                    symbol="diamond", line=dict(width=1.5, color="white"),
                ),
                customdata=list(zip(t_months, [f"{v:,.0f}" for v in t_volumes])),
                hovertemplate=(
                    "Kaufdatum: %{x|%Y-%m-%d}<br>"
                    "Preis: %{y:.2f} €/MWh<br>"
                    "Lieferung: %{customdata[0]}<br>"
                    "Volumen: %{customdata[1]} MWh<extra></extra>"
                ),
            ))

        fig.update_layout(
            title=f"Tranche Execution Timeline – {result.strategy.name}",
            xaxis_title="Kaufdatum", yaxis_title="EUR/MWh",
            template="plotly_dark", height=460,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        return fig

    @staticmethod
    def waterfall_chart(
        baseline_result: BacktestResult,
        comparison_result: BacktestResult,
    ) -> go.Figure:
        """Waterfall-Chart: Erklärt Kostendifferenz zwischen zwei Strategien."""

        baseline_cost = baseline_result.total_cost
        comp_cost = comparison_result.total_cost
        diff = comp_cost - baseline_cost

        # Zerlege in Komponenten
        baseline_spot = baseline_result.total_spot_cost
        baseline_fwd = baseline_result.total_forward_cost

        comp_spot = comparison_result.total_spot_cost
        comp_fwd = comparison_result.total_forward_cost

        spot_delta = comp_spot - baseline_spot
        fwd_delta = comp_fwd - baseline_fwd

        fig = go.Figure(go.Waterfall(
            x=[
                baseline_result.strategy.name,
                "Δ Forward-Kosten",
                "Δ Spot-Kosten",
                comparison_result.strategy.name,
            ],
            measure=["absolute", "relative", "relative", "total"],
            y=[baseline_cost, fwd_delta, spot_delta, comp_cost],
            text=[
                f"{baseline_cost:,.0f}€",
                f"{'+' if fwd_delta >= 0 else ''}{fwd_delta:,.0f}€",
                f"{'+' if spot_delta >= 0 else ''}{spot_delta:,.0f}€",
                f"{comp_cost:,.0f}€",
            ],
            textposition="outside",
            connector=dict(line=dict(color="rgba(63,63,63,0.7)")),
            increasing=dict(marker_color=Config.COLORS["danger"]),
            decreasing=dict(marker_color=Config.COLORS["success"]),
            totals=dict(marker_color=Config.COLORS["info"]),
        ))

        fig.update_layout(
            title="Kosten-Waterfall: Warum ist eine Strategie besser/schlechter?",
            yaxis_title="EUR", template="plotly_dark", height=440,
        )
        return fig

    @staticmethod
    def optimization_heatmap(grid_results: pd.DataFrame) -> go.Figure:
        """2D-Heatmap: Strategie × Hedge-Quote → Gesamtkosten."""

        pivot = grid_results.pivot_table(
            values="avg_price", index="strategy", columns="hedge_ratio", aggfunc="mean"
        )

        # Spalten formatieren
        col_labels = [f"{int(c * 100)}%" for c in pivot.columns]

        fig = go.Figure(go.Heatmap(
            z=pivot.values,
            x=col_labels,
            y=pivot.index.tolist(),
            colorscale="RdYlGn_r",
            colorbar_title="Ø €/MWh",
            text=np.round(pivot.values, 1),
            texttemplate="%{text}",
            textfont=dict(size=10),
        ))

        # Minimum markieren
        min_val = pivot.values.min()
        min_pos = np.unravel_index(pivot.values.argmin(), pivot.values.shape)

        fig.add_annotation(
            x=col_labels[min_pos[1]], y=pivot.index[min_pos[0]],
            text="★ OPTIMUM",
            showarrow=True, arrowhead=2, arrowsize=1.5,
            arrowcolor="white", font=dict(size=14, color="white"),
        )

        fig.update_layout(
            title="Szenario-Matrix: Ø Blended Price (€/MWh)",
            xaxis_title="Hedge-Quote", yaxis_title="Strategie",
            template="plotly_dark", height=max(350, 60 * len(pivot.index) + 100),
        )
        return fig

    @staticmethod
    def hedge_ratio_curve(grid_results: pd.DataFrame, strategy_name: str) -> go.Figure:
        """Kostenkurve über Hedge-Quoten für eine einzelne Strategie."""
        df = grid_results[grid_results["strategy"] == strategy_name].sort_values("hedge_ratio")

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(go.Scatter(
            x=df["hedge_ratio"] * 100, y=df["avg_price"],
            mode="lines+markers", name="Ø Blended Price",
            line=dict(color=Config.COLORS["actual"], width=3),
            marker=dict(size=6),
        ), secondary_y=False)

        fig.add_trace(go.Scatter(
            x=df["hedge_ratio"] * 100, y=df["total_cost"],
            mode="lines+markers", name="Gesamtkosten",
            line=dict(color=Config.COLORS["info"], width=2, dash="dot"),
        ), secondary_y=True)

        # Minimum markieren
        if len(df) > 0:
            min_idx = df["avg_price"].idxmin()
            min_row = df.loc[min_idx]
            fig.add_vline(
                x=min_row["hedge_ratio"] * 100,
                line_dash="dash", line_color=Config.COLORS["warning"],
                annotation_text=f"Optimum: {min_row['hedge_ratio']:.0%}",
            )

        fig.update_layout(
            title=f"Kosten vs. Hedge-Quote – {strategy_name}",
            xaxis_title="Hedge-Quote (%)",
            template="plotly_dark", height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        fig.update_yaxes(title_text="€/MWh", secondary_y=False)
        fig.update_yaxes(title_text="Gesamtkosten (EUR)", secondary_y=True)
        return fig

    @staticmethod
    def strategy_comparison_table(results: List[BacktestResult]) -> pd.DataFrame:
        """Vergleichstabelle aller Strategien."""
        records = []
        for res in results:
            records.append({
                "Strategie": res.strategy.name,
                "Hedge-Quote": f"{res.effective_hedge_ratio:.0%}",
                "Ø Preis (€/MWh)": round(res.avg_blended_price, 2),
                "Gesamtkosten (€)": round(res.total_cost, 0),
                "Fwd-Kosten (€)": round(res.total_forward_cost, 0),
                "Spot-Kosten (€)": round(res.total_spot_cost, 0),
                "Tranchen": len(res.all_tranches),
                "Nachfrage (MWh)": round(res.total_demand, 0),
            })

        df = pd.DataFrame(records)

        # Ranking hinzufügen
        if len(df) > 0:
            df["Rang"] = df["Gesamtkosten (€)"].rank().astype(int)
            df = df.sort_values("Rang")

        return df

    @staticmethod
    def savings_vs_benchmark(results: List[BacktestResult], benchmark: BacktestResult) -> go.Figure:
        """Balkendiagramm: Ersparnis vs. Benchmark (z.B. 100% Spot)."""
        names = []
        savings_pct = []
        savings_abs = []

        bm_cost = benchmark.total_cost

        for res in results:
            if res.strategy.name == benchmark.strategy.name:
                continue
            saving = bm_cost - res.total_cost
            pct = saving / bm_cost * 100 if bm_cost > 0 else 0
            names.append(res.strategy.name)
            savings_pct.append(pct)
            savings_abs.append(saving)

        colors = [Config.COLORS["success"] if s > 0 else Config.COLORS["danger"] for s in savings_pct]

        fig = go.Figure(go.Bar(
            x=names, y=savings_pct,
            marker_color=colors,
            text=[f"{s:+.1f}%" for s in savings_pct],
            textposition="outside",
            customdata=[f"{s:+,.0f} €" for s in savings_abs],
            hovertemplate="%{x}<br>Einsparung: %{text}<br>Absolut: %{customdata}<extra></extra>",
        ))
        fig.update_layout(
            title=f"Einsparung vs. Benchmark ({benchmark.strategy.name})",
            yaxis_title="Einsparung (%)",
            template="plotly_dark", height=400,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="white", line_width=0.5)
        return fig


# ============================================================================
# 16. PAGE: BACKTESTING
# ============================================================================

def page_backtesting():
    st.header("🔄 Backtesting Engine")
    st.markdown(
        """
        Kontrafaktische Analyse: **Was wäre gewesen, wenn…?**
        Testen Sie verschiedene Beschaffungsstrategien gegen historische Marktdaten –
        mit Look-Ahead Bias Prevention und realistischen Transaktionskosten.
        """
    )

    # --- Datenprüfung ---
    spot = DataStore.get("spot_prices")
    sett = DataStore.get("settlement_prices")
    load = DataStore.get("load_profile")
    purch = DataStore.get("purchases")

    missing = []
    if spot is None: missing.append("Spotpreise")
    if sett is None: missing.append("Settlement-Preise")
    if load is None: missing.append("Lastprofil")

    if missing:
        st.error(f"❌ Fehlende Daten: **{', '.join(missing)}**. Bitte unter Daten-Management laden.")
        return

    # --- Engine initialisieren ---
    @st.cache_resource(show_spinner=False)
    def get_engine(_spot, _sett, _load, _purch, _hash):
        return BacktestEngine(_spot, _sett, _load, _purch)

    # Hash für Cache-Invalidierung
    data_hash = f"{len(spot)}_{len(sett)}_{len(load)}_{len(purch) if purch is not None else 0}"

    with st.spinner("⚙️ Backtesting-Engine wird initialisiert..."):
        engine = get_engine(spot, sett, load, purch, data_hash)

    st.success(
        f"✅ Engine bereit: **{len(engine.available_months)} Liefermonate** verfügbar "
        f"({engine.available_months[0]} → {engine.available_months[-1]})"
    )

    # === TABS ===
    tab_single, tab_compare, tab_optimize = st.tabs([
        "🎛️ Einzelstrategie",
        "⚖️ Strategievergleich",
        "🔍 Optimierung & Grid Search",
    ])

    # =====================================================================
    # TAB 1: EINZELSTRATEGIE
    # =====================================================================
    with tab_single:
        st.subheader("🎛️ Strategie konfigurieren & testen")

        col_cfg, col_adv = st.columns([3, 2])

        with col_cfg:
            strategy_type = st.selectbox(
                "Strategietyp",
                [s for s in StrategyType if s not in (StrategyType.BENCHMARK_SPOT, StrategyType.BENCHMARK_FORWARD)],
                format_func=lambda x: x.value,
                key="bt_single_type",
            )

            c1, c2 = st.columns(2)
            with c1:
                hedge_ratio = st.slider(
                    "Hedge-Quote (Forward-Anteil)",
                    min_value=0, max_value=100, value=70, step=5,
                    format="%d%%", key="bt_single_hr",
                )
            with c2:
                n_tranches = st.slider(
                    "Anzahl Tranchen",
                    min_value=1, max_value=24, value=6,
                    key="bt_single_nt",
                )

            buying_window = st.slider(
                "Einkaufsfenster (Monate vor Lieferung)",
                min_value=1, max_value=12, value=6,
                key="bt_single_bw",
            )

        with col_adv:
            st.markdown("**🔧 Erweiterte Parameter**")

            if strategy_type == StrategyType.LIMIT_BASED:
                limit_pctl = st.slider(
                    "Limit-Perzentil",
                    min_value=5, max_value=80, value=40, step=5,
                    help="Kaufe wenn Preis unter dem X. Perzentil des Beobachtungsfensters",
                    key="bt_single_lp",
                )
            else:
                limit_pctl = 40

            if strategy_type == StrategyType.RSI_TRIGGERED:
                rsi_period = st.slider("RSI-Periode", 5, 30, 14, key="bt_single_rsi_p")
                rsi_thresh = st.slider("RSI-Schwelle", 10, 60, 35, key="bt_single_rsi_t")
            else:
                rsi_period, rsi_thresh = 14, 35

            if strategy_type == StrategyType.SEASONAL:
                buy_months = st.multiselect(
                    "Kaufmonate",
                    list(range(1, 13)),
                    default=[4, 5, 6, 7],
                    format_func=lambda m: [
                        "Jan", "Feb", "Mär", "Apr", "Mai", "Jun",
                        "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"
                    ][m - 1],
                    key="bt_single_sm",
                )
            else:
                buy_months = [4, 5, 6, 7]

            st.divider()
            st.markdown("**💰 Transaktionskosten**")
            slippage = st.number_input(
                "Slippage (Basis Points)",
                min_value=0.0, max_value=50.0, value=5.0, step=1.0,
                key="bt_single_slip",
            )
            broker_fee = st.number_input(
                "Brokergebühr (€/MWh)",
                min_value=0.0, max_value=2.0, value=0.05, step=0.01,
                key="bt_single_bf",
            )

        # --- Backtest ausführen ---
        if st.button("🚀 Backtest starten", type="primary", use_container_width=True, key="bt_single_run"):
            config = StrategyConfig(
                name=f"{strategy_type.value} ({hedge_ratio}%)",
                strategy_type=strategy_type,
                hedge_ratio=hedge_ratio / 100,
                n_tranches=n_tranches,
                buying_window_months=buying_window,
                limit_percentile=limit_pctl,
                rsi_period=rsi_period,
                rsi_threshold=rsi_thresh,
                buy_months=buy_months,
                slippage_bps=slippage,
                broker_fee_eur_mwh=broker_fee,
            )

            # Benchmark: 100% Spot
            benchmark_config = StrategyConfig(
                name="100% Spot (Benchmark)",
                strategy_type=StrategyType.BENCHMARK_SPOT,
                hedge_ratio=0.0,
            )

            with st.spinner("⏳ Backtesting läuft..."):
                result = engine.run_single(config)
                benchmark = engine.run_single(benchmark_config)

            st.session_state["bt_single_result"] = result
            st.session_state["bt_single_benchmark"] = benchmark

        # --- Ergebnisse anzeigen ---
        result = st.session_state.get("bt_single_result")
        benchmark = st.session_state.get("bt_single_benchmark")

        if result is not None and benchmark is not None:
            st.divider()
            st.subheader("📊 Ergebnisse")

            # KPIs
            saving_abs = benchmark.total_cost - result.total_cost
            saving_pct = saving_abs / benchmark.total_cost * 100 if benchmark.total_cost > 0 else 0

            kpi_cols = st.columns(6)
            kpi_data = [
                ("Gesamtkosten", f"€ {result.total_cost:,.0f}", None),
                ("Ø Blended Price", f"{result.avg_blended_price:.2f} €/MWh", None),
                ("Eff. Hedge-Quote", f"{result.effective_hedge_ratio:.0%}", None),
                ("vs. 100% Spot", f"{saving_pct:+.1f}%",
                 f"{'Ersparnis' if saving_abs > 0 else 'Mehrkosten'}: € {abs(saving_abs):,.0f}"),
                ("Tranchen", f"{len(result.all_tranches)}", None),
                ("Engine-Zeit", f"{result.execution_time_ms:.0f} ms", None),
            ]

            for col, (label, value, delta) in zip(kpi_cols, kpi_data):
                if delta:
                    col.metric(label, value, delta)
                else:
                    col.metric(label, value)

            # Charts
            c1, c2 = st.columns(2)

            with c1:
                fig_cost = BacktestVisualizer.monthly_cost_breakdown(result)
                st.plotly_chart(fig_cost, use_container_width=True)

            with c2:
                fig_price = BacktestVisualizer.blended_price_timeline(result)
                st.plotly_chart(fig_price, use_container_width=True)

            # Tranche Timeline
            fig_tranche = BacktestVisualizer.tranche_timeline(result, engine.forward_curves)
            st.plotly_chart(fig_tranche, use_container_width=True)

            # Kumulative Kosten vs. Benchmark
            fig_cum = BacktestVisualizer.cumulative_cost_comparison([benchmark, result])
            st.plotly_chart(fig_cum, use_container_width=True)

            # Waterfall
            fig_wf = BacktestVisualizer.waterfall_chart(benchmark, result)
            st.plotly_chart(fig_wf, use_container_width=True)

            # Detailtabelle
            with st.expander("📋 Monatliche Detailergebnisse"):
                df_detail = result.to_dataframe()
                st.dataframe(
                    df_detail.style.format({
                        "demand_mwh": "{:,.0f}",
                        "hedged_mwh": "{:,.0f}",
                        "open_mwh": "{:,.0f}",
                        "hedge_ratio": "{:.0%}",
                        "fwd_cost_eur": "{:,.0f}",
                        "spot_cost_eur": "{:,.0f}",
                        "total_cost_eur": "{:,.0f}",
                        "avg_fwd_price": "{:.2f}",
                        "avg_spot_price": "{:.2f}",
                        "blended_price": "{:.2f}",
                    }).background_gradient(subset=["blended_price"], cmap="RdYlGn_r"),
                    use_container_width=True,
                    height=500,
                )

    # =====================================================================
    # TAB 2: STRATEGIEVERGLEICH
    # =====================================================================
    with tab_compare:
        st.subheader("⚖️ Multi-Strategie-Vergleich")
        st.markdown(
            "Konfigurieren und vergleichen Sie bis zu **6 Strategien** gleichzeitig."
        )

        # Gemeinsame Parameter
        with st.expander("🔧 Gemeinsame Parameter", expanded=True):
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                comp_window = st.slider("Einkaufsfenster (Monate)", 1, 12, 6, key="comp_bw")
            with cc2:
                comp_slip = st.number_input("Slippage (bps)", 0.0, 50.0, 5.0, key="comp_slip")
            with cc3:
                comp_fee = st.number_input("Broker (€/MWh)", 0.0, 2.0, 0.05, key="comp_fee")

        # Strategie-Konfiguratoren
        st.markdown("---")
        n_strategies = st.number_input(
            "Anzahl Strategien", min_value=2, max_value=6, value=4, key="comp_n"
        )

        configs = []
        cols = st.columns(min(int(n_strategies), 3))

        strategy_presets = [
            (StrategyType.DCA, 70, "DCA 70%"),
            (StrategyType.DCA, 30, "DCA 30%"),
            (StrategyType.LIMIT_BASED, 60, "Limit 60%"),
            (StrategyType.RSI_TRIGGERED, 50, "RSI 50%"),
            (StrategyType.SEASONAL, 80, "Saisonal 80%"),
            (StrategyType.FRONT_LOADED, 70, "FrontLoad 70%"),
        ]

        for i in range(int(n_strategies)):
            col_idx = i % len(cols)
            preset = strategy_presets[i] if i < len(strategy_presets) else strategy_presets[0]

            with cols[col_idx]:
                st.markdown(f"**Strategie {i + 1}**")

                st_type_options = [s for s in StrategyType
                                  if s not in (StrategyType.BENCHMARK_SPOT, StrategyType.BENCHMARK_FORWARD)]
                st_idx = st_type_options.index(preset[0]) if preset[0] in st_type_options else 0

                s_type = st.selectbox(
                    "Typ", st_type_options,
                    index=st_idx,
                    format_func=lambda x: x.value,
                    key=f"comp_type_{i}",
                )
                s_hr = st.slider(
                    "Hedge-Quote %", 0, 100, preset[1], 5,
                    key=f"comp_hr_{i}",
                )
                s_nt = st.slider(
                    "Tranchen", 1, 24, 6,
                    key=f"comp_nt_{i}",
                )
                s_name = st.text_input(
                    "Name", value=preset[2] if i < len(strategy_presets) else f"Strategie {i+1}",
                    key=f"comp_name_{i}",
                )

                configs.append(StrategyConfig(
                    name=s_name,
                    strategy_type=s_type,
                    hedge_ratio=s_hr / 100,
                    n_tranches=s_nt,
                    buying_window_months=comp_window,
                    slippage_bps=comp_slip,
                    broker_fee_eur_mwh=comp_fee,
                ))

        # Benchmarks immer hinzufügen
        configs_with_bm = [
            StrategyConfig(
                name="100% Spot",
                strategy_type=StrategyType.BENCHMARK_SPOT,
                hedge_ratio=0,
            ),
        ] + configs

        if purch is not None:
            configs_with_bm.append(
                StrategyConfig(
                    name="Actual (Echte Beschaffung)",
                    strategy_type=StrategyType.ACTUAL,
                    hedge_ratio=1.0,
                )
            )

        # Backtest ausführen
        if st.button("🚀 Vergleich starten", type="primary", use_container_width=True, key="comp_run"):
            with st.spinner(f"⏳ {len(configs_with_bm)} Strategien werden simuliert..."):
                all_results = engine.run_multi_strategy(configs_with_bm)

            st.session_state["comp_results"] = all_results

        # Ergebnisse anzeigen
        comp_results = st.session_state.get("comp_results")

        if comp_results is not None:
            st.divider()
            st.subheader("📊 Vergleichsergebnis")

            # Vergleichstabelle
            comp_table = BacktestVisualizer.strategy_comparison_table(comp_results)
            st.dataframe(
                comp_table.style.format({
                    "Gesamtkosten (€)": "{:,.0f}",
                    "Fwd-Kosten (€)": "{:,.0f}",
                    "Spot-Kosten (€)": "{:,.0f}",
                    "Nachfrage (MWh)": "{:,.0f}",
                    "Ø Preis (€/MWh)": "{:.2f}",
                }).background_gradient(subset=["Ø Preis (€/MWh)"], cmap="RdYlGn_r"),
                use_container_width=True,
                hide_index=True,
            )

            # Charts
            # Kumulative Kosten
            fig_cum_comp = BacktestVisualizer.cumulative_cost_comparison(comp_results)
            st.plotly_chart(fig_cum_comp, use_container_width=True)

            # Einsparung vs. Benchmark
            benchmark_spot = comp_results[0]  # 100% Spot ist immer der erste
            fig_sav = BacktestVisualizer.savings_vs_benchmark(comp_results, benchmark_spot)
            st.plotly_chart(fig_sav, use_container_width=True)

            # Waterfall für beste vs. schlechteste Strategie
            costs = [r.total_cost for r in comp_results]
            best_idx = np.argmin(costs)
            worst_idx = np.argmax(costs)
            if best_idx != worst_idx:
                st.subheader("🔍 Waterfall: Beste vs. Schlechteste")
                fig_wf_comp = BacktestVisualizer.waterfall_chart(
                    comp_results[worst_idx], comp_results[best_idx]
                )
                st.plotly_chart(fig_wf_comp, use_container_width=True)

            # Monatlicher Preisvergleich
            st.subheader("📈 Monatlicher Blended Price – alle Strategien")
            fig_monthly_all = go.Figure()
            colors = [
                Config.COLORS["spot"], Config.COLORS["forward"],
                Config.COLORS["actual"], Config.COLORS["hedge"],
                Config.COLORS["warning"], Config.COLORS["info"],
                "#e74c3c", "#1abc9c",
            ]
            for i, res in enumerate(comp_results):
                df_r = res.to_dataframe()
                fig_monthly_all.add_trace(go.Scatter(
                    x=df_r["delivery_month"], y=df_r["blended_price"],
                    mode="lines+markers", name=res.strategy.name,
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=4),
                ))
            fig_monthly_all.update_layout(
                yaxis_title="Blended Price (€/MWh)",
                template="plotly_dark", height=460,
                xaxis_tickangle=-45, hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_monthly_all, use_container_width=True)

    # =====================================================================
    # TAB 3: OPTIMIERUNG & GRID SEARCH
    # =====================================================================
    with tab_optimize:
        st.subheader("🔍 Systematische Optimierung")
        st.markdown(
            """
            **Grid Search** über alle Strategietypen und Hedge-Quoten.
            Findet automatisch die historisch optimale Konfiguration.

            ⚠️ *Hinweis: Vergangene Ergebnisse sind keine Garantie für die Zukunft.
            Overfitting auf historische Daten vermeiden!*
            """
        )

        # Konfiguration
        c1, c2 = st.columns(2)
        with c1:
            opt_strategies = st.multiselect(
                "Strategietypen",
                [s for s in StrategyType if s not in (
                    StrategyType.BENCHMARK_SPOT, StrategyType.BENCHMARK_FORWARD, StrategyType.ACTUAL
                )],
                default=[StrategyType.DCA, StrategyType.LIMIT_BASED, StrategyType.RSI_TRIGGERED],
                format_func=lambda x: x.value,
                key="opt_strats",
            )

        with c2:
            hr_step = st.select_slider(
                "Hedge-Quoten-Schrittweite",
                options=[2, 5, 10, 20],
                value=5,
                format_func=lambda x: f"{x}%",
                key="opt_step",
            )

        opt_window = st.slider(
            "Einkaufsfenster (Monate)", 1, 12, 6, key="opt_bw"
        )

        hedge_ratios = [i / 100 for i in range(0, 101, hr_step)]

        n_combinations = len(opt_strategies) * len(hedge_ratios)
        st.info(f"📊 **{n_combinations} Kombinationen** werden getestet.")

        if st.button(
            "🧬 Optimierung starten", type="primary",
            use_container_width=True, key="opt_run"
        ):
            base_cfg = StrategyConfig(
                name="base", strategy_type=StrategyType.DCA,
                buying_window_months=opt_window,
            )

            progress = st.progress(0, text="Starte Grid Search...")
            all_grids = []

            for i, st_type in enumerate(opt_strategies):
                progress.progress(
                    (i + 1) / len(opt_strategies),
                    text=f"Teste {st_type.value}... ({i + 1}/{len(opt_strategies)})",
                )
                grid = engine.run_grid_search(
                    st_type, hedge_ratios, base_config=base_cfg
                )
                all_grids.append(grid)

            progress.progress(1.0, text="Fertig!")

            full_grid = pd.concat(all_grids, ignore_index=True)

            # Benchmark hinzufügen
            bm_result = engine.run_single(StrategyConfig(
                name="100% Spot", strategy_type=StrategyType.BENCHMARK_SPOT, hedge_ratio=0
            ))
            full_grid["savings_vs_spot_pct"] = (
                (bm_result.total_cost - full_grid["total_cost"]) / bm_result.total_cost * 100
            )

            st.session_state["opt_grid"] = full_grid
            st.session_state["opt_benchmark"] = bm_result

        # --- Ergebnisse ---
        grid = st.session_state.get("opt_grid")
        bm = st.session_state.get("opt_benchmark")

        if grid is not None and bm is not None:
            st.divider()

            # Optimales Ergebnis
            best = grid.loc[grid["avg_price"].idxmin()]
            worst = grid.loc[grid["avg_price"].idxmax()]

            st.subheader("🏆 Optimales Ergebnis")
            bcols = st.columns(4)
            bcols[0].metric("Beste Strategie", best["strategy"][:30])
            bcols[1].metric("Optimale Hedge-Quote", f"{best['hedge_ratio']:.0%}")
            bcols[2].metric("Ø Preis", f"{best['avg_price']:.2f} €/MWh")
            bcols[3].metric(
                "Ersparnis vs. 100% Spot",
                f"{best['savings_vs_spot_pct']:.1f}%",
                f"€ {bm.total_cost - best['total_cost']:,.0f}",
            )

            # Heatmap
            st.subheader("🗺️ Szenario-Heatmap")
            fig_hm = BacktestVisualizer.optimization_heatmap(grid)
            st.plotly_chart(fig_hm, use_container_width=True)

            # Kosten-Kurven pro Strategie
            st.subheader("📈 Kosten vs. Hedge-Quote")

            strategies_in_grid = grid["strategy"].unique()
            n_cols = min(len(strategies_in_grid), 2)
            chart_cols = st.columns(n_cols)

            for i, strat_name in enumerate(strategies_in_grid):
                with chart_cols[i % n_cols]:
                    fig_curve = BacktestVisualizer.hedge_ratio_curve(grid, strat_name)
                    st.plotly_chart(fig_curve, use_container_width=True)

            # Top 10 Tabelle
            st.subheader("🏅 Top 10 Konfigurationen")
            top10 = grid.nsmallest(10, "avg_price")[
                ["strategy", "hedge_ratio", "avg_price", "total_cost", "savings_vs_spot_pct"]
            ].copy()
            top10.columns = ["Strategie", "Hedge-Quote", "Ø Preis (€/MWh)", "Gesamtkosten (€)", "Ersparnis (%)"]
            top10.index = range(1, len(top10) + 1)
            top10.index.name = "Rang"

            st.dataframe(
                top10.style.format({
                    "Hedge-Quote": "{:.0%}",
                    "Ø Preis (€/MWh)": "{:.2f}",
                    "Gesamtkosten (€)": "{:,.0f}",
                    "Ersparnis (%)": "{:+.1f}%",
                }).background_gradient(subset=["Ø Preis (€/MWh)"], cmap="RdYlGn_r"),
                use_container_width=True,
            )

            # Spreizung (bester vs. schlechtester Fall)
            st.subheader("📊 Gesamtanalyse")
            spread = worst["avg_price"] - best["avg_price"]
            spread_pct = spread / worst["avg_price"] * 100

            acols = st.columns(3)
            acols[0].metric(
                "Preisspreizung (Best ↔ Worst)",
                f"{spread:.2f} €/MWh",
                f"{spread_pct:.1f}% Bandbreite",
            )
            acols[1].metric(
                "100% Spot Benchmark",
                f"{bm.avg_blended_price:.2f} €/MWh",
                f"€ {bm.total_cost:,.0f}",
            )
            acols[2].metric(
                "Getestete Kombinationen",
                f"{len(grid)}",
                f"{len(grid['strategy'].unique())} Strategien × {len(grid['hedge_ratio'].unique())} Quoten",
            )

            # Download
            csv = grid.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Grid-Search-Ergebnisse als CSV",
                csv,
                "grid_search_results.csv",
                "text/csv",
                use_container_width=True,
      )
  # ============================================================================
# ============================================================================
#
#  MODULE 3 :  RISK ENGINE & PERFORMANCE ANALYTICS
#  VaR · CVaR · Monte Carlo · MtM · Hedge Effectiveness · Stress Testing
#
# ============================================================================
# ============================================================================

from scipy import stats as scipy_stats
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================================
# 19. RISIKO-METRIKEN BERECHNUNG
# ============================================================================

class RiskMetrics:
    """
    Institutionelle Risikokennzahlen für Energieportfolios.

    Berechnet:
    - Parametrischer VaR (Gauß + Cornish-Fisher)
    - Historischer VaR
    - Monte-Carlo VaR
    - Conditional VaR / Expected Shortfall
    - Volatilitätsmaße
    - Drawdown-Metriken
    """

    # ---- Value at Risk ----

    @staticmethod
    def parametric_var(
        returns: pd.Series,
        confidence: float = 0.95,
        holding_period: int = 1,
    ) -> float:
        """
        Parametrischer VaR unter Normalverteilungsannahme.
        Negatives Vorzeichen = potentieller Verlust.
        """
        mu = returns.mean()
        sigma = returns.std()
        z = scipy_stats.norm.ppf(1 - confidence)
        var = -(mu * holding_period + z * sigma * np.sqrt(holding_period))
        return float(var)

    @staticmethod
    def cornish_fisher_var(
        returns: pd.Series,
        confidence: float = 0.95,
        holding_period: int = 1,
    ) -> float:
        """
        Cornish-Fisher-VaR: Korrigiert für Schiefe und Kurtosis.
        Wesentlich realistischer für Energiemärkte mit Fat Tails.
        """
        mu = returns.mean()
        sigma = returns.std()
        skew = returns.skew()
        kurt = returns.kurtosis()  # Excess Kurtosis

        z = scipy_stats.norm.ppf(1 - confidence)

        # Cornish-Fisher-Expansion
        z_cf = (
            z
            + (z**2 - 1) * skew / 6
            + (z**3 - 3 * z) * kurt / 24
            - (2 * z**3 - 5 * z) * skew**2 / 36
        )

        var = -(mu * holding_period + z_cf * sigma * np.sqrt(holding_period))
        return float(var)

    @staticmethod
    def historical_var(
        returns: pd.Series,
        confidence: float = 0.95,
    ) -> float:
        """Historischer VaR: Direkt aus der empirischen Verteilung."""
        return float(-returns.quantile(1 - confidence))

    @staticmethod
    def conditional_var(
        returns: pd.Series,
        confidence: float = 0.95,
    ) -> float:
        """
        Conditional VaR (Expected Shortfall / CVaR):
        Durchschnittlicher Verlust jenseits des VaR.
        """
        var_threshold = returns.quantile(1 - confidence)
        tail_losses = returns[returns <= var_threshold]
        if len(tail_losses) == 0:
            return float(-var_threshold)
        return float(-tail_losses.mean())

    # ---- Volatilitätsmaße ----

    @staticmethod
    def annualized_volatility(
        returns: pd.Series,
        periods_per_year: int = 252,
    ) -> float:
        return float(returns.std() * np.sqrt(periods_per_year))

    @staticmethod
    def rolling_volatility(
        returns: pd.Series,
        window: int = 30,
        periods_per_year: int = 252,
    ) -> pd.Series:
        return returns.rolling(window, min_periods=max(5, window // 4)).std() * np.sqrt(periods_per_year)

    @staticmethod
    def ewma_volatility(
        returns: pd.Series,
        span: int = 30,
        periods_per_year: int = 252,
    ) -> pd.Series:
        """Exponentiell gewichtete Volatilität (RiskMetrics-Style)."""
        return returns.ewm(span=span).std() * np.sqrt(periods_per_year)

    # ---- Drawdown-Analyse ----

    @staticmethod
    def compute_drawdowns(cumulative_costs: pd.Series) -> pd.DataFrame:
        """
        Berechnet Drawdowns relativ zum Running Minimum der kumulativen Kosten.
        Bei Kosten ist ein 'Drawdown' = Kosten steigen über das bisherige Minimum.
        """
        running_min = cumulative_costs.cummin()
        drawdown = cumulative_costs - running_min
        drawdown_pct = drawdown / running_min.replace(0, np.nan) * 100

        return pd.DataFrame({
            "cumulative_cost": cumulative_costs,
            "running_min": running_min,
            "drawdown_abs": drawdown,
            "drawdown_pct": drawdown_pct.fillna(0),
        })

    @staticmethod
    def max_drawdown(cumulative_costs: pd.Series) -> Dict[str, Any]:
        """Maximum Drawdown mit Start-/End-Datum."""
        dd = RiskMetrics.compute_drawdowns(cumulative_costs)
        max_dd_idx = dd["drawdown_abs"].idxmax()

        if pd.isna(max_dd_idx):
            return {"max_drawdown_abs": 0, "max_drawdown_pct": 0, "peak_date": None, "trough_date": None}

        max_dd_abs = dd.loc[max_dd_idx, "drawdown_abs"]
        max_dd_pct = dd.loc[max_dd_idx, "drawdown_pct"]

        # Peak finden (letztes Running-Min vor dem Drawdown-Maximum)
        prior = dd.loc[:max_dd_idx]
        peak_date = prior["running_min"].idxmin() if len(prior) > 0 else max_dd_idx

        return {
            "max_drawdown_abs": float(max_dd_abs),
            "max_drawdown_pct": float(max_dd_pct),
            "peak_date": peak_date,
            "trough_date": max_dd_idx,
        }

    # ---- Tail-Risk-Metriken ----

    @staticmethod
    def tail_ratio(returns: pd.Series, percentile: float = 5.0) -> float:
        """Verhältnis rechter/linker Tail – > 1 = mehr Upside als Downside."""
        right = np.percentile(returns.dropna(), 100 - percentile)
        left = abs(np.percentile(returns.dropna(), percentile))
        return float(right / left) if left != 0 else float("inf")

    @staticmethod
    def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
        """Omega Ratio: Wahrscheinlichkeitsgewichtetes Gain/Loss-Verhältnis."""
        excess = returns - threshold
        gains = excess[excess > 0].sum()
        losses = abs(excess[excess <= 0].sum())
        return float(gains / losses) if losses != 0 else float("inf")


# ============================================================================
# 20. MARK-TO-MARKET ENGINE
# ============================================================================

class MarkToMarketEngine:
    """
    Berechnet den täglichen Mark-to-Market-Wert offener Forward-Positionen.

    MtM = (Aktueller Settlement - Kaufpreis) × Restvolumen

    Positiver MtM = Position im Gewinn (Markt ist gestiegen seit Kauf)
    Negativer MtM = Position im Verlust
    """

    @staticmethod
    def compute_mtm_for_tranches(
        tranches: List[TrancheExecution],
        settlement_prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Berechnet tägliche MtM-Bewertung für alle Tranchen.
        """
        if not tranches:
            return pd.DataFrame(columns=[
                "date", "total_mtm", "unrealized_pnl", "n_open_positions",
                "total_exposure_mwh",
            ])

        sett_base = settlement_prices[
            settlement_prices["product_type"].str.lower() == "base"
        ].copy()

        records = []

        # Alle relevanten Handelstage
        all_dates = sorted(sett_base["trade_date"].dt.date.unique())

        for eval_date in all_dates:
            eval_ts = pd.Timestamp(eval_date)
            total_mtm = 0.0
            n_open = 0
            total_exposure = 0.0

            for tranche in tranches:
                # Position ist offen zwischen Kaufdatum und Lieferbeginn
                delivery_start = pd.Timestamp(f"{tranche.delivery_month}-01")

                if tranche.execution_date <= eval_ts < delivery_start:
                    # Aktuelle Marktbewertung holen
                    product = f"M-{tranche.delivery_month}"
                    current_sett = sett_base[
                        (sett_base["trade_date"].dt.date <= eval_date)
                        & (sett_base["product"] == product)
                    ]

                    if current_sett.empty:
                        continue

                    current_price = current_sett.iloc[-1]["settlement_price"]
                    mtm = (current_price - tranche.price_eur_mwh) * tranche.volume_mwh

                    total_mtm += mtm
                    n_open += 1
                    total_exposure += tranche.volume_mwh

            records.append({
                "date": eval_date,
                "total_mtm": total_mtm,
                "unrealized_pnl": total_mtm,
                "n_open_positions": n_open,
                "total_exposure_mwh": total_exposure,
            })

        return pd.DataFrame(records)


# ============================================================================
# 21. HEDGE EFFECTIVENESS ANALYSER
# ============================================================================

class HedgeEffectivenessAnalyser:
    """
    Berechnet die Hedge-Effektivität mittels:
    1. Dollar-Offset-Methode (einfachste)
    2. Regressionsbasiert (IAS 39 / IFRS 9 konform)
    3. Varianzreduktions-Methode
    """

    @staticmethod
    def dollar_offset(
        hedge_pnl: pd.Series,
        underlying_pnl: pd.Series,
    ) -> pd.DataFrame:
        """
        Dollar-Offset-Ratio: Veränderung Hedge / Veränderung Underlying.
        Effektiver Hedge: Ratio nahe -1.0 (zwischen -0.8 und -1.25 gilt als effektiv).
        """
        # Kumuliert
        cum_hedge = hedge_pnl.cumsum()
        cum_underlying = underlying_pnl.cumsum()

        ratio = cum_hedge / cum_underlying.replace(0, np.nan)
        ratio = ratio.fillna(0)

        return pd.DataFrame({
            "date": hedge_pnl.index,
            "cum_hedge_pnl": cum_hedge.values,
            "cum_underlying_pnl": cum_underlying.values,
            "dollar_offset_ratio": ratio.values,
        })

    @staticmethod
    def regression_analysis(
        hedge_returns: pd.Series,
        underlying_returns: pd.Series,
    ) -> Dict[str, Any]:
        """
        Regressionsbasierte Hedge-Effektivität.

        hedge_returns = α + β × underlying_returns + ε

        Effektiv wenn:
        - R² > 0.80 (IAS 39: 80-125% Dollar Offset)
        - β nahe -1.0
        - α nahe 0
        """
        valid = pd.DataFrame({
            "hedge": hedge_returns,
            "underlying": underlying_returns,
        }).dropna()

        if len(valid) < 10:
            return {
                "r_squared": 0, "beta": 0, "alpha": 0, "p_value": 1,
                "std_error": 0, "n_obs": len(valid), "is_effective": False,
                "effectiveness_pct": 0,
            }

        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
            valid["underlying"], valid["hedge"]
        )

        r_sq = r_value**2
        is_effective = r_sq >= 0.80 and -1.25 <= slope <= -0.80

        return {
            "r_squared": float(r_sq),
            "beta": float(slope),
            "alpha": float(intercept),
            "p_value": float(p_value),
            "std_error": float(std_err),
            "n_obs": len(valid),
            "is_effective": is_effective,
            "effectiveness_pct": float(r_sq * 100),
        }

    @staticmethod
    def variance_reduction(
        hedged_costs: pd.Series,
        unhedged_costs: pd.Series,
    ) -> Dict[str, float]:
        """
        Varianzreduktions-Methode:
        VR = 1 - Var(hedged) / Var(unhedged)
        """
        var_hedged = hedged_costs.var()
        var_unhedged = unhedged_costs.var()

        vr = 1 - var_hedged / var_unhedged if var_unhedged > 0 else 0

        return {
            "variance_hedged": float(var_hedged),
            "variance_unhedged": float(var_unhedged),
            "variance_reduction": float(vr),
            "variance_reduction_pct": float(vr * 100),
            "std_hedged": float(np.sqrt(var_hedged)),
            "std_unhedged": float(np.sqrt(var_unhedged)),
        }


# ============================================================================
# 22. MONTE-CARLO SIMULATION ENGINE
# ============================================================================

class MonteCarloEngine:
    """
    Monte-Carlo-Simulation für Energiepreise.

    Implementiert:
    - Geometric Brownian Motion (GBM)
    - Ornstein-Uhlenbeck (Mean-Reverting)
    - Mean-Reverting Jump Diffusion (Merton-Modell für Energie)

    Kalibriert auf historische Spotpreise.
    """

    @staticmethod
    def calibrate_ou_parameters(prices: pd.Series, dt: float = 1/252) -> Dict[str, float]:
        """
        Kalibriert Ornstein-Uhlenbeck-Parameter aus historischen Preisen.

        dX = θ(μ - X)dt + σdW

        Methode: OLS auf diskretisierter Form: X(t+1) - X(t) = a + b·X(t) + ε
        """
        X = prices.values[:-1]
        dX = np.diff(prices.values)

        # OLS
        n = len(X)
        if n < 20:
            return {"theta": 0.1, "mu": prices.mean(), "sigma": prices.std() * 0.1, "n_obs": n}

        A = np.column_stack([np.ones(n), X])
        params, residuals, _, _ = np.linalg.lstsq(A, dX, rcond=None)

        a, b = params
        theta = -b / dt
        mu = -a / b if abs(b) > 1e-10 else float(prices.mean())
        sigma_resid = np.std(dX - a - b * X)
        sigma = sigma_resid / np.sqrt(dt)

        # Plausibilität
        theta = max(0.001, min(theta, 100))
        mu = max(0, mu)
        sigma = max(0.1, sigma)

        return {
            "theta": float(theta),
            "mu": float(mu),
            "sigma": float(sigma),
            "n_obs": n,
            "half_life_days": float(np.log(2) / theta) if theta > 0 else float("inf"),
        }

    @staticmethod
    def calibrate_jump_parameters(
        prices: pd.Series,
        jump_threshold: float = 3.0,
    ) -> Dict[str, float]:
        """
        Kalibriert Jump-Parameter durch Identifikation von Ausreißern.
        Jump = Return > threshold × std.
        """
        returns = prices.pct_change().dropna()
        if len(returns) < 20:
            return {"jump_intensity": 0.02, "jump_mean": 0.05, "jump_std": 0.1}

        std = returns.std()
        jumps = returns[returns.abs() > jump_threshold * std]

        jump_intensity = len(jumps) / len(returns)
        jump_mean = float(jumps.mean()) if len(jumps) > 0 else 0.0
        jump_std = float(jumps.std()) if len(jumps) > 1 else std * 2

        return {
            "jump_intensity": float(jump_intensity),
            "jump_mean": jump_mean,
            "jump_std": float(jump_std),
            "n_jumps": len(jumps),
        }

    @staticmethod
    @st.cache_data(show_spinner=False)
    def simulate_ou(
        n_paths: int,
        n_steps: int,
        theta: float,
        mu: float,
        sigma: float,
        x0: float,
        dt: float = 1/252,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Ornstein-Uhlenbeck-Simulation.
        Returns: (n_steps+1, n_paths) Array
        """
        np.random.seed(seed)
        paths = np.zeros((n_steps + 1, n_paths))
        paths[0, :] = x0

        for t in range(1, n_steps + 1):
            dW = np.random.normal(0, np.sqrt(dt), n_paths)
            paths[t, :] = (
                paths[t - 1, :]
                + theta * (mu - paths[t - 1, :]) * dt
                + sigma * dW
            )

        return paths

    @staticmethod
    @st.cache_data(show_spinner=False)
    def simulate_mrjd(
        n_paths: int,
        n_steps: int,
        theta: float,
        mu: float,
        sigma: float,
        jump_intensity: float,
        jump_mean: float,
        jump_std: float,
        x0: float,
        dt: float = 1/252,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Mean-Reverting Jump Diffusion (Merton-Modell angepasst für Energie).

        dX = θ(μ - X)dt + σdW + J·dN

        wobei:
        - dN ~ Poisson(λ·dt)
        - J ~ Normal(μ_J, σ_J)
        """
        np.random.seed(seed)
        paths = np.zeros((n_steps + 1, n_paths))
        paths[0, :] = x0

        for t in range(1, n_steps + 1):
            dW = np.random.normal(0, np.sqrt(dt), n_paths)

            # Mean Reversion
            drift = theta * (mu - paths[t - 1, :]) * dt
            diffusion = sigma * dW

            # Jumps
            n_jumps = np.random.poisson(jump_intensity * dt, n_paths)
            jump_sizes = np.zeros(n_paths)
            for i in range(n_paths):
                if n_jumps[i] > 0:
                    jump_sizes[i] = np.sum(
                        np.random.normal(jump_mean, jump_std, n_jumps[i])
                    )

            paths[t, :] = paths[t - 1, :] + drift + diffusion + jump_sizes
            # Floor: Energiepreise können negativ sein, aber Floor bei -500
            paths[t, :] = np.maximum(paths[t, :], -500)

        return paths

    @staticmethod
    def compute_mc_statistics(paths: np.ndarray) -> Dict[str, np.ndarray]:
        """Berechnet Statistiken über alle Pfade."""
        return {
            "mean": np.mean(paths, axis=1),
            "median": np.median(paths, axis=1),
            "std": np.std(paths, axis=1),
            "p5": np.percentile(paths, 5, axis=1),
            "p25": np.percentile(paths, 25, axis=1),
            "p75": np.percentile(paths, 75, axis=1),
            "p95": np.percentile(paths, 95, axis=1),
            "p1": np.percentile(paths, 1, axis=1),
            "p99": np.percentile(paths, 99, axis=1),
            "min": np.min(paths, axis=1),
            "max": np.max(paths, axis=1),
        }


# ============================================================================
# 23. STRESS TESTING ENGINE
# ============================================================================

class StressTestEngine:
    """
    Stress-Tests für Beschaffungsstrategien.
    Historische Szenarien + hypothetische Schocks.
    """

    HISTORICAL_SCENARIOS = {
        "Energiekrise 2022 (Jun-Sep)": {
            "description": "Extremer Preisanstieg durch Gas-Lieferstopp",
            "spot_multiplier": 4.5,
            "forward_multiplier": 3.8,
            "duration_months": 4,
            "volatility_multiplier": 3.0,
        },
        "Energiekrise 2022 Peak (Aug-Sep)": {
            "description": "Höhepunkt der Krise mit Spotpreisen > 700 EUR/MWh",
            "spot_multiplier": 8.0,
            "forward_multiplier": 5.0,
            "duration_months": 2,
            "volatility_multiplier": 5.0,
        },
        "Moderater Anstieg (+50%)": {
            "description": "Graduelle Preissteigerung über 6 Monate",
            "spot_multiplier": 1.5,
            "forward_multiplier": 1.4,
            "duration_months": 6,
            "volatility_multiplier": 1.3,
        },
        "Preisverfall (-40%)": {
            "description": "Starker Rückgang der Energiepreise",
            "spot_multiplier": 0.6,
            "forward_multiplier": 0.65,
            "duration_months": 6,
            "volatility_multiplier": 1.5,
        },
        "Flash-Spike (1 Woche)": {
            "description": "Kurzfristiger extremer Preisanstieg",
            "spot_multiplier": 6.0,
            "forward_multiplier": 2.0,
            "duration_months": 0.25,
            "volatility_multiplier": 8.0,
        },
        "Negative Spotpreise": {
            "description": "Erneuerbare Überproduktion → negative Preise",
            "spot_multiplier": -0.3,
            "forward_multiplier": 0.7,
            "duration_months": 1,
            "volatility_multiplier": 4.0,
        },
    }

    @staticmethod
    def apply_stress_to_spot(
        spot_prices: pd.Series,
        multiplier: float,
        start_idx: int,
        duration_steps: int,
        ramp_up_steps: int = 10,
    ) -> pd.Series:
        """
        Wendet einen Stress-Multiplikator auf einen Abschnitt
        der Spotpreise an, mit Auf-/Abrampe.
        """
        stressed = spot_prices.copy()
        n = len(stressed)
        end_idx = min(start_idx + duration_steps, n)

        for i in range(start_idx, end_idx):
            # Ramp-Up
            if i - start_idx < ramp_up_steps:
                progress = (i - start_idx) / ramp_up_steps
                current_mult = 1.0 + (multiplier - 1.0) * progress
            # Ramp-Down
            elif end_idx - i < ramp_up_steps:
                progress = (end_idx - i) / ramp_up_steps
                current_mult = 1.0 + (multiplier - 1.0) * progress
            else:
                current_mult = multiplier

            stressed.iloc[i] = stressed.iloc[i] * current_mult

        return stressed

    @staticmethod
    def compute_stress_impact(
        normal_result: BacktestResult,
        stressed_result: BacktestResult,
    ) -> Dict[str, Any]:
        """Berechnet den Stress-Impact."""
        cost_diff = stressed_result.total_cost - normal_result.total_cost
        cost_pct = cost_diff / normal_result.total_cost * 100 if normal_result.total_cost > 0 else 0
        price_diff = stressed_result.avg_blended_price - normal_result.avg_blended_price

        return {
            "cost_impact_eur": float(cost_diff),
            "cost_impact_pct": float(cost_pct),
            "price_impact_eur_mwh": float(price_diff),
            "stressed_total_cost": float(stressed_result.total_cost),
            "normal_total_cost": float(normal_result.total_cost),
            "stressed_avg_price": float(stressed_result.avg_blended_price),
            "normal_avg_price": float(normal_result.avg_blended_price),
        }


# ============================================================================
# 24. PnL ATTRIBUTION ENGINE
# ============================================================================

class PnLAttributionEngine:
    """
    Zerlegt die Gesamtkosten einer Strategie in Erklärungskomponenten.

    Komponenten:
    1. Baseline (100% Spot)
    2. Hedge-Effekt (Forward vs. Spot Differenz)
    3. Timing-Effekt (Wann gekauft?)
    4. Volumen-Effekt (Wie viel gehedgt?)
    5. Transaktionskosten
    """

    @staticmethod
    def decompose(
        strategy_result: BacktestResult,
        spot_benchmark: BacktestResult,
    ) -> pd.DataFrame:
        """
        Vollständige PnL-Attribution pro Liefermonat.
        """
        records = []

        for s_month, bm_month in zip(
            strategy_result.monthly_results,
            spot_benchmark.monthly_results,
        ):
            baseline_cost = bm_month.total_cost_eur  # 100% Spot
            actual_cost = s_month.total_cost_eur

            # Hedge-Effekt: (Spot-Preis - Forward-Preis) × gehedgtes Volumen
            if s_month.hedged_volume_mwh > 0 and bm_month.avg_spot_price > 0:
                hedge_effect = (
                    (bm_month.avg_spot_price - s_month.avg_forward_price)
                    * s_month.hedged_volume_mwh
                )
            else:
                hedge_effect = 0.0

            # Transaktionskosten (aus Forward-Seite)
            tx_cost_rate = strategy_result.strategy.total_transaction_cost_pct
            broker = strategy_result.strategy.broker_fee_eur_mwh
            tx_costs = s_month.hedged_volume_mwh * (
                s_month.avg_forward_price * tx_cost_rate + broker
            ) if s_month.hedged_volume_mwh > 0 else 0.0

            # Residuum
            total_effect = baseline_cost - actual_cost
            residuum = total_effect - hedge_effect + tx_costs

            records.append({
                "delivery_month": s_month.delivery_month,
                "baseline_cost": baseline_cost,
                "actual_cost": actual_cost,
                "total_effect": total_effect,
                "hedge_effect": hedge_effect,
                "transaction_costs": -tx_costs,
                "timing_residual": residuum,
                "hedge_ratio": s_month.hedge_ratio_actual,
                "avg_spot": bm_month.avg_spot_price,
                "avg_forward": s_month.avg_forward_price,
            })

        return pd.DataFrame(records)


# ============================================================================
# 25. RISK-MODUL VISUALISIERUNGEN
# ============================================================================

class RiskVisualizer:
    """Visualisierungen für das Risiko-Modul."""

    @staticmethod
    def var_waterfall(
        var_results: Dict[str, float],
        confidence: float,
    ) -> go.Figure:
        """Vergleich verschiedener VaR-Methoden als Balkendiagramm."""
        methods = list(var_results.keys())
        values = list(var_results.values())

        colors = [
            Config.COLORS["info"],
            Config.COLORS["warning"],
            Config.COLORS["danger"],
            Config.COLORS["spot"],
        ]

        fig = go.Figure(go.Bar(
            x=methods, y=values,
            marker_color=colors[:len(methods)],
            text=[f"{v:.2f}%" for v in values],
            textposition="outside",
        ))
        fig.update_layout(
            title=f"Value at Risk – {confidence:.0%} Konfidenzniveau",
            yaxis_title="VaR (%)",
            template="plotly_dark", height=380,
        )
        return fig

    @staticmethod
    def mc_fan_chart(
        paths: np.ndarray,
        stats: Dict[str, np.ndarray],
        historical: Optional[pd.Series] = None,
        title: str = "Monte-Carlo-Simulation",
    ) -> go.Figure:
        """Fan-Chart: Konfidenzintervalle aus Monte-Carlo-Pfaden."""
        n_steps = paths.shape[0]
        x = list(range(n_steps))

        fig = go.Figure()

        # 1%-99% Band
        fig.add_trace(go.Scatter(
            x=x, y=stats["p99"], mode="lines",
            line=dict(width=0), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=x, y=stats["p1"], mode="lines",
            line=dict(width=0), fill="tonexty",
            fillcolor="rgba(231,76,60,0.08)",
            name="1-99% Konfidenz",
        ))

        # 5%-95% Band
        fig.add_trace(go.Scatter(
            x=x, y=stats["p95"], mode="lines",
            line=dict(width=0), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=x, y=stats["p5"], mode="lines",
            line=dict(width=0), fill="tonexty",
            fillcolor="rgba(243,156,18,0.15)",
            name="5-95% Konfidenz",
        ))

        # 25%-75% Band
        fig.add_trace(go.Scatter(
            x=x, y=stats["p75"], mode="lines",
            line=dict(width=0), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=x, y=stats["p25"], mode="lines",
            line=dict(width=0), fill="tonexty",
            fillcolor="rgba(52,152,219,0.25)",
            name="25-75% Konfidenz",
        ))

        # Median
        fig.add_trace(go.Scatter(
            x=x, y=stats["median"],
            mode="lines", name="Median",
            line=dict(color="white", width=2),
        ))

        # Mean
        fig.add_trace(go.Scatter(
            x=x, y=stats["mean"],
            mode="lines", name="Mittelwert",
            line=dict(color=Config.COLORS["actual"], width=2, dash="dash"),
        ))

        # Beispielpfade
        n_sample = min(20, paths.shape[1])
        for i in range(n_sample):
            fig.add_trace(go.Scattergl(
                x=x, y=paths[:, i],
                mode="lines", name=f"Pfad {i+1}",
                line=dict(width=0.3, color="rgba(150,150,150,0.2)"),
                showlegend=False,
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Tage",
            yaxis_title="EUR/MWh",
            template="plotly_dark", height=520,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        return fig

    @staticmethod
    def mc_terminal_distribution(
        paths: np.ndarray,
        title: str = "Verteilung am Simulationsende",
    ) -> go.Figure:
        """Histogram der Terminal-Werte."""
        terminal = paths[-1, :]

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=terminal, nbinsx=80,
            marker_color=Config.COLORS["info"],
            opacity=0.75, name="Terminal-Preise",
        ))

        # VaR-Linien
        var_95 = np.percentile(terminal, 5)
        var_99 = np.percentile(terminal, 1)
        mean = np.mean(terminal)

        fig.add_vline(x=mean, line_dash="solid", line_color="white",
                      annotation_text=f"Mean: {mean:.1f}")
        fig.add_vline(x=var_95, line_dash="dash", line_color=Config.COLORS["warning"],
                      annotation_text=f"VaR 95%: {var_95:.1f}")
        fig.add_vline(x=var_99, line_dash="dash", line_color=Config.COLORS["danger"],
                      annotation_text=f"VaR 99%: {var_99:.1f}")

        fig.update_layout(
            title=title,
            xaxis_title="EUR/MWh", yaxis_title="Häufigkeit",
            template="plotly_dark", height=400,
        )
        return fig

    @staticmethod
    def drawdown_chart(
        drawdown_df: pd.DataFrame,
        title: str = "Drawdown-Analyse",
    ) -> go.Figure:
        """Drawdown-Chart mit absoluten Werten."""
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.6, 0.4],
            vertical_spacing=0.08,
            subplot_titles=["Kumulative Kosten", "Drawdown"],
        )

        fig.add_trace(go.Scatter(
            x=drawdown_df.index, y=drawdown_df["cumulative_cost"],
            mode="lines", name="Kumulierte Kosten",
            line=dict(color=Config.COLORS["info"], width=2),
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=drawdown_df.index, y=drawdown_df["running_min"],
            mode="lines", name="Running Minimum",
            line=dict(color=Config.COLORS["success"], width=1.5, dash="dot"),
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=drawdown_df.index, y=-drawdown_df["drawdown_pct"],
            mode="lines", name="Drawdown (%)",
            line=dict(color=Config.COLORS["danger"], width=1.5),
            fill="tozeroy", fillcolor="rgba(231,76,60,0.2)",
        ), row=2, col=1)

        fig.update_layout(
            title=title, template="plotly_dark", height=550,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        fig.update_yaxes(title_text="EUR", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        return fig

    @staticmethod
    def hedge_effectiveness_scatter(
        hedge_returns: pd.Series,
        underlying_returns: pd.Series,
        regression: Dict[str, Any],
    ) -> go.Figure:
        """Streudiagramm mit Regressionsgerade."""
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=underlying_returns, y=hedge_returns,
            mode="markers", name="Datenpunkte",
            marker=dict(color=Config.COLORS["info"], size=5, opacity=0.5),
        ))

        # Regressionsgerade
        x_range = np.linspace(underlying_returns.min(), underlying_returns.max(), 100)
        y_hat = regression["alpha"] + regression["beta"] * x_range

        fig.add_trace(go.Scatter(
            x=x_range, y=y_hat,
            mode="lines", name=f"Regression (β={regression['beta']:.3f}, R²={regression['r_squared']:.3f})",
            line=dict(color=Config.COLORS["danger"], width=2.5),
        ))

        # Perfekte Hedge-Linie (slope = -1)
        fig.add_trace(go.Scatter(
            x=x_range, y=-x_range,
            mode="lines", name="Perfekter Hedge (β=-1)",
            line=dict(color="white", width=1, dash="dash"),
        ))

        effectiveness = "✅ Effektiv" if regression["is_effective"] else "❌ Ineffektiv"

        fig.update_layout(
            title=f"Hedge Effectiveness – Regressionsanalyse ({effectiveness})",
            xaxis_title="Underlying Returns (Spot)",
            yaxis_title="Hedge Returns (Forward PnL)",
            template="plotly_dark", height=440,
        )
        return fig

    @staticmethod
    def pnl_attribution_waterfall(attribution_df: pd.DataFrame) -> go.Figure:
        """Aggregierter Waterfall der PnL-Attribution."""
        totals = attribution_df[
            ["baseline_cost", "hedge_effect", "transaction_costs", "timing_residual", "actual_cost"]
        ].sum()

        fig = go.Figure(go.Waterfall(
            x=[
                "Baseline\n(100% Spot)",
                "Hedge-\nEffekt",
                "Transaktions-\nkosten",
                "Timing &\nResiduum",
                "Endkosten\n(Strategie)",
            ],
            measure=["absolute", "relative", "relative", "relative", "total"],
            y=[
                totals["baseline_cost"],
                -totals["hedge_effect"],
                totals["transaction_costs"],
                -totals["timing_residual"],
                totals["actual_cost"],
            ],
            text=[
                f"€{totals['baseline_cost']:,.0f}",
                f"€{-totals['hedge_effect']:+,.0f}",
                f"€{totals['transaction_costs']:+,.0f}",
                f"€{-totals['timing_residual']:+,.0f}",
                f"€{totals['actual_cost']:,.0f}",
            ],
            textposition="outside",
            connector=dict(line=dict(color="rgba(63,63,63,0.5)")),
            increasing=dict(marker_color=Config.COLORS["danger"]),
            decreasing=dict(marker_color=Config.COLORS["success"]),
            totals=dict(marker_color=Config.COLORS["info"]),
        ))

        fig.update_layout(
            title="PnL-Attribution: Warum kostet die Strategie mehr/weniger als 100% Spot?",
            yaxis_title="EUR", template="plotly_dark", height=440,
        )
        return fig

    @staticmethod
    def stress_test_comparison(
        stress_results: Dict[str, Dict[str, Any]],
    ) -> go.Figure:
        """Balkendiagramm: Kosten-Impact verschiedener Stress-Szenarien."""
        scenarios = list(stress_results.keys())
        impacts_pct = [v["cost_impact_pct"] for v in stress_results.values()]
        impacts_abs = [v["cost_impact_eur"] for v in stress_results.values()]

        colors = [
            Config.COLORS["danger"] if v > 10 else
            Config.COLORS["warning"] if v > 0 else
            Config.COLORS["success"]
            for v in impacts_pct
        ]

        fig = go.Figure(go.Bar(
            x=scenarios, y=impacts_pct,
            marker_color=colors,
            text=[f"{v:+.1f}%" for v in impacts_pct],
            textposition="outside",
            customdata=[f"€{v:+,.0f}" for v in impacts_abs],
            hovertemplate="%{x}<br>Impact: %{text}<br>Absolut: %{customdata}<extra></extra>",
        ))
        fig.add_hline(y=0, line_color="white", line_width=0.5)
        fig.update_layout(
            title="Stress-Test: Kosten-Impact verschiedener Szenarien",
            yaxis_title="Kosten-Impact (%)",
            template="plotly_dark", height=420,
            xaxis_tickangle=-25,
        )
        return fig


# ============================================================================
# 26. PAGE: RISK & PERFORMANCE ANALYTICS
# ============================================================================

def page_risk_analytics():
    st.header("📐 Risiko- & Performance-Analytics")
    st.markdown(
        """
        Institutionelle Risikoanalyse: **VaR · CVaR · Monte Carlo · MtM · 
        Hedge Effectiveness · Drawdown · PnL-Attribution · Stress Testing**
        """
    )

    # Datenprüfung
    spot = DataStore.get("spot_prices")
    sett = DataStore.get("settlement_prices")
    load = DataStore.get("load_profile")
    purch = DataStore.get("purchases")

    if spot is None:
        st.error("❌ Spotpreise fehlen. Bitte unter Daten-Management laden.")
        return

    # Tabs
    tab_var, tab_mc, tab_mtm, tab_hedge, tab_pnl, tab_stress = st.tabs([
        "📊 VaR & Risikomaße",
        "🎲 Monte Carlo",
        "💹 Mark-to-Market",
        "🛡️ Hedge Effectiveness",
        "📈 PnL Attribution",
        "⚡ Stress Testing",
    ])

    # ================================================================
    # TAB 1: VaR & RISIKOMASSE
    # ================================================================
    with tab_var:
        st.subheader("📊 Value at Risk & Risikokennzahlen")

        spot_s = spot.set_index("timestamp")["price_eur_mwh"].sort_index()

        # Daily returns
        daily_spot = spot_s.resample("D").mean().dropna()
        returns = daily_spot.pct_change().dropna()

        # Konfiguration
        c1, c2, c3 = st.columns(3)
        with c1:
            confidence = st.select_slider(
                "Konfidenzniveau",
                options=[0.90, 0.95, 0.975, 0.99],
                value=0.95, format_func=lambda x: f"{x:.1%}",
                key="var_conf",
            )
        with c2:
            holding = st.selectbox(
                "Haltedauer (Tage)", [1, 5, 10, 21, 63], index=0,
                key="var_hold",
            )
        with c3:
            vol_window = st.slider(
                "Volatilitätsfenster (Tage)", 10, 120, 30,
                key="var_vol_w",
            )

        # Berechnung
        var_param = RiskMetrics.parametric_var(returns, confidence, holding)
        var_cf = RiskMetrics.cornish_fisher_var(returns, confidence, holding)
        var_hist = RiskMetrics.historical_var(returns, confidence)
        cvar = RiskMetrics.conditional_var(returns, confidence)
        ann_vol = RiskMetrics.annualized_volatility(returns)
        tail_r = RiskMetrics.tail_ratio(returns)
        omega = RiskMetrics.omega_ratio(returns)

        # KPIs
        st.markdown("### 📋 Risikokennzahlen")
        m_cols = st.columns(4)
        m_cols[0].metric("Parametrischer VaR", f"{var_param:.2%}")
        m_cols[1].metric("Cornish-Fisher VaR", f"{var_cf:.2%}")
        m_cols[2].metric("Historischer VaR", f"{var_hist:.2%}")
        m_cols[3].metric(f"CVaR / ES ({confidence:.0%})", f"{cvar:.2%}")

        m_cols2 = st.columns(4)
        m_cols2[0].metric("Ann. Volatilität", f"{ann_vol:.1%}")
        m_cols2[1].metric("Tail Ratio", f"{tail_r:.2f}")
        m_cols2[2].metric("Omega Ratio", f"{omega:.2f}")
        m_cols2[3].metric("Skewness", f"{returns.skew():.3f}")

        # VaR-Methodenvergleich
        var_results = {
            "Parametrisch (Gauß)": var_param * 100,
            "Cornish-Fisher": var_cf * 100,
            "Historisch": var_hist * 100,
            f"CVaR/ES": cvar * 100,
        }
        fig_var = RiskVisualizer.var_waterfall(var_results, confidence)
        st.plotly_chart(fig_var, use_container_width=True)

        # Returns-Verteilung
        c1, c2 = st.columns(2)
        with c1:
            fig_ret = go.Figure()
            fig_ret.add_trace(go.Histogram(
                x=returns * 100, nbinsx=100,
                marker_color=Config.COLORS["info"], opacity=0.7,
                name="Historisch",
            ))
            # Normalverteilungs-Overlay
            x_norm = np.linspace(returns.min(), returns.max(), 200)
            y_norm = scipy_stats.norm.pdf(x_norm, returns.mean(), returns.std())
            y_norm_scaled = y_norm * len(returns) * (returns.max() - returns.min()) / 100 * 100
            fig_ret.add_trace(go.Scatter(
                x=x_norm * 100, y=y_norm_scaled,
                mode="lines", name="Normalverteilung",
                line=dict(color=Config.COLORS["warning"], width=2, dash="dash"),
            ))
            var_line = -var_hist * 100
            fig_ret.add_vline(x=var_line, line_dash="dash", line_color=Config.COLORS["danger"],
                              annotation_text=f"VaR {confidence:.0%}")
            fig_ret.update_layout(
                title="Returns-Verteilung vs. Normalverteilung",
                xaxis_title="Tägliche Returns (%)", yaxis_title="Häufigkeit",
                template="plotly_dark", height=400,
            )
            st.plotly_chart(fig_ret, use_container_width=True)

        with c2:
            # QQ-Plot
            theoretical_q = scipy_stats.norm.ppf(
                np.linspace(0.001, 0.999, len(returns))
            )
            empirical_q = np.sort(returns.values)

            fig_qq = go.Figure()
            fig_qq.add_trace(go.Scatter(
                x=theoretical_q, y=empirical_q,
                mode="markers", name="Quantile",
                marker=dict(color=Config.COLORS["info"], size=3, opacity=0.5),
            ))
            line_range = [min(theoretical_q), max(theoretical_q)]
            fig_qq.add_trace(go.Scatter(
                x=line_range, y=[returns.mean() + returns.std() * lr for lr in line_range],
                mode="lines", name="Normalverteilung",
                line=dict(color=Config.COLORS["danger"], width=2),
            ))
            fig_qq.update_layout(
                title="Q-Q Plot (Fat Tails Analyse)",
                xaxis_title="Theoretische Quantile (Normal)",
                yaxis_title="Empirische Quantile",
                template="plotly_dark", height=400,
            )
            st.plotly_chart(fig_qq, use_container_width=True)

        # Rollende Volatilität
        st.subheader("📉 Rollende Risikokennzahlen")
        roll_vol = RiskMetrics.rolling_volatility(returns, vol_window)
        ewma_vol = RiskMetrics.ewma_volatility(returns, vol_window)

        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=roll_vol.index, y=roll_vol * 100,
            mode="lines", name=f"Rollende Vol. ({vol_window}d)",
            line=dict(color=Config.COLORS["warning"], width=1.5),
        ))
        fig_vol.add_trace(go.Scatter(
            x=ewma_vol.index, y=ewma_vol * 100,
            mode="lines", name=f"EWMA Vol. (Span={vol_window})",
            line=dict(color=Config.COLORS["danger"], width=1.5),
        ))
        fig_vol.update_layout(
            title="Annualisierte Volatilität über Zeit",
            yaxis_title="Volatilität (%)", template="plotly_dark", height=380,
            hovermode="x unified",
        )
        st.plotly_chart(fig_vol, use_container_width=True)

        # Drawdown auf Spotpreise
        st.subheader("📉 Spot-Preis Drawdown")
        dd = RiskMetrics.compute_drawdowns(daily_spot)
        max_dd = RiskMetrics.max_drawdown(daily_spot)

        st.metric(
            "Maximum Drawdown",
            f"{max_dd['max_drawdown_pct']:.1f}%",
            f"€{max_dd['max_drawdown_abs']:.2f}/MWh",
        )

        fig_dd = RiskVisualizer.drawdown_chart(dd, "Spot-Preis Drawdown-Analyse")
        st.plotly_chart(fig_dd, use_container_width=True)

    # ================================================================
    # TAB 2: MONTE CARLO
    # ================================================================
    with tab_mc:
        st.subheader("🎲 Monte-Carlo-Simulation")
        st.markdown(
            """
            Stochastische Preissimulation mit kalibrierten Modellen.
            Testet Beschaffungsstrategien gegen **tausende mögliche Zukunftsszenarien**.
            """
        )

        spot_daily = spot.set_index("timestamp")["price_eur_mwh"].resample("D").mean().dropna()

        # Kalibrierung
        ou_params = MonteCarloEngine.calibrate_ou_parameters(spot_daily)
        jump_params = MonteCarloEngine.calibrate_jump_parameters(spot_daily)

        with st.expander("🔬 Kalibrierte Modellparameter", expanded=True):
            p_cols = st.columns(4)
            p_cols[0].metric("θ (Mean-Rev.-Speed)", f"{ou_params['theta']:.4f}")
            p_cols[1].metric("μ (Langfrist-Mittel)", f"{ou_params['mu']:.2f} €/MWh")
            p_cols[2].metric("σ (Volatilität)", f"{ou_params['sigma']:.2f}")
            p_cols[3].metric("Halbwertszeit", f"{ou_params['half_life_days']:.0f} Tage")

            j_cols = st.columns(3)
            j_cols[0].metric("Jump-Intensität", f"{jump_params['jump_intensity']:.4f}")
            j_cols[1].metric("Jump-Mittel", f"{jump_params['jump_mean']:.4f}")
            j_cols[2].metric("Erkannte Jumps", f"{jump_params['n_jumps']}")

        # Simulationsparameter
        st.markdown("### ⚙️ Simulation konfigurieren")
        mc_c1, mc_c2, mc_c3, mc_c4 = st.columns(4)

        with mc_c1:
            mc_model = st.selectbox(
                "Modell",
                ["Ornstein-Uhlenbeck", "Mean-Reverting Jump Diffusion"],
                index=1, key="mc_model",
            )
        with mc_c2:
            mc_paths = st.select_slider(
                "Pfade",
                options=[100, 500, 1000, 2000, 5000, 10000],
                value=1000, key="mc_paths",
            )
        with mc_c3:
            mc_days = st.slider(
                "Horizont (Tage)", 30, 730, 365, key="mc_days",
            )
        with mc_c4:
            mc_seed = st.number_input("Seed", value=42, key="mc_seed")

        # Manuelle Overrides
        with st.expander("📝 Parameter manuell anpassen"):
            ov_c1, ov_c2, ov_c3 = st.columns(3)
            with ov_c1:
                ov_theta = st.number_input("θ", value=ou_params["theta"], format="%.4f", key="mc_ov_t")
                ov_mu = st.number_input("μ", value=ou_params["mu"], format="%.2f", key="mc_ov_m")
            with ov_c2:
                ov_sigma = st.number_input("σ", value=ou_params["sigma"], format="%.2f", key="mc_ov_s")
            with ov_c3:
                ov_ji = st.number_input("Jump-Intensität", value=jump_params["jump_intensity"],
                                        format="%.4f", key="mc_ov_ji")
                ov_jm = st.number_input("Jump-Mittel", value=jump_params["jump_mean"],
                                        format="%.4f", key="mc_ov_jm")
                ov_js = st.number_input("Jump-Std", value=jump_params["jump_std"],
                                        format="%.4f", key="mc_ov_js")

        if st.button("🚀 Simulation starten", type="primary", use_container_width=True, key="mc_run"):
            x0 = float(spot_daily.iloc[-1])

            with st.spinner(f"⏳ Simuliere {mc_paths:,} Pfade über {mc_days} Tage..."):
                if mc_model == "Ornstein-Uhlenbeck":
                    paths = MonteCarloEngine.simulate_ou(
                        mc_paths, mc_days, ov_theta, ov_mu, ov_sigma,
                        x0, seed=int(mc_seed),
                    )
                else:
                    paths = MonteCarloEngine.simulate_mrjd(
                        mc_paths, mc_days, ov_theta, ov_mu, ov_sigma,
                        ov_ji, ov_jm, ov_js,
                        x0, seed=int(mc_seed),
                    )

            mc_stats = MonteCarloEngine.compute_mc_statistics(paths)
            st.session_state["mc_paths"] = paths
            st.session_state["mc_stats"] = mc_stats

        # Ergebnisse
        mc_paths_data = st.session_state.get("mc_paths")
        mc_stats_data = st.session_state.get("mc_stats")

        if mc_paths_data is not None and mc_stats_data is not None:
            st.divider()

            # KPIs
            terminal = mc_paths_data[-1, :]
            mc_kpi = st.columns(5)
            mc_kpi[0].metric("Ø Terminal-Preis", f"{np.mean(terminal):.2f} €")
            mc_kpi[1].metric("Median Terminal", f"{np.median(terminal):.2f} €")
            mc_kpi[2].metric("5%-Quantil", f"{np.percentile(terminal, 5):.2f} €")
            mc_kpi[3].metric("95%-Quantil", f"{np.percentile(terminal, 95):.2f} €")
            mc_kpi[4].metric("Prob. > 200€", f"{(terminal > 200).mean():.1%}")

            # Fan-Chart
            fig_fan = RiskVisualizer.mc_fan_chart(
                mc_paths_data, mc_stats_data,
                title=f"Monte-Carlo: {mc_paths_data.shape[1]:,} Pfade, {mc_paths_data.shape[0]-1} Tage",
            )
            st.plotly_chart(fig_fan, use_container_width=True)

            # Terminal-Verteilung
            c1, c2 = st.columns(2)
            with c1:
                fig_term = RiskVisualizer.mc_terminal_distribution(
                    mc_paths_data, "Terminal-Preisverteilung"
                )
                st.plotly_chart(fig_term, use_container_width=True)

            with c2:
                # Monte-Carlo VaR
                mc_var_95 = float(np.percentile(terminal, 5))
                mc_var_99 = float(np.percentile(terminal, 1))
                mc_cvar_95 = float(terminal[terminal <= np.percentile(terminal, 5)].mean())
                mc_es = float(terminal[terminal <= np.percentile(terminal, 1)].mean()) if (terminal <= np.percentile(terminal, 1)).any() else mc_var_99

                current_price = float(spot_daily.iloc[-1])

                st.markdown("### 📊 Monte-Carlo Risikomaße")
                st.metric("Aktueller Spotpreis", f"{current_price:.2f} €/MWh")
                st.metric("MC VaR 95% (Terminal)", f"{mc_var_95:.2f} €/MWh",
                           f"{(mc_var_95 - current_price)/current_price*100:+.1f}%")
                st.metric("MC VaR 99% (Terminal)", f"{mc_var_99:.2f} €/MWh",
                           f"{(mc_var_99 - current_price)/current_price*100:+.1f}%")
                st.metric("MC CVaR 95% (Terminal)", f"{mc_cvar_95:.2f} €/MWh")

            # Kosten-Simulation
            st.subheader("💰 MC-Kosten-Simulation für verschiedene Strategien")
            st.markdown("Durchschnittliche erwartete Kosten über alle simulierten Pfade.")

            hedge_ratios_mc = [0.0, 0.3, 0.5, 0.7, 1.0]
            current_fwd_price = ov_mu  # Approximation

            mc_cost_records = []
            for hr in hedge_ratios_mc:
                fwd_costs = hr * current_fwd_price * terminal * 0  # Fixed forward price
                fwd_part = hr * current_fwd_price
                spot_part = (1 - hr) * terminal
                blended = fwd_part + spot_part

                mc_cost_records.append({
                    "hedge_ratio": f"{hr:.0%}",
                    "mean_price": float(np.mean(blended)),
                    "median_price": float(np.median(blended)),
                    "std_price": float(np.std(blended)),
                    "var_95": float(np.percentile(blended, 95)),
                    "var_5": float(np.percentile(blended, 5)),
                    "worst_case": float(np.percentile(blended, 99)),
                })

            mc_cost_df = pd.DataFrame(mc_cost_records)
            st.dataframe(
                mc_cost_df.style.format({
                    "mean_price": "{:.2f} €",
                    "median_price": "{:.2f} €",
                    "std_price": "{:.2f} €",
                    "var_95": "{:.2f} €",
                    "var_5": "{:.2f} €",
                    "worst_case": "{:.2f} €",
                }),
                use_container_width=True,
            )

    # ================================================================
    # TAB 3: MARK-TO-MARKET
    # ================================================================
    with tab_mtm:
        st.subheader("💹 Mark-to-Market Tracking")

        bt_result = st.session_state.get("bt_single_result")

        if bt_result is None or sett is None:
            st.warning("⚠️ Bitte führen Sie zuerst einen Backtest durch (Modul 2).")
            st.info("Alternative: Nutzen Sie echte Beschaffungsdaten.")

            if purch is not None and sett is not None:
                st.markdown("---")
                st.markdown("**MtM für echte Beschaffungen:**")

                fwd_purch = purch[purch["purchase_type"].str.lower() == "forward"]
                if not fwd_purch.empty:
                    actual_tranches = []
                    for _, row in fwd_purch.iterrows():
                        del_start = pd.Timestamp(row["delivery_start"])
                        hours = pd.Timestamp(row["delivery_end"]) - del_start
                        hours_n = max(hours.total_seconds() / 3600, 1)
                        actual_tranches.append(TrancheExecution(
                            execution_date=pd.Timestamp(row["purchase_date"]),
                            delivery_month=del_start.strftime("%Y-%m"),
                            volume_mwh=float(row["volume_mw"]) * hours_n,
                            price_eur_mwh=float(row["price_eur_mwh"]),
                        ))

                    with st.spinner("Berechne MtM..."):
                        mtm_df = MarkToMarketEngine.compute_mtm_for_tranches(actual_tranches, sett)

                    if not mtm_df.empty:
                        mtm_df["date"] = pd.to_datetime(mtm_df["date"])

                        # MtM Chart
                        fig_mtm = go.Figure()
                        fig_mtm.add_trace(go.Scatter(
                            x=mtm_df["date"], y=mtm_df["total_mtm"],
                            mode="lines", name="MtM (EUR)",
                            line=dict(width=2),
                            fill="tozeroy",
                            fillcolor="rgba(46,204,113,0.15)" if mtm_df["total_mtm"].iloc[-1] > 0
                            else "rgba(231,76,60,0.15)",
                        ))
                        fig_mtm.update_layout(
                            title="Mark-to-Market: Unrealisierter PnL",
                            yaxis_title="EUR", template="plotly_dark", height=420,
                            hovermode="x unified",
                        )
                        fig_mtm.add_hline(y=0, line_color="white", line_width=0.5)
                        st.plotly_chart(fig_mtm, use_container_width=True)

                        # KPIs
                        mtm_kpis = st.columns(4)
                        last_mtm = mtm_df["total_mtm"].iloc[-1] if len(mtm_df) > 0 else 0
                        max_mtm = mtm_df["total_mtm"].max()
                        min_mtm = mtm_df["total_mtm"].min()
                        max_exp = mtm_df["total_exposure_mwh"].max()
                        mtm_kpis[0].metric("Aktueller MtM", f"€{last_mtm:,.0f}")
                        mtm_kpis[1].metric("Max. MtM (Gewinn)", f"€{max_mtm:,.0f}")
                        mtm_kpis[2].metric("Min. MtM (Verlust)", f"€{min_mtm:,.0f}")
                        mtm_kpis[3].metric("Max. Exposure", f"{max_exp:,.0f} MWh")
            return

        # MtM aus Backtest-Ergebnis
        if bt_result.all_tranches and sett is not None:
            with st.spinner("Berechne MtM für Backtesting-Tranchen..."):
                mtm_df = MarkToMarketEngine.compute_mtm_for_tranches(
                    bt_result.all_tranches, sett
                )

            if not mtm_df.empty:
                mtm_df["date"] = pd.to_datetime(mtm_df["date"])

                fig_mtm = go.Figure()
                colors = np.where(mtm_df["total_mtm"] >= 0, Config.COLORS["success"], Config.COLORS["danger"])

                fig_mtm.add_trace(go.Scatter(
                    x=mtm_df["date"], y=mtm_df["total_mtm"],
                    mode="lines", name="MtM (EUR)",
                    line=dict(color=Config.COLORS["info"], width=2),
                    fill="tozeroy",
                ))
                fig_mtm.add_hline(y=0, line_color="white", line_width=0.5)
                fig_mtm.update_layout(
                    title=f"Mark-to-Market – {bt_result.strategy.name}",
                    yaxis_title="EUR", template="plotly_dark", height=450,
                    hovermode="x unified",
                )
                st.plotly_chart(fig_mtm, use_container_width=True)

                # Exposure über Zeit
                fig_exp = go.Figure()
                fig_exp.add_trace(go.Scatter(
                    x=mtm_df["date"], y=mtm_df["total_exposure_mwh"],
                    mode="lines", name="Exposure (MWh)",
                    line=dict(color=Config.COLORS["warning"], width=2),
                    fill="tozeroy", fillcolor="rgba(243,156,18,0.15)",
                ))
                fig_exp.add_trace(go.Scatter(
                    x=mtm_df["date"], y=mtm_df["n_open_positions"],
                    mode="lines", name="Offene Positionen",
                    line=dict(color=Config.COLORS["info"], width=1.5, dash="dot"),
                    yaxis="y2",
                ))
                fig_exp.update_layout(
                    title="Exposure & Offene Positionen",
                    yaxis_title="MWh",
                    yaxis2=dict(title="Positionen", overlaying="y", side="right"),
                    template="plotly_dark", height=380,
                    hovermode="x unified",
                )
                st.plotly_chart(fig_exp, use_container_width=True)

    # ================================================================
    # TAB 4: HEDGE EFFECTIVENESS
    # ================================================================
    with tab_hedge:
        st.subheader("🛡️ Hedge Effectiveness")

        if sett is None or load is None:
            st.warning("Settlement-Preise und Lastprofil werden benötigt.")
            return

        bt_result = st.session_state.get("bt_single_result")
        bm_result = st.session_state.get("bt_single_benchmark")

        if bt_result is None or bm_result is None:
            st.info("Bitte führen Sie zuerst einen Backtest durch (Tab 'Einzelstrategie').")
            return

        st.markdown(
            """
            Misst, wie gut die Forward-Beschaffung das Spotpreisrisiko reduziert hat.

            **Methoden:**
            - 📐 **Regressionsanalyse** (IFRS 9 / IAS 39): R² > 0.80 = effektiv
            - 💵 **Dollar-Offset**: Ratio nahe -1.0
            - 📊 **Varianzreduktion**: Vergleich der Kostenvolatilität
            """
        )

        # Daten vorbereiten
        bt_df = bt_result.to_dataframe()
        bm_df = bm_result.to_dataframe()

        # Monatliche Returns
        hedged_costs = bt_df.set_index("delivery_month")["total_cost_eur"]
        unhedged_costs = bm_df.set_index("delivery_month")["total_cost_eur"]

        hedged_returns = hedged_costs.pct_change().dropna()
        unhedged_returns = unhedged_costs.pct_change().dropna()

        # Hedge PnL = Forward-Position PnL
        hedge_pnl = bt_df.set_index("delivery_month")["fwd_cost_eur"]
        underlying_pnl = bm_df.set_index("delivery_month")["spot_cost_eur"]

        # Regressionanalyse
        if len(hedged_returns) >= 5 and len(unhedged_returns) >= 5:
            common = hedged_returns.index.intersection(unhedged_returns.index)
            h_ret = hedged_returns.loc[common]
            u_ret = unhedged_returns.loc[common]

            regression = HedgeEffectivenessAnalyser.regression_analysis(h_ret, u_ret)

            # KPIs
            he_cols = st.columns(4)
            he_color = Config.COLORS["success"] if regression["is_effective"] else Config.COLORS["danger"]
            he_cols[0].metric("R²", f"{regression['r_squared']:.4f}")
            he_cols[1].metric("Beta (β)", f"{regression['beta']:.4f}")
            he_cols[2].metric("Alpha (α)", f"{regression['alpha']:.6f}")
            he_cols[3].metric(
                "Status",
                "✅ Effektiv" if regression["is_effective"] else "❌ Ineffektiv",
            )

            # Scatter mit Regression
            fig_he = RiskVisualizer.hedge_effectiveness_scatter(h_ret, u_ret, regression)
            st.plotly_chart(fig_he, use_container_width=True)

        # Varianzreduktion
        st.subheader("📊 Varianzreduktion")
        vr = HedgeEffectivenessAnalyser.variance_reduction(
            hedged_costs, unhedged_costs
        )

        vr_cols = st.columns(3)
        vr_cols[0].metric("Varianz (gehedgt)", f"{vr['std_hedged']:,.0f} €")
        vr_cols[1].metric("Varianz (ungehedgt)", f"{vr['std_unhedged']:,.0f} €")

        vr_color = Config.COLORS["success"] if vr["variance_reduction"] > 0 else Config.COLORS["danger"]
        vr_cols[2].metric(
            "Varianzreduktion",
            f"{vr['variance_reduction_pct']:.1f}%",
        )

        # Vergleich der monatlichen Kosten
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            x=unhedged_costs.index, y=unhedged_costs.values,
            name="Ungehedgt (100% Spot)",
            marker_color=Config.COLORS["spot"], opacity=0.7,
        ))
        fig_comp.add_trace(go.Bar(
            x=hedged_costs.index, y=hedged_costs.values,
            name=f"Gehedgt ({bt_result.strategy.name})",
            marker_color=Config.COLORS["forward"], opacity=0.7,
        ))
        fig_comp.update_layout(
            title="Monatliche Kostenvolatilität: Gehedgt vs. Ungehedgt",
            barmode="group", yaxis_title="EUR",
            template="plotly_dark", height=400,
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    # ================================================================
    # TAB 5: PnL ATTRIBUTION
    # ================================================================
    with tab_pnl:
        st.subheader("📈 PnL-Attribution")

        bt_result = st.session_state.get("bt_single_result")
        bm_result = st.session_state.get("bt_single_benchmark")

        if bt_result is None or bm_result is None:
            st.info("Bitte führen Sie zuerst einen Backtest durch (Tab 'Einzelstrategie').")
            return

        st.markdown(
            """
            **Zerlegt die Kostendifferenz** zwischen Ihrer Strategie und 100% Spot
            in erklärbare Komponenten:

            | Komponente | Erklärung |
            |---|---|
            | **Baseline** | Kosten bei 100% Spot-Beschaffung |
            | **Hedge-Effekt** | (Spot - Forward) × Hedge-Volumen |
            | **Transaktionskosten** | Slippage + Gebühren |
            | **Timing & Residuum** | Wann genau gekauft wurde |
            """
        )

        with st.spinner("Berechne PnL-Attribution..."):
            attribution = PnLAttributionEngine.decompose(bt_result, bm_result)

        if attribution.empty:
            st.warning("Keine Attribution möglich.")
            return

        # Aggregierter Waterfall
        fig_attr = RiskVisualizer.pnl_attribution_waterfall(attribution)
        st.plotly_chart(fig_attr, use_container_width=True)

        # Monatliche Attribution
        st.subheader("📊 Monatliche PnL-Zerlegung")
        fig_monthly_attr = go.Figure()
        fig_monthly_attr.add_trace(go.Bar(
            x=attribution["delivery_month"], y=attribution["hedge_effect"],
            name="Hedge-Effekt", marker_color=Config.COLORS["success"],
        ))
        fig_monthly_attr.add_trace(go.Bar(
            x=attribution["delivery_month"], y=attribution["transaction_costs"],
            name="Transaktionskosten", marker_color=Config.COLORS["danger"],
        ))
        fig_monthly_attr.add_trace(go.Bar(
            x=attribution["delivery_month"], y=attribution["timing_residual"],
            name="Timing & Residuum", marker_color=Config.COLORS["warning"],
        ))
        fig_monthly_attr.update_layout(
            barmode="relative",
            title="Monatliche PnL-Attribution",
            yaxis_title="EUR", template="plotly_dark", height=420,
            xaxis_tickangle=-45,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        fig_monthly_attr.add_hline(y=0, line_color="white", line_width=0.5)
        st.plotly_chart(fig_monthly_attr, use_container_width=True)

        # Forward-Preis-Vorteil pro Monat
        st.subheader("💰 Forward vs. Spot Preisvergleich pro Monat")
        fig_fwd_spot = go.Figure()
        fig_fwd_spot.add_trace(go.Bar(
            x=attribution["delivery_month"],
            y=attribution["avg_spot"] - attribution["avg_forward"],
            marker_color=[
                Config.COLORS["success"] if v > 0 else Config.COLORS["danger"]
                for v in (attribution["avg_spot"] - attribution["avg_forward"])
            ],
            text=[f"{v:+.1f}" for v in (attribution["avg_spot"] - attribution["avg_forward"])],
            textposition="outside",
        ))
        fig_fwd_spot.add_hline(y=0, line_color="white", line_width=0.5)
        fig_fwd_spot.update_layout(
            title="Preisdifferenz: Spot − Forward (positiv = Forward war günstiger)",
            yaxis_title="€/MWh", template="plotly_dark", height=380,
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig_fwd_spot, use_container_width=True)

        # Detailtabelle
        with st.expander("📋 Detaillierte Attribution"):
            st.dataframe(
                attribution.style.format({
                    "baseline_cost": "{:,.0f}",
                    "actual_cost": "{:,.0f}",
                    "total_effect": "{:+,.0f}",
                    "hedge_effect": "{:+,.0f}",
                    "transaction_costs": "{:+,.0f}",
                    "timing_residual": "{:+,.0f}",
                    "hedge_ratio": "{:.0%}",
                    "avg_spot": "{:.2f}",
                    "avg_forward": "{:.2f}",
                }),
                use_container_width=True,
            )

    # ================================================================
    # TAB 6: STRESS TESTING
    # ================================================================
    with tab_stress:
        st.subheader("⚡ Stress Testing")
        st.markdown(
            """
            Testen Sie Ihre Beschaffungsstrategie gegen **historische Extremszenarien**
            und **hypothetische Schocks**. Wie resilient ist Ihre Hedge-Strategie?
            """
        )

        if load is None or sett is None:
            st.warning("Spot, Settlement und Lastprofil werden benötigt.")
            return

        # Strategie-Konfiguration
        st.markdown("### 🎛️ Zu testende Strategie")
        stress_c1, stress_c2 = st.columns(2)
        with stress_c1:
            stress_type = st.selectbox(
                "Strategietyp",
                [StrategyType.DCA, StrategyType.LIMIT_BASED, StrategyType.RSI_TRIGGERED],
                format_func=lambda x: x.value, key="stress_type",
            )
        with stress_c2:
            stress_hr = st.slider(
                "Hedge-Quote", 0, 100, 70, 5, key="stress_hr",
            )

        # Szenarien auswählen
        st.markdown("### ⚡ Szenarien")
        available_scenarios = list(StressTestEngine.HISTORICAL_SCENARIOS.keys())
        selected_scenarios = st.multiselect(
            "Szenarien auswählen",
            available_scenarios,
            default=available_scenarios[:4],
            key="stress_scenarios",
        )

        for sc_name in selected_scenarios:
            sc = StressTestEngine.HISTORICAL_SCENARIOS[sc_name]
            st.caption(f"**{sc_name}**: {sc['description']} "
                       f"(Spot ×{sc['spot_multiplier']}, {sc['duration_months']} Mon.)")

        if st.button("🧪 Stress-Test durchführen", type="primary",
                     use_container_width=True, key="stress_run"):

            stress_config = StrategyConfig(
                name=f"{stress_type.value} ({stress_hr}%)",
                strategy_type=stress_type,
                hedge_ratio=stress_hr / 100,
                n_tranches=6,
                buying_window_months=6,
            )

            data_hash = f"{len(spot)}_{len(sett)}_{len(load)}"
            engine = BacktestEngine(spot, sett, load, purch)

            # Normal-Ergebnis
            with st.spinner("Berechne Baseline..."):
                normal_result = engine.run_single(stress_config)
                spot_benchmark = engine.run_single(StrategyConfig(
                    name="100% Spot", strategy_type=StrategyType.BENCHMARK_SPOT, hedge_ratio=0
                ))

            stress_results = {}

            progress = st.progress(0, text="Stresse...")

            for i, sc_name in enumerate(selected_scenarios):
                sc = StressTestEngine.HISTORICAL_SCENARIOS[sc_name]
                progress.progress(
                    (i + 1) / len(selected_scenarios),
                    text=f"Teste: {sc_name}...",
                )

                # Spotpreise stressen
                spot_stressed = spot.copy()
                n_hours = len(spot_stressed)
                dur_hours = int(sc["duration_months"] * 30 * 24)
                start_hour = max(0, n_hours // 3)

                spot_series = spot_stressed["price_eur_mwh"].copy()
                stressed = StressTestEngine.apply_stress_to_spot(
                    spot_series, sc["spot_multiplier"],
                    start_hour, dur_hours,
                )
                spot_stressed["price_eur_mwh"] = stressed

                # Engine mit gestressten Daten
                stressed_engine = BacktestEngine(
                    spot_stressed, sett, load, purch,
                )

                # Backtest
                stressed_result = stressed_engine.run_single(stress_config)
                stressed_spot_bm = stressed_engine.run_single(StrategyConfig(
                    name="100% Spot", strategy_type=StrategyType.BENCHMARK_SPOT, hedge_ratio=0
                ))

                impact = StressTestEngine.compute_stress_impact(normal_result, stressed_result)
                impact["spot_benchmark_stressed"] = stressed_spot_bm.total_cost
                impact["hedge_protection"] = (
                    (stressed_spot_bm.total_cost - stressed_result.total_cost)
                    / stressed_spot_bm.total_cost * 100
                    if stressed_spot_bm.total_cost > 0 else 0
                )

                stress_results[sc_name] = impact

            progress.progress(1.0, text="Fertig!")
            st.session_state["stress_results"] = stress_results
            st.session_state["stress_normal"] = normal_result

        # Ergebnisse anzeigen
        stress_results = st.session_state.get("stress_results")
        stress_normal = st.session_state.get("stress_normal")

        if stress_results is not None and stress_normal is not None:
            st.divider()

            # Impact-Chart
            fig_stress = RiskVisualizer.stress_test_comparison(stress_results)
            st.plotly_chart(fig_stress, use_container_width=True)

            # Detailtabelle
            st.subheader("📋 Stress-Test-Ergebnisse")
            stress_table = []
            for sc_name, impact in stress_results.items():
                sc = StressTestEngine.HISTORICAL_SCENARIOS.get(sc_name, {})
                stress_table.append({
                    "Szenario": sc_name,
                    "Normal-Kosten (€)": impact["normal_total_cost"],
                    "Stress-Kosten (€)": impact["stressed_total_cost"],
                    "Kosten-Impact (€)": impact["cost_impact_eur"],
                    "Impact (%)": impact["cost_impact_pct"],
                    "Hedge-Schutz (%)": impact.get("hedge_protection", 0),
                    "Spot-Multiplikator": sc.get("spot_multiplier", "?"),
                })

            stress_df = pd.DataFrame(stress_table)
            st.dataframe(
                stress_df.style.format({
                    "Normal-Kosten (€)": "{:,.0f}",
                    "Stress-Kosten (€)": "{:,.0f}",
                    "Kosten-Impact (€)": "{:+,.0f}",
                    "Impact (%)": "{:+.1f}%",
                    "Hedge-Schutz (%)": "{:.1f}%",
                }).background_gradient(
                    subset=["Impact (%)"],
                    cmap="RdYlGn_r",
                ),
                use_container_width=True,
                hide_index=True,
            )

            # Schlussfolgerung
            worst_scenario = max(stress_results.items(), key=lambda x: x[1]["cost_impact_pct"])
            best_protection = max(stress_results.items(), key=lambda x: x[1].get("hedge_protection", 0))

            st.markdown("### 💡 Zusammenfassung")
            summary_cols = st.columns(2)
            with summary_cols[0]:
                st.error(
                    f"**Schlimmster Fall:** {worst_scenario[0]}\n\n"
                    f"Kostenexplosion: **{worst_scenario[1]['cost_impact_pct']:+.1f}%** "
                    f"(€{worst_scenario[1]['cost_impact_eur']:+,.0f})"
                )
            with summary_cols[1]:
                st.success(
                    f"**Bester Hedge-Schutz:** {best_protection[0]}\n\n"
                    f"Hedge reduziert Kosten um **{best_protection[1].get('hedge_protection', 0):.1f}%** "
                    f"gegenüber 100% Spot"
      )
# ============================================================================
# ============================================================================
#
#  MODULE 4 :  EXECUTIVE DASHBOARD · REPORTING · SCENARIO MANAGEMENT
#  Summary · Live Calculator · Alerts · Scenarios · Export · Reports
#
# ============================================================================
# ============================================================================

import base64
from io import BytesIO
from jinja2 import Template

# ============================================================================
# 29. SCENARIO MANAGER  –  Speichern, Laden, Vergleichen
# ============================================================================

class ScenarioManager:
    """
    Verwaltet gespeicherte Strategieszenarien in Session State.
    Ermöglicht Benennung, Versionierung und Vergleich.
    """

    SESSION_KEY = "saved_scenarios"

    @classmethod
    def init(cls):
        if cls.SESSION_KEY not in st.session_state:
            st.session_state[cls.SESSION_KEY] = {}

    @classmethod
    def save(
        cls,
        name: str,
        config: StrategyConfig,
        result: BacktestResult,
        notes: str = "",
    ):
        cls.init()
        st.session_state[cls.SESSION_KEY][name] = {
            "config": config,
            "result": result,
            "notes": notes,
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary": result.summary_dict(),
        }

    @classmethod
    def load(cls, name: str) -> Optional[Dict]:
        cls.init()
        return st.session_state[cls.SESSION_KEY].get(name)

    @classmethod
    def list_all(cls) -> Dict[str, Dict]:
        cls.init()
        return st.session_state[cls.SESSION_KEY]

    @classmethod
    def delete(cls, name: str):
        cls.init()
        if name in st.session_state[cls.SESSION_KEY]:
            del st.session_state[cls.SESSION_KEY][name]

    @classmethod
    def clear_all(cls):
        st.session_state[cls.SESSION_KEY] = {}

    @classmethod
    def get_comparison_df(cls) -> pd.DataFrame:
        """Erstellt Vergleichs-DataFrame aller gespeicherten Szenarien."""
        cls.init()
        scenarios = st.session_state[cls.SESSION_KEY]
        if not scenarios:
            return pd.DataFrame()

        records = []
        for name, data in scenarios.items():
            summary = data["summary"]
            records.append({
                "Szenario": name,
                "Gespeichert": data["saved_at"],
                "Strategie": summary.get("Typ", ""),
                "Hedge-Quote": summary.get("Ziel-Hedge-Quote", ""),
                "Ø Preis (€/MWh)": summary.get("Ø Preis (€/MWh)", ""),
                "Gesamtkosten (€)": summary.get("Gesamtkosten (€)", ""),
                "Tranchen": summary.get("Ausgeführte Tranchen", ""),
                "Notizen": data.get("notes", ""),
            })

        return pd.DataFrame(records)


# ============================================================================
# 30. ALERTING & EMPFEHLUNGSLOGIK
# ============================================================================

@dataclass
class Alert:
    """Einzelne Warnung oder Empfehlung."""
    severity: str          # CRITICAL, WARNING, INFO, OPPORTUNITY
    category: str          # HEDGE, COST, RISK, MARKET, DATA
    title: str
    message: str
    action: str = ""
    metric_value: Optional[float] = None

    @property
    def icon(self) -> str:
        return {
            "CRITICAL": "🔴",
            "WARNING": "🟡",
            "INFO": "🔵",
            "OPPORTUNITY": "🟢",
        }.get(self.severity, "⚪")

    @property
    def css_class(self) -> str:
        return {
            "CRITICAL": "val-error",
            "WARNING": "val-warn",
            "INFO": "val-info",
            "OPPORTUNITY": "val-info",
        }.get(self.severity, "val-info")


class AlertEngine:
    """
    Regelbasiertes Alerting-System.
    Analysiert Portfolio-Zustand und generiert Handlungsempfehlungen.
    """

    @staticmethod
    def analyze_portfolio(
        spot: Optional[pd.DataFrame],
        sett: Optional[pd.DataFrame],
        purch: Optional[pd.DataFrame],
        load: Optional[pd.DataFrame],
        bt_result: Optional[BacktestResult] = None,
        bt_benchmark: Optional[BacktestResult] = None,
    ) -> List[Alert]:
        alerts: List[Alert] = []

        # ---- Daten-Alerts ----
        if spot is None:
            alerts.append(Alert("CRITICAL", "DATA", "Keine Spotpreise",
                                "Spotpreise fehlen – keine Analyse möglich.",
                                "Laden Sie Spotpreise unter Daten-Management."))
            return alerts

        if sett is None:
            alerts.append(Alert("WARNING", "DATA", "Keine Settlements",
                                "Settlement-Preise fehlen – Backtesting eingeschränkt.",
                                "Laden Sie Settlement-Preise hoch."))

        if load is None:
            alerts.append(Alert("WARNING", "DATA", "Kein Lastprofil",
                                "Ohne Lastprofil kann kein Volumen-Matching erfolgen."))

        # ---- Markt-Alerts ----
        if spot is not None and len(spot) > 30:
            spot_s = spot.set_index("timestamp")["price_eur_mwh"].sort_index()
            daily = spot_s.resample("D").mean().dropna()

            if len(daily) > 30:
                current = daily.iloc[-1]
                ma30 = daily.iloc[-30:].mean()
                ma90 = daily.iloc[-90:].mean() if len(daily) > 90 else ma30
                vol_30d = daily.pct_change().iloc[-30:].std() * np.sqrt(252)

                # Preis vs. Moving Average
                pct_vs_ma30 = (current - ma30) / ma30 * 100
                if pct_vs_ma30 > 20:
                    alerts.append(Alert(
                        "WARNING", "MARKET",
                        f"Spotpreis {pct_vs_ma30:+.0f}% über MA-30",
                        f"Aktuell: {current:.2f} €/MWh vs. MA-30: {ma30:.2f} €/MWh. "
                        f"Markt ist überhitzt.",
                        "Erwägen Sie, offene Positionen über Forward abzusichern.",
                        pct_vs_ma30,
                    ))
                elif pct_vs_ma30 < -20:
                    alerts.append(Alert(
                        "OPPORTUNITY", "MARKET",
                        f"Spotpreis {pct_vs_ma30:+.0f}% unter MA-30",
                        f"Aktuell: {current:.2f} €/MWh vs. MA-30: {ma30:.2f} €/MWh. "
                        f"Markt ist günstig.",
                        "Guter Zeitpunkt für Forward-Beschaffung / Aufstockung.",
                        pct_vs_ma30,
                    ))

                # Volatilität
                if vol_30d > 0.8:
                    alerts.append(Alert(
                        "WARNING", "RISK",
                        f"Hohe Volatilität: {vol_30d:.0%} (ann.)",
                        "30-Tage-Volatilität liegt über 80%. "
                        "Erhöhtes Preisrisiko für offene Positionen.",
                        "Erhöhen Sie die Hedge-Quote oder nutzen Sie DCA.",
                        vol_30d * 100,
                    ))

                # Trend
                if len(daily) > 90:
                    trend_pct = (daily.iloc[-1] - daily.iloc[-90]) / daily.iloc[-90] * 100
                    if trend_pct > 30:
                        alerts.append(Alert(
                            "WARNING", "MARKET",
                            f"Aufwärtstrend: {trend_pct:+.0f}% in 90 Tagen",
                            "Anhaltender Preisanstieg über 3 Monate.",
                            "Forward-Beschaffung prüfen, bevor Preise weiter steigen.",
                        ))

        # ---- Hedge-Alerts ----
        if bt_result is not None:
            hr = bt_result.effective_hedge_ratio
            if hr < 0.3:
                alerts.append(Alert(
                    "CRITICAL", "HEDGE",
                    f"Niedrige Hedge-Quote: {hr:.0%}",
                    "Weniger als 30% des Volumens ist abgesichert. "
                    "Hohes Spotpreisrisiko.",
                    "Erhöhen Sie die Forward-Beschaffung.",
                    hr * 100,
                ))
            elif hr > 1.1:
                alerts.append(Alert(
                    "WARNING", "HEDGE",
                    f"Überhedging: {hr:.0%}",
                    "Mehr als 110% des Verbrauchs ist abgesichert. "
                    "Risiko der Rückvermarktung.",
                    "Prüfen Sie offene Positionen.",
                    hr * 100,
                ))

            # Kosten vs. Benchmark
            if bt_benchmark is not None:
                saving = (bt_benchmark.total_cost - bt_result.total_cost)
                saving_pct = saving / bt_benchmark.total_cost * 100 if bt_benchmark.total_cost > 0 else 0

                if saving_pct > 10:
                    alerts.append(Alert(
                        "OPPORTUNITY", "COST",
                        f"Strategie spart {saving_pct:.1f}% vs. 100% Spot",
                        f"Absolute Ersparnis: €{saving:,.0f}. "
                        f"Ø Preis: {bt_result.avg_blended_price:.2f} vs. "
                        f"{bt_benchmark.avg_blended_price:.2f} €/MWh.",
                    ))
                elif saving_pct < -10:
                    alerts.append(Alert(
                        "WARNING", "COST",
                        f"Strategie kostet {-saving_pct:.1f}% mehr als 100% Spot",
                        f"Forward-Beschaffung war teurer. "
                        f"Prüfen Sie Timing und Hedge-Quote.",
                        "Erwägen Sie eine niedrigere Hedge-Quote oder andere Strategie.",
                    ))

        # ---- Beschaffungs-Alerts ----
        if purch is not None and len(purch) > 0:
            fwd = purch[purch["purchase_type"].str.lower() == "forward"]
            if len(fwd) > 0:
                max_price = fwd["price_eur_mwh"].max()
                min_price = fwd["price_eur_mwh"].min()
                spread = max_price - min_price
                avg_price = fwd["price_eur_mwh"].mean()

                if spread / avg_price > 0.5:
                    alerts.append(Alert(
                        "INFO", "COST",
                        f"Große Preisspanne in Käufen: {spread:.2f} €/MWh",
                        f"Min: {min_price:.2f} – Max: {max_price:.2f} €/MWh. "
                        f"Timing hatte großen Einfluss.",
                    ))

        # Sortiere: CRITICAL zuerst
        severity_order = {"CRITICAL": 0, "WARNING": 1, "OPPORTUNITY": 2, "INFO": 3}
        alerts.sort(key=lambda a: severity_order.get(a.severity, 99))

        return alerts


# ============================================================================
# 31. EXCEL EXPORT ENGINE
# ============================================================================

class ExcelExporter:
    """
    Multi-Sheet Excel-Export mit Formatierung.
    """

    @staticmethod
    def export_full_report(
        spot: Optional[pd.DataFrame],
        sett: Optional[pd.DataFrame],
        purch: Optional[pd.DataFrame],
        load: Optional[pd.DataFrame],
        bt_result: Optional[BacktestResult] = None,
        bt_benchmark: Optional[BacktestResult] = None,
        comp_results: Optional[List[BacktestResult]] = None,
        grid_results: Optional[pd.DataFrame] = None,
    ) -> bytes:
        """Erzeugt vollständigen Excel-Report."""
        output = BytesIO()

        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            # Sheet 1: Summary
            summary_data = {"Metrik": [], "Wert": []}

            if bt_result:
                for k, v in bt_result.summary_dict().items():
                    summary_data["Metrik"].append(k)
                    summary_data["Wert"].append(str(v))

            if bt_benchmark:
                summary_data["Metrik"].append("---")
                summary_data["Wert"].append("---")
                summary_data["Metrik"].append("BENCHMARK (100% Spot)")
                summary_data["Wert"].append("")
                for k, v in bt_benchmark.summary_dict().items():
                    summary_data["Metrik"].append(f"BM: {k}")
                    summary_data["Wert"].append(str(v))

                saving = bt_benchmark.total_cost - bt_result.total_cost
                saving_pct = saving / bt_benchmark.total_cost * 100 if bt_benchmark.total_cost > 0 else 0
                summary_data["Metrik"].append("Ersparnis vs. Spot (€)")
                summary_data["Wert"].append(f"{saving:,.0f}")
                summary_data["Metrik"].append("Ersparnis vs. Spot (%)")
                summary_data["Wert"].append(f"{saving_pct:.1f}%")

            pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)

            # Sheet 2: Monatliche Ergebnisse
            if bt_result:
                bt_result.to_dataframe().to_excel(
                    writer, sheet_name="Monatsergebnisse", index=False
                )

            # Sheet 3: Tranchen
            if bt_result and bt_result.all_tranches:
                tranche_records = []
                for t in bt_result.all_tranches:
                    tranche_records.append({
                        "Kaufdatum": t.execution_date,
                        "Liefermonat": t.delivery_month,
                        "Volumen (MWh)": t.volume_mwh,
                        "Preis (€/MWh)": t.price_eur_mwh,
                        "Kosten (€)": t.cost_eur,
                    })
                pd.DataFrame(tranche_records).to_excel(
                    writer, sheet_name="Tranchen", index=False
                )

            # Sheet 4: Strategievergleich
            if comp_results:
                comp_table = BacktestVisualizer.strategy_comparison_table(comp_results)
                comp_table.to_excel(writer, sheet_name="Strategievergleich", index=False)

            # Sheet 5: Grid Search
            if grid_results is not None and not grid_results.empty:
                grid_results.to_excel(writer, sheet_name="Grid Search", index=False)

            # Sheet 6: Spotpreise (Aggregat)
            if spot is not None:
                spot_daily = spot.set_index("timestamp")["price_eur_mwh"].resample("D").agg(
                    ["mean", "min", "max", "std"]
                ).reset_index()
                spot_daily.columns = ["Datum", "Ø Preis", "Min", "Max", "StdAbw"]
                spot_daily.to_excel(writer, sheet_name="Spotpreise (täglich)", index=False)

            # Sheet 7: Beschaffungen
            if purch is not None:
                purch.to_excel(writer, sheet_name="Beschaffungen", index=False)

            # Sheet 8: Lastprofil (Monatlich)
            if load is not None:
                load_monthly = load.set_index("timestamp")["volume_mwh"].resample("MS").agg(
                    ["sum", "mean", "max"]
                ).reset_index()
                load_monthly.columns = ["Monat", "Summe (MWh)", "Ø (MWh)", "Max (MWh)"]
                load_monthly.to_excel(writer, sheet_name="Lastprofil (monatl.)", index=False)

        return output.getvalue()


# ============================================================================
# 32. HTML REPORT GENERATOR
# ============================================================================

class HTMLReportGenerator:
    """Erzeugt einen formatierten HTML-Bericht zum Download."""

    TEMPLATE = """
    <!DOCTYPE html>
    <html lang="de">
    <head>
        <meta charset="UTF-8">
        <title>Energy Portfolio Report</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', -apple-system, sans-serif;
                background: #0e1117; color: #ecf0f1;
                padding: 40px; line-height: 1.6;
            }
            .header {
                text-align: center; margin-bottom: 40px;
                border-bottom: 2px solid #3498db; padding-bottom: 20px;
            }
            .header h1 {
                font-size: 2rem; font-weight: 800;
                background: linear-gradient(135deg, #3498db, #e74c3c, #f39c12);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            }
            .header .subtitle { color: #7f8c8d; font-size: 0.95rem; margin-top: 5px; }
            .header .date { color: #95a5a6; font-size: 0.85rem; margin-top: 8px; }
            .section { margin: 30px 0; }
            .section h2 {
                font-size: 1.4rem; color: #3498db;
                border-left: 4px solid #3498db; padding-left: 12px;
                margin-bottom: 15px;
            }
            .kpi-grid {
                display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 12px; margin: 15px 0;
            }
            .kpi-card {
                background: linear-gradient(135deg, #1a1a2e, #16213e);
                border: 1px solid #2c3e6b; border-radius: 10px;
                padding: 16px; text-align: center;
            }
            .kpi-card .label { color: #7f8c8d; font-size: 0.75rem; text-transform: uppercase; }
            .kpi-card .value { color: #ecf0f1; font-size: 1.5rem; font-weight: 700; }
            .kpi-card .delta { font-size: 0.85rem; }
            .delta-pos { color: #2ecc71; }
            .delta-neg { color: #e74c3c; }
            table {
                width: 100%; border-collapse: collapse;
                margin: 15px 0; font-size: 0.9rem;
            }
            th {
                background: #1a1a2e; color: #3498db;
                padding: 10px 12px; text-align: left;
                border-bottom: 2px solid #2c3e6b;
            }
            td {
                padding: 8px 12px; border-bottom: 1px solid #2c3e50;
            }
            tr:hover td { background: rgba(52,152,219,0.08); }
            .alert-critical { border-left: 4px solid #e74c3c; background: #2c0b0e; padding: 12px; margin: 6px 0; border-radius: 0 6px 6px 0; }
            .alert-warning { border-left: 4px solid #f39c12; background: #2c2200; padding: 12px; margin: 6px 0; border-radius: 0 6px 6px 0; }
            .alert-info { border-left: 4px solid #3498db; background: #0a1628; padding: 12px; margin: 6px 0; border-radius: 0 6px 6px 0; }
            .alert-opportunity { border-left: 4px solid #2ecc71; background: #0a2818; padding: 12px; margin: 6px 0; border-radius: 0 6px 6px 0; }
            .footer {
                text-align: center; margin-top: 50px;
                border-top: 1px solid #2c3e50; padding-top: 15px;
                color: #7f8c8d; font-size: 0.8rem;
            }
            @media print {
                body { background: white; color: #333; padding: 20px; }
                .kpi-card { border: 1px solid #ccc; }
                .kpi-card .value { color: #333; }
                th { background: #f0f0f0; color: #333; }
                td { border-bottom: 1px solid #ddd; }
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>⚡ Energy Portfolio Report</h1>
            <div class="subtitle">{{ subtitle }}</div>
            <div class="date">Erstellt: {{ report_date }}</div>
        </div>

        <!-- KPI Section -->
        <div class="section">
            <h2>📊 Key Performance Indicators</h2>
            <div class="kpi-grid">
                {% for kpi in kpis %}
                <div class="kpi-card">
                    <div class="label">{{ kpi.label }}</div>
                    <div class="value">{{ kpi.value }}</div>
                    {% if kpi.delta %}
                    <div class="delta {{ 'delta-pos' if kpi.delta_positive else 'delta-neg' }}">
                        {{ kpi.delta }}
                    </div>
                    {% endif %}
                </div>
                {% endfor %}
            </div>
        </div>

        <!-- Strategy Summary -->
        {% if strategy_summary %}
        <div class="section">
            <h2>🔄 Strategie-Zusammenfassung</h2>
            <table>
                <tr><th>Parameter</th><th>Wert</th></tr>
                {% for key, value in strategy_summary.items() %}
                <tr><td>{{ key }}</td><td>{{ value }}</td></tr>
                {% endfor %}
            </table>
        </div>
        {% endif %}

        <!-- Monthly Results -->
        {% if monthly_table %}
        <div class="section">
            <h2>📋 Monatliche Ergebnisse</h2>
            <table>
                <tr>
                    {% for col in monthly_columns %}
                    <th>{{ col }}</th>
                    {% endfor %}
                </tr>
                {% for row in monthly_table %}
                <tr>
                    {% for val in row %}
                    <td>{{ val }}</td>
                    {% endfor %}
                </tr>
                {% endfor %}
            </table>
        </div>
        {% endif %}

        <!-- Alerts -->
        {% if alerts %}
        <div class="section">
            <h2>⚠️ Alerts & Empfehlungen</h2>
            {% for alert in alerts %}
            <div class="alert-{{ alert.css_class }}">
                <strong>{{ alert.icon }} {{ alert.title }}</strong><br>
                {{ alert.message }}
                {% if alert.action %}<br><em>→ {{ alert.action }}</em>{% endif %}
            </div>
            {% endfor %}
        </div>
        {% endif %}

        <!-- Comparison -->
        {% if comparison_table %}
        <div class="section">
            <h2>⚖️ Strategievergleich</h2>
            <table>
                <tr>
                    {% for col in comparison_columns %}
                    <th>{{ col }}</th>
                    {% endfor %}
                </tr>
                {% for row in comparison_table %}
                <tr>
                    {% for val in row %}
                    <td>{{ val }}</td>
                    {% endfor %}
                </tr>
                {% endfor %}
            </table>
        </div>
        {% endif %}

        <div class="footer">
            <p>Energy Portfolio Backtesting System v{{ version }}</p>
            <p>Generiert am {{ report_date }} · Vertraulich</p>
        </div>
    </body>
    </html>
    """

    @staticmethod
    def generate(
        bt_result: Optional[BacktestResult] = None,
        bt_benchmark: Optional[BacktestResult] = None,
        comp_results: Optional[List[BacktestResult]] = None,
        alerts: Optional[List[Alert]] = None,
        spot: Optional[pd.DataFrame] = None,
        subtitle: str = "Backtesting & Procurement Analytics",
    ) -> str:
        """Erzeugt vollständigen HTML-Report."""

        # KPIs
        kpis = []
        if bt_result:
            kpis.append({"label": "Gesamtkosten", "value": f"€{bt_result.total_cost:,.0f}",
                         "delta": None, "delta_positive": True})
            kpis.append({"label": "Ø Blended Price", "value": f"{bt_result.avg_blended_price:.2f} €/MWh",
                         "delta": None, "delta_positive": True})
            kpis.append({"label": "Hedge-Quote", "value": f"{bt_result.effective_hedge_ratio:.0%}",
                         "delta": None, "delta_positive": True})
            kpis.append({"label": "Tranchen", "value": f"{len(bt_result.all_tranches)}",
                         "delta": None, "delta_positive": True})

            if bt_benchmark:
                saving = bt_benchmark.total_cost - bt_result.total_cost
                saving_pct = saving / bt_benchmark.total_cost * 100 if bt_benchmark.total_cost > 0 else 0
                kpis.append({
                    "label": "vs. 100% Spot",
                    "value": f"{saving_pct:+.1f}%",
                    "delta": f"€{saving:+,.0f}",
                    "delta_positive": saving > 0,
                })

        if spot is not None and len(spot) > 0:
            kpis.append({"label": "Spot Ø", "value": f"{spot['price_eur_mwh'].mean():.2f} €",
                         "delta": None, "delta_positive": True})

        # Strategy Summary
        strategy_summary = bt_result.summary_dict() if bt_result else None

        # Monthly Table
        monthly_table = None
        monthly_columns = []
        if bt_result:
            df = bt_result.to_dataframe()
            monthly_columns = ["Monat", "Nachfrage", "Gehedgt", "Offen",
                               "Fwd-Kosten", "Spot-Kosten", "Gesamt", "Ø Preis"]
            monthly_table = []
            for _, row in df.iterrows():
                monthly_table.append([
                    row["delivery_month"],
                    f"{row['demand_mwh']:,.0f}",
                    f"{row['hedged_mwh']:,.0f}",
                    f"{row['open_mwh']:,.0f}",
                    f"€{row['fwd_cost_eur']:,.0f}",
                    f"€{row['spot_cost_eur']:,.0f}",
                    f"€{row['total_cost_eur']:,.0f}",
                    f"{row['blended_price']:.2f}",
                ])

        # Alerts
        alert_data = []
        if alerts:
            for a in alerts:
                css_map = {"CRITICAL": "critical", "WARNING": "warning",
                           "INFO": "info", "OPPORTUNITY": "opportunity"}
                alert_data.append({
                    "icon": a.icon,
                    "title": a.title,
                    "message": a.message,
                    "action": a.action,
                    "css_class": css_map.get(a.severity, "info"),
                })

        # Comparison
        comparison_table = None
        comparison_columns = []
        if comp_results and len(comp_results) > 1:
            comp_df = BacktestVisualizer.strategy_comparison_table(comp_results)
            comparison_columns = comp_df.columns.tolist()
            comparison_table = comp_df.values.tolist()

        template = Template(HTMLReportGenerator.TEMPLATE)
        html = template.render(
            subtitle=subtitle,
            report_date=datetime.now().strftime("%d.%m.%Y %H:%M"),
            version=Config.VERSION,
            kpis=kpis,
            strategy_summary=strategy_summary,
            monthly_table=monthly_table,
            monthly_columns=monthly_columns,
            alerts=alert_data,
            comparison_table=comparison_table,
            comparison_columns=comparison_columns,
        )

        return html


# ============================================================================
# 33. LIVE STRATEGY CALCULATOR (Echtzeit-Neuberechnung)
# ============================================================================

def render_live_calculator(engine: BacktestEngine):
    """
    Interaktiver Echtzeit-Strategierechner.
    Slider-Änderungen triggern sofortige Neuberechnung.
    """
    st.subheader("🎛️ Live Strategy Calculator")
    st.markdown(
        "Ziehen Sie die Slider – die Ergebnisse aktualisieren sich **in Echtzeit**."
    )

    # Slider-Reihe
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        live_hr = st.slider(
            "Hedge-Quote", 0, 100, 50, 5,
            key="live_hr", format="%d%%",
        )
    with col2:
        live_type = st.selectbox(
            "Strategie",
            [StrategyType.DCA, StrategyType.LIMIT_BASED,
             StrategyType.RSI_TRIGGERED, StrategyType.SEASONAL],
            format_func=lambda x: x.value.split(" ")[0],
            key="live_type",
        )
    with col3:
        live_tranches = st.slider(
            "Tranchen", 1, 18, 6, key="live_tranches",
        )
    with col4:
        live_window = st.slider(
            "Fenster (Mon.)", 1, 12, 6, key="live_window",
        )

    # Berechnung
    config = StrategyConfig(
        name=f"Live: {live_type.value.split(' ')[0]} {live_hr}%",
        strategy_type=live_type,
        hedge_ratio=live_hr / 100,
        n_tranches=live_tranches,
        buying_window_months=live_window,
    )

    benchmark_cfg = StrategyConfig(
        name="100% Spot", strategy_type=StrategyType.BENCHMARK_SPOT, hedge_ratio=0,
    )

    result = engine.run_single(config)
    benchmark = engine.run_single(benchmark_cfg)

    saving = benchmark.total_cost - result.total_cost
    saving_pct = saving / benchmark.total_cost * 100 if benchmark.total_cost > 0 else 0

    # KPI-Reihe
    kpi_cols = st.columns(5)
    kpi_cols[0].metric("Gesamtkosten", f"€{result.total_cost:,.0f}")
    kpi_cols[1].metric("Ø Preis", f"{result.avg_blended_price:.2f} €/MWh")
    kpi_cols[2].metric("Eff. Hedge", f"{result.effective_hedge_ratio:.0%}")
    kpi_cols[3].metric("vs. Spot", f"{saving_pct:+.1f}%", f"€{saving:+,.0f}")
    kpi_cols[4].metric("Tranchen", f"{len(result.all_tranches)}")

    # Mini-Charts
    chart_cols = st.columns(2)

    with chart_cols[0]:
        df_r = result.to_dataframe()
        fig_mini = go.Figure()
        fig_mini.add_trace(go.Bar(
            x=df_r["delivery_month"], y=df_r["fwd_cost_eur"],
            name="Forward", marker_color=Config.COLORS["forward"],
        ))
        fig_mini.add_trace(go.Bar(
            x=df_r["delivery_month"], y=df_r["spot_cost_eur"],
            name="Spot", marker_color=Config.COLORS["spot"],
        ))
        fig_mini.update_layout(
            barmode="stack", title="Monatliche Kosten",
            template="plotly_dark", height=300,
            showlegend=True, xaxis_tickangle=-45,
            margin=dict(l=40, r=20, t=40, b=60),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_mini, use_container_width=True)

    with chart_cols[1]:
        bm_df = benchmark.to_dataframe()
        fig_mini2 = go.Figure()
        fig_mini2.add_trace(go.Scatter(
            x=bm_df["delivery_month"], y=bm_df["blended_price"].cumsum(),
            name="100% Spot (kum.)", mode="lines",
            line=dict(color=Config.COLORS["spot"], width=2),
        ))
        fig_mini2.add_trace(go.Scatter(
            x=df_r["delivery_month"], y=df_r["blended_price"].cumsum(),
            name="Strategie (kum.)", mode="lines",
            line=dict(color=Config.COLORS["actual"], width=2),
        ))
        fig_mini2.update_layout(
            title="Kum. Blended Price",
            template="plotly_dark", height=300,
            xaxis_tickangle=-45,
            margin=dict(l=40, r=20, t=40, b=60),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_mini2, use_container_width=True)

    return result, benchmark


# ============================================================================
# 34. PAGE: EXECUTIVE DASHBOARD
# ============================================================================

def page_executive_dashboard():
    st.header("🏛️ Executive Dashboard")

    spot = DataStore.get("spot_prices")
    sett = DataStore.get("settlement_prices")
    load = DataStore.get("load_profile")
    purch = DataStore.get("purchases")

    if spot is None:
        st.warning("⚠️ Laden Sie zuerst Daten unter **Daten-Management**.")
        return

    # ---- ALERTING ----
    bt_result = st.session_state.get("bt_single_result")
    bt_benchmark = st.session_state.get("bt_single_benchmark")

    alerts = AlertEngine.analyze_portfolio(
        spot, sett, purch, load, bt_result, bt_benchmark
    )

    if alerts:
        with st.expander(f"⚠️ **{len(alerts)} Alerts & Empfehlungen**", expanded=True):
            for alert in alerts:
                st.markdown(
                    f'<div class="{alert.css_class}">'
                    f'{alert.icon} <strong>{alert.title}</strong><br>'
                    f'{alert.message}'
                    f'{"<br><em>→ " + alert.action + "</em>" if alert.action else ""}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ---- MARKT-OVERVIEW ----
    st.subheader("📈 Marktüberblick")

    spot_s = spot.set_index("timestamp")["price_eur_mwh"].sort_index()
    daily = spot_s.resample("D").mean().dropna()

    if len(daily) > 0:
        current = daily.iloc[-1]
        prev = daily.iloc[-2] if len(daily) > 1 else current
        change_1d = (current - prev) / prev * 100 if prev != 0 else 0

        ma7 = daily.iloc[-7:].mean() if len(daily) >= 7 else current
        ma30 = daily.iloc[-30:].mean() if len(daily) >= 30 else current
        ma90 = daily.iloc[-90:].mean() if len(daily) >= 90 else current
        vol_30d = daily.pct_change().iloc[-30:].std() * np.sqrt(252) * 100 if len(daily) >= 30 else 0
        ytd_change = (current - daily.iloc[0]) / daily.iloc[0] * 100 if daily.iloc[0] != 0 else 0

        market_cols = st.columns(6)
        market_cols[0].metric("Aktueller Spot", f"{current:.2f} €", f"{change_1d:+.1f}%")
        market_cols[1].metric("MA-7", f"{ma7:.2f} €")
        market_cols[2].metric("MA-30", f"{ma30:.2f} €")
        market_cols[3].metric("MA-90", f"{ma90:.2f} €")
        market_cols[4].metric("Vol. (30d ann.)", f"{vol_30d:.0f}%")
        market_cols[5].metric("YTD", f"{ytd_change:+.1f}%")

        # Sparkline
        fig_spark = go.Figure()
        fig_spark.add_trace(go.Scatter(
            x=daily.index[-90:], y=daily.values[-90:],
            mode="lines", name="Spot (90d)",
            line=dict(color=Config.COLORS["spot"], width=2),
            fill="tozeroy", fillcolor="rgba(231,76,60,0.1)",
        ))
        ma_30_series = daily.rolling(30).mean()
        fig_spark.add_trace(go.Scatter(
            x=ma_30_series.index[-90:], y=ma_30_series.values[-90:],
            mode="lines", name="MA-30",
            line=dict(color=Config.COLORS["forward"], width=1.5, dash="dash"),
        ))
        fig_spark.update_layout(
            height=250, template="plotly_dark",
            margin=dict(l=40, r=20, t=10, b=30),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig_spark, use_container_width=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ---- PORTFOLIO-STATUS ----
    if bt_result and bt_benchmark:
        st.subheader("📊 Portfolio-Performance")

        saving = bt_benchmark.total_cost - bt_result.total_cost
        saving_pct = saving / bt_benchmark.total_cost * 100 if bt_benchmark.total_cost > 0 else 0

        perf_cols = st.columns(5)
        perf_cols[0].metric("Strategie", bt_result.strategy.name[:25])
        perf_cols[1].metric("Gesamtkosten", f"€{bt_result.total_cost:,.0f}")
        perf_cols[2].metric("Ø Preis", f"{bt_result.avg_blended_price:.2f} €/MWh")
        perf_cols[3].metric("vs. 100% Spot", f"{saving_pct:+.1f}%", f"€{saving:+,.0f}")
        perf_cols[4].metric("Hedge-Quote", f"{bt_result.effective_hedge_ratio:.0%}")

        # Pie Chart: Kostenaufschlüsselung
        pie_c1, pie_c2 = st.columns(2)

        with pie_c1:
            fig_pie = go.Figure(go.Pie(
                labels=["Forward-Kosten", "Spot-Kosten"],
                values=[bt_result.total_forward_cost, bt_result.total_spot_cost],
                marker_colors=[Config.COLORS["forward"], Config.COLORS["spot"]],
                hole=0.55, textinfo="percent+label",
            ))
            fig_pie.update_layout(
                title="Kostenaufteilung", template="plotly_dark", height=320,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with pie_c2:
            # Gauge für Hedge-Quote
            hr_val = bt_result.effective_hedge_ratio * 100
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=hr_val,
                number=dict(suffix="%"),
                delta=dict(reference=70, suffix="%"),
                gauge=dict(
                    axis=dict(range=[0, 120], tickwidth=1),
                    bar=dict(color=Config.COLORS["forward"]),
                    steps=[
                        dict(range=[0, 30], color="rgba(231,76,60,0.3)"),
                        dict(range=[30, 60], color="rgba(243,156,18,0.3)"),
                        dict(range=[60, 100], color="rgba(46,204,113,0.3)"),
                        dict(range=[100, 120], color="rgba(52,152,219,0.3)"),
                    ],
                    threshold=dict(
                        line=dict(color="white", width=3),
                        thickness=0.8, value=100,
                    ),
                ),
                title=dict(text="Hedge-Quote", font=dict(size=16)),
            ))
            fig_gauge.update_layout(
                template="plotly_dark", height=320,
                margin=dict(l=20, r=20, t=60, b=20),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Kumulative Performance
        df_bt = bt_result.to_dataframe()
        df_bm = bt_benchmark.to_dataframe()

        fig_perf = go.Figure()
        fig_perf.add_trace(go.Scatter(
            x=df_bm["delivery_month"],
            y=df_bm["total_cost_eur"].cumsum(),
            name="100% Spot", mode="lines",
            line=dict(color=Config.COLORS["spot"], width=2, dash="dot"),
        ))
        fig_perf.add_trace(go.Scatter(
            x=df_bt["delivery_month"],
            y=df_bt["total_cost_eur"].cumsum(),
            name=bt_result.strategy.name, mode="lines",
            line=dict(color=Config.COLORS["actual"], width=3),
        ))
        # Differenz schattiert
        cum_bm = df_bm["total_cost_eur"].cumsum()
        cum_bt = df_bt["total_cost_eur"].cumsum()

        fig_perf.update_layout(
            title="Kumulative Performance vs. Benchmark",
            yaxis_title="Kumulative Kosten (€)",
            template="plotly_dark", height=380,
            xaxis_tickangle=-45, hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_perf, use_container_width=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ---- LIVE CALCULATOR ----
    if sett is not None and load is not None:
        data_hash = f"{len(spot)}_{len(sett)}_{len(load)}_{len(purch) if purch is not None else 0}"

        @st.cache_resource(show_spinner=False)
        def get_exec_engine(_spot, _sett, _load, _purch, _hash):
            return BacktestEngine(_spot, _sett, _load, _purch)

        engine = get_exec_engine(spot, sett, load, purch, data_hash)
        live_result, live_bm = render_live_calculator(engine)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ---- BESCHAFFUNGSHISTORIE ----
    if purch is not None and len(purch) > 0:
        st.subheader("🛒 Beschaffungshistorie (Letzte Transaktionen)")
        recent = purch.sort_values("purchase_date", ascending=False).head(10)
        display_cols = ["purchase_date", "product", "volume_mw", "price_eur_mwh", "purchase_type"]
        existing_cols = [c for c in display_cols if c in recent.columns]

        st.dataframe(
            recent[existing_cols].style.format({
                "volume_mw": "{:.0f}",
                "price_eur_mwh": "{:.2f}",
            }),
            use_container_width=True,
            height=300,
        )


# ============================================================================
# 35. PAGE: SCENARIO MANAGER
# ============================================================================

def page_scenario_manager():
    st.header("💾 Szenario-Manager")
    st.markdown(
        "Speichern, vergleichen und exportieren Sie verschiedene Strategieszenarien."
    )

    ScenarioManager.init()

    tab_save, tab_compare, tab_export = st.tabs([
        "💾 Speichern",
        "⚖️ Vergleichen",
        "📤 Exportieren",
    ])

    # ---- SPEICHERN ----
    with tab_save:
        st.subheader("💾 Aktuelles Ergebnis speichern")

        bt_result = st.session_state.get("bt_single_result")

        if bt_result is None:
            st.info("Führen Sie zuerst einen Backtest durch (Backtesting Engine → Einzelstrategie).")
        else:
            st.success(
                f"Aktiver Backtest: **{bt_result.strategy.name}** | "
                f"€{bt_result.total_cost:,.0f} | "
                f"Ø {bt_result.avg_blended_price:.2f} €/MWh"
            )

            c1, c2 = st.columns([2, 3])
            with c1:
                sc_name = st.text_input(
                    "Szenario-Name",
                    value=f"{bt_result.strategy.name} – {datetime.now().strftime('%d.%m %H:%M')}",
                    key="sc_save_name",
                )
            with c2:
                sc_notes = st.text_area(
                    "Notizen",
                    placeholder="z.B. 'Aggressive Strategie für Winter 2024'",
                    key="sc_save_notes",
                    height=80,
                )

            if st.button("💾 Szenario speichern", type="primary", key="sc_save_btn"):
                ScenarioManager.save(sc_name, bt_result.strategy, bt_result, sc_notes)
                st.success(f"✅ Szenario **'{sc_name}'** gespeichert!")
                st.rerun()

        # Gespeicherte Szenarien
        st.divider()
        st.subheader("📋 Gespeicherte Szenarien")

        scenarios = ScenarioManager.list_all()
        if not scenarios:
            st.info("Noch keine Szenarien gespeichert.")
        else:
            for name, data in scenarios.items():
                with st.expander(f"📁 {name}", expanded=False):
                    s_cols = st.columns([3, 1])
                    with s_cols[0]:
                        summary = data["summary"]
                        s_mcols = st.columns(4)
                        s_mcols[0].metric("Strategie", summary.get("Typ", "")[:25])
                        s_mcols[1].metric("Hedge-Quote", summary.get("Ziel-Hedge-Quote", ""))
                        s_mcols[2].metric("Ø Preis", summary.get("Ø Preis (€/MWh)", ""))
                        s_mcols[3].metric("Kosten", summary.get("Gesamtkosten (€)", ""))

                        if data.get("notes"):
                            st.caption(f"📝 {data['notes']}")
                        st.caption(f"Gespeichert: {data['saved_at']}")

                    with s_cols[1]:
                        if st.button("🗑️ Löschen", key=f"sc_del_{name}"):
                            ScenarioManager.delete(name)
                            st.rerun()

    # ---- VERGLEICHEN ----
    with tab_compare:
        st.subheader("⚖️ Szenario-Vergleich")

        scenarios = ScenarioManager.list_all()
        if len(scenarios) < 2:
            st.info("Speichern Sie mindestens 2 Szenarien zum Vergleichen.")
        else:
            comp_df = ScenarioManager.get_comparison_df()
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            # Kosten-Vergleichs-Chart
            if not comp_df.empty:
                names = comp_df["Szenario"].tolist()
                costs = []
                prices = []
                for name in names:
                    data = scenarios[name]
                    costs.append(data["result"].total_cost)
                    prices.append(data["result"].avg_blended_price)

                fig_sc = make_subplots(specs=[[{"secondary_y": True}]])
                fig_sc.add_trace(go.Bar(
                    x=names, y=costs,
                    name="Gesamtkosten (€)",
                    marker_color=Config.COLORS["forward"], opacity=0.7,
                ), secondary_y=False)
                fig_sc.add_trace(go.Scatter(
                    x=names, y=prices,
                    name="Ø Preis (€/MWh)",
                    mode="lines+markers",
                    line=dict(color=Config.COLORS["spot"], width=2.5),
                    marker=dict(size=10),
                ), secondary_y=True)
                fig_sc.update_layout(
                    title="Szenario-Vergleich",
                    template="plotly_dark", height=420,
                    xaxis_tickangle=-25,
                )
                fig_sc.update_yaxes(title_text="EUR", secondary_y=False)
                fig_sc.update_yaxes(title_text="€/MWh", secondary_y=True)
                st.plotly_chart(fig_sc, use_container_width=True)

                # Kumulative Kosten aller gespeicherten Szenarien
                st.subheader("📈 Kumulative Kosten aller Szenarien")
                fig_cum_sc = go.Figure()
                colors = [
                    Config.COLORS["spot"], Config.COLORS["forward"],
                    Config.COLORS["actual"], Config.COLORS["hedge"],
                    Config.COLORS["warning"], Config.COLORS["info"],
                    "#e74c3c", "#1abc9c", "#9b59b6", "#e67e22",
                ]
                for i, (name, data) in enumerate(scenarios.items()):
                    df_sc = data["result"].to_dataframe()
                    fig_cum_sc.add_trace(go.Scatter(
                        x=df_sc["delivery_month"],
                        y=df_sc["total_cost_eur"].cumsum(),
                        mode="lines+markers", name=name,
                        line=dict(color=colors[i % len(colors)], width=2),
                        marker=dict(size=3),
                    ))
                fig_cum_sc.update_layout(
                    yaxis_title="Kumulative Kosten (€)",
                    template="plotly_dark", height=450,
                    xaxis_tickangle=-45, hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig_cum_sc, use_container_width=True)

    # ---- EXPORTIEREN ----
    with tab_export:
        st.subheader("📤 Reports & Datenexport")

        spot = DataStore.get("spot_prices")
        sett = DataStore.get("settlement_prices")
        load = DataStore.get("load_profile")
        purch = DataStore.get("purchases")
        bt_result = st.session_state.get("bt_single_result")
        bt_benchmark = st.session_state.get("bt_single_benchmark")
        comp_results = st.session_state.get("comp_results")
        grid_results = st.session_state.get("opt_grid")

        st.markdown("### 📊 Excel-Export (Multi-Sheet)")
        st.caption(
            "Enthält: Summary, Monatsergebnisse, Tranchen, Strategievergleich, "
            "Grid Search, Spotpreise, Beschaffungen, Lastprofil"
        )

        if st.button("📊 Excel-Report generieren", type="primary",
                      use_container_width=True, key="export_excel"):
            with st.spinner("Generiere Excel..."):
                excel_data = ExcelExporter.export_full_report(
                    spot, sett, purch, load,
                    bt_result, bt_benchmark, comp_results, grid_results,
                )

            st.download_button(
                label="📥 Excel herunterladen",
                data=excel_data,
                file_name=f"energy_portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        st.divider()

        st.markdown("### 📄 HTML-Report")
        st.caption(
            "Druckbarer Bericht mit KPIs, Monatstabelle, Alerts und Strategievergleich. "
            "Kann als PDF gedruckt werden (Browser → Drucken → Als PDF speichern)."
        )

        report_subtitle = st.text_input(
            "Report-Untertitel",
            value="Backtesting & Procurement Analytics",
            key="report_subtitle",
        )

        if st.button("📄 HTML-Report generieren", type="primary",
                      use_container_width=True, key="export_html"):
            alerts = AlertEngine.analyze_portfolio(
                spot, sett, purch, load, bt_result, bt_benchmark
            )

            with st.spinner("Generiere Report..."):
                html = HTMLReportGenerator.generate(
                    bt_result, bt_benchmark, comp_results, alerts,
                    spot, report_subtitle,
                )

            st.download_button(
                label="📥 HTML-Report herunterladen",
                data=html.encode("utf-8"),
                file_name=f"energy_report_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
                mime="text/html",
                use_container_width=True,
            )

            # Preview
            with st.expander("👁️ Report-Vorschau"):
                st.components.v1.html(html, height=800, scrolling=True)

        st.divider()

        # CSV-Exports
        st.markdown("### 📁 Einzelne Daten-Exports (CSV)")

        csv_cols = st.columns(4)

        with csv_cols[0]:
            if spot is not None:
                csv = spot.to_csv(index=False).encode("utf-8")
                st.download_button("⚡ Spotpreise", csv, "spot_prices.csv", "text/csv")

        with csv_cols[1]:
            if sett is not None:
                csv = sett.to_csv(index=False).encode("utf-8")
                st.download_button("📋 Settlements", csv, "settlements.csv", "text/csv")

        with csv_cols[2]:
            if purch is not None:
                csv = purch.to_csv(index=False).encode("utf-8")
                st.download_button("🛒 Beschaffungen", csv, "purchases.csv", "text/csv")

        with csv_cols[3]:
            if load is not None:
                csv = load.to_csv(index=False).encode("utf-8")
                st.download_button("📊 Lastprofil", csv, "load_profile.csv", "text/csv")

        # Backtest-Ergebnisse
        if bt_result is not None:
            st.divider()
            st.markdown("### 🔄 Backtest-Ergebnis Export")

            bt_csv_cols = st.columns(3)
            with bt_csv_cols[0]:
                csv = bt_result.to_dataframe().to_csv(index=False).encode("utf-8")
                st.download_button("📊 Monatsergebnisse", csv, "backtest_monthly.csv", "text/csv")

            with bt_csv_cols[1]:
                if bt_result.all_tranches:
                    tranche_records = [{
                        "date": t.execution_date, "month": t.delivery_month,
                        "volume": t.volume_mwh, "price": t.price_eur_mwh, "cost": t.cost_eur,
                    } for t in bt_result.all_tranches]
                    csv = pd.DataFrame(tranche_records).to_csv(index=False).encode("utf-8")
                    st.download_button("📋 Tranchen", csv, "tranches.csv", "text/csv")

            with bt_csv_cols[2]:
                if grid_results is not None:
                    csv = grid_results.to_csv(index=False).encode("utf-8")
                    st.download_button("🔍 Grid Search", csv, "grid_search.csv", "text/csv")

        # Szenarien-Export
        scenarios = ScenarioManager.list_all()
        if scenarios:
            st.divider()
            st.markdown("### 💾 Szenarien-Export")
            comp_df = ScenarioManager.get_comparison_df()
            csv = comp_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                f"💾 Alle {len(scenarios)} Szenarien als CSV",
                csv, "scenarios_comparison.csv", "text/csv",
                use_container_width=True,
            )


# ============================================================================
# 36. PAGE: DOKUMENTATION & METHODIK
# ============================================================================

def page_documentation():
    st.header("📚 Dokumentation & Methodik")

    st.markdown("""
    ## System-Architektur

    Das **Energy Portfolio Backtesting System** ist ein institutionelles
    Decision Support System (DSS) für quantitatives Energiebeschaffungs-Management.

    ---

    ### 📊 Module

    | Modul | Funktion | Status |
    |---|---|---|
    | **Daten-Management** | Upload, Validierung, Sample-Generierung | ✅ |
    | **Daten-Explorer** | Interaktive Zeitreihenanalyse, Heatmaps, Statistik | ✅ |
    | **Shaping Engine** | Terminmarkt → stündliche Profile | ✅ |
    | **Backtesting Engine** | Strategiesimulation, Grid Search, Optimierung | ✅ |
    | **Risiko & Performance** | VaR, CVaR, Monte Carlo, MtM, Hedge Effectiveness | ✅ |
    | **Executive Dashboard** | KPIs, Live Calculator, Alerts | ✅ |
    | **Szenario-Manager** | Speichern, Vergleichen, Exportieren | ✅ |

    ---

    ### 🔬 Stochastische Modelle

    #### Ornstein-Uhlenbeck (Mean-Reverting)
    ```
    dX = θ(μ - X)dt + σdW
    ```
    - **θ**: Mean-Reversion-Geschwindigkeit
    - **μ**: Langfristiges Gleichgewichtsniveau
    - **σ**: Volatilität
    - Kalibrierung via OLS auf diskretisierter Form

    #### Mean-Reverting Jump Diffusion
    ```
    dX = θ(μ - X)dt + σdW + J·dN
    ```
    - **dN ~ Poisson(λ·dt)**: Jump-Arrival
    - **J ~ Normal(μ_J, σ_J)**: Jump-Größe
    - Geeignet für Energiemärkte mit Preisspitzen

    ---

    ### 📐 Risikokennzahlen

    | Metrik | Formel / Methode | Anwendung |
    |---|---|---|
    | **Parametrischer VaR** | μ + z_α · σ · √T | Normalverteilungsannahme |
    | **Cornish-Fisher VaR** | CF-Expansion mit Skewness/Kurtosis | Fat-Tail-Korrektur |
    | **Historischer VaR** | Empirisches Quantil | Modellunabhängig |
    | **CVaR / Expected Shortfall** | E[X | X ≤ VaR] | Tail-Risiko |
    | **Hedge Effectiveness (R²)** | Regression: Hedge ~ Underlying | IFRS 9 konform |
    | **Varianzreduktion** | 1 - Var(hedged)/Var(unhedged) | Risikoabsenkung |

    ---

    ### 🔄 Beschaffungsstrategien

    | Strategie | Logik |
    |---|---|
    | **DCA** | Gleichmäßige Tranchen über Einkaufsfenster |
    | **Front-Loaded** | 2/3 der Tranchen in erster Hälfte |
    | **Back-Loaded** | 2/3 der Tranchen in zweiter Hälfte |
    | **Limit-basiert** | Kaufe wenn Preis < X. Perzentil (rollend) |
    | **RSI-getriggert** | Kaufe wenn RSI < Schwelle (überverkauft) |
    | **Saisonal** | Kaufe nur in bestimmten Monaten |

    ---

    ### ⚠️ Look-Ahead Bias Prevention

    Das System verhindert Look-Ahead Bias durch **As-of-Date Lookups**:
    Bei der Simulation eines Kaufs am Tag T wird nur die Forward-Kurve verwendet,
    die am Tag T (oder dem letzten vorangegangenen Handelstag) bekannt war.

    ---

    ### 📊 Datenformate

    #### Spotpreise
    | Spalte | Typ | Beschreibung |
    |---|---|---|
    | `timestamp` | datetime | Zeitstempel (15min, stündlich, täglich) |
    | `price_eur_mwh` | float | Spotpreis in EUR/MWh |

    #### Settlement-Preise
    | Spalte | Typ | Beschreibung |
    |---|---|---|
    | `trade_date` | datetime | Handelstag |
    | `product` | string | Produktname (z.B. M-2024-06) |
    | `product_type` | string | Base, Peak, OffPeak |
    | `delivery_start` | datetime | Lieferbeginn |
    | `delivery_end` | datetime | Lieferende |
    | `settlement_price` | float | Abrechnungspreis EUR/MWh |

    #### Beschaffungen
    | Spalte | Typ | Beschreibung |
    |---|---|---|
    | `purchase_date` | datetime | Kaufdatum |
    | `product` | string | Produktname |
    | `volume_mw` | float | Volumen in MW |
    | `price_eur_mwh` | float | Kaufpreis |
    | `delivery_start` | datetime | Lieferbeginn |
    | `delivery_end` | datetime | Lieferende |
    | `purchase_type` | string | forward / spot |

    #### Lastprofil
    | Spalte | Typ | Beschreibung |
    |---|---|---|
    | `timestamp` | datetime | Zeitstempel |
    | `volume_mwh` | float | Verbrauch in MWh |

    ---

    ### 🔒 Disclaimer

    Dieses Tool dient ausschließlich zu **Analyse- und Bildungszwecken**.
    Vergangene Backtesting-Ergebnisse sind keine Garantie für zukünftige Performance.
    Overfitting auf historische Daten ist ein reales Risiko.
    Konsultieren Sie qualifizierte Berater für reale Beschaffungsentscheidungen.
    """)
  # ============================================================================
# 37. SIDEBAR (FINAL – alle Module)
# ============================================================================

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("### ⚡ Navigation")

        page = st.radio(
            "Modul:",
            [
                "🏛️ Executive Dashboard",
                "📊 Daten-Management",
                "🔬 Daten-Explorer",
                "⚙️ Shaping Engine",
                "🔄 Backtesting Engine",
                "📐 Risiko & Performance",
                "💾 Szenario-Manager",
                "📚 Dokumentation",
            ],
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("### 📋 Datenstatus")

        status = DataStore.status()
        for name, ok in status.items():
            badge = "badge-ok" if ok else "badge-miss"
            icon = "✅" if ok else "❌"
            st.markdown(f'{icon} <span class="{badge}">{name}</span>', unsafe_allow_html=True)

        # Quick Stats
        spot = DataStore.get("spot_prices")
        if spot is not None:
            st.divider()
            st.markdown("### 📈 Quick Stats")
            st.caption(f"**Spot:** {len(spot):,} Punkte")
            st.caption(
                f"{spot['timestamp'].min().strftime('%Y-%m-%d')} → "
                f"{spot['timestamp'].max().strftime('%Y-%m-%d')}"
            )
            st.caption(f"Ø {spot['price_eur_mwh'].mean():.1f} €/MWh")

        sett = DataStore.get("settlement_prices")
        if sett is not None:
            st.caption(f"**Sett:** {len(sett):,} ({sett['product'].nunique()} Prod.)")

        purch = DataStore.get("purchases")
        if purch is not None:
            st.caption(f"**Käufe:** {len(purch)} ({purch['volume_mw'].sum():.0f} MW)")

        load = DataStore.get("load_profile")
        if load is not None:
            st.caption(f"**Last:** {len(load):,} Punkte")

        # Backtest-Status
        bt_result = st.session_state.get("bt_single_result")
        if bt_result is not None:
            st.divider()
            st.markdown("### 🔄 Backtest")
            st.caption(f"**{bt_result.strategy.name[:30]}**")
            st.caption(f"€{bt_result.total_cost:,.0f} | Ø {bt_result.avg_blended_price:.2f}")

        # Szenarien
        ScenarioManager.init()
        n_scenarios = len(ScenarioManager.list_all())
        if n_scenarios > 0:
            st.caption(f"💾 {n_scenarios} Szenarien gespeichert")

        # Reset
        st.divider()
        if st.button("🗑️ Alles zurücksetzen", use_container_width=True):
            DataStore.clear_all()
            ScenarioManager.clear_all()
            keys_to_clear = [
                "bt_single_result", "bt_single_benchmark", "comp_results",
                "opt_grid", "opt_benchmark", "mc_paths", "mc_stats",
                "stress_results", "stress_normal",
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

        st.divider()
        st.caption(f"v{Config.VERSION}")
        st.caption("© 2025 Energy Portfolio System")

    return page


# ============================================================================
# 38. MAIN (FINAL – alle Module)
# ============================================================================

def main():
    st.set_page_config(**Config.PAGE_CONFIG)
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    DataStore.init()
    ScenarioManager.init()

    st.markdown(f'<p class="main-title">{Config.APP_TITLE}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-title">{Config.APP_SUBTITLE}</p>', unsafe_allow_html=True)

    page = render_sidebar()

    if page == "🏛️ Executive Dashboard":
        page_executive_dashboard()
    elif page == "📊 Daten-Management":
        page_data_management()
    elif page == "🔬 Daten-Explorer":
        page_data_explorer()
    elif page == "⚙️ Shaping Engine":
        page_shaping_engine()
    elif page == "🔄 Backtesting Engine":
        page_backtesting()
    elif page == "📐 Risiko & Performance":
        page_risk_analytics()
    elif page == "💾 Szenario-Manager":
        page_scenario_manager()
    elif page == "📚 Dokumentation":
        page_documentation()


if __name__ == "__main__":
    main()
