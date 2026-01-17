import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    'SMOOTHING_WINDOW': 3,
    'LOOKBACK_MIN_PERIODS': 4,
    'INVERSION_THRESHOLD': -0.50,
    'REAL_RATE_THRESHOLD': 2.0,
    # CHANGED TO 1: Reacts INSTANTLY to data changes (Like Model A)
    'PERSISTENCE_WEEKS': 1  
}

# MERGED RISK LOGIC (Best of Model B)
INDICATOR_WEIGHTS = {
    'GROWTH': {
        'GDP QoQ': 1.0, 
        'Retail Sales MoM': 0.8, 
        'Ind Prod MoM': 0.7, 
        'NFP MoM': 1.0,
        'Unemployment Rate': -0.9, 
        'ISM Mfg PMI': 0.85, 
        'ISM Services PMI': 0.85,
        'Housing Starts MoM': 0.6, 
        'Durable Goods MoM': 0.7,
        'U. of Mich. Consumer Sentiment': 0.6,
        'High Yield Spread': -0.8,  # Risk Off = Negative Growth
        'Initial Jobless Claims': -0.7
    },
    'INFLATION': {
        'CPI MoM': 1.0, 
        'Core CPI MoM': 1.0, 
        'PPI MoM': 0.8,
        'Core PPI MoM': 0.8,
        'PCE Price Index MoM': 0.9,
        'Core PCE Price Index MoM': 0.9,
        'Avg Hourly Earnings MoM': 0.7,
        'Unit Labor Costs': 0.6
    }
}

# --- FINAL "REAL WORLD" CFD MAPPING ---
ASSET_MAPPING = {
    'Q1': {
        'Theme': 'Goldilocks (US Growth > World)',
        'Bias': 'Long Tech / Neutral Dollar',
        'Long': [
            'NAS100 (Nasdaq) - Best Trade',
            'BTCUSD (Bitcoin)', 
            'US500 (S&P 500)',
            'US10Y (Bond Price - Mild Buy)' 
        ],
        'Short': [
            'V75 (VIX Volatility)', 
            'XAUUSD (Gold - Neutral/Trim)', 
            # REMOVED DXY SHORT - Don't fight the Fed in Q1
        ]
    },
    'Q2': {
        'Theme': 'Reflation (Inflation is Back)',
        'Bias': 'Long Commodities / Short Bonds',
        'Long': [
            'USOIL (WTI Crude)', 
            'XCUUSD (Copper)', 
            'US30 (Dow Jones - Value)', 
            'USDJPY (The "Carry" Trade)'
        ],
        'Short': [
            'US10Y (Sell Bond Price) - #1 Trade', 
            'NAS100 (Tech hates Rates)', 
            'EURUSD (Rates Divergence)'
        ]
    },
    'Q3': {
        'Theme': 'Stagflation (Fear)',
        'Bias': 'Cash is King',
        'Long': [
            'XAUUSD (Gold - Fear Hedge)', 
            'USDOLLAR (DXY)', 
            'USDCHF (Safety)'
        ],
        'Short': [
            'US30 (Dow)', 
            'US2000 (Small Caps)', 
            'GBPUSD', 
            'AUDUSD'
        ]
    },
    'Q4': {
        'Theme': 'Deflation (Recession)',
        'Bias': 'Long Bonds / Short Risk',
        'Long': [
            'US10Y (Buy Bond Price) - #1 Trade', 
            'JPY (Short USDJPY)', 
            'XAUUSD (Gold)'
        ],
        'Short': [
            'USOIL (WTI)', 
            'US500 (S&P 500)', 
            'GBPJPY'
        ]
    }
}

# ==========================================
# 2. CALCULATION ENGINE
# ==========================================
class MacroRegimeEngine:
    def __init__(self, df_input, lookback_weeks):
        self.raw = df_input.copy()
        self.lookback = int(lookback_weeks)
        self.processed = False
        self.regime_df = None

    def process_data(self):
        df = self.raw.copy()
        for c in ['Previous', 'Forecast', 'Actual']:
            df[c] = pd.to_numeric(df[c], errors='coerce')

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        
        # Blending: Actual > Forecast > Previous (Fill Forward)
        df['Value_Used'] = df['Actual'].fillna(df['Forecast'])
        
        self.wide_vals = df.pivot_table(index='Date', columns='Indicator', values='Value_Used', aggfunc='last')
        self.wide_vals = self.wide_vals.resample('W-FRI').last().ffill()

        # Z-Scores
        dynamic_min = min(12, max(CONFIG['LOOKBACK_MIN_PERIODS'], int(self.lookback / 2)))
        rolling_mean = self.wide_vals.rolling(window=self.lookback, min_periods=dynamic_min).mean()
        rolling_std = self.wide_vals.rolling(window=self.lookback, min_periods=dynamic_min).std()

        self.z_scores = (self.wide_vals - rolling_mean) / rolling_std

        # Derivatives
        self.struct_accel = self.z_scores.diff(4).rolling(CONFIG['SMOOTHING_WINDOW']).mean()
        self.tactical_impulse = self.z_scores.diff(1).rolling(CONFIG['SMOOTHING_WINDOW']).mean()
        self.processed = True

    def calculate_regimes(self):
        if not self.processed: self.process_data()
        dates = self.struct_accel.index
        results = []

        for d in dates:
            row_res = {'Date': d}
            
            def get_weighted_score(source_df, category):
                score, denom = 0, 0
                for ind, w in INDICATOR_WEIGHTS[category].items():
                    if ind in source_df.columns:
                        val = source_df.loc[d, ind]
                        if not np.isnan(val):
                            score += val * w
                            denom += abs(w)
                return score / denom if denom > 0 else 0

            # Structural
            row_res['Growth_Struct'] = get_weighted_score(self.struct_accel, 'GROWTH')
            row_res['Inflation_Struct'] = get_weighted_score(self.struct_accel, 'INFLATION')
            row_res['Growth_Tactical'] = get_weighted_score(self.tactical_impulse, 'GROWTH')
            
            # Raw Quad
            g, i = row_res['Growth_Struct'], row_res['Inflation_Struct']
            if g >= 0 and i < 0: row_res['Raw_Quad'] = 'Q1'
            elif g >= 0 and i >= 0: row_res['Raw_Quad'] = 'Q2'
            elif g < 0 and i >= 0: row_res['Raw_Quad'] = 'Q3'
            else: row_res['Raw_Quad'] = 'Q4'
            
            results.append(row_res)
            
        df_res = pd.DataFrame(results).set_index('Date')
        
        # PERSISTENCE LOGIC
        persisted_quads = []
        current_valid_quad = None
        consecutive_count = 0
        last_raw = None
        
        for raw_q in df_res['Raw_Quad']:
            if raw_q == last_raw:
                consecutive_count += 1
            else:
                consecutive_count = 1
                last_raw = raw_q
            
            # If threshold met (or if config is 1), update immediately
            if consecutive_count >= CONFIG['PERSISTENCE_WEEKS']:
                current_valid_quad = raw_q
            
            persisted_quads.append(current_valid_quad)
            
        df_res['Final_Quad'] = persisted_quads
        df_res['Final_Quad'] = df_res['Final_Quad'].bfill() 
        
        self.regime_df = df_res

    def generate_signal(self):
        if self.regime_df.empty: return {}
        latest = self.regime_df.iloc[-1]
        
        # Warnings
        warnings = []
        yc = self.wide_vals.get('Yield Curve 10Y-2Y', pd.Series([0])).iloc[-1]
        rr = self.wide_vals.get('Real Rate', pd.Series([0])).iloc[-1]
        
        if yc < CONFIG['INVERSION_THRESHOLD']: warnings.append(f"DEEP INVERSION ({yc:.2f}): Reduce Equity Beta")
        if rr > CONFIG['REAL_RATE_THRESHOLD']: warnings.append(f"HIGH REAL RATES ({rr:.2f}): Gold Headwind")

        # Impulse
        g_s = latest['Growth_Struct']
        g_t = latest['Growth_Tactical']
        tac_bias = "Add Risk" if np.sign(g_s) == np.sign(g_t) else "Trim/Wait (Divergence)"
        
        # Probabilities (Markov Chain)
        quad_hist = self.regime_df['Final_Quad']
        trans = quad_hist.groupby([quad_hist, quad_hist.shift(-1)]).size().unstack(fill_value=0)
        current_q = latest['Final_Quad']
        
        if current_q in trans.index:
            curr_trans = trans.loc[current_q]
            if curr_trans.sum() > 0:
                probs = (curr_trans / curr_trans.sum()).to_dict()
            else:
                probs = {'Q1':0.25, 'Q2':0.25, 'Q3':0.25, 'Q4':0.25}
        else:
            probs = {'Q1':0.25, 'Q2':0.25, 'Q3':0.25, 'Q4':0.25}

        return {
            "Date": latest.name,
            "Macro_Quad": latest['Final_Quad'],
            "Struct_Regime": f"Growth {latest['Growth_Struct']:.2f} | Inf {latest['Inflation_Struct']:.2f}",
            "Tactical_Impulse": tac_bias,
            "Risk_Warnings": warnings,
            "Probs": probs,
            "Bias": ASSET_MAPPING.get(latest['Final_Quad'], {})
        }

# ==========================================
# 3. DASHBOARD UI
# ==========================================
def get_template_csv():
    return pd.DataFrame({
        'Date': ['2024-01-01'],
        'Indicator': ['GDP QoQ'],
        'Previous': [2.0],
        'Forecast': [2.1],
        'Actual': [2.1]
    })

def main():
    st.set_page_config(page_title="Macro Quant Pro", layout="wide")
    st.markdown("""<style>.stApp{background-color:#0e1117;} .metric-box{background:#1f1f1f;padding:15px;border-radius:8px;border:1px solid #333;}</style>""", unsafe_allow_html=True)

    if 'data_history' not in st.session_state:
        st.session_state.data_history = None

    with st.sidebar:
        st.header("⚙️ Settings")
        lookback_months = st.slider("Lookback (Months)", 1, 60, 12)
        lookback_weeks = int(lookback_months * 4.33)
        lookback_weeks = max(lookback_weeks, 4)
        
        st.markdown("---")
        st.subheader("1. Load Data")
        uploaded_file = st.file_uploader("Upload CSV History", type=['csv'])
        
        if uploaded_file is not None and st.session_state.data_history is None:
            st.session_state.data_history = pd.read_csv(uploaded_file)
            st.success("File Loaded!")
        
        st.markdown("---")
        st.download_button("Template CSV", get_template_csv().to_csv(index=False), "macro_template.csv", "text/csv")

    st.title("⚡ Macro Quant: Live Forecast Engine")

    if st.session_state.data_history is not None:
        st.markdown("### 📝 Live Data Editor")
        edited_df = st.data_editor(st.session_state.data_history, num_rows="dynamic", use_container_width=True, height=300)
        st.session_state.data_history = edited_df
        
        try:
            engine = MacroRegimeEngine(edited_df, lookback_weeks)
            engine.calculate_regimes()
            signal = engine.generate_signal()
            
            if not signal:
                st.warning("Not enough data history.")
                st.stop()
                
            c1, c2, c3, c4 = st.columns(4)
            qc = {'Q1':'#00CC96','Q2':'#EF553B','Q3':'#AB63FA','Q4':'#636EFA'}.get(signal['Macro_Quad'], '#fff')
            
            with c1: st.markdown(f"<div class='metric-box' style='border-left:5px solid {qc}'><h3>{signal['Macro_Quad']}</h3><p>{signal['Struct_Regime']}</p></div>", unsafe_allow_html=True)
            with c2: st.markdown(f"<div class='metric-box'><h3>Impulse</h3><p style='color:{'#00CC96' if 'Add' in signal['Tactical_Impulse'] else 'orange'}'>{signal['Tactical_Impulse']}</p></div>", unsafe_allow_html=True)
            with c3: 
                next_q = max(signal['Probs'], key=signal['Probs'].get) if signal['Probs'] else "N/A"
                val = int(signal['Probs'][next_q]*100) if signal['Probs'] else 0
                st.markdown(f"<div class='metric-box'><h3>Next Prob</h3><p>{val}% &rarr; {next_q}</p></div>", unsafe_allow_html=True)
            with c4: st.markdown(f"<div class='metric-box'><h3>Latest Data</h3><p>{signal['Date'].strftime('%Y-%m-%d')}</p></div>", unsafe_allow_html=True)

            st.markdown("---")
            g1, g2 = st.columns([2,1])
            
            latest = engine.regime_df.index[-1]
            start = latest - timedelta(weeks=lookback_weeks)
            chart_df = engine.regime_df[engine.regime_df.index >= start]

            with g1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['Growth_Struct'], name='Growth', line=dict(color='#00CC96')))
                fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['Inflation_Struct'], name='Inflation', line=dict(color='#EF553B')))
                fig.add_hline(y=0, line_dash='dash', line_color='gray')
                fig.update_layout(title="Structural Acceleration", template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with g2:
                fig_s = px.scatter(chart_df, x='Growth_Struct', y='Inflation_Struct', color='Final_Quad', 
                                   color_discrete_map={'Q1':'#00CC96','Q2':'#EF553B','Q3':'#AB63FA','Q4':'#636EFA'},
                                   title="Regime Compass")
                fig_s.add_vline(x=0); fig_s.add_hline(y=0)
                st.plotly_chart(fig_s, use_container_width=True)

            st.markdown("### 🛑 Trade Execution")
            t1, t2 = st.columns(2)
            if 'Long' in signal['Bias']:
                with t1:
                    st.success(f"**LONG:** {', '.join(signal['Bias']['Long'])}")
                    st.error(f"**SHORT:** {', '.join(signal['Bias']['Short'])}")
            with t2:
                if signal['Risk_Warnings']:
                    for w in signal['Risk_Warnings']: st.warning(f"⚠️ {w}")
                else:
                    st.success("✅ Rates & Policy Filters: Passed")

        except Exception as e:
            st.error(f"Calculation Error: {e}")

    else:
        st.info("Please upload your Macro CSV to begin.")

if __name__ == "__main__":
    main()