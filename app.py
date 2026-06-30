import os
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, render_template_string

# Core Machine Learning Framework Dependencies
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.io as pio

app = Flask(__name__)

# Hardcoded File References matching the target Excel structure
EXCEL_FILE = "bankdata.xlsx"

# ==============================================================================
# 1. DATA INGESTION & DATA CLEANING ENGINE (LIGHT CONTEXT)
# ==============================================================================
def parse_and_clean_excel_statement(sheet_name, account_label):
    """Bypasses printed bank statement headers, cleans currency strings and filters tables directly from Excel."""
    excel_path = EXCEL_FILE
    
    if not os.path.exists(excel_path):
        xlsx_files = [f for f in os.listdir('.') if f.endswith('.xlsx')]
        if xlsx_files:
            excel_path = xlsx_files[0]
        else:
            print(f"[CRITICAL ERROR] Target statement file missing from root directory.")
            return pd.DataFrame()
        
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, header=6)
        df.columns = [str(c).strip() for c in df.columns]
        
        def parse_naira_strings(val):
            if pd.isna(val) or str(val).strip() == '--' or str(val).strip() == '' or str(val).strip() == '-':
                return 0.0
            clean_str = str(val).replace('₦', '').replace(',', '').strip()
            try:
                return float(clean_str)
            except:
                return 0.0

        df = df.rename(columns={
            'Debit(₦)': 'Debit', 'Credit(₦)': 'Credit', 
            'Balance After(₦)': 'Balance', 'Balance After': 'Balance'
        })
        
        df['Debit'] = df['Debit'].apply(parse_naira_strings)
        df['Credit'] = df['Credit'].apply(parse_naira_strings)
        df['Balance'] = df['Balance'].apply(parse_naira_strings)
        df['Trans. Date'] = pd.to_datetime(df['Trans. Date'], errors='coerce')
        df['Description'] = df['Description'].fillna('Unknown Transaction').astype(str)
        
        df = df.dropna(subset=['Trans. Date']).sort_values('Trans. Date')
        df['Account_Type'] = account_label
        return df
    except Exception as e:
        print(f"[ENGINE REJECTION] Failed to compute sheet: {sheet_name}. Details: {str(e)}")
        return pd.DataFrame()

# Initialize data engines
df_wallet = parse_and_clean_excel_statement("Wallet Account Transactions", "Wallet Account")
df_savings = parse_and_clean_excel_statement("Savings Account Transactions", "Savings Account")

# Compile unified data ledger matrix safely
unified_ledger = pd.concat([df_wallet, df_savings], ignore_index=True).sort_values('Trans. Date')

# ==============================================================================
# 2. NATURAL LANGUAGE PROCESSING TRANSACTION CATEGORIZER (SUPERVISED ML)
# ==============================================================================
def train_financial_text_classifier():
    """Trains a Random Forest model contextually mapped to OPay statement nomenclatures."""
    corpus = [
        ("Transfer from AUSTIN CHUKWUNYERE OBIOMA", "Inbound Remittances"),
        ("Transfer from HENRY OHWOJERO", "Inbound Remittances"),
        ("Transfer to ENIAME ENTERPRISE", "Peer Outbound Transfers"),
        ("Transfer to MD FROZEN BUSINESSES", "Peer Outbound Transfers"),
        ("Airtime | 7038316123 | MTN", "Telecom & Infrastructure Cost"),
        ("Airtime | 9015027942 | AIR", "Telecom & Infrastructure Cost"),
        ("Data Purchase", "Telecom & Infrastructure Cost"),
        ("Auto-save to OWealth Balance", "Automated Wealth Sweeps"),
        ("Spend & Save Deposit", "Micro-Savings Schemes"),
        ("OWealth Interest Earned", "Passive Capital Dividend"),
        ("OWealth Withdrawal(Transaction Payment)", "Savings Liquidation Drawdown"),
        ("Stamp Duty", "Regulatory Fees / Levies"),
        ("VAT on Transfer Fee", "Regulatory Fees / Levies")
    ]
    
    texts = [x[0] for x in corpus]
    labels = [x[1] for x in corpus]
    
    vec = TfidfVectorizer(ngram_range=(1, 2))
    X_train = vec.fit_transform(texts)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, labels)
    return vec, rf

vectorizer, classifier = train_financial_text_classifier()

if not unified_ledger.empty:
    X_master = vectorizer.transform(unified_ledger['Description'])
    unified_ledger['ML_Inferred_Category'] = classifier.predict(X_master)

# ==============================================================================
# 3. CORE WEB APPLICATION ROUTING CONTROLLERS
# ==============================================================================
@app.route('/')
def overview_dashboard():
    wallet_credit = df_wallet['Credit'].sum() if not df_wallet.empty else 0.0
    wallet_debit = df_wallet['Debit'].sum() if not df_wallet.empty else 0.0
    savings_credit = df_savings['Credit'].sum() if not df_savings.empty else 0.0
    savings_debit = df_savings['Debit'].sum() if not df_savings.empty else 0.0
    
    grand_credits = wallet_credit + savings_credit
    grand_debits = wallet_debit + savings_debit
    portfolio_net = grand_credits - grand_debits
    
    # 3.1 Plotly Graph 1: Timeline (Light mode white theme configuration)
    fig_line = px.line(
        unified_ledger, x='Trans. Date', y='Balance', color='Account_Type',
        template="plotly_white", color_discrete_map={'Wallet Account': '#1f6feb', 'Savings Account': '#238636'}
    )
    fig_line.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
        height=380, margin=dict(t=15, b=15, l=15, r=15)
    )
    chart_timeline = pio.to_html(fig_line, full_html=False, include_plotlyjs='cdn')
    
    # 3.2 Plotly Graph 2: Pie Chart Layout Overlap Fix
    debits_only = unified_ledger[unified_ledger['Debit'] > 0]
    fig_pie = px.pie(
        debits_only, values='Debit', names='ML_Inferred_Category',
        template="plotly_white", hole=0.4
    )
    fig_pie.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
        height=340, margin=dict(t=15, b=80, l=15, r=15),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5)
    )
    chart_pie = pio.to_html(fig_pie, full_html=False, include_plotlyjs=False)
    
    ledger_records = unified_ledger.tail(30).to_dict(orient='records')
    
    return render_template_string(
        HTML_VIEW_LAYOUT,
        current_tab='overview',
        owner_name="PAUL OYIBO",
        wallet_credit=wallet_credit, wallet_debit=wallet_debit,
        savings_credit=savings_credit, savings_debit=savings_debit,
        grand_credits=grand_credits, grand_debits=grand_debits, portfolio_net=portfolio_net,
        chart_timeline=chart_timeline, chart_pie=chart_pie,
        ledger_records=ledger_records
    )

@app.route('/anomalies')
def anomaly_audit_dashboard():
    features = unified_ledger[['Debit', 'Credit', 'Balance']].copy()
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(features)
    
    iso_forest = IsolationForest(contamination=0.02, random_state=42)
    unified_ledger['Anomaly_State'] = iso_forest.fit_predict(scaled_matrix)
    unified_ledger['Status_Cluster'] = np.where(unified_ledger['Anomaly_State'] == -1, "Statistical Outlier", "Normal Operation")
    
    anomalies_only = unified_ledger[unified_ledger['Anomaly_State'] == -1]
    
    fig_scat = px.scatter(
        unified_ledger, x='Trans. Date', y='Debit', color='Status_Cluster',
        color_discrete_map={"Normal Operation": "#1f6feb", "Statistical Outlier": "#da3637"},
        template="plotly_white", title="Spatial Transaction Clustering Scatter Matrix"
    )
    fig_scat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=380, margin=dict(t=15, b=15, l=15, r=15))
    chart_scatter = pio.to_html(fig_scat, full_html=False, include_plotlyjs='cdn')
    
    anomaly_logs = anomalies_only.to_dict(orient='records')
    
    return render_template_string(
        HTML_VIEW_LAYOUT,
        current_tab='anomalies',
        chart_scatter=chart_scatter,
        anomaly_logs=anomaly_logs,
        total_anomalies=len(anomaly_logs)
    )

# ==============================================================================
# 4. TAILWIND INTEGRATED 2026 LIGHT THEME VIEW INFRASTRUCTURE
# ==============================================================================
HTML_VIEW_LAYOUT = """
<!DOCTYPE html>
<html lang="en" class="h-full bg-[#f8fafc]">
<head>
    <meta charset="UTF-8">
    <title>Intelligent Financial Analytics Framework</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="h-full flex text-[#0f172a] font-sans antialiased bg-[#f8fafc]">

    <div class="w-64 bg-white border-r border-[#e2e8f0] flex flex-col justify-between h-full fixed">
        <div>
            <div class="p-6 border-b border-[#e2e8f0]">
                <h2 class="text-lg font-bold tracking-wider text-[#1e40af]">📈 Paul FinAI System</h2>
                <p class="text-[10px] text-[#64748b] uppercase tracking-widest mt-1">Machine Learning Platform</p>
            </div>
            <nav class="mt-6 px-4 space-y-2">
                <a href="/" class="flex items-center px-4 py-3 text-sm rounded-lg {% if current_tab == 'overview' %}bg-[#1e40af] text-white font-semibold{% else %}text-[#64748b] hover:bg-[#f1f5f9] hover:text-[#0f172a]{% endif %} transition">
                    📊 Visual Analysis Engine
                </a>
                <a href="/anomalies" class="flex items-center px-4 py-3 text-sm rounded-lg {% if current_tab == 'anomalies' %}bg-[#1e40af] text-white font-semibold{% else %}text-[#64748b] hover:bg-[#f1f5f9] hover:text-[#0f172a]{% endif %} transition">
                    🚨 Outlier Anomaly Audit
                </a>
            </nav>
        </div>
        <div class="p-4 border-t border-[#e2e8f0] text-center text-xs font-mono text-[#64748b]">
            SYSTEM COMPILING INTEL v2.0
        </div>
    </div>

    <div class="flex-1 pl-64 flex flex-col min-h-screen">
        <main class="flex-1 p-8 overflow-y-auto">
            
            {% if current_tab == 'overview' %}
            <div class="mb-6">
                <h1 class="text-3xl font-extrabold text-[#0f172a]">Statement Analytical Insights & Data Visualizations</h1>
                <p class="text-sm text-[#64748b] mt-1">Processing multi-channel banking tables for portfolio profile: <b>{{ owner_name }}</b></p>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                <div class="bg-white p-5 border border-[#e2e8f0] rounded-xl shadow-sm">
                    <p class="text-xs text-[#64748b] uppercase font-semibold tracking-wider">Cumulative Statement Credits (Inflows)</p>
                    <h3 class="text-2xl font-bold text-[#16a34a] mt-2">₦{{ "{:,.2f}".format(grand_credits) }}</h3>
                    <p class="text-[11px] text-[#475569] mt-1">Wallet: ₦{{ "{:,.2f}".format(wallet_credit) }} | Savings: ₦{{ "{:,.2f}".format(savings_credit) }}</p>
                </div>
                <div class="bg-white p-5 border border-[#e2e8f0] rounded-xl shadow-sm">
                    <p class="text-xs text-[#64748b] uppercase font-semibold tracking-wider">Cumulative Statement Debits (Outflows)</p>
                    <h3 class="text-2xl font-bold text-[#dc2626] mt-2">₦{{ "{:,.2f}".format(grand_debits) }}</h3>
                    <p class="text-[11px] text-[#475569] mt-1">Wallet: ₦{{ "{:,.2f}".format(wallet_debit) }} | Savings: ₦{{ "{:,.2f}".format(savings_debit) }}</p>
                </div>
                <div class="bg-white p-5 border border-[#e2e8f0] rounded-xl shadow-sm">
                    <p class="text-xs text-[#64748b] uppercase font-semibold tracking-wider">Integrated Net Velocity Differential</p>
                    <h3 class="text-2xl font-bold text-[#2563eb] mt-2">₦{{ "{:,.2f}".format(portfolio_net) }}</h3>
                    <p class="text-[11px] text-[#475569] mt-1">Algorithmic Structural Residual Buffer</p>
                </div>
            </div>

            <div class="grid grid-cols-1 xl:grid-cols-3 gap-6 mb-8">
                <div class="xl:col-span-2 bg-white p-6 border border-[#e2e8f0] rounded-xl shadow-sm">
                    <h4 class="text-sm font-bold text-[#0f172a] mb-4 uppercase tracking-wide">Chronological Structural Sub-Account Balance Projections</h4>
                    {{ chart_timeline | safe }}
                </div>
                <div class="bg-white p-6 border border-[#e2e8f0] rounded-xl shadow-sm flex flex-col justify-between">
                    <div>
                        <h4 class="text-sm font-bold text-[#0f172a] mb-4 uppercase tracking-wide">Supervised Random Forest Outflow Mappings</h4>
                        {{ chart_pie | safe }}
                    </div>
                    <div class="border-t border-[#e2e8f0] pt-4 mt-2">
                        <p class="text-xs text-[#475569] leading-relaxed">
                            <b>Model Definition:</b> This text breakdown represents feature label metrics output by the predictive Random Forest multi-class model after reading structural string transaction patterns across execution indices.
                        </p>
                    </div>
                </div>
            </div>

            <div class="bg-white border border-[#e2e8f0] rounded-xl p-6 shadow-sm">
                <h4 class="text-sm font-bold text-[#0f172a] mb-4 uppercase tracking-wide">Sanitized Multi-Sheet Master Ledger Records</h4>
                <div class="overflow-x-auto">
                    <table class="w-full text-left border-collapse">
                        <thead>
                            <tr class="border-b border-[#e2e8f0] text-xs text-[#64748b] uppercase font-bold bg-[#f1f5f9]">
                                <th class="py-3 px-4">Transaction Date</th>
                                <th class="py-3 px-4">Account Origin</th>
                                <th class="py-3 px-4">Descriptor Label</th>
                                <th class="py-3 px-4">ML Classified Category</th>
                                <th class="py-3 px-4 text-right">Debit (₦)</th>
                                <th class="py-3 px-4 text-right">Credit (₦)</th>
                                <th class="py-3 px-4 text-right">Equilibrium (₦)</th>
                            </tr>
                        </thead>
                        <tbody class="text-xs divide-y divide-[#e2e8f0]">
                            {% for record in ledger_records %}
                            <tr class="hover:bg-[#f8fafc] transition">
                                <td class="py-3 px-4 text-[#475569]">{{ record['Trans. Date'] }}</td>
                                <td class="py-3 px-4 font-semibold {% if record['Account_Type'] == 'Wallet Account' %}text-[#2563eb]{% else %}text-[#16a34a]{% endif %}">{{ record['Account_Type'] }}</td>
                                <td class="py-3 px-4 font-mono max-w-xs truncate text-[#334155]">{{ record['Description'] }}</td>
                                <td class="py-3 px-4 text-[#1e40af] font-medium italic">{{ record['ML_Inferred_Category'] }}</td>
                                <td class="py-3 px-4 text-right text-[#dc2626] font-medium">{% if record['Debit'] > 0 %}₦{{ "{:,.2f}".format(record['Debit']) }}{% else %}--{% endif %}</td>
                                <td class="py-3 px-4 text-right text-[#16a34a] font-medium">{% if record['Credit'] > 0 %}₦{{ "{:,.2f}".format(record['Credit']) }}{% else %}--{% endif %}</td>
                                <td class="py-3 px-4 text-right text-[#0f172a] font-bold">₦{{ "{:,.2f}".format(record['Balance']) }}</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>

            {% elif current_tab == 'anomalies' %}
            <div class="mb-6">
                <h1 class="text-3xl font-extrabold text-[#0f172a]">Unsupervised Isolation Forest Outlier Parsing Matrix</h1>
                <p class="text-sm text-[#64748b] mt-1">Tracking transaction variances using multi-feature cluster maps.</p>
            </div>

            <div class="bg-white p-6 border border-[#e2e8f0] rounded-xl mb-6 shadow-sm">
                <h4 class="text-sm font-bold text-[#0f172a] mb-4 uppercase tracking-wide">Clustering Boundary Anomalous Outlier Dispersion Plot</h4>
                {{ chart_scatter | safe }}
            </div>

            <div class="bg-white border border-[#e2e8f0] rounded-xl p-6 shadow-sm">
                <div class="flex justify-between items-center mb-4">
                    <h4 class="text-md font-bold text-[#0f172a]">System Isolated Threat Vector Anomalies ({{ total_anomalies }} incidents flagged)</h4>
                    <span class="bg-[#fee2e2] text-[#991b1b] border border-[#fca5a5] px-3 py-1 rounded-full text-xs font-mono font-bold">Contamination Limit: 2%</span>
                </div>
                <div class="overflow-x-auto">
                    <table class="w-full text-left border-collapse">
                        <thead>
                            <tr class="border-b border-[#e2e8f0] text-xs text-[#64748b] uppercase font-bold bg-[#f1f5f9]">
                                <th class="py-3 px-4">Execution Timestamp</th>
                                <th class="py-3 px-4">Account Node</th>
                                <th class="py-3 px-4">Description Core</th>
                                <th class="py-3 px-4 text-right">Debit Size (₦)</th>
                                <th class="py-3 px-4 text-right">Credit Size (₦)</th>
                                <th class="py-3 px-4 text-right">Residual Balance Buffer (₦)</th>
                            </tr>
                        </thead>
                        <tbody class="text-xs divide-y divide-[#e2e8f0]">
                            {% for log in anomaly_logs %}
                            <tr class="bg-[#fef2f2] hover:bg-[#fee2e2] transition">
                                <td class="py-3 px-4 text-[#475569]">{{ log['Trans. Date'] }}</td>
                                <td class="py-3 px-4 font-bold text-[#b91c1c]">{{ log['Account_Type'] }}</td>
                                <td class="py-3 px-4 font-mono text-[#334155]">{{ log['Description'] }}</td>
                                <td class="py-3 px-4 text-right text-[#b91c1c] font-semibold">₦{{ "{:,.2f}".format(log['Debit']) }}</td>
                                <td class="py-3 px-4 text-right text-[#16a34a]">₦{{ "{:,.2f}".format(log['Credit']) }}</td>
                                <td class="py-3 px-4 text-right font-bold text-[#0f172a]">₦{{ "{:,.2f}".format(log['Balance']) }}</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
            {% endif %}

        </main>
    </div>

</body>
</html>
"""

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)