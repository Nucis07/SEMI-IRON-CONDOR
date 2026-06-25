# ============================================================================
#  WEB-APP STRATEGIA MIBO - put-protected strangle (browser, PC + telefono)
#  Stessa logica dello script: GARCH-FHS, strike, difesa squeeze, edge.
#  Inserisci bid/ask -> calcola mid, slippage, edge -> genera la riga per il foglio.
#  Avvio locale:  streamlit run app_strategia.py
#  Online (gratis): vedi README_webapp.md
# ============================================================================
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ---------------- parametri (come negli script) ----------------
TICKER="FTSEMIB.MI"; FINESTRA=750; ORIZZONTE=4; N_SIM=50_000
MOLT=2.5; STEP=100
SHORT_PUT_PCT=25.0; SHORT_CALL_PCT=75.0; DIST_ALA=1500.0
VRP_MARKUP=1.25; COSTO_GAMBA=1.0   # in PUNTI: 1 pt = 2,5€ -> commissione 2,5€/gamba = 1.0
USA_EVT_TAIL=True; SOGLIA_EVT=0.10
MARGINE_PCT=0.104; VOL_PCT_MEDIA=50.0; VOL_PCT_ALTA=75.0   # margine dinamico = 10,4% del fair (era SOGLIA_EDGE_PT=20)
SHORT_CALL_SQUEEZE=90.0; VOL_PCT_SQUEEZE=80.0; DD_SQUEEZE_PCT=15.0; FIN_MAX_DD=63

def gjr(s2,e,om,al,ga,be): return om+al*e**2+ga*(e**2)*(e<0)+be*s2
def fhs(s2s,pool,om,al,ga,be,H,N,evt):
    s2=np.full(N,s2s); cum=np.zeros(N)
    for _ in range(H):
        z=np.random.choice(pool,N)
        if evt is not None:
            u,c_,sc_=evt; m=z<u; k=int(m.sum())
            if k: z[m]=u-stats.genpareto.rvs(c_,0,sc_,size=k)
        e=np.sqrt(s2)*z; cum+=e/100.0; s2=gjr(s2,e,om,al,ga,be)
    return cum
def bs(S,K,v,kind):
    if v<=0: return max(0.0,(S-K) if kind=='c' else (K-S))
    d1=(np.log(S/K)+0.5*v*v)/v; d2=d1-v
    return S*norm.cdf(d1)-K*norm.cdf(d2) if kind=='c' else K*norm.cdf(-d2)-S*norm.cdf(-d1)
def iv_imp(prezzo,S,K,kind):
    try: return brentq(lambda v: bs(S,K,v,kind)-prezzo,1e-5,3.0)
    except Exception: return np.nan

@st.cache_data(ttl=1800, show_spinner="Scarico dati e stimo il GARCH...")
def calcola_modello():
    np.random.seed(42)
    df=yf.download(TICKER,period="5y",progress=False,auto_adjust=False)
    if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.droplevel(1)
    df=df.dropna()
    px=df["Close"].values
    close=df["Close"].iloc[-FINESTRA:]; rend=close.pct_change().dropna().values*100.0
    m=arch_model(rend,mean="Constant",vol="GARCH",p=1,o=1,q=1,dist="skewt",rescale=False).fit(disp="off")
    om=m.params["omega"]; al=m.params["alpha[1]"]; ga=m.params["gamma[1]"]; be=m.params["beta[1]"]
    var0=float(m.forecast(horizon=1,reindex=False).variance.iloc[-1,0])
    pool=np.asarray(m.std_resid); pool=pool[~np.isnan(pool)]
    evt=None
    if USA_EVT_TAIL and len(pool)>30:
        u=np.quantile(pool,SOGLIA_EVT); ex=u-pool[pool<u]
        c_,_,sc_=stats.genpareto.fit(ex,floc=0); evt=(u,c_,sc_)
    cum=fhs(var0,pool,om,al,ga,be,ORIZZONTE,N_SIM,evt)
    ratios=np.exp(cum); vol=cum.std()
    # distribuzione storica vol realizzata (per il percentile/regime)
    logret=np.diff(np.log(px))
    roll=pd.Series(logret).rolling(20).std().dropna().values*np.sqrt(ORIZZONTE)
    return dict(px_last=float(close.iloc[-1]), ratios=ratios, vol=float(vol),
                roll=roll, px_recent=px[-FIN_MAX_DD:], data=str(df.index[-1].date()))

# ============================================================================
st.set_page_config(page_title="Strategia MIBO", page_icon="📈", layout="centered")

# ---------------- autenticazione ----------------
PASSWORD = "Overthemoon?"   # <-- cambia questa password

if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    st.title("🔒 Accesso richiesto")
    pwd = st.text_input("Password", type="password")
    if st.button("Entra"):
        if pwd == PASSWORD:
            st.session_state.auth = True
            st.rerun()
        else:
            st.error("Password errata.")
    st.stop()

st.title("📈 Strategia MIBO — put-protected strangle")

col1,col2=st.columns([3,1])
with col2:
    if st.button("🔄 Aggiorna dati"): st.cache_data.clear(); st.rerun()

M=calcola_modello()
with col1:
    st.caption(f"Ultimo dato: {M['data']} · clicca Aggiorna per riscaricare")

# prezzo manuale (ritardo yfinance)
pm=st.number_input("Prezzo FTSE MIB (lascia 0 per usare yfinance: %.0f)" % M["px_last"],
                   min_value=0.0, value=0.0, step=1.0, format="%.0f")
P0 = pm if pm>0 else M["px_last"]
fonte = "MANUALE" if pm>0 else "yfinance"

PT=P0*M["ratios"]; vol=M["vol"]; iv=vol*VRP_MARKUP; vol_w=vol*100
vol_pct=(M["roll"]<vol).mean()*100
regime="BASSA" if vol_pct<VOL_PCT_MEDIA else ("MEDIA" if vol_pct<VOL_PCT_ALTA else "ALTA")
peak=M["px_recent"].max(); dd_pct=(P0/peak-1)*100
squeeze=(vol_pct>VOL_PCT_SQUEEZE) and (dd_pct<=-DD_SQUEEZE_PCT)

Kps=np.percentile(PT,SHORT_PUT_PCT); Kpw=Kps-DIST_ALA
Kcs=np.percentile(PT,SHORT_CALL_PCT); Kc90=np.percentile(PT,SHORT_CALL_SQUEEZE)
Kcall=Kc90 if squeeze else Kcs
rnd=lambda x: round(x/STEP)*STEP
def gamba(K,kind):
    fair=float(np.mean(np.maximum((PT-K) if kind=='c' else (K-PT),0.0))); return fair, bs(P0,K,iv,kind)
f_c,b_c=gamba(Kcall,'c'); f_p,b_p=gamba(Kps,'p'); f_l,b_l=gamba(Kpw,'p')

# ---------------- pannello livelli ----------------
c1,c2,c3=st.columns(3)
c1.metric("FTSE MIB", f"{P0:,.0f}", fonte)
c2.metric("Vol attesa", f"{vol_w:.2f}%", f"perc. {vol_pct:.0f}% · {regime}")
c3.metric("Drawdown 3m", f"{dd_pct:+.1f}%")
if squeeze:
    st.warning(f"🔺 FINESTRA SQUEEZE attiva (vol alta + drawdown >{DD_SQUEEZE_PCT:.0f}%): "
               f"call spostata al {SHORT_CALL_SQUEEZE:.0f}°. Tieni size ridotta.")

st.subheader("Strike da operare")
tab=pd.DataFrame({
    "gamba":["VENDI CALL"+(" (90° squeeze)" if squeeze else ""),"VENDI PUT","COMPRA LONG PUT"],
    "strike":[rnd(Kcall),rnd(Kps),rnd(Kpw)],
    "equo (VRP0)":[round(f_c),round(f_p),round(f_l)],
    "BS@IV":[round(b_c),round(b_p),round(b_l)]})
st.table(tab.set_index("gamba"))

# ---------------- inserimento prezzi ----------------
st.subheader("Prezzi dal book (bid / ask)")
def leg_input(nome):
    a,b,c=st.columns(3)
    bid=a.number_input(f"{nome} BID",min_value=0.0,value=0.0,step=1.0,format="%.0f",key=nome+"b")
    ask=b.number_input(f"{nome} ASK",min_value=0.0,value=0.0,step=1.0,format="%.0f",key=nome+"a")
    exe=c.number_input(f"{nome} eseguito (opz)",min_value=0.0,value=0.0,step=1.0,format="%.0f",key=nome+"e")
    return bid,ask,exe
cb,ca,ce=leg_input("CALL")
pb,pa,pe=leg_input("PUT")
lb,la,le=leg_input("LONG PUT")
ncontr=st.number_input("N contratti",min_value=1,value=1,step=1)

def mid(b,a): return (b+a)/2 if (b>0 and a>0) else 0.0
mc,mp,ml=mid(cb,ca),mid(pb,pa),mid(lb,la)
# eseguito: se non inserito, uso il mid (assume fill al mid)
ec=ce if ce>0 else mc; ep=pe if pe>0 else mp; el=le if le>0 else ml

pronto = (mc>0 and mp>0 and ml>0)
if pronto:
    # equo arrotondati come finiranno nel foglio (per far combaciare i numeri)
    fcr,fpr,flr=round(f_c),round(f_p),round(f_l)
    net_mid=mc+mp-ml          # prezzo di mercato "pulito" (mid)
    net_exe=ec+ep-el          # eseguito (per slippage e P&L)
    net_equo=fcr+fpr-flr      # valore equo (modello)
    edge=net_mid-net_equo-3*COSTO_GAMBA        # edge sul MID, AL NETTO commissioni (3 gambe)
    soglia_pt=MARGINE_PCT*net_equo if net_equo>0 else float("inf")  # margine dinamico (10,4% del fair)
    markup=net_mid/net_equo if net_equo>0 else float("nan")   # -> VRP_MARKUP
    slippage=net_mid-net_exe                    # mid - eseguito -> COSTO_GAMBA
    ivw=iv_imp(el,P0,Kpw,'p'); skew=ivw/iv if (iv>0 and not np.isnan(ivw)) else float("nan")
    verdetto = "NEG" if edge<=0 else ("SOTT" if edge<soglia_pt else "POS")

    st.subheader("Decisione")
    d1,d2,d3=st.columns(3)
    d1.metric("EDGE netto (mid−comm)", f"{edge:+.0f} pt", f"{edge*MOLT:+,.0f} €")
    d2.metric("Markup (mid, lordo comm.)", f"{markup:.3f}x" if markup==markup else "—")
    d3.metric("Slippage (mid-eseg.)", f"{slippage:+.0f} pt")
    if edge<=0:
        st.error("⛔ SALTA: edge ≤ 0, valore atteso negativo.")
    elif edge<soglia_pt:
        st.warning(f"⚠️ VALUTA: edge netto {edge:.0f} pt < soglia {soglia_pt:.0f} pt "
                   f"(comm. incluse). Size minima o salta.")
    else:
        size = "BASSA" if regime=="BASSA" else ("NORMALE" if regime=="MEDIA" else "ALTA")
        st.success(f"✅ OPERA — SIZE {size}  (edge netto {edge:.0f} pt ≥ soglia {soglia_pt:.0f} pt, "
                   f"comm. incluse, vol {regime})")
    if skew==skew:
        st.caption(f"Ala long put: IV implicita {ivw*100:.2f}% vs modello {iv*100:.2f}% = {skew:.2f}x "
                   f"({'equa' if skew<=1.5 else 'cara' if skew<=2.5 else 'molto cara'})")

    # ---------------- riga per il foglio ----------------
    st.subheader("Riga per il foglio Google")
    oggi=datetime.now().strftime("%d/%m/%Y")
    # ordine colonne A..W del Diario (E,O,P,Q sono valori, il resto input)
    riga=[oggi, f"{P0:.0f}", f"{rnd(Kcall):.0f}", f"{rnd(Kps):.0f}", f"{rnd(Kpw):.0f}",
          f"{ec:.0f}", f"{ep:.0f}", f"{el:.0f}",
          f"{cb:.0f}", f"{ca:.0f}", f"{pb:.0f}", f"{pa:.0f}", f"{lb:.0f}", f"{la:.0f}",
          f"{mc:g}", f"{mp:g}", f"{ml:g}",
          f"{vol_w:.2f}", f"{fcr:.0f}", f"{fpr:.0f}", f"{flr:.0f}", verdetto, f"{ncontr:d}"]
    st.code("\t".join(riga), language=None)
    st.caption("Copia la riga e incollala nella cella **A** della prima riga vuota del foglio "
               "(colonne A→W). Le colonne calcolate da X in poi restano con le formule.")
else:
    st.info("Inserisci almeno bid e ask delle tre opzioni per calcolare edge e generare la riga.")
