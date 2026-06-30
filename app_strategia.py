# ============================================================================
#  WEB-APP STRATEGIA MIBO - put-protected strangle (browser, PC + telefono)
#  Stessa logica dello script: GARCH-FHS, strike, difesa squeeze, edge.
#  Inserisci bid/ask -> calcola mid, slippage, edge -> genera la riga per il foglio.
#  >>> STRIKE EDITABILI: puoi cambiare i 3 strike (es. long put piu' vicina di
#      1500 se non c'e') e fair/edge/markup si ricalcolano su QUELLI. Gli strike
#      operati PERSISTONO quando rigiri il modello (mattina -> sera).
#  >>> RIGA FOGLIO: come eseguito si copia il MID (niente fill manuale nel foglio);
#      i campi "eseguito" restano nell'app per il controllo serale e lo slippage.
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
    vol_real=float(roll[-1])                                   # vol realizzata corrente (20g, weekly)
    vol_recent=float(np.std(logret[-5:])*np.sqrt(ORIZZONTE))   # realizzata ultimi 5g (per shock recente)
    return dict(px_last=float(close.iloc[-1]), ratios=ratios, vol=float(vol),
                roll=roll, vol_real=vol_real, vol_recent=vol_recent,
                px_recent=px[-FIN_MAX_DD:], data=str(df.index[-1].date()))

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

# ---------------- DOPPIA LENTE: forward (GARCH) vs realizzata ----------------
vol_real=M["vol_real"]; vol_real_pct=(M["roll"]<vol_real).mean()*100
vol_real_w=vol_real*100
shock_recente=M["vol_recent"]>1.30*vol_real          # movimento brusco negli ultimi 5g
gap=vol_pct-vol_real_pct                              # forward - realizzata (percentili)
if gap>15 and shock_recente:
    lente=("🔴","GARCH reagisce a uno shock recente: vol probabilmente resta alta. "
                "Size ridotta, occhio alla difesa call.")
elif gap>15:
    lente=("⚪","forward sopra la realizzata ma nessun movimento recente: scarto di "
                "calibrazione tra finestre, NON un segnale. Ignora.")
elif gap<-15:
    lente=("🟠","vol in normalizzazione (realizzata ancora alta): finestra di premio, "
                "ma e' la zona dello squeeze sulla call.")
else:
    lente=("🟢","forward e realizzata concordi: lettura del regime robusta.")

# ---------------- strike CONSIGLIATI dal modello ----------------
Kps=np.percentile(PT,SHORT_PUT_PCT); Kpw=Kps-DIST_ALA
Kcs=np.percentile(PT,SHORT_CALL_PCT); Kc90=np.percentile(PT,SHORT_CALL_SQUEEZE)
Kcall=Kc90 if squeeze else Kcs
rnd=lambda x: round(x/STEP)*STEP
def gamba(K,kind):
    fair=float(np.mean(np.maximum((PT-K) if kind=='c' else (K-PT),0.0))); return fair, bs(P0,K,iv,kind)
# fair ai consigliati (solo per la tabella di riferimento)
fc_r,bc_r=gamba(Kcall,'c'); fp_r,bp_r=gamba(Kps,'p'); fl_r,bl_r=gamba(Kpw,'p')
# strike consigliati arrotondati (default dei campi editabili)
rec_call, rec_ps, rec_pw = float(rnd(Kcall)), float(rnd(Kps)), float(rnd(Kpw))

# ---------------- pannello livelli ----------------
c1,c2,c3=st.columns(3)
c1.metric("FTSE MIB", f"{P0:,.0f}", fonte)
c2.metric("Vol attesa", f"{vol_w:.2f}%", f"perc. {vol_pct:.0f}% · {regime}")
c3.metric("Drawdown 3m", f"{dd_pct:+.1f}%")
if squeeze:
    st.warning(f"🔺 FINESTRA SQUEEZE attiva (vol alta + drawdown >{DD_SQUEEZE_PCT:.0f}%): "
               f"call spostata al {SHORT_CALL_SQUEEZE:.0f}°. Tieni size ridotta.")

# ---- doppia lente: forward vs realizzata ----
l1,l2=st.columns(2)
l1.metric("Vol FORWARD (GARCH)", f"{vol_w:.2f}%", f"{vol_pct:.0f}° perc.")
l2.metric("Vol REALIZZATA (20g)", f"{vol_real_w:.2f}%", f"{vol_real_pct:.0f}° perc.")
st.caption(f"{lente[0]} {lente[1]}")

st.subheader("Strike consigliati dal modello")
tab=pd.DataFrame({
    "gamba":["VENDI CALL"+(" (90° squeeze)" if squeeze else ""),"VENDI PUT","COMPRA LONG PUT"],
    "strike":[rnd(Kcall),rnd(Kps),rnd(Kpw)],
    "equo (VRP0)":[round(fc_r),round(fp_r),round(fl_r)],
    "BS@IV":[round(bc_r),round(bp_r),round(bl_r)]})
st.table(tab.set_index("gamba"))

# ---------------- STRIKE OPERATI (editabili, persistenti) ----------------
st.subheader("Strike operati — modificabili")
for k, v in [("k_call", rec_call), ("k_ps", rec_ps), ("k_pw", rec_pw)]:
    if k not in st.session_state:
        st.session_state[k] = float(v)
    else:
        st.session_state[k] = float(st.session_state[k])   # evita value di tipo str

bcol1, bcol2 = st.columns([1,3])
with bcol1:
    if st.button("↺ usa consigliati"):
        st.session_state.k_call = rec_call
        st.session_state.k_ps   = rec_ps
        st.session_state.k_pw   = rec_pw
        st.rerun()
with bcol2:
    st.caption("Cambia gli strike (es. long put piu' vicina se 1500 non c'e'). "
               "Fair/edge/markup si ricalcolano su questi. I valori restano se rigiri il modello.")

e1,e2,e3=st.columns(3)
Kcall_op=e1.number_input("CALL operata",     min_value=0.0, step=float(STEP), key="k_call", format="%.0f")
Kps_op  =e2.number_input("PUT operata",      min_value=0.0, step=float(STEP), key="k_ps",   format="%.0f")
Kpw_op  =e3.number_input("LONG PUT operata", min_value=0.0, step=float(STEP), key="k_pw",   format="%.0f")

ala_op = Kps_op - Kpw_op
if ala_op <= 0:
    st.error("La long put deve stare SOTTO la put corta (strike piu' basso).")
elif abs(ala_op-DIST_ALA) >= 1:
    st.info(f"⚠️ Ala long put operata: {ala_op:,.0f} pt (consigliata {DIST_ALA:,.0f}) — "
            f"protezione {'piu STRETTA' if ala_op<DIST_ALA else 'piu LARGA'}: "
            f"perdita massima lato put ≈ {ala_op:,.0f} pt − premio.")
else:
    st.caption(f"Ala long put operata: {ala_op:,.0f} pt")

# fair/BS RICALCOLATI sugli strike operati -> tutto a valle usa questi
f_c,b_c=gamba(Kcall_op,'c'); f_p,b_p=gamba(Kps_op,'p'); f_l,b_l=gamba(Kpw_op,'p')

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
# eseguito: se non inserito, uso il mid (assume fill al mid) -> serve in app per slippage serale
ec=ce if ce>0 else mc; ep=pe if pe>0 else mp; el=le if le>0 else ml

pronto = (mc>0 and mp>0 and ml>0 and ala_op>0)
if pronto:
    # equo arrotondati come finiranno nel foglio (per far combaciare i numeri)
    fcr,fpr,flr=round(f_c),round(f_p),round(f_l)
    net_mid=mc+mp-ml          # prezzo di mercato "pulito" (mid)
    net_exe=ec+ep-el          # eseguito (per-gamba: eseguito o mid di fallback)
    net_equo=fcr+fpr-flr      # valore equo (modello) AGLI STRIKE OPERATI
    eseguito_inserito = (ce>0 or pe>0 or le>0)
    net_op = net_exe if eseguito_inserito else net_mid   # edge su ESEGUITO se inserito, altrimenti MID
    base = "ESEGUITO" if eseguito_inserito else "MID"
    edge=net_op-net_equo-3*COSTO_GAMBA          # edge AL NETTO commissioni (3 gambe)
    soglia_pt=MARGINE_PCT*net_equo if net_equo>0 else float("inf")  # margine dinamico (10,4% del fair)
    markup=net_op/net_equo if net_equo>0 else float("nan")
    slippage=net_mid-net_exe                    # mid - eseguito (controllo serale)
    ivw=iv_imp(el,P0,Kpw_op,'p'); skew=ivw/iv if (iv>0 and not np.isnan(ivw)) else float("nan")
    verdetto = "NEG" if edge<=0 else ("SOTT" if edge<soglia_pt else "POS")

    st.subheader("Decisione")
    d1,d2,d3=st.columns(3)
    d1.metric(f"EDGE netto ({base}−comm)", f"{edge:+.0f} pt", f"{edge*MOLT:+,.0f} €")
    d2.metric(f"Markup ({base}, lordo comm.)", f"{markup:.3f}x" if markup==markup else "—")
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
    # NB: strike = OPERATI (editabili); eseguito (F/G/H) = MID (slippage 0 nel foglio)
    riga=[oggi, f"{P0:.0f}", f"{Kcall_op:.0f}", f"{Kps_op:.0f}", f"{Kpw_op:.0f}",
          f"{mc:.0f}", f"{mp:.0f}", f"{ml:.0f}",
          f"{cb:.0f}", f"{ca:.0f}", f"{pb:.0f}", f"{pa:.0f}", f"{lb:.0f}", f"{la:.0f}",
          f"{mc:.0f}", f"{mp:.0f}", f"{ml:.0f}",
          f"{vol_w:.2f}", f"{fcr:.0f}", f"{fpr:.0f}", f"{flr:.0f}", verdetto, f"{ncontr:d}"]
    st.code("\t".join(riga), language=None)
    st.caption("Copia la riga e incollala nella cella **A** della prima riga vuota del foglio "
               "(colonne A→W). Eseguito = MID. Le colonne calcolate da X in poi restano con le formule.")
else:
    st.info("Inserisci almeno bid e ask delle tre opzioni per calcolare edge e generare la riga.")
