# ============================================================================
#  WEB-APP STRATEGIA MIBO - IRON CONDOR a 4 gambe (browser, PC + telefono)
#  Stessa logica dello script: GARCH-FHS, strike, regola stacca-call, edge.
#  Inserisci bid/ask -> calcola mid, slippage, edge -> genera la riga per il foglio.
#  >>> STRIKE EDITABILI: puoi cambiare i 3 strike (es. long put piu' vicina di
#      1500 se non c'e') e fair/edge/markup si ricalcolano su QUELLI. Gli strike
#      operati PERSISTONO quando rigiri il modello (mattina -> sera).
#  >>> QUATTRO GAMBE: short put / long put a -1500 · short call / long call a +1000.
#      La long call rende definita anche la perdita al rialzo, quindi il MARGINE
#      diventa calcolabile: max(ampiezza put, ampiezza call) x 2,5 x lotti.
#      L'app lo mostra e avvisa se supera il capitale impostato.
#  >>> RIPIEGO PUT SPREAD: se la struttura a 4 gambe non supera le soglie, l'app
#      verifica se le supera il solo PUT SPREAD (short put + long put). Spesso e'
#      la CALL prezzata al fair value che schiaccia il markup aggregato mentre la
#      PUT resta ricca. Se il ripiego passa si opera a 2 gambe, SEMPRE a 1 lotto:
#      nella banda 1.15-1.35 il put spread da solo e' la fascia peggiore, quindi
#      la size tripla NON si applica.
#  >>> STACCA-CALL: solo BANDA MARKUP [1.42, 1.68). Il ramo DRAWDOWN e' spento
#      (USA_STACCA_DRAWDOWN=False): con la long call non serve piu'.
#  >>> FILTRO AMPIEZZA: si salta la settimana se (call-put)/spot < 2.2%.
#  >>> VERDETTO: POS4 (4 gambe) / POS2 (call staccata dalla banda markup)
#      / POS2R (ripiego sul put spread) / SOTT / NEG. Il Riepilogo del foglio usa
#      "POS*" per aggregare e ha righe separate per POS3, POS2 e POS2R.
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
TICKER="FTSEMIB.MI"; FINESTRA=756; ORIZZONTE_DEFAULT=4
N_SIM=100_000   # come il backtest; riduci a 50_000 solo se il cloud soffre di RAM
MOLT=2.5; STEP=100
SHORT_PUT_PCT=25.0; SHORT_CALL_PCT=75.0
DIST_ALA=1500.0          # long put:  punti SOTTO la put corta
DIST_ALA_CALL=1000.0     # long call: punti SOPRA la call corta (definisce il margine)
VRP_MARKUP=1.25; COSTO_GAMBA=1.0   # in PUNTI: 1 pt = 2,5€ -> commissione 2,5€/gamba = 1.0
USA_EVT_TAIL=True; SOGLIA_EVT=0.10
MARGINE_PCT=0.104; SOGLIA_OPER=1.15   # pavimento economico e soglia operativa sul markup
VOL_PCT_MEDIA=50.0; VOL_PCT_ALTA=75.0
FIN_MAX_DD=63           # giorni di borsa su cui si misura il drawdown 3 mesi

# --- STACCA-CALL da DRAWDOWN: DISATTIVATO ---------------------------------
# Nasceva quando la call era NUDA: dopo un crollo il rimbalzo la colpiva senza
# limite. Con la long call a +1000 la perdita al rialzo e' gia' definita, e la
# regola e' diventata inerte: nei dati 2020-2026 si e' attivata 12 volte (tutte
# marzo-maggio 2020 e 2022), di cui solo 6 hanno superato il filtro d'ingresso,
# per un effetto netto di -59 EUR su 6 anni e mezzo. Metti True per riattivarla.
USA_STACCA_DRAWDOWN=False
DD_STACCA_CALL=15.0     # soglia usata solo se USA_STACCA_DRAWDOWN=True

# --- STACCA-CALL da BANDA MARKUP: attivo ----------------------------------
# In questa banda la sola parte CALL ha valore atteso negativo (-137 EUR per
# lotto su 45 settimane): il mercato prezza tensione senza crollo e la call
# viene colpita piu' spesso di quanto il modello preveda. Vale +6.170 EUR
# rispetto al non averla (IC90% +304 / +13.654, P=96%).
# La banda deriva dalla mappatura per quantile della (1.50,1.80) validata sulla
# struttura a 3 gambe, NON da una griglia ottimizzata sul P&L. L'intorno e' un
# plateau largo (1.40-1.70 danno tutti 34.900-35.700), quindi la posizione
# esatta conta poco. None disattiva.
MK_BANDA=(1.42, 1.68)

# --- FILTRO AMPIEZZA: non operare quando il modello stringe troppo ---------
# ampiezza = (strike call - strike put) / spot. Sotto questa soglia il premio
# incassato scende ma l'ala resta larga uguale, quindi il rapporto premio/
# rischio peggiora e l'EV per lotto crolla. Migliora tutte e quattro le
# combinazioni testate (4 gambe e put spread, size 1/3 e 3/3) e non peggiora
# mai il drawdown. ATTENZIONE: il guadagno e' concentrato quasi tutto nel 2026,
# quindi il dato NON e' robusto quanto lo stacca-call. 2.2% e' scelto sul
# fianco della curva, non sul massimo (che era 2.41% e cadeva sul 20esimo
# percentile del campione, cioe' un numero letto dagli stessi dati).
# None disattiva.
AMPIEZZA_MIN=0.022
RIPIEGO_PUT_SPREAD=True # se la struttura a 4 gambe non passa, prova il solo put spread
SIZE_SOLO_4GAMBE=True   # il triplo vale SOLO sulla struttura a 4 gambe: su qualunque
                        # put spread (stacca-call o ripiego) la banda 1.15-1.35 ha EV
                        # per lotto ~+15 EUR contro i +246 delle 4 gambe, quindi
                        # amplificarla aggiunge varianza senza valore atteso
BANDA_SIZE=(1.15, 1.35) # banda di markup in cui si triplica la size
LOTTI_BASE=1; LOTTI_BANDA=3
CAPITALE=15_000.0       # usato solo per l'avviso sul margine
VOL_STOP=35.0           # vol GARCH ANNUALIZZATA oltre la quale l'app sconsiglia di
                        # operare. NON e' un filtro di qualita': ad alta vol l'edge
                        # per lotto e' il MIGLIORE (+316 EUR a 22-28 contro +11 sotto
                        # 15). Serve solo da interruttore: sopra ~35 le perdite in
                        # punti diventano abbastanza grandi da mangiare il margine e
                        # bloccare le settimane successive, che sono le piu' ricche.
                        # Tarato su 2 sole settimane (marzo 2020): avviso, non blocco.
PREMIO_RISCHIO_MIN=0.10 # premio incassato / ampiezza dell'ala. Il markup non vede
                        # questa grandezza: nelle settimane calme resta 1.33 ma il
                        # premio scende a 147 pt su 1500 di rischio (9.8%) e l'EV
                        # crolla a +12 EUR per lotto. Indicatore, non regola.

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
def calcola_modello(H):
    np.random.seed(42)
    df=yf.download(TICKER,period="5y",progress=False,auto_adjust=False)
    if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.droplevel(1)
    df=df.dropna()
    px=df["Close"].values
    close=df["Close"].iloc[-FINESTRA:]; rend=(np.log(close/close.shift(1))).dropna().values*100.0
    m=arch_model(rend,mean="Zero",vol="GARCH",p=1,o=1,q=1,dist="skewt",rescale=False).fit(disp="off")
    om=m.params["omega"]; al=m.params["alpha[1]"]; ga=m.params["gamma[1]"]; be=m.params["beta[1]"]
    var0=float(m.forecast(horizon=1,reindex=False).variance.iloc[-1,0])
    pool=np.asarray(m.std_resid); pool=pool[~np.isnan(pool)]
    evt=None
    if USA_EVT_TAIL and len(pool)>30:
        u=np.quantile(pool,SOGLIA_EVT); ex=u-pool[pool<u]
        c_,_,sc_=stats.genpareto.fit(ex,floc=0); evt=(u,c_,sc_)
    cum=fhs(var0,pool,om,al,ga,be,H,N_SIM,evt)
    ratios=np.exp(cum); vol=cum.std()
    # distribuzione storica vol realizzata (per il percentile/regime)
    logret=np.diff(np.log(px))
    roll=pd.Series(logret).rolling(20).std().dropna().values*np.sqrt(H)
    vol_real=float(roll[-1])                                   # vol realizzata corrente (20g, weekly)
    vol_recent=float(np.std(logret[-5:])*np.sqrt(H))   # realizzata ultimi 5g (per shock recente)
    return dict(px_last=float(close.iloc[-1]), ratios=ratios, vol=float(vol),
                roll=roll, vol_real=vol_real, vol_recent=vol_recent,
                px_recent=px[-FIN_MAX_DD:], data=str(df.index[-1].date()))

# ============================================================================
st.set_page_config(page_title="Strategia MIBO", page_icon="📈", layout="centered")

st.title("📈 Strategia MIBO — Iron Condor")

col1,col2=st.columns([3,1])
with col2:
    if st.button("🔄 Aggiorna dati"): st.cache_data.clear(); st.rerun()

H = st.number_input("Orizzonte H (giorni di borsa fino al regolamento)",
                    min_value=1, max_value=20, value=ORIZZONTE_DEFAULT, step=1,
                    help="4 = settimana normale (ven chiusura -> ven asta). Alza a 5 se giri "
                         "il giovedi'/venerdi' mattina o se la settimana ha un giorno in piu'; "
                         "abbassa se la scadenza e' anticipata (es. festivi). Oltre ~10 giorni "
                         "la calibrazione del motore weekly non e' verificata: usa con cautela.")
M=calcola_modello(int(H))
with col1:
    st.caption(f"Ultimo dato: {M['data']} · H={int(H)} · clicca Aggiorna per riscaricare")

# prezzo manuale (ritardo yfinance)
pm=st.number_input("Prezzo FTSE MIB (lascia 0 per usare yfinance: %.0f)" % M["px_last"],
                   min_value=0.0, value=0.0, step=1.0, format="%.0f")
P0 = pm if pm>0 else M["px_last"]
fonte = "MANUALE" if pm>0 else "yfinance"

PT=P0*M["ratios"]; vol=M["vol"]; iv=vol*VRP_MARKUP; vol_w=vol*100
vol_pct=(M["roll"]<vol).mean()*100
regime="BASSA" if vol_pct<VOL_PCT_MEDIA else ("MEDIA" if vol_pct<VOL_PCT_ALTA else "ALTA")
peak=M["px_recent"].max(); dd_pct=(P0/peak-1)*100
stacca_dd = USA_STACCA_DRAWDOWN and (dd_pct <= -DD_STACCA_CALL)
stacca_call = stacca_dd   # la banda markup si valuta dopo i prezzi

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
                "ma e' la zona del rimbalzo violento sulla call: con la long call a +1000 "
                "la perdita al rialzo e' comunque limitata.")
else:
    lente=("🟢","forward e realizzata concordi: lettura del regime robusta.")

# ---------------- strike CONSIGLIATI dal modello ----------------
Kps=np.percentile(PT,SHORT_PUT_PCT); Kpw=Kps-DIST_ALA
Kcall=np.percentile(PT,SHORT_CALL_PCT)
rnd_giu=lambda x: np.floor(x/STEP)*STEP   # put: verso il basso
rnd_su =lambda x: np.ceil(x/STEP)*STEP    # call: verso l'alto
def gamba(K,kind):
    fair=float(np.mean(np.maximum((PT-K) if kind=='c' else (K-PT),0.0))); return fair, bs(P0,K,iv,kind)
# fair ai consigliati (solo per la tabella di riferimento)
Kcw=Kcall+DIST_ALA_CALL
fc_r,bc_r=gamba(Kcall,'c'); fp_r,bp_r=gamba(Kps,'p'); fl_r,bl_r=gamba(Kpw,'p')
flc_r,blc_r=gamba(Kcw,'c')
# strike consigliati arrotondati (default dei campi editabili)
rec_ps  = float(rnd_giu(Kps))
rec_call = float(rnd_su(Kcall))
rec_pw  = rec_ps - DIST_ALA        # in griglia per costruzione
rec_cw  = rec_call + DIST_ALA_CALL # idem

# ---------------- pannello livelli ----------------
c1,c2,c3=st.columns(3)
c1.metric("FTSE MIB", f"{P0:,.0f}", fonte)
c2.metric("Vol attesa", f"{vol_w:.2f}%", f"perc. {vol_pct:.0f}% · {regime}")
c3.metric("Drawdown 3m", f"{dd_pct:+.1f}%")
if stacca_dd:
    st.warning(f"✂️ CALL STACCATA — sensore DRAWDOWN (dd 3m {dd_pct:+.1f}% ≤ -{DD_STACCA_CALL:.0f}%): "
               "struttura a 2 GAMBE = short put + long put. I campi della call vengono "
               "ignorati. Regola validata sul 2020: dopo crolli profondi il rimbalzo "
               "violento colpisce la call nuda, il VRP ricco resta sulle put.")

# ---- doppia lente: forward vs realizzata ----
l1,l2=st.columns(2)
vol_ann = vol_w*np.sqrt(252/H) if H>0 else float("nan")
l1.metric("Vol FORWARD (GARCH)", f"{vol_w:.2f}%",
          f"{vol_ann:.1f}% annua · {vol_pct:.0f}° perc.")
l2.metric("Vol REALIZZATA (20g)", f"{vol_real_w:.2f}%", f"{vol_real_pct:.0f}° perc.")
st.caption(f"{lente[0]} {lente[1]}")

st.subheader("Strike consigliati dal modello")
if stacca_call:
    tab=pd.DataFrame({
        "gamba":["LONG PUT (compri)","SHORT PUT (vendi)"],
        "strike":[int(rnd_giu(Kps)-DIST_ALA),int(rnd_giu(Kps))],
        "equo (VRP0)":[round(fl_r),round(fp_r)],
        "BS@IV":[round(bl_r),round(bp_r)]})
else:
    tab=pd.DataFrame({
        "gamba":["LONG PUT (compri)","SHORT PUT (vendi)",
                 "LONG CALL (compri)","SHORT CALL (vendi)"],
        "strike":[int(rnd_giu(Kps)-DIST_ALA),int(rnd_giu(Kps)),
                  int(rec_cw),int(rnd_su(Kcall))],
        "equo (VRP0)":[round(fl_r),round(fp_r),round(flc_r),round(fc_r)],
        "BS@IV":[round(bl_r),round(bp_r),round(blc_r),round(bc_r)]})
st.table(tab.set_index("gamba"))

# ---------------- STRIKE OPERATI (editabili, persistenti) ----------------
st.subheader("Strike operati — modificabili")
for k, v in [("k_call", rec_call), ("k_ps", rec_ps), ("k_pw", rec_pw), ("k_cw", rec_cw)]:
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
        st.session_state.k_cw   = rec_cw
        st.rerun()
with bcol2:
    st.caption("Cambia gli strike (es. long put piu' vicina se 1500 non c'e'). "
               "Fair/edge/markup si ricalcolano su questi. I valori restano se rigiri il modello.")

e1,e2,e3,e4=st.columns(4)
Kpw_op  =e1.number_input("LONG PUT operata",  min_value=0.0, step=float(STEP), key="k_pw",   format="%.0f")
Kps_op  =e2.number_input("SHORT PUT operata", min_value=0.0, step=float(STEP), key="k_ps",   format="%.0f")
Kcw_op  =e3.number_input("LONG CALL operata", min_value=0.0, step=float(STEP), key="k_cw",   format="%.0f")
Kcall_op=e4.number_input("SHORT CALL operata",min_value=0.0, step=float(STEP), key="k_call", format="%.0f")

ala_op    = Kps_op - Kpw_op
ala_c_op  = Kcw_op - Kcall_op
if ala_op <= 0:
    st.error("La long put deve stare SOTTO la put corta (strike piu' basso).")
elif abs(ala_op-DIST_ALA) >= 1:
    st.info(f"⚠️ Ala long put operata: {ala_op:,.0f} pt (consigliata {DIST_ALA:,.0f}) — "
            f"protezione {'piu STRETTA' if ala_op<DIST_ALA else 'piu LARGA'}: "
            f"perdita massima lato put ≈ {ala_op:,.0f} pt − premio.")
else:
    st.caption(f"Ala long put operata: {ala_op:,.0f} pt")

if not stacca_dd:
    if ala_c_op <= 0:
        st.error("La long call deve stare SOPRA la call corta (strike piu' alto).")
    elif abs(ala_c_op-DIST_ALA_CALL) >= 1:
        st.info(f"⚠️ Ala long call operata: {ala_c_op:,.0f} pt (consigliata {DIST_ALA_CALL:,.0f}) — "
                f"il MARGINE si dimensiona sul lato piu' largo: "
                f"max({ala_op:,.0f}, {ala_c_op:,.0f}) pt.")
    else:
        st.caption(f"Ala long call operata: {ala_c_op:,.0f} pt")

# ---- FILTRO AMPIEZZA: quanto il modello ha stretto gli strike short ----
ampiezza_pct = (Kcall_op - Kps_op)/P0 if (P0>0 and Kcall_op>Kps_op) else float("nan")
troppo_stretta = (AMPIEZZA_MIN is not None and ampiezza_pct==ampiezza_pct
                  and ampiezza_pct < AMPIEZZA_MIN and not stacca_dd)
if ampiezza_pct==ampiezza_pct:
    _a = "⚠️" if troppo_stretta else "✔️"
    _sg = f" · soglia {100*AMPIEZZA_MIN:.1f}%" if AMPIEZZA_MIN is not None else " · filtro disattivato"
    st.caption(f"{_a} Ampiezza short: **{100*ampiezza_pct:.2f}%** dello spot "
               f"({Kcall_op-Kps_op:,.0f} pt fra put e call){_sg}")
if troppo_stretta:
    st.error(f"⛔ SETTIMANA STRETTA — ampiezza {100*ampiezza_pct:.2f}% sotto la soglia del "
             f"{100*AMPIEZZA_MIN:.1f}%. Il modello ha stretto perche' la vol e' bassa, e la "
             f"calibrazione regge (sforamenti 26.7% contro 25% atteso): il problema non e' il "
             f"rischio ma il compenso. Il premio scende mentre l'ala resta larga 1.500 pt, "
             f"quindi il rapporto premio/rischio passa da ~13% a ~10% e l'EV per lotto va a "
             f"zero. Meglio non operare. Il dato e' pero' concentrato nel 2026: metti "
             f"AMPIEZZA_MIN=None per disattivare.")

# fair/BS RICALCOLATI sugli strike operati -> tutto a valle usa questi
f_c,b_c=gamba(Kcall_op,'c'); f_p,b_p=gamba(Kps_op,'p'); f_l,b_l=gamba(Kpw_op,'p')
f_lc,b_lc=gamba(Kcw_op,'c')

# ---------------- inserimento prezzi ----------------
st.subheader("Prezzi dal book (bid / ask)")
def leg_input(nome, chiave=None):
    """nome = etichetta a schermo · chiave = prefisso di session_state.
       Le due cose sono separate cosi' l'ordine di visualizzazione si puo'
       cambiare senza azzerare i valori gia' inseriti.
       La quarta colonna mostra il MID appena bid e ask sono compilati, con lo
       scarto dell'eseguito quando c'e': serve a vedere subito quanto si sta
       lasciando sul tavolo gamba per gamba."""
    k = chiave if chiave is not None else nome
    a,b,c,d=st.columns([1,1,1,1])
    bid=a.number_input(f"{nome} BID",min_value=0.0,value=0.0,step=1.0,format="%.0f",key=k+"b")
    ask=b.number_input(f"{nome} ASK",min_value=0.0,value=0.0,step=1.0,format="%.0f",key=k+"a")
    exe=c.number_input(f"{nome} eseguito (opz)",min_value=0.0,value=0.0,step=1.0,format="%.0f",key=k+"e")
    if bid>0 and ask>0:
        m=(bid+ask)/2; sp=ask-bid
        if exe>0:
            venduta = not nome.startswith("LONG")
            scarto = (exe-m) if venduta else (m-exe)   # positivo = meglio del mid
            d.metric("mid", f"{m:.1f}", f"{scarto:+.1f} vs mid",
                     delta_color="normal" if scarto>=0 else "inverse")
        else:
            d.metric("mid", f"{m:.1f}", f"spread {sp:.0f}", delta_color="off")
    else:
        d.metric("mid", "—")
    return bid,ask,exe
# ordine a schermo: LONG PUT · SHORT PUT · LONG CALL · SHORT CALL
lb,la,le    = leg_input("LONG PUT",   "LONG PUT")
pb,pa,pe    = leg_input("SHORT PUT",  "PUT")
lcb,lca,lce = leg_input("LONG CALL",  "LONG CALL")
cb,ca,ce    = leg_input("SHORT CALL", "CALL")
ncontr=st.number_input("N contratti (0 = usa la size suggerita dalla banda)",min_value=0,value=0,step=1)

def mid(b,a): return (b+a)/2 if (b>0 and a>0) else 0.0
mc,mp,ml,mlc=mid(cb,ca),mid(pb,pa),mid(lb,la),mid(lcb,lca)

# --- SOLO PUT SPREAD -------------------------------------------------------
# Capita di riempire il put spread e di esaurire il budget di slippage prima di
# aprire le due gambe call. In quel caso la soglia va ricalcolata sulla
# struttura che stai davvero aprendo: il pavimento a 2 gambe e' net_equo*1.104
# + 2 punti invece di + 4, e il net_equo non contiene le gambe call.
# Sui dati 2020-2026 il put spread supera comunque la propria soglia nel 92%
# delle settimane in cui passavano le 4 gambe, e rinunciare alla call costa in
# media 28 EUR per lotto (t=0.49, indistinguibile da zero).
put_pronto  = (mp>0 and ml>0)
call_pronta = (mc>0 and mlc>0)
solo_ps_auto = put_pronto and (not call_pronta)
solo_ps = st.checkbox("Valuta SOLO il put spread (call non aperta)",
                      value=solo_ps_auto,
                      help="Si attiva da sola se metti i prezzi di put e long put "
                           "ma non quelli delle due gambe call.")
if solo_ps and solo_ps_auto:
    st.info("↩️ Mancano i prezzi delle gambe call: valuto il **solo put spread** "
            "con la soglia a 2 gambe. Se poi riesci ad aprire anche la call, "
            "inserisci i suoi prezzi e il verdetto si ricalcola da solo.")
# eseguito: se non inserito, uso il mid (assume fill al mid) -> serve in app per slippage serale
ec=ce if ce>0 else mc; ep=pe if pe>0 else mp; el=le if le>0 else ml
elc=lce if lce>0 else mlc

pronto = (put_pronto and ala_op>0
           and (solo_ps or stacca_call or (call_pronta and ala_c_op>0)))
if pronto:
    # equo arrotondati come finiranno nel foglio (per far combaciare i numeri)
    fcr,fpr,flr,flcr=round(f_c),round(f_p),round(f_l),round(f_lc)
    # sensore BANDA: markup 4 gambe misurato sul book (serve la call quotata)
    ne4_ = fcr+fpr-flr-flcr
    mk4_ = (mc+mp-ml-mlc)/ne4_ if (call_pronta and ne4_>0) else float("nan")
    stacca_band = (MK_BANDA is not None and not stacca_dd and mk4_==mk4_
                   and MK_BANDA[0] <= mk4_ < MK_BANDA[1])
    stacca_call = solo_ps or stacca_dd or stacca_band
    if stacca_band:
        st.warning(f"✂️ CALL STACCATA — sensore BANDA MARKUP: markup 4 gambe {mk4_:.3f} "
                   f"∈ [{MK_BANDA[0]:.2f}, {MK_BANDA[1]:.2f}) = stress prezzato senza crollo. "
                   "Si opera il PUT SPREAD; call e long call restano fuori.")
    # ---- RIPIEGO SUL SOLO PUT SPREAD ----------------------------------------
    # Se la struttura a 3 gambe non supera le due soglie, si verifica se le supera
    # il put spread da solo. Motivo: la call viene spesso prezzata al fair value
    # del modello (markup ~1.0) e trascina giu' il markup aggregato anche quando
    # la put e' ricca. In quel caso si vende solo la put, protetta dall'ala.
    ripiego_ps = False
    if RIPIEGO_PUT_SPREAD and not stacca_call and call_pronta and ne4_>0:
        nm4_ = mc+mp-ml-mlc
        passa4 = (nm4_ >= ne4_*(1+MARGINE_PCT)+4*COSTO_GAMBA) and (nm4_ >= ne4_*SOGLIA_OPER)
        ne2_ = fpr-flr
        nm2_ = mp-ml
        passa2 = (ne2_>0 and nm2_ >= ne2_*(1+MARGINE_PCT)+2*COSTO_GAMBA
                  and nm2_ >= ne2_*SOGLIA_OPER)
        if (not passa4) and passa2:
            ripiego_ps = True
            stacca_call = True
            st.warning(f"↩️ RIPIEGO SUL PUT SPREAD — la struttura a 4 gambe non passa "
                       f"(markup 4g {mk4_:.3f}), ma il solo put spread si': markup "
                       f"{nm2_/ne2_:.3f}. Si vendono solo PUT + long put; call e long call "
                       f"restano fuori. Size fissa a 1 LOTTO: nella banda 1.15-1.35 il put "
                       f"spread da solo e' la fascia con il rapporto peggiore, quindi la "
                       f"regola del triplo non si applica.")

    N_GAMBE = 2 if stacca_call else 4
    net_mid=((mc-mlc) if not stacca_call else 0)+mp-ml
    net_exe=((ec-elc) if not stacca_call else 0)+ep-el
    net_equo=((fcr-flcr) if not stacca_call else 0)+fpr-flr   # equo AGLI STRIKE OPERATI
    eseguito_inserito = (ce>0 or pe>0 or le>0 or lce>0)
    net_op = net_exe if eseguito_inserito else net_mid   # edge su ESEGUITO se inserito, altrimenti MID
    base = "ESEGUITO" if eseguito_inserito else "MID"
    edge=net_op-net_equo-N_GAMBE*COSTO_GAMBA    # edge AL NETTO commissioni
    markup=net_op/net_equo if net_equo>0 else float("nan")
    slippage=net_mid-net_exe                    # mid - eseguito (controllo serale)
    # doppia soglia: pavimento economico (1.104 + costi) e soglia OPERATIVA (markup 1.15)
    net_min_pav = net_equo*(1+MARGINE_PCT)+N_GAMBE*COSTO_GAMBA
    net_min_oper = net_equo*SOGLIA_OPER
    cuscino = net_mid - net_min_oper            # budget di slippage vs il mid
    # --- markup AL MID (decide la SIZE, come nel backtest) ---
    markup_mid = net_mid/net_equo if net_equo>0 else float("nan")
    in_banda_size = (markup_mid==markup_mid) and (BANDA_SIZE[0] <= markup_mid < BANDA_SIZE[1])
    if SIZE_SOLO_4GAMBE and N_GAMBE == 2:
        in_banda_size = False                 # nessun triplo sulle strutture a 2 gambe
    lotti_reg = LOTTI_BANDA if in_banda_size else LOTTI_BASE
    # --- MARGINE: con entrambe le ali la perdita massima e' definita ---
    ampiezza = ala_op if stacca_call else max(ala_op, ala_c_op)
    margine  = ampiezza*MOLT*lotti_reg
    # --- markup LIVE (aggiornamento durante l'esecuzione: eseguito dove inserito, mid altrove) ---
    #     net_op e' gia' costruito cosi'; markup_live = net_op/net_equo
    markup_live = markup     # = net_op/net_equo (mid se nessun eseguito, poi si muove)
    punti_residui = net_op - net_min_oper     # quanto manca alla soglia 1.15 coi fill correnti
    guardia_B = eseguito_inserito and (markup_live < SOGLIA_OPER)
    ivw=iv_imp(el,P0,Kpw_op,'p'); skew=ivw/iv if (iv>0 and not np.isnan(ivw)) else float("nan")
    ivc=iv_imp(elc,P0,Kcw_op,'c') if (not stacca_call and elc>0) else float("nan")
    skew_c=ivc/iv if (iv>0 and not np.isnan(ivc)) else float("nan")
    # verdetto allineato al foglio: NEG sotto il pavimento, SOTT tra pavimento e
    # operativa, POS sopra la soglia operativa 1.15
    # POS3 = struttura a 3 gambe · POS2 = put spread (stacca-call o ripiego)
    if net_op < net_min_pav:
        verdetto = "NEG"
    elif net_op < net_min_oper:
        verdetto = "SOTT"
    else:
        verdetto = ("POS2R" if ripiego_ps else "POS2") if N_GAMBE == 2 else "POS4"

    st.subheader("Decisione")
    d1,d2,d3=st.columns(3)
    d1.metric(f"Markup ({base})", f"{markup_live:.3f}x" if markup_live==markup_live else "—",
              f"al mid {markup_mid:.3f}" if markup_mid==markup_mid else "—")
    d2.metric("Punti residui vs 1.15", f"{punti_residui:+.0f} pt",
              "budget slippage" if punti_residui>0 else "SOTTO SOGLIA")
    d3.metric(f"EDGE netto ({base}−comm)", f"{edge:+.0f} pt", f"{edge*MOLT:+,.0f} €")
    _sz = ('3 LOTTI (banda 1.15-1.35)' if in_banda_size
           else ('1 lotto (put spread: niente triplo)' if N_GAMBE == 2 else '1 lotto'))
    st.caption(f"SIZE decisa sul mid: {_sz} "
               f"· markup mid {markup_mid:.3f}  |  markup live {markup_live:.3f}")
    # premio incassato rispetto al rischio massimo: il markup non lo misura
    pr = (net_op/ampiezza) if ampiezza>0 else float("nan")
    if pr==pr:
        _pr = "✔️" if pr >= PREMIO_RISCHIO_MIN else "⚠️"
        st.caption(f"{_pr} Premio/rischio: **{100*pr:.1f}%** ({net_op:.0f} pt su "
                   f"{ampiezza:,.0f} pt di ala). Sotto il {100*PREMIO_RISCHIO_MIN:.0f}% "
                   f"il rischio e' lo stesso ma il premio no: nelle settimane calme il "
                   f"markup resta alto mentre l'EV per lotto crolla.")
    if vol_ann==vol_ann and vol_ann > VOL_STOP:
        st.error(f"🛑 VOL {vol_ann:.1f}% ANNUA, sopra la soglia di {VOL_STOP:.0f}% — "
                 f"valuta di NON operare. Non perche' il premio sia cattivo (ad alta vol "
                 f"e' il migliore), ma perche' una perdita a questa volatilita' puo' "
                 f"consumare il margine e impedirti di operare le settimane successive, "
                 f"che sono le piu' ricche. Soglia tarata su pochissime osservazioni: "
                 f"e' un interruttore di emergenza, non un filtro.")
    _q = "⚠️" if margine > CAPITALE else "✔️"
    st.caption(f"{_q} MARGINE richiesto: **{margine:,.0f} €** "
               f"({ampiezza:,.0f} pt × 2,5 × {lotti_reg} lotti) su capitale {CAPITALE:,.0f} € — "
               f"perdita massima definita su entrambi i lati."
               + ("  **Capitale insufficiente: non aprire.**" if margine > CAPITALE else ""))
    st.caption(f"Net dal book ({base}): {net_op:.0f} pt · pavimento economico: {net_min_pav:.0f} pt "
               f"· minimo OPERATIVO (markup {SOGLIA_OPER:.2f}): {net_min_oper:.0f} pt "
               f"· slippage speso finora: {slippage:+.0f} pt")
    if verdetto=="NEG":
        st.error(f"⛔ SALTA: net {net_op:.0f} pt sotto il pavimento economico "
                 f"({net_min_pav:.0f} pt). Il rischio non e' pagato.")
    elif verdetto=="SOTT":
        st.error(f"⛔ SALTA: net {net_op:.0f} pt sopra il pavimento economico ma sotto il "
                 f"minimo operativo {net_min_oper:.0f} pt (markup {SOGLIA_OPER:.2f}). In questa "
                 f"fascia l'edge teorico non copre l'esecuzione: nel backtest e' in perdita. "
                 f"Niente trade, nemmeno a size ridotta.")
    elif troppo_stretta:
        st.error(f"⛔ NON APRIRE — l'edge c'e' ({verdetto}) ma l'ampiezza degli short e' "
                 f"{100*ampiezza_pct:.2f}%, sotto la soglia. Il premio non compensa un'ala "
                 f"che resta larga uguale.")
    elif margine > CAPITALE:
        st.error(f"⛔ NON APRIRE — margine richiesto {margine:,.0f} € sopra il capitale "
                 f"{CAPITALE:,.0f} €. Riduci la size o salta la settimana: aprire senza "
                 f"copertura di margine espone a chiusure forzate dal broker.")
    elif guardia_B:
        st.error(f"⛔ NON APRIRE — markup live sceso a {markup_live:.3f} < {SOGLIA_OPER:.2f} "
                 f"coi fill correnti ({punti_residui:+.0f} pt sotto soglia). Il trade a questi "
                 f"prezzi non esiste: chiudi le eventuali gambe gia' aperte e passa la settimana. "
                 f"(La size non conta: sotto 1.15 non si opera.)")
    else:
        lotti_txt = ("3 LOTTI (banda qualita' 1.15-1.35)" if in_banda_size
                     else ("1 lotto — PUT SPREAD (la banda vale solo a 4 gambe)"
                           if N_GAMBE == 2 else "1 lotto"))
        st.success(f"✅ OPERA — {lotti_txt}. Punti residui {punti_residui:+.0f} pt: puoi concedere "
                   f"fino a {punti_residui:.0f} pt ancora dal punto attuale lavorando gli ordini "
                   f"(prima l'ala, poi le short) e restare sopra {SOGLIA_OPER:.2f}. "
                   f"Ordine consigliato: limite al mid, poi migliora di un tick finche' i punti residui > 0.")
    if skew==skew:
        st.caption(f"Ala long put: IV implicita {ivw*100:.2f}% vs modello {iv*100:.2f}% = {skew:.2f}x "
                   f"({'equa' if skew<=1.5 else 'cara' if skew<=2.5 else 'molto cara'})")
    if skew_c==skew_c:
        st.caption(f"Ala long call: IV implicita {ivc*100:.2f}% vs modello {iv*100:.2f}% = {skew_c:.2f}x "
                   f"— costo {mlc:.0f} pt contro fair {flcr:.0f} pt")

    # ---------------- riga per il foglio ----------------
    st.subheader("Riga per il foglio Google")
    oggi=datetime.now().strftime("%d/%m/%Y")
    # ordine colonne A..W del Diario (E,O,P,Q sono valori, il resto input)
    # Convenzione registro: MID sempre; ESEGUITO (F/G/H) solo se realmente inserito,
    # altrimenti VUOTO. Il pattern delle celle eseguito racconta l'esito:
    #   3 pieni = trade eseguito · parziali = saltato live per slippage · vuoti = non operato.
    c_ = (lambda x: "" if stacca_call else x)   # call e long call vuote se staccate
    # eseguito per gamba: valore solo se l'utente l'ha messo (ce/pe/le/lce > 0)
    exe_c  = "" if stacca_call else (f"{ce:.0f}"  if ce>0  else "")
    exe_p  = f"{pe:.0f}" if pe>0 else ""
    exe_l  = f"{le:.0f}" if le>0 else ""
    exe_lc = "" if stacca_call else (f"{lce:.0f}" if lce>0 else "")
    n_exe = sum(1 for x in (([exe_c, exe_lc] if not stacca_call else []) + [exe_p, exe_l]) if x != "")
    n_att = (2 if stacca_call else 4)
    esito_fill = "completo" if n_exe==n_att else ("parziale" if n_exe>0 else "non operato")
    # se non si opera (NEG/SOTT/guardia/margine) la size va a ZERO: la riga si registra
    # comunque per lo storico, ma il foglio non deve contarla come trade
    si_opera = ((verdetto.startswith("POS")) and (not guardia_B)
                and (margine <= CAPITALE) and (not troppo_stretta))
    n_contr = (ncontr if ncontr > 0 else lotti_reg) if si_opera else 0
    # BLOCCO UNICO A->Y: nel registro tutte le colonne di input sono contigue,
    # le calcolate cominciano da AB. Z e AA (scadenza, settle) si compilano a mano.
    riga=[oggi, f"{P0:.0f}",
          c_(f"{Kcall_op:.0f}"), f"{Kps_op:.0f}", f"{Kpw_op:.0f}", c_(f"{Kcw_op:.0f}"),
          exe_c, exe_p, exe_l, exe_lc,
          c_(f"{cb:.0f}"), c_(f"{ca:.0f}"), f"{pb:.0f}", f"{pa:.0f}",
          f"{lb:.0f}", f"{la:.0f}", c_(f"{lcb:.0f}"), c_(f"{lca:.0f}"),
          f"{vol_w:.2f}",
          c_(f"{fcr:.0f}"), f"{fpr:.0f}", f"{flr:.0f}", c_(f"{flcr:.0f}"),
          verdetto, f"{n_contr:d}"]
    st.code("\t".join(riga), language=None)
    if not si_opera:
        st.caption("⚠️ Settimana NON operata: N contratti = 0. La riga resta per lo storico, "
                   "ma il foglio non la conta come trade e il P&L resta vuoto.")
    st.caption("Incolla nella cella **A** della prima riga vuota: sono 25 colonne, **A→Y**, "
               "tutte di input. **Z** (data scadenza) e **AA** (settle) li compili a mano a "
               "scadenza. Da **AB** in poi e' tutto calcolato — mid, markup, margine, payoff, "
               "P&L — e non va toccato.")
else:
    st.info("Inserisci almeno bid e ask delle quattro opzioni per calcolare edge e generare la riga.")