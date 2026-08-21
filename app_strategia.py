# ============================================================================
#  PUT SPREAD MIBO — operativita' del venerdi' (browser, PC + telefono)
#  Derivata da app_strategiav12.py, ma per la struttura a 2 GAMBE e con il
#  motore riscalato sulla previsione HAR.
#
#  COSA CAMBIA RISPETTO ALLA v12
#  - Solo PUT: short put al 30o percentile FHS + long put 1.000 punti sotto.
#    La call non si vende mai, quindi niente stacca-call, niente banda di size,
#    niente ripiego: la struttura e' una sola.
#  - MOTORE RISCALATO HAR. Il GJR-GARCH sbaglia il livello della volatilita'
#    attesa (sovrastima di circa il 9%, e i percentili sforavano il 25% invece
#    del 30%). La distribuzione FHS viene tenuta com'e' — asimmetria, code,
#    arricchimento EVT — e se ne corregge solo la SCALA con il rapporto fra la
#    previsione HAR e quella GARCH. Sul campione 2016-2026 la calibrazione dei
#    quantili diventa non rifiutabile a tutti i percentili (10,3% al 10o,
#    19,2% al 20o, 27,0% al 30o) mentre col motore base era rifiutata ovunque.
#  - SIZE DAL MARGINE: si aprono tutti i lotti che entrano nel capitale, cioe'
#    parte intera di (capitale / (ampiezza x 2,5)). Con 10.000 EUR e un'ala da
#    1.000 punti sono 4 lotti.
#
#  PARAMETRI FISSATI QUI E NON MODIFICABILI A SCHERMO
#  - percentile della short put: 30
#  - ala: 1.000 punti
#  - soglia markup: 1,15 · pavimento economico: fair x 1,104 + 2 punti
#  - filtro ampiezza (call-put)/spot: attivo al 2,2%
#  Sono i valori scelti dopo il confronto sistematico sul campione 2016-2026;
#  cambiarli settimana per settimana significherebbe scegliere la struttura
#  guardando il mercato, che e' esattamente cio' che i test hanno mostrato non
#  funzionare. Per modificarli si tocca il blocco PARAMETRI qui sotto.
#
#  MODIFICABILI A SCHERMO
#  - capitale (default 10.000 EUR), da cui discende il numero di lotti
#  - prezzo del FTSE MIB, se yfinance e' in ritardo
#  - i due strike operati, se quelli suggeriti non sono quotati
#
#  Avvio locale:  streamlit run app_putspread_live.py
# ============================================================================
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from arch import arch_model
from scipy import stats
from scipy.optimize import brentq
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ----------------------------- PARAMETRI (solo da codice) -------------------
TICKER = "FTSEMIB.MI"
FINESTRA = 756                # sedute per la stima del GARCH
N_SIM = 100_000               # cammini FHS
ORIZZONTE_DEFAULT = 4         # giorni di borsa fino al regolamento
MOLT = 2.5                    # EUR per punto
STEP = 100.0                  # passo della griglia strike

PCT_SHORT = 30.0              # percentile FHS della short put
PCT_CALL_RIF = 75.0           # serve solo a calcolare l'ampiezza del filtro
DIST_ALA = 1000.0             # punti sotto la short put
SOGLIA_OPER = 1.15            # markup minimo
MARGINE_PCT = 0.104           # pavimento economico sul fair value
COSTO_GAMBA = 1.0             # punti per gamba (1 pt = 2,5 EUR)
N_GAMBE = 2
AMPIEZZA_MIN = 0.022          # (strike call 75o - strike put 30o) / spot

USA_EVT_TAIL = True
SOGLIA_EVT = 0.10             # coda sinistra arricchita con GPD sotto il 10o pct
LAG_HAR = (1, 5, 22, 66)      # orizzonti della regressione HAR
MIN_FATT, MAX_FATT = 0.50, 2.00   # limiti del fattore di riscalatura

# Scenario di esecuzione su cui poggiano i risultati storici: sulla gamba
# venduta si assume di prendere il 60% del mezzo-spread, sulla comprata l'80%.
F_RIF, G_RIF = 0.60, 0.80

VRP_MARKUP = 1.25             # solo per la IV di riferimento nella diagnostica


# ----------------------------- motore ---------------------------------------
def gjr(s2, e, om, al, ga, be):
    return om + al * e ** 2 + ga * (e ** 2) * (e < 0) + be * s2


def fhs(var0, pool, om, al, ga, be, H, N, evt, rng):
    s2 = np.full(N, var0)
    cum = np.zeros(N)
    for _ in range(H):
        z = rng.choice(pool, N)
        if evt is not None:
            u, c_, sc_ = evt
            m = z < u
            k = int(m.sum())
            if k:
                z[m] = u - stats.genpareto.rvs(c_, 0, sc_, size=k, random_state=rng)
        e = np.sqrt(s2) * z
        cum += e / 100.0
        s2 = gjr(s2, e, om, al, ga, be)
    return cum


def previsione_har(px, H):
    """Volatilita' annualizzata attesa su H giorni, regressione log-lineare
       sulla varianza realizzata a 1, 5, 22 e 66 giorni. Stimata solo su dati
       disponibili alla data: le coppie usate hanno t+H <= oggi."""
    rv = (px ** 2).clip(lower=1e-8)
    X = pd.concat([rv.rolling(l).mean() for l in LAG_HAR], axis=1).dropna()
    y = rv.rolling(H).mean().shift(-H).reindex(X.index)
    ok = y.notna() & (y > 0) & np.isfinite(np.log(X)).all(axis=1)
    if ok.sum() < 300:
        return np.nan
    Xa = np.column_stack([np.ones(int(ok.sum())), np.log(X[ok].values)])
    ya = np.log(y[ok].values)
    if not (np.isfinite(Xa).all() and np.isfinite(ya).all()):
        return np.nan
    beta = np.linalg.pinv(Xa) @ ya
    xs = np.concatenate([[1.0], np.log(X.iloc[-1].values)])
    return float(np.sqrt(np.exp(float(xs @ beta))) * np.sqrt(252))


def bs(S, K, v, kind):
    if v <= 0:
        return max(0.0, (S - K) if kind == "c" else (K - S))
    d1 = (np.log(S / K) + 0.5 * v * v) / v
    d2 = d1 - v
    return (S * norm.cdf(d1) - K * norm.cdf(d2)) if kind == "c" \
        else (K * norm.cdf(-d2) - S * norm.cdf(-d1))


def iv_imp(prezzo, S, K, kind):
    try:
        return brentq(lambda v: bs(S, K, v, kind) - prezzo, 1e-5, 3.0)
    except Exception:
        return np.nan


@st.cache_data(ttl=1800, show_spinner="Scarico i dati, stimo il GARCH e il HAR...")
def calcola_modello(H):
    df = yf.download(TICKER, period="10y", progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df.dropna()
    px = df["Close"]
    lr = (np.log(px / px.shift(1))).dropna() * 100.0

    rend = lr.iloc[-FINESTRA:].values
    m = arch_model(rend, mean="Zero", vol="GARCH", p=1, o=1, q=1,
                   dist="skewt", rescale=False).fit(disp="off")
    om = m.params["omega"]; al = m.params["alpha[1]"]
    ga = m.params["gamma[1]"]; be = m.params["beta[1]"]
    var0 = float(m.forecast(horizon=1, reindex=False).variance.iloc[-1, 0])
    pool = np.asarray(m.std_resid)
    pool = pool[~np.isnan(pool)]

    evt = None
    if USA_EVT_TAIL and len(pool) > 30:
        u = np.quantile(pool, SOGLIA_EVT)
        ex = u - pool[pool < u]
        if len(ex) >= 10:
            c_, _, sc_ = stats.genpareto.fit(ex, floc=0)
            evt = (u, c_, sc_)

    rng = np.random.default_rng(42)
    cum = fhs(var0, pool, om, al, ga, be, H, N_SIM, evt, rng)
    vol_g = float(cum.std()) * 100 * np.sqrt(252 / H)
    vol_h = previsione_har(lr, H)

    fatt, capped = 1.0, False
    if np.isfinite(vol_h) and vol_g > 0:
        fatt = vol_h / vol_g
        if fatt < MIN_FATT or fatt > MAX_FATT:
            capped = True
            fatt = min(max(fatt, MIN_FATT), MAX_FATT)

    # volatilita' realizzata a 20 giorni, per la doppia lente
    real20 = float(lr.iloc[-20:].std() * np.sqrt(252))

    return dict(px_last=float(px.iloc[-1]), cum=cum * fatt, cum_garch=cum,
                vol_garch=vol_g, vol_har=vol_h, fattore=fatt, capped=capped,
                real20=real20, data=str(df.index[-1].date()))


# ============================================================================
st.set_page_config(page_title="Put spread MIBO — HAR", page_icon="📉", layout="centered")
st.title("📉 Put spread MIBO — motore riscalato HAR")

c1, c2 = st.columns([3, 1])
with c2:
    if st.button("🔄 Aggiorna dati"):
        st.cache_data.clear()
        st.rerun()

H = st.number_input(
    "Orizzonte H (giorni di borsa fino al regolamento)",
    min_value=1, max_value=20, value=ORIZZONTE_DEFAULT, step=1,
    help="4 = settimana normale (ven chiusura -> ven asta di apertura). Alza a 5 se "
         "giri il giovedi'; abbassa se la scadenza e' anticipata da festivi.")
M = calcola_modello(int(H))
with c1:
    st.caption(f"Ultimo dato: {M['data']} · H={int(H)} · struttura fissa "
               f"{int(PCT_SHORT)}/{int(DIST_ALA)} · soglia {SOGLIA_OPER:.2f}")

capitale = st.number_input("Capitale (EUR)", min_value=1000, max_value=1_000_000,
                           value=10000, step=1000,
                           help="Da qui discende il numero di lotti: parte intera di "
                                "capitale diviso il margine per lotto.")
pm = st.number_input("Prezzo FTSE MIB (lascia 0 per usare yfinance: %.0f)" % M["px_last"],
                     min_value=0.0, value=0.0, step=1.0, format="%.0f")
P0 = pm if pm > 0 else M["px_last"]
fonte = "MANUALE" if pm > 0 else "yfinance"

PT = P0 * np.exp(M["cum"])
vol_w = float(M["cum"].std()) * 100
vol_ann = vol_w * np.sqrt(252 / int(H))
sigma_pt = P0 * (vol_ann / 100) * np.sqrt(int(H) / 252)

# ---------------- pannello del motore ----------------
m1, m2, m3 = st.columns(3)
m1.metric("FTSE MIB", f"{P0:,.0f}", fonte)
m2.metric("Vol attesa (riscalata)", f"{vol_ann:.1f}%",
          f"GARCH {M['vol_garch']:.1f}% · HAR {M['vol_har']:.1f}%")
m3.metric("Fattore di riscalatura", f"{M['fattore']:.3f}",
          "cap attivo" if M["capped"] else f"sigma {sigma_pt:,.0f} pt")
if M["capped"]:
    st.warning(f"⚠️ Il fattore HAR/GARCH era fuori dall'intervallo "
               f"[{MIN_FATT:.2f}, {MAX_FATT:.2f}] ed e' stato limitato. Succede in meno "
               f"dell'1% delle settimane: guarda i due numeri di volatilita' prima di operare.")

# ---------------- strike suggeriti ----------------
rnd_giu = lambda x: np.floor(x / STEP) * STEP
rnd_su = lambda x: np.ceil(x / STEP) * STEP
rec_ps = float(rnd_giu(np.percentile(PT, PCT_SHORT)))
rec_pw = rec_ps - DIST_ALA
k_call_rif = float(rnd_su(np.percentile(PT, PCT_CALL_RIF)))   # solo per il filtro


def fair(K):
    """valore atteso del payoff della put allo strike K, sulla distribuzione riscalata"""
    return float(np.mean(np.maximum(K - PT, 0.0)))


st.subheader("Strike suggeriti dal modello")
st.table(pd.DataFrame({
    "gamba": ["SHORT PUT (vendi)", "LONG PUT (compri)"],
    "strike": [int(rec_ps), int(rec_pw)],
    "distanza dallo spot": [f"{100*(rec_ps/P0-1):+.2f}%", f"{100*(rec_pw/P0-1):+.2f}%"],
    "in sigma": [f"{(P0-rec_ps)/sigma_pt:.2f}σ", f"{(P0-rec_pw)/sigma_pt:.2f}σ"],
    "fair value": [round(fair(rec_ps)), round(fair(rec_pw))],
}).set_index("gamba"))

# ---------------- strike operati (editabili, persistenti) ----------------
st.subheader("Strike operati — modificabili")
for k, v in (("k_ps", rec_ps), ("k_pw", rec_pw)):
    if k not in st.session_state:
        st.session_state[k] = float(v)
    else:
        st.session_state[k] = float(st.session_state[k])

b1, b2 = st.columns([1, 3])
with b1:
    if st.button("↺ usa suggeriti"):
        st.session_state.k_ps = rec_ps
        st.session_state.k_pw = rec_pw
        st.rerun()
with b2:
    st.caption("Cambia gli strike se quelli suggeriti non sono quotati. Fair value, "
               "markup e numero di lotti si ricalcolano su questi. I valori restano "
               "se rigiri il modello.")

e1, e2 = st.columns(2)
Kps = e1.number_input("SHORT PUT operata", min_value=0.0, step=STEP, key="k_ps", format="%.0f")
Kpw = e2.number_input("LONG PUT operata", min_value=0.0, step=STEP, key="k_pw", format="%.0f")

ala = Kps - Kpw
if ala <= 0:
    st.error("La long put deve stare SOTTO la short put (strike piu' basso).")
    st.stop()

marg_lotto = ala * MOLT
lotti = int(capitale // marg_lotto)
margine = marg_lotto * lotti

a1, a2, a3 = st.columns(3)
a1.metric("Ala operata", f"{ala:,.0f} pt", f"{ala/sigma_pt:.2f}σ · {100*ala/P0:.2f}% spot")
a2.metric("Lotti", f"{lotti}", f"margine {marg_lotto:,.0f} €/lotto")
a3.metric("Margine impegnato", f"{margine:,.0f} €",
          f"{100*margine/capitale:.0f}% del capitale")

if lotti < 1:
    st.error(f"⛔ Il margine di un solo lotto ({marg_lotto:,.0f} €) supera il capitale "
             f"({capitale:,.0f} €). Riduci l'ala o aumenta il capitale.")
    st.stop()
if abs(ala - DIST_ALA) >= 1:
    st.info(f"ℹ️ Ala operata {ala:,.0f} pt invece dei {DIST_ALA:,.0f} previsti. "
            f"Perdita massima {ala - 0:,.0f} pt meno premio, per lotto; il numero di "
            f"lotti si e' adattato di conseguenza.")

# ---- filtro ampiezza: quanto il modello ha stretto gli strike ----
ampiezza_pct = (k_call_rif - Kps) / P0 if P0 > 0 else float("nan")
troppo_stretta = ampiezza_pct == ampiezza_pct and ampiezza_pct < AMPIEZZA_MIN
_a = "⚠️" if troppo_stretta else "✔️"
st.caption(f"{_a} Ampiezza short: **{100*ampiezza_pct:.2f}%** dello spot "
           f"(fra il 30° percentile put e il 75° percentile call, {k_call_rif:,.0f}) · "
           f"soglia {100*AMPIEZZA_MIN:.1f}%. La call non si vende: questo numero misura "
           f"solo quanto il modello ha stretto, ed e' il filtro che sui dati storici "
           f"migliora ogni configurazione testata.")

fv_p, fv_l = fair(Kps), fair(Kpw)
net_equo = fv_p - fv_l

# ---------------- prezzi dal book ----------------
st.subheader("Prezzi dal book (bid / ask)")


def leg_input(nome, chiave, venduta):
    """f = (mid - eseguito)/(mezzo spread) sulla VENDUTA, 0=mid 1=bid
       g = (eseguito - mid)/(mezzo spread) sulla COMPRATA, 0=mid 1=ask
       Il riferimento non e' un obiettivo ma il metro su cui poggiano i numeri
       storici: sopra, quei risultati per te sono ottimistici."""
    etich = "f (venduta)" if venduta else "g (comprata)"
    rif = F_RIF if venduta else G_RIF
    a, b, c, d, e_ = st.columns([1, 1, 1, 1, 1])
    bid = a.number_input(f"{nome} BID", min_value=0.0, value=0.0, step=1.0,
                         format="%.0f", key=chiave + "b")
    ask = b.number_input(f"{nome} ASK", min_value=0.0, value=0.0, step=1.0,
                         format="%.0f", key=chiave + "a")
    exe = c.number_input(f"{nome} eseguito", min_value=0.0, value=0.0, step=1.0,
                         format="%.0f", key=chiave + "e")
    fg = float("nan")
    if bid > 0 and ask > 0:
        m_ = (bid + ask) / 2
        sp = ask - bid
        hs = sp / 2.0
        if exe > 0:
            scarto = (exe - m_) if venduta else (m_ - exe)
            d.metric("mid", f"{m_:.1f}", f"{scarto:+.1f} vs mid",
                     delta_color="normal" if scarto >= 0 else "inverse")
            if hs > 0:
                fg = -scarto / hs
                e_.metric(etich, f"{fg:.2f}", f"{fg-rif:+.2f} vs {rif:.2f}",
                          delta_color="inverse")
            else:
                e_.metric(etich, "—", "spread 0", delta_color="off")
        else:
            d.metric("mid", f"{m_:.1f}", f"spread {sp:.0f}", delta_color="off")
            e_.metric(etich, "—", f"rif {rif:.2f}", delta_color="off")
    else:
        d.metric("mid", "—")
        e_.metric(etich, "—")
    return bid, ask, exe, fg


pb, pa, pe, fg_sp = leg_input("SHORT PUT", "PUT", True)
lb, la, le, fg_lp = leg_input("LONG PUT", "LONGPUT", False)

mid = lambda b, a: (b + a) / 2 if (b > 0 and a > 0) else 0.0
mp, ml = mid(pb, pa), mid(lb, la)

if not (mp > 0 and ml > 0):
    st.info("Inserisci bid e ask di entrambe le gambe per calcolare markup, edge e "
            "generare la riga del foglio.")
    st.stop()

# ---------------- decisione ----------------
fpr, flr = round(fv_p), round(fv_l)
net_equo_r = fpr - flr
ep = pe if pe > 0 else mp
el = le if le > 0 else ml
net_mid = mp - ml
net_exe = ep - el
eseguito_inserito = (pe > 0 or le > 0)
net_op = net_exe if eseguito_inserito else net_mid
base = "ESEGUITO" if eseguito_inserito else "MID"

if net_equo_r <= 0:
    st.error("⛔ Fair value netto non positivo: la long put vale quanto la short. "
             "Controlla gli strike.")
    st.stop()

markup = net_op / net_equo_r
markup_mid = net_mid / net_equo_r
net_min_pav = net_equo_r * (1 + MARGINE_PCT) + N_GAMBE * COSTO_GAMBA
net_min_oper = net_equo_r * SOGLIA_OPER
soglia_vera = max(net_min_pav, net_min_oper)
punti_residui = net_op - soglia_vera
edge = net_op - net_equo_r - N_GAMBE * COSTO_GAMBA
slippage = net_mid - net_exe

# Il verdetto deve distinguere PERCHE' non si opera, altrimenti il registro non
# puo' calcolare il tasso di settimane operabili: una settimana con edge buono ma
# bloccata dall'ampiezza non e' la stessa cosa di una senza edge.
if net_op < net_min_pav:
    verdetto = "NEG"            # il premio non copre il pavimento economico
elif net_op < net_min_oper:
    verdetto = "SOTT"           # sopra il pavimento ma sotto la soglia markup
elif troppo_stretta:
    verdetto = "STRETTA"        # edge c'e', ma il modello ha stretto troppo gli strike
elif margine > capitale:
    verdetto = "MARGINE"        # edge c'e', ma il capitale non basta
else:
    verdetto = "POS"

st.subheader("Decisione")
d1, d2, d3 = st.columns(3)
d1.metric(f"Markup ({base})", f"{markup:.3f}x", f"al mid {markup_mid:.3f}")
d2.metric("Punti residui vs soglia", f"{punti_residui:+.0f} pt",
          "budget slippage" if punti_residui > 0 else "SOTTO SOGLIA")
d3.metric(f"EDGE netto ({base}−comm)", f"{edge:+.0f} pt",
          f"{edge*MOLT*lotti:+,.0f} € su {lotti} lott{'o' if lotti==1 else 'i'}")

premio_tot = net_op * MOLT * lotti
perdita_max = (ala - net_op) * MOLT * lotti
st.caption(f"Premio incassato **{premio_tot:,.0f} €** · perdita massima "
           f"**{perdita_max:,.0f} €** ({100*perdita_max/capitale:.0f}% del capitale) · "
           f"la long put entra a **{100*(Kpw/P0-1):+.2f}%** dallo spot")
st.caption(f"Net dal book ({base}): {net_op:.0f} pt · pavimento economico "
           f"{net_min_pav:.0f} pt · minimo operativo (markup {SOGLIA_OPER:.2f}) "
           f"{net_min_oper:.0f} pt · slippage speso finora {slippage:+.0f} pt")

# qualita' di esecuzione contro lo scenario di riferimento
_hs_tot = _slip_tot = _hs_rif = 0.0
_n = 0
_det = []
for _nm, _b, _a, _e, _v in (("SHORT PUT", pb, pa, pe, True), ("LONG PUT", lb, la, le, False)):
    if _b > 0 and _a > 0 and _e > 0:
        _hs = (_a - _b) / 2.0
        if _hs <= 0:
            continue
        _sl = ((_b + _a) / 2.0 - _e) if _v else (_e - (_b + _a) / 2.0)
        _slip_tot += _sl
        _hs_tot += _hs
        _hs_rif += _hs * (F_RIF if _v else G_RIF)
        _n += 1
        _det.append(f"{_nm} {'f' if _v else 'g'}={_sl/_hs:.2f}")
if _n:
    _delta = _slip_tot - _hs_rif
    _ok = "✔️" if _delta <= 0 else "⚠️"
    st.caption(f"{_ok} **Esecuzione vs riferimento** ({_n}/{N_GAMBE} gambe): hai pagato "
               f"**{_slip_tot:+.1f} pt** contro i **{_hs_rif:.1f} pt** dello scenario "
               f"f={F_RIF:.2f} / g={G_RIF:.2f} = "
               f"**{'meglio' if _delta<=0 else 'peggio'} di {abs(_delta):.1f} pt** "
               f"({-_delta*MOLT*lotti:+,.0f} €). · " + " · ".join(_det))

guardia = eseguito_inserito and (net_op < soglia_vera)

if verdetto == "NEG":
    st.error(f"⛔ SALTA: net {net_op:.0f} pt sotto il pavimento economico "
             f"({net_min_pav:.0f} pt). Il rischio non e' pagato.")
elif verdetto == "SOTT":
    st.error(f"⛔ SALTA: net {net_op:.0f} pt sopra il pavimento ma sotto il minimo "
             f"operativo {net_min_oper:.0f} pt (markup {SOGLIA_OPER:.2f}).")
elif verdetto == "STRETTA":
    st.error(f"⛔ NON APRIRE — l'edge c'e' ma l'ampiezza degli strike e' "
             f"{100*ampiezza_pct:.2f}%, sotto la soglia del {100*AMPIEZZA_MIN:.1f}%. "
             f"Il modello ha stretto perche' la vol e' bassa: il premio scende mentre "
             f"l'ala resta larga uguale.")
elif verdetto == "MARGINE":
    st.error(f"⛔ NON APRIRE — margine {margine:,.0f} € sopra il capitale "
             f"{capitale:,.0f} €.")
elif guardia:
    st.error(f"⛔ NON APRIRE — coi fill correnti il net e' sceso a {net_op:.0f} pt, "
             f"sotto la soglia di {soglia_vera:.0f}. Chiudi le gambe gia' aperte e "
             f"passa la settimana.")
else:
    st.success(f"✅ OPERA — {lotti} lott{'o' if lotti==1 else 'i'}, margine "
               f"{margine:,.0f} €. Punti residui {punti_residui:+.0f}: puoi concedere "
               f"fino a {punti_residui:.0f} pt lavorando gli ordini (prima l'ala, poi la "
               f"short) e restare sopra soglia. Ordine consigliato: limite al mid, poi "
               f"migliora di un tick finche' i punti residui restano positivi.")

# diagnostica sull'ala
ivw = iv_imp(el, P0, Kpw, "p")
iv_mod = float(M["cum"].std()) * VRP_MARKUP
if ivw == ivw and iv_mod > 0:
    st.caption(f"Ala long put: IV implicita {ivw*100:.2f}% contro {iv_mod*100:.2f}% del "
               f"modello = {ivw/iv_mod:.2f}x — costo {ml:.0f} pt contro fair {flr:.0f} pt")

# ---------------- riga per il foglio ----------------
st.subheader("Riga per il foglio")
oggi = datetime.now().strftime("%d/%m/%Y")
si_opera = (verdetto == "POS") and (not guardia)
n_contr = lotti if si_opera else 0
# Ordine identico alle colonne A..T del foglio Diario del registro, tutte di input.
# Le celle "eseguito" restano VUOTE finche' non le compili davvero: il pattern delle
# celle vuote racconta l'esito (due piene = eseguito, una = saltato per slippage,
# nessuna = non operato).
riga = [oggi,                                   # A  Data apertura
        f"{P0:.0f}",                            # B  Spot
        f"{Kps:.0f}",                           # C  Strike PUT (vendi)
        f"{Kpw:.0f}",                           # D  Strike LONG PUT (compri)
        f"{pb:.0f}", f"{pa:.0f}",               # E  F  Put BID/ASK
        f"{lb:.0f}", f"{la:.0f}",               # G  H  LongPut BID/ASK
        f"{pe:.0f}" if pe > 0 else "",          # I  Eseguito PUT
        f"{le:.0f}" if le > 0 else "",          # J  Eseguito LONG PUT
        f"{vol_ann:.2f}",                       # K  Vol riscalata %
        f"{M['vol_garch']:.2f}",                # L  Vol GARCH %
        f"{M['fattore']:.3f}",                  # M  Fattore HAR
        f"{sigma_pt:.0f}",                      # N  Sigma settimanale (punti)
        f"{fpr:.0f}",                           # O  Equo PUT
        f"{flr:.0f}",                           # P  Equo LONG PUT
        verdetto,                               # Q  Verdetto
        f"{n_contr:d}",                         # R  Lotti
        f"{margine:.0f}" if si_opera else "0",  # S  Margine impegnato
        f"{capitale:.0f}"]                      # T  Capitale
st.code("\t".join(riga), language=None)
if not si_opera:
    st.caption("⚠️ Settimana NON operata: lotti e margine a zero. La riga si registra "
               "comunque, perche' il tasso di settimane operabili e' uno dei fili "
               "d'inciampo e ha bisogno anche delle settimane saltate.")
st.caption("Incolla nella cella **A** della prima riga vuota del foglio *Diario*: sono "
           "20 colonne, **A→T**, tutte di input. **U** (data scadenza) e **V** (settle "
           "dall'asta di apertura) si compilano il venerdi' successivo. Da **W** in poi "
           "e' tutto calcolato e non va toccato.")
