# ============================================================================
#  PUT SPREAD MIBO — operativita' del venerdi' (browser, PC + telefono)
#  v6 — agosto 2026. Derivata da app_strategiav12.py, ma per la struttura a
#  2 GAMBE e con il motore riscalato sulla previsione HAR.
#
#  AVVIO:  python -m streamlit run app_putspread_live6.py
#  (il prefisso "python -m" evita lo shim streamlit.exe, che i criteri di
#   controllo applicazioni di Windows possono bloccare perche' non firmato)
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
#    VERIFICATO FUORI CAMPIONE sul 2014-2015, mai usati per tarare nulla: 25,5%
#    al 30o, nessuna deviazione significativa a nessun percentile, e nel blocco
#    piu' volatile dell'intero campione. Il motore base senza riscalatura sta a
#    0,72-0,84 del nominale sul 2016-2026 ma a 0,83-1,10 sul 2014-2015: e'
#    tarato bene solo quando la volatilita' e' alta, ed e' esattamente cio' che
#    la riscalatura HAR corregge.
#    Lo scarto fra il 30% nominale e il ~27% osservato NON e' scalibratura: lo
#    strike viene arrotondato per difetto alla griglia da 100 punti, quindi
#    finisce in media 50 punti (circa 0,1 sigma) piu' lontano di dove il modello
#    lo indica. Arrotondando per eccesso lo sforamento sale al 33,9%, al piu'
#    vicino al 30,2%. E' conservativita' deliberata e misurata, non errore.
#  - SIZE DAL MARGINE CON TETTO: si aprono tutti i lotti che entrano nel
#    capitale, cioe' parte intera di (capitale / (ampiezza x 2,5)), MA mai piu'
#    di quanti ne darebbe l'ala NOMINALE da 1.000 punti. Con 10.000 EUR sono 4
#    lotti, con 20.000 sono 8. Il tetto morde solo quando la catena non quota la
#    long a 1.000 punti e l'ala ripiega piu' corta: li' il margine per lotto
#    scende e la size salirebbe (a 400 punti sarebbero 10 lotti su 10.000 EUR)
#    senza che il rischio diminuisca. Sul campione 2016-2026 il tetto riduce del
#    10% la deviazione standard settimanale e del 20% il capitale complessivamente
#    versato, a P&L quasi invariato.
#
#  PARAMETRI FISSATI QUI E NON MODIFICABILI A SCHERMO
#  - percentile della short put: 30
#  - ala: 1.000 punti
#  - pavimento economico: fair x 1,10 + 2 punti  (UNICO filtro d'ingresso)
#  Percentile e ala vengono dal confronto sistematico sul campione 2016-2026;
#  cambiarli settimana per settimana significherebbe scegliere la struttura
#  guardando il mercato, che e' esattamente cio' che i test hanno mostrato non
#  funzionare. Per modificarli si tocca il blocco PARAMETRI qui sotto.
#
#  Il 10% del pavimento e' invece un cuscinetto sull'errore di stima scelto PER
#  PRINCIPIO, non il massimo di una curva. Sul campione 2014-2026 nessun valore
#  fra 0,00 e 0,20 e' distinguibile dagli altri: il rendimento oscilla di quindici
#  punti senza schema, il leave-one-year-out perde contro entrambi gli estremi e
#  il drawdown non e' monotono. L'apparente ottimo a 0,16 nasce da TRE venerdi'
#  (2020-06-05, 2022-02-18, 2026-06-19): tolti quelli, la curva diventa monotona
#  decrescente e il massimo torna a 0,10. NON ritararlo.
#
#  Nota su f e g: stanno DENTRO il numeratore del pavimento, quindi la sua
#  selettivita' dipende dall'esecuzione. La barriera effettiva a f=0,60/g=0,80 e'
#  circa il 26% sopra il fair value: 10 punti di cuscinetto e 16 di bid-ask. Il
#  pavimento si autoregola — se lo spread si comprime opera piu' settimane da
#  solo — ed e' anche, di fatto, un filtro sulla liquidita': le settimane che
#  esclude hanno spread relativo mediano 0,219 contro 0,159 delle ammesse.
#
#  FILTRO AMPIEZZA — RIMOSSO (agosto 2026). Fino a questa versione l'ampiezza
#  (call 75o - short put) / spot bloccava l'operativita' sotto il 2,2%, con il
#  verdetto STRETTA. E' stato tolto per due ragioni misurate.
#
#  1. Non era lo stesso filtro del backtest. Qui l'ampiezza si calcola sulla
#     distribuzione RISCALATA e sul percentile 30 (lo strike che vendi davvero);
#     nel backtest si leggeva da strikes_settimanali_v2.csv, cioe' dal motore
#     ORIGINALE e sul percentile 25. Stessa soglia, due grandezze diverse: la
#     versione del backtest bloccava il 4,3% delle settimane, questa il ~33%.
#     Il numero validato non era quello che girava il venerdi'.
#  2. Non funziona in nessuna delle due definizioni. Sui dieci anni interi
#     2016-2025, rendimento sul capitale impiegato: 284,5% senza filtro, 278,8%
#     con la versione del backtest, 266,3% con questa. Il +317,7% attribuito al
#     filtro in §8 veniva per intero dal 2026, anno parziale, e da DUE settimane.
#     Nel 2025 il filtro non si e' attivato nemmeno una volta.
#
#  L'ampiezza resta CALCOLATA E MOSTRATA come diagnostica: dice quanto il
#  modello ha stretto, ed e' un'informazione utile da leggere. Non decide piu'.
#  Il verdetto STRETTA non viene piu' emesso.

#  SOGLIA MARKUP — RIMOSSA (agosto 2026). Era 1,15, verificata su 13 anni.
#  1. Sotto 1,12 e' INERTE: il pavimento economico morde per primo. Fra 1,00 e
#     1,15 cambiano 13 settimane su 445.
#  2. Sopra, la curva e' frastagliata e risale fra 1,35 e 1,40. Il valore
#     economico dell'intera soglia nasce da QUATTRO settimane, una del 2020 e
#     tre del 2022; negli altri nove anni in cui morde, distrugge valore.
#  3. Il leave-one-year-out su 13 anni sceglie 1,18 (su 11 sceglieva 1,15): un
#     parametro che si sposta aggiungendo due anni non e' stimato, e' adattato.
#     E la soglia scelta fuori campione rende MENO del non filtrare affatto.
#  4. Sul 2014-2015, mai usati per tarare nulla, 1,15 costa 4.050 EUR rispetto
#     a 1,00, e non ha aiutato in nessuno degli ultimi quattro anni.
#  5. A parita' di premio incassato, con effetti fissi anno su 633 settimane, il
#     markup ha coefficiente -13,8 con t=-1,59. Tutto il suo potere apparente
#     passa dal numeratore: markup alto = premio alto, e il premio alto paga. Il
#     denominatore (fair value del modello) aggiunge rumore, non segnale.
#  Il massimo drawdown del campione e' -14.828 con e senza la soglia.
#  Il verdetto SOTT non si emette piu'.
#
#  Il markup resta CALCOLATO E MOSTRATO come diagnostica. Non decide piu'.
#
#  VERDETTO NQ (nuovo, agosto 2026) — la catena settimanale non quota strike
#  utili sotto lo spot. Succede il venerdi' di un crollo violento, quando la
#  scala degli strike non ha ancora seguito l'indice: la borsa la estende con un
#  giorno di ritardo. Tre volte in 633 settimane, sempre con la seduta oltre il
#  -4,5%: 2021-11-26 (Omicron), 2022-03-04 (Ucraina), 2025-04-04 (dazi). In quelle
#  date la put piu' bassa quotata stava rispettivamente a +0,2%, +2,4% e +3,9%
#  SOPRA lo spot: nessuna put OTM, nemmeno una.
#  NON e' deducibile dall'app, che vede solo lo spot e non il book: lo dichiara
#  l'operatore con la casella dedicata. E non e' un rifiuto di prezzo, e'
#  indisponibilita' dello strumento: va distinto nel registro.
#
#  UNICA VARIABILE CHE PREDICE — otto sono state provate con effetti fissi anno
#  e controllo per il premio: markup, volatilita' HAR, fattore di riscalatura,
#  drawdown a 3 mesi, distanza dello short, struttura a termine dell'implicita,
#  scarto implicita-realizzata, skew allo strike venduto. Nessuna sopravvive.
#  L'unica informazione utile disponibile il venerdi' e' QUANTO TI STANNO
#  PAGANDO, ed e' esattamente cio' su cui il pavimento gia' decide.
#
#  MODIFICABILI A SCHERMO
#  - capitale (default 10.000 EUR), da cui discende il numero di lotti
#  - prezzo del FTSE MIB, se yfinance e' in ritardo
#  - i due strike operati, se quelli suggeriti non sono quotati
#
#  Avvio locale:  python -m streamlit run app_putspread_live6.py
# ============================================================================
import warnings
from datetime import date, timedelta

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
H_MIN = 3                     # sotto le 3 sedute non si opera: la griglia da 100
                              # punti vale il 72-96% della distanza dello strike
MOLT = 2.5                    # EUR per punto
STEP = 100.0                  # passo della griglia strike

PCT_SHORT = 30.0              # percentile FHS della short put
PCT_CALL_RIF = 75.0           # serve solo alla diagnostica sull'ampiezza
DIST_ALA = 1000.0             # punti sotto la short put
# SOGLIA_OPER e' stata RIMOSSA in agosto 2026: vedi intestazione. L'unico filtro
# d'ingresso e' il pavimento economico.
MARGINE_PCT = 0.10            # cuscinetto sull'errore di stima, scelto per principio
COSTO_GAMBA = 1.0             # punti per gamba (1 pt = 2,5 EUR)
N_GAMBE = 2
# Riferimento storico dell'ampiezza, NON una soglia operativa: serve solo a
# colorare la didascalia. Il filtro e' stato rimosso (vedi intestazione).
AMPIEZZA_RIF = 0.022

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
    # NON usare df.dropna(): scarta la riga intera se una qualunque colonna e' NaN,
    # e sull'indice il volume o l'adjusted close dell'ultima barra spesso lo sono.
    # Il risultato e' che l'app perde l'ultima seduta senza dirlo.
    px = df["Close"].dropna()
    # La barra di OGGI, se il mercato e' aperto, e' PARZIALE: il suo rendimento non
    # e' una seduta completa e contaminerebbe la stima del GARCH e il pool dei
    # residui. Si stima sulle sole sedute chiuse; il prezzo corrente entra dopo,
    # come P0, tramite prezzo_corrente().
    px = px[px.index.date < date.today()]
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
                real20=real20, data=str(px.index[-1].date()))


@st.cache_data(ttl=60, show_spinner=False)
def prezzo_corrente():
    """Ultimo prezzo battuto, aggiornato ogni minuto. A mercato aperto e' il prezzo
       in tempo reale; a mercato chiuso e' l'ultima chiusura. Separato dal modello
       perche' quello si stima sulle sedute chiuse e si ricalcola ogni mezz'ora,
       mentre questo deve seguire il mercato."""
    try:
        fi = yf.Ticker(TICKER).fast_info
        p = float(fi["last_price"])
        if p > 0:
            return p, "yfinance live"
    except Exception:
        pass
    try:
        h = yf.Ticker(TICKER).history(period="5d", interval="1m")["Close"].dropna()
        if len(h):
            return float(h.iloc[-1]), "yfinance 1m"
    except Exception:
        pass
    return None, None


# ============================================================================
st.set_page_config(page_title="Put spread MIBO — HAR", page_icon="📉", layout="centered")
st.title("📉 Put spread MIBO — motore riscalato HAR")

c1, c2 = st.columns([3, 1])
with c2:
    if st.button("🔄 Aggiorna dati"):
        st.cache_data.clear()
        st.rerun()

# ---------------- orizzonte ----------------
# Le festivita' di Borsa Italiana sono nove e sono FISSE: 1 gennaio, venerdi'
# santo, lunedi' dell'Angelo, 1 maggio, 15 agosto, 24-25-26 dicembre, 31 dicembre.
# Verificate contro 4.319 giorni feriali dal 2010 al 2026: ZERO discordanze.
# ATTENZIONE, e' il punto che sorprende: la borsa NON chiude nelle altre feste
# civili italiane. Epifania, 25 aprile, 2 giugno, Ognissanti e Immacolata sono
# giorni di contrattazione normali — usare un elenco generico di festivita'
# italiane darebbe risultati sbagliati.
def _pasqua(y):
    a = y % 19; b = y // 100; c = y % 100; d = b // 4; e = b % 4
    f = (b + 8) // 25; g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30; i = c // 4; k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7; m = (a + 11 * h + 22 * l) // 451
    mo = (h + l - 7 * m + 114) // 31; da = ((h + l - 7 * m + 114) % 31) + 1
    return date(y, mo, da)


def _festivi(y):
    p = _pasqua(y)
    return {date(y, 1, 1), p - timedelta(days=2), p + timedelta(days=1),
            date(y, 5, 1), date(y, 8, 15),
            date(y, 12, 24), date(y, 12, 25), date(y, 12, 26), date(y, 12, 31)}


def _seduta(d):
    return d.weekday() < 5 and d not in _festivi(d.year)


def _sedute_fra(a, b):
    """sedute STRETTAMENTE comprese fra a e b"""
    n = 0
    x = a + timedelta(days=1)
    while x < b:
        if _seduta(x):
            n += 1
        x += timedelta(days=1)
    return n


def _settlement(scad):
    g = scad
    while not _seduta(g):
        g -= timedelta(days=1)
    return g


def orizzonte(data_op):
    """(scadenza, giorno di settlement, H) per la settimanale aperta in data_op.
       Il regolamento e' l'asta di apertura della scadenza; se cade in un giorno
       di chiusura si arretra all'ultima seduta utile. H conta le sedute
       STRETTAMENTE comprese fra apertura e settlement.
       Se la prima scadenza utile darebbe H=0 — apertura di giovedi', oppure il
       venerdi' e' festivo e il regolamento cadrebbe oggi stesso — si passa alla
       settimanale successiva, che e' quella che si opera davvero.
       Riproduce H_bor del calendario storico su 642 settimane su 645."""
    scad = data_op + timedelta(days=(4 - data_op.weekday()) % 7 or 7)
    g = _settlement(scad)
    if g <= data_op or _sedute_fra(data_op, g) == 0:
        scad += timedelta(days=7)
        g = _settlement(scad)
    return scad, g, _sedute_fra(data_op, g)


c_d, c_h = st.columns([1, 1])
with c_d:
    data_op = st.date_input("Data di apertura", value=date.today(), key="data_op",
                            help="Normalmente oggi. Cambiala solo per simulare "
                                 "un'altra settimana.")
_scad, _gset, _H_auto = orizzonte(data_op)
with c_h:
    H = st.number_input(
        "Orizzonte H (sedute di borsa fino al regolamento)",
        min_value=1, max_value=20, value=int(_H_auto), step=1,
        help="Calcolato dal calendario di Borsa Italiana. Resta modificabile: se il "
             "book mostra una scadenza diversa da quella prevista, correggilo a mano.")

st.caption(f"Scadenza {_scad:%d/%m/%Y} · regolamento all'apertura del "
           f"{_gset:%d/%m/%Y} · **{_H_auto} sedute**"
           + ("" if int(H) == int(_H_auto) else
              f" — stai usando H={int(H)} invece di {_H_auto}"))

if int(H) < H_MIN:
    st.error(
        f"⛔ SALTA — solo {int(H)} sedut{'a' if int(H) == 1 else 'e'} fino al "
        f"regolamento. Sotto le {H_MIN} sedute il passo della griglia MIBO (100 punti) "
        f"vale il 72-96% della distanza dello strike: tutti i percentili cadono sullo "
        f"stesso punto di griglia e non stai piu' vendendo il 30°, stai vendendo "
        f"l'unico strike disponibile. A H=1 la frequenza di sforamento misurata e' la "
        f"stessa al 30° e al 40° percentile. Registra la riga e passa la settimana.")
    st.caption("Se sei convinto che il calendario sbagli — la scadenza non e' quella "
               "prevista — correggi H qui sopra e la pagina riparte.")
    st.stop()

M = calcola_modello(int(H))
with c1:
    st.caption(f"Ultimo dato: {M['data']} · H={int(H)} · struttura fissa "
               f"{int(PCT_SHORT)}/{int(DIST_ALA)} · pavimento "
               f"fair×{1 + MARGINE_PCT:.2f}+{N_GAMBE * COSTO_GAMBA:.0f}pt")

capitale = st.number_input("Capitale (EUR)", min_value=1000, max_value=1_000_000,
                           value=10000, step=1000,
                           help="Da qui discende il numero di lotti: parte intera di "
                                "capitale diviso il margine per lotto.")
_p_live, _fonte_live = prezzo_corrente()
_p_auto = _p_live if _p_live else M["px_last"]
_desc = (f"{_fonte_live}, aggiornato ogni minuto" if _p_live
         else f"ultima chiusura del {M['data']} — prezzo live non disponibile")
pm = st.number_input("Prezzo FTSE MIB (lascia 0 per usare %s: %.0f)" % (_desc, _p_auto),
                     min_value=0.0, value=0.0, step=1.0, format="%.0f")
P0 = pm if pm > 0 else _p_auto
fonte = "MANUALE" if pm > 0 else (_fonte_live or "ultima chiusura")
st.caption(f"Spot **{P0:,.0f}** ({fonte}) · modello stimato sulle sedute chiuse fino "
           f"al {M['data']} · orizzonte calcolato da {data_op:%d/%m/%Y}, non dall'ultimo "
           f"dato di borsa")

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
# TETTO AI LOTTI. Se la catena non quota la long a 1.000 punti e si ripiega piu'
# vicino, il margine per lotto scende e la size dal margine salirebbe: con 10.000 EUR
# e un'ala ripiegata a 400 punti sarebbero 10 lotti invece di 4. Non e' un vantaggio:
# la perdita massima resta pari al capitale e la si tocca molto piu' spesso. Il tetto
# e' la size che darebbe l'ala NOMINALE, cioe' quella che opereresti se entrambe le
# put fossero quotate dove le vuoi. Sul campione 2016-2026 riduce del 10% la
# deviazione standard settimanale e del 20% il capitale complessivamente versato.
lotti_liberi = int(capitale // marg_lotto)
lotti_nom = int(capitale // (DIST_ALA * MOLT))
lotti = min(lotti_liberi, lotti_nom)
tetto_morde = lotti_liberi > lotti_nom
margine = marg_lotto * lotti

a1, a2, a3 = st.columns(3)
a1.metric("Ala operata", f"{ala:,.0f} pt", f"{ala/sigma_pt:.2f}σ · {100*ala/P0:.2f}% spot")
a2.metric("Lotti", f"{lotti}", f"margine {marg_lotto:,.0f} €/lotto"
                               + (f" · tetto {lotti_nom}" if tetto_morde else ""))
a3.metric("Margine impegnato", f"{margine:,.0f} €",
          f"{100*margine/capitale:.0f}% del capitale")

if lotti < 1:
    st.error(f"⛔ Il margine di un solo lotto ({marg_lotto:,.0f} €) supera il capitale "
             f"({capitale:,.0f} €). Riduci l'ala o aumenta il capitale.")
    st.stop()
if tetto_morde:
    st.info(f"ℹ️ Ala ripiegata a {ala:,.0f} pt: il margine per lotto scende e la size "
            f"dal margine sarebbe **{lotti_liberi} lotti**. Si opera comunque a "
            f"**{lotti}**, la size dell'ala nominale da {DIST_ALA:,.0f} pt. Aprirne di "
            f"piu' non ridurrebbe la perdita massima — resta pari al capitale — ma la "
            f"farebbe toccare due o tre volte piu' spesso, e raddoppierebbe le gambe "
            f"da eseguire. Margine impegnato {100*margine/capitale:.0f}% invece del 100%.")
elif abs(ala - DIST_ALA) >= 1:
    st.info(f"ℹ️ Ala operata {ala:,.0f} pt invece dei {DIST_ALA:,.0f} previsti. "
            f"Perdita massima {ala - 0:,.0f} pt meno premio, per lotto; il numero di "
            f"lotti si e' adattato di conseguenza.")

# ---- ampiezza: DIAGNOSTICA, non piu' un filtro ----
# Misura quanto il modello ha stretto la distribuzione. Sotto il riferimento
# storico del 2,2% significa "settimana di volatilita' attesa bassa": si opera
# comunque, ma vale la pena saperlo, perche' e' anche la condizione in cui l'ala
# fissa da 1.000 punti vale meno in percentuale dello spot.
ampiezza_pct = (k_call_rif - Kps) / P0 if P0 > 0 else float("nan")
stretta = ampiezza_pct == ampiezza_pct and ampiezza_pct < AMPIEZZA_RIF
_a = "🔎" if stretta else "✔️"
st.caption(f"{_a} Ampiezza short: **{100*ampiezza_pct:.2f}%** dello spot "
           f"(fra il 30° percentile put e il 75° percentile call, {k_call_rif:,.0f}) · "
           f"riferimento storico {100*AMPIEZZA_RIF:.1f}%. "
           + ("**Sotto il riferimento**: il modello ha stretto, vol attesa bassa. "
              "Non blocca l'operativita' — il filtro e' stato rimosso perche' sui "
              "dieci anni interi 2016-2025 costava rendimento invece di aggiungerne. "
              if stretta else
              "La call non si vende: questo numero misura solo quanto il modello ha "
              "stretto, ed e' una diagnostica, non un filtro. ")
           + f"Ala operata {100*ala/P0:.2f}% dello spot.")

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

# ---------------- NQ: strumento non disponibile ----------------
# La catena non quota strike utili sotto lo spot. Non e' deducibile dall'app, che
# vede solo lo spot e non il book: lo dichiara l'operatore. Il controllo non puo'
# stare piu' avanti, nella catena dei verdetti, perche' senza quote non c'e' mid
# e il flusso si ferma prima di arrivarci.
nq = st.checkbox(
    "⛔ La catena non quota strike utili sotto lo spot",
    key="nq",
    help="Spuntalo se sul book non esiste nessuna put quotata sotto lo spot, "
         "oppure se la piu' bassa disponibile resta sopra la short suggerita e "
         "non lascia spazio per l'ala. Genera la riga NQ per il registro e "
         "chiude la settimana.")

if nq:
    st.error(
        "⛔ SALTA — strumento non disponibile, non e' una decisione di prezzo.\n\n"
        "Non alzare la short per trovare una copertura: sulle tre date del "
        "campione l'unica struttura possibile erano due strike adiacenti "
        "entrambi dentro i soldi, ala 100 punti, con premio pari al massimo "
        "teorico. Non e' una vendita di volatilita', e con l'ala a 100 punti il "
        "margine per lotto crolla e la size esploderebbe. Il 2025-04-04 il "
        "premio mid del miglior spread possibile era perfino NEGATIVO: quelle "
        "quote erano stantie, nessuno stava piu' prezzando quegli strike.")
    st.subheader("Riga per il foglio")
    riga_nq = [data_op.strftime("%d/%m/%Y"),          # A  Data apertura
               f"{P0:.0f}",                           # B  Spot
               f"{Kps:.0f}",                          # C  Strike PUT (indicato)
               f"{Kpw:.0f}",                          # D  Strike LONG PUT (indicato)
               "", "", "", "",                        # E..H  quote assenti
               "", "",                                # I  J   eseguiti assenti
               f"{vol_ann:.2f}",                      # K  Vol riscalata %
               f"{M['vol_garch']:.2f}",               # L  Vol GARCH %
               f"{M['fattore']:.3f}",                 # M  Fattore HAR
               f"{sigma_pt:.0f}",                     # N  Sigma settimanale
               f"{round(fv_p):.0f}",                  # O  Equo PUT
               f"{round(fv_l):.0f}",                  # P  Equo LONG PUT
               "NQ",                                  # Q  Verdetto
               "0", "0",                              # R  S  lotti e margine
               f"{capitale:.0f}"]                     # T  Capitale
    st.code("\t".join(riga_nq), language=None)
    st.caption("Gli strike restano scritti: sono quelli che il modello indicava e "
               "che il mercato non quotava — l'informazione che servira' fra due "
               "anni per sapere quanto era lontana la richiesta dall'offerta. Le "
               "colonne delle quote e degli eseguiti restano vuote. Registra la "
               "riga e passa la settimana.")
    st.stop()

if not (mp > 0 and ml > 0):
    st.info("Inserisci bid e ask di entrambe le gambe per calcolare markup, edge e "
            "generare la riga del foglio. Se sul book non c'e' nulla di utile sotto "
            "lo spot, spunta la casella qui sopra.")
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
soglia_vera = net_min_pav
punti_residui = net_op - soglia_vera
edge = net_op - net_equo_r - N_GAMBE * COSTO_GAMBA
slippage = net_mid - net_exe

# Il verdetto deve distinguere PERCHE' non si opera, altrimenti il registro non
# puo' calcolare il tasso di settimane operabili: una settimana senza edge non e'
# la stessa cosa di una in cui il capitale non basta.
# RITIRATI: STRETTA (agosto 2026, col filtro ampiezza) e SOTT (agosto 2026, con la
# soglia markup). Le righe gia' nel registro restano valide come storico; da qui in
# avanti quelle settimane producono POS oppure NEG.
# NQ e' nuovo: la catena settimanale non quota nessuno strike sotto lo spot. Succede
# il venerdi' di un crollo violento, quando la scala degli strike non ha ancora
# seguito l'indice — 2021-11-26, 2022-03-04, 2025-04-04 nel campione. Non e' una
# scelta, e' indisponibilita' dello strumento, e va distinta da un rifiuto di prezzo.
if net_op < net_min_pav:
    verdetto = "NEG"            # il premio non copre il pavimento economico
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
           f"{net_min_pav:.0f} pt (unica soglia) · markup {markup:.2f}x, "
           f"diagnostico e non piu' vincolante · slippage speso finora "
           f"{slippage:+.0f} pt")

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

# Delta dello spread. BS qui e' unita' di misura, non modello di prezzo: la
# trasformazione prezzo -> IV -> delta e' biunivoca e l'errore del modello si
# applica a entrambe le gambe allo stesso modo.
ivs = iv_imp(ep, P0, Kps, "p")
if ivs == ivs and ivw == ivw and ivs > 0 and ivw > 0:
    d_s = -norm.cdf(-((np.log(P0 / Kps) + 0.5 * ivs * ivs) / ivs))
    d_l = -norm.cdf(-((np.log(P0 / Kpw) + 0.5 * ivw * ivw) / ivw))
    d_net = -(d_s - d_l)
    st.caption(f"Delta dello spread **{d_net:+.3f}** per lotto → "
               f"**{d_net * MOLT * lotti:+.1f} € per punto di indice** su {lotti} "
               f"lott{'o' if lotti == 1 else 'i'}. Storico: mediana 0,278 per lotto, "
               f"stabile fra 0,24 e 0,31 in tredici anni. E' esposizione lunga al "
               f"mercato, non rumore: parte del rendimento viene da li'.")

# ---------------- riga per il foglio ----------------
st.subheader("Riga per il foglio")
# la data della riga e' quella di apertura scelta sopra, non l'orologio:
# se stai simulando un'altra settimana il registro deve dirlo
oggi = data_op.strftime("%d/%m/%Y")
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
