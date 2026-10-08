# ============================================================================
#  PUT SPREAD MIBO — operativita' del venerdi' (browser, PC + telefono)
#  v9 — ottobre 2026. Derivata dalla v8: stesso motore, stesso flusso, stesso
#  pavimento. Cambia il CENTRO della distribuzione, secondo una regola fissa.
#
#  AVVIO:  python -m streamlit run app_putspread_live9.py
#
#  COSA CAMBIA RISPETTO ALLA v8
#  - REGOLA DEL DRIFT A 1 ANNO. Il GARCH e' stimato a media zero, ma il FHS
#    pesca dai residui standardizzati delle ultime 756 sedute SENZA ricentrarli:
#    la simulazione eredita il rendimento medio degli ultimi tre anni (il
#    "drift"). Dopo tre anni di rialzo forte questo sposta la distribuzione
#    verso l'alto e AVVICINA lo short allo spot: a luglio 2026 il 30o percentile
#    stava a 230 punti invece di 383.
#    Su 1.225 venerdi' 2001-2026 il drift della finestra NON predice il
#    rendimento della settimana dopo (pendenza -0,03, IC 90% [-0,90, +0,72];
#    probabilita' che si realizzi per intero: 1%). Togliendolo sempre, pero', la
#    calibrazione peggiora dopo i ribassi (errore 15,8 contro 9,3), perche' li'
#    il drift negativo compensa una distribuzione troppo stretta.
#    Regola adottata: se lo spot del venerdi' e' SOPRA la chiusura di 252 sedute
#    prima, il drift si azzera (residui ricentrati); altrimenti si tiene il
#    motore com'e'. Errore di calibrazione 2001-2026: 9,1 contro 9,8 del motore
#    v8 e 10,7 del drift sempre a zero. Orizzonti di 1, 2 e 3 anni equivalenti.
#    Backtest 2014-2026 con le regole di produzione: P&L +147.749 contro
#    +134.747, drawdown -18.515 contro -20.811, perdite massime 13 contro 17.
#    LIMITI DICHIARATI: il vantaggio di P&L dipende da poche settimane e non e'
#    distinguibile dal caso (anno mobile: meglio nel 50% delle finestre, peggio
#    nel 47%; cambia segno con il margine del pavimento). Il drawdown piu' basso
#    regge in 19 combinazioni su 20. E' un'assicurazione a costo quasi nullo
#    contro la rottura del trend, non una fonte di rendimento. Vedi
#    claude/drift_casualita_e_regola_mista_risultati.md e lo studio del 06/10.
#  - L'app calcola ENTRAMBE le varianti e mostra entrambi gli strike; quello da
#    operare e' evidenziato in rosso. L'altro si registra (colonna L del Diario)
#    per verificare la regola in avanti.
#  - PROBABILITA' DI PERDITA MASSIMA CON IL HAR CORRETTO, solo informativa.
#    La regressione HAR e' in logaritmi: exp(previsione) e' la MEDIANA della
#    varianza futura, non la media, e la volatilita' esce bassa di circa il 20%
#    (fattore di smearing 1,20-1,24). Per lo short conta poco (il 30o percentile
#    resta calibrato), ma la coda e' troppo sottile: il motore prevede la
#    perdita massima al 2-5%, osservata 6-7% nel 2024-2026. Con il HAR corretto
#    la previsione torna in linea (2024-2026: 5,3% prevista, 5,7% osservata).
#    Come motore operativo il HAR corretto NON e' stato adottato: alza il fair,
#    dimezza le settimane operate e costa P&L (claude/har_corretto_risultati.md).
#    Qui serve solo a misurare: l'app mostra la probabilita' corretta, il net
#    equo corretto e il verdetto che darebbe il pavimento corretto. I due numeri
#    vanno nel registro (colonne M e N) e fra un anno diranno se la coda
#    corretta era giusta e se le settimane che il pavimento corretto boccia sono
#    davvero quelle che fanno male.
#  - La riga per il registro ha 14 colonne (A..N) invece di 10. Alla scadenza le
#    celle da compilare si spostano di quattro: O data, P e Q i due risultati
#    Directa, R il settlement a mano.
#
#  INVARIATO DALLA v8: percentile 30, ala 1.000 punti, pavimento fair x 1,10 +
#  2 punti calcolato sul motore OPERATIVO (non su quello corretto), tetto ai
#  lotti, verdetto NQ, rendimento del venerdi' dentro il modello.
#
#  ----------------------------------------------------------------------------
#  STORIA PRECEDENTE (dalla v8, invariata)
#
#  COSA CAMBIA RISPETTO ALLA v7
#  - IL RENDIMENTO DEL VENERDI' ENTRA NEL MODELLO. La v7 stimava GARCH e HAR
#    sulle sedute chiuse fino a giovedi' e usava lo spot del venerdi' solo come
#    centro della distribuzione: la larghezza non sapeva che oggi l'indice
#    aveva fatto -3%. Il backtest invece condiziona alla CHIUSURA del venerdi'.
#    Misurato sulle 508 settimane appaiate 2016-2025 con le regole di produzione
#    invariate (studio_venerdi.py): condizionare al giovedi' costa il 22% del
#    P&L (94.761 contro 121.339 EUR) e porta il drawdown massimo da -15.376 a
#    -22.698, a sforamento e tasso operabile identici; la differenza sta nei
#    venerdi' con |r| > 2%, dove la vol riscalata era piu' bassa di due punti.
#    Da questa versione il rendimento del venerdi' in corso, log(spot/chiusura
#    di giovedi'), entra come innovazione di oggi: un passo del GJR sul var0 e
#    un punto in piu' nel campione del HAR. Il fit del GARCH resta sulle sedute
#    chiuse. Intraday si sta fra il vecchio e il backtest; a fine seduta si
#    coincide con il backtest. A mercato chiuso, o simulando un'altra data,
#    non si applica nulla.
#  - La finestra del HAR (10 anni di yfinance contro tutta la storia del
#    backtest) e' stata verificata nello stesso studio ed e' innocua: P&L +1%,
#    sforamento e tasso operabile uguali. Resta a 10 anni.
#
#  COSA CAMBIA RISPETTO ALLA v6
#  - f E g SONO RITIRATI. Misuravano lo scarto dal mid e richiedevano un
#    bid/ask CONTEMPORANEO al fill. Nella pratica il book cambia fra la
#    trascrizione e l'invio dell'ordine, e a eseguito avvenuto non si sa piu'
#    quale fosse lo spread in quel momento: il numero registrato non era
#    rumoroso, era un'altra grandezza. Il filo d'inciampo sull'esecuzione
#    diventa il MARKUP ESEGUITO (net eseguito / net equo), che usa il fair
#    value del motore: deterministico, disponibile prima dell'ordine, non
#    scade. Le costanti F_RIF e G_RIF non esistono piu'.
#  - FLUSSO IN DUE PASSI. Si trascrive dal book SOLO il bid/ask della LONG
#    PUT, che essendo lontana dai soldi e' molto piu' stabile. Si compra
#    l'ala, si inserisce il prezzo eseguito, e l'app calcola il PEGGIOR BID
#    ACCETTABILE sulla short perche' la struttura resti sopra il pavimento.
#    Prima dell'acquisto il bid minimo viene mostrato usando l'ASK della long,
#    cioe' nel caso peggiore: serve a sapere in anticipo se la settimana e'
#    operabile senza aver gia' impegnato nulla.
#  (il prefisso "python -m" evita lo shim streamlit.exe, che i criteri di
#   controllo applicazioni di Windows possono bloccare perche' non firmato)
#
#  COSA CAMBIA RISPETTO ALLA v12
#  - Solo PUT: short put al 30o percentile FHS + long put 1.000 punti sotto.
#    La call non si vende mai, quindi niente stacca-call, niente banda di size,
#    niente ripiego: la struttura e' una sola.
#  - MOTORE RISCALATO HAR. Il GJR-GARCH sbaglia il livello della volatilita'
#    attesa e i percentili sforavano meno del nominale. La distribuzione FHS
#    viene tenuta com'e' — asimmetria, code, arricchimento EVT — e se ne
#    corregge solo la SCALA con il rapporto fra la previsione HAR e quella
#    GARCH. Sul campione 2016-2026 la calibrazione del 30o percentile allo
#    strike operato e' 27,0%. Lo scarto dal 30% nominale viene in parte
#    dall'arrotondamento per difetto alla griglia da 100 punti.
#    NOTA OTTOBRE 2026: parte di quello che qui si chiamava "sovrastima del
#    GARCH" e' in realta' la previsione HAR che sta bassa (vedi sopra, smearing).
#  - SIZE DAL MARGINE CON TETTO: si aprono tutti i lotti che entrano nel
#    capitale, cioe' parte intera di (capitale / (ampiezza x 2,5)), MA mai piu'
#    di quanti ne darebbe l'ala NOMINALE da 1.000 punti. Con 10.000 EUR sono 4
#    lotti, con 20.000 sono 8. Il tetto morde solo quando la catena non quota la
#    long a 1.000 punti e l'ala ripiega piu' corta.
#
#  PARAMETRI FISSATI QUI E NON MODIFICABILI A SCHERMO
#  - percentile della short put: 30
#  - ala: 1.000 punti
#  - pavimento economico: fair x 1,10 + 2 punti  (UNICO filtro d'ingresso)
#  - regola del drift: confronto con la chiusura di 252 sedute prima
#  Il 10% del pavimento e' un cuscinetto sull'errore di stima scelto PER
#  PRINCIPIO, non il massimo di una curva: fra 0,00 e 0,20 nessun valore e'
#  distinguibile. NON ritararlo. Il pavimento da solo vale circa 28.000 EUR sul
#  2014-2026: le settimane che scarta, operate, avrebbero perso in media 170 EUR.
#
#  Filtro ampiezza e soglia markup sono stati RIMOSSI in agosto 2026 e restano
#  solo come diagnostica. Il verdetto NQ (la catena non quota strike utili
#  sotto lo spot) lo dichiara l'operatore con la casella dedicata.
#
#  MODIFICABILI A SCHERMO
#  - capitale (default 10.000 EUR), da cui discende il numero di lotti
#  - prezzo del FTSE MIB, se yfinance e' in ritardo
#  - i due strike operati, se quelli suggeriti non sono quotati
#
#  Avvio locale:  python -m streamlit run app_putspread_live9.py
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
MARGINE_PCT = 0.10            # cuscinetto sull'errore di stima, scelto per principio
COSTO_GAMBA = 1.0             # punti per gamba (1 pt = 2,5 EUR)
N_GAMBE = 2
AMPIEZZA_RIF = 0.022          # riferimento storico dell'ampiezza, solo diagnostica

SEDUTE_DRIFT = 252            # regola del drift: spot contro la chiusura di 252 sedute prima

USA_EVT_TAIL = True
SOGLIA_EVT = 0.10             # coda sinistra arricchita con GPD sotto il 10o pct
LAG_HAR = (1, 5, 22, 66)      # orizzonti della regressione HAR
MIN_FATT, MAX_FATT = 0.50, 2.00   # limiti del fattore di riscalatura

VRP_MARKUP = 1.25             # solo per la IV di riferimento nella diagnostica

NOMI = {"drift": "con drift (motore v8)", "zero": "senza drift"}


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
       disponibili alla data: le coppie usate hanno t+H <= oggi.
       Restituisce (vol, smear): vol e' quella del motore operativo (mediana
       della varianza futura, come nella v8); smear e' il fattore che la porta
       alla MEDIA, cioe' la radice della media di exp(residui) della stessa
       regressione. Serve solo al HAR corretto, informativo."""
    rv = (px ** 2).clip(lower=1e-8)
    X = pd.concat([rv.rolling(l).mean() for l in LAG_HAR], axis=1).dropna()
    y = rv.rolling(H).mean().shift(-H).reindex(X.index)
    ok = y.notna() & (y > 0) & np.isfinite(np.log(X)).all(axis=1)
    if ok.sum() < 300:
        return np.nan, np.nan
    Xa = np.column_stack([np.ones(int(ok.sum())), np.log(X[ok].values)])
    ya = np.log(y[ok].values)
    if not (np.isfinite(Xa).all() and np.isfinite(ya).all()):
        return np.nan, np.nan
    beta = np.linalg.pinv(Xa) @ ya
    smear = float(np.sqrt(np.mean(np.exp(ya - Xa @ beta))))
    xs = np.concatenate([[1.0], np.log(X.iloc[-1].values)])
    return float(np.sqrt(np.exp(float(xs @ beta))) * np.sqrt(252)), smear


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


def _fattore(vol, vol_g):
    """rapporto di riscalatura limitato a [MIN_FATT, MAX_FATT]: (fattore, limitato?)"""
    if not (np.isfinite(vol) and vol_g > 0):
        return 1.0, False
    f = vol / vol_g
    if f < MIN_FATT or f > MAX_FATT:
        return min(max(f, MIN_FATT), MAX_FATT), True
    return f, False


@st.cache_data(ttl=1800, show_spinner="Scarico i dati, stimo il GARCH e il HAR...")
def calcola_modello(H, r_oggi=None):
    """r_oggi: rendimento logaritmico in % della seduta in corso (spot contro
       l'ultima chiusura), oppure None se non c'e' una seduta in corso. Entra
       come innovazione di oggi nel GJR e come ultimo punto del HAR; il fit del
       GARCH resta sulle sole sedute chiuse. E' arrotondato a monte in modo che
       la cache non si invalidi a ogni tick.
       Simula DUE varianti, con lo stesso GARCH e lo stesso seme:
         drift  il pool dei residui com'e' (motore v8)
         zero   il pool ricentrato a media zero
       e per ciascuna due scale: HAR operativo e HAR corretto (informativo)."""
    df = yf.download(TICKER, period="10y", progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    # NON usare df.dropna(): scarta la riga intera se una qualunque colonna e' NaN,
    # e sull'indice il volume o l'adjusted close dell'ultima barra spesso lo sono.
    px = df["Close"].dropna()
    # La barra di OGGI, se il mercato e' aperto, e' PARZIALE: si stima sulle sole
    # sedute chiuse; il prezzo corrente entra dopo, come P0.
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

    # Rendimento del venerdi' in corso: un passo del GJR sul var0 e un punto in
    # piu' nel campione del HAR, come nel backtest (vedi v8).
    lr_har = lr
    if r_oggi is not None:
        var0 = float(gjr(var0, r_oggi, om, al, ga, be))
        lr_har = pd.concat([lr, pd.Series([r_oggi], index=[px.index[-1] + pd.Timedelta(days=1)])])

    vol_h, smear = previsione_har(lr_har, H)
    vol_c = vol_h * smear if np.isfinite(smear) else np.nan

    out = dict(px_last=float(px.iloc[-1]), data=str(px.index[-1].date()), r_oggi=r_oggi,
               vol_har=vol_h, vol_har_corr=vol_c, smear=smear, media_pool=float(pool.mean()),
               real20=float(lr.iloc[-20:].std() * np.sqrt(252)),
               # chiusure di riferimento per la regola del drift: 252 sedute prima
               # del venerdi'. Se il venerdi' e' in corso, px finisce a giovedi' e
               # il riferimento e' px[-252]; se il venerdi' e' gia' chiuso ed e'
               # l'ultima barra di px, il riferimento e' px[-253].
               ref_in_corso=(float(px.iloc[-SEDUTE_DRIFT]), str(px.index[-SEDUTE_DRIFT].date())),
               ref_chiuso=(float(px.iloc[-SEDUTE_DRIFT - 1]), str(px.index[-SEDUTE_DRIFT - 1].date())),
               cum={}, cum_corr={}, vol_garch={}, fattore={}, capped={})
    for nome, pl in (("drift", pool), ("zero", pool - pool.mean())):
        evt = None
        if USA_EVT_TAIL and len(pl) > 30:
            u = np.quantile(pl, SOGLIA_EVT)
            ex = u - pl[pl < u]
            if len(ex) >= 10:
                c_, _, sc_ = stats.genpareto.fit(ex, floc=0)
                evt = (u, c_, sc_)
        rng = np.random.default_rng(42)
        cum = fhs(var0, pl, om, al, ga, be, H, N_SIM, evt, rng)
        vol_g = float(cum.std()) * 100 * np.sqrt(252 / H)
        f, cap = _fattore(vol_h, vol_g)
        fc, _ = _fattore(vol_c, vol_g)
        out["cum"][nome] = cum * f
        out["cum_corr"][nome] = cum * fc
        out["vol_garch"][nome] = vol_g
        out["fattore"][nome] = f
        out["capped"][nome] = cap
    return out


@st.cache_data(ttl=60, show_spinner=False)
def prezzo_corrente():
    """Ultimo prezzo battuto, aggiornato ogni minuto."""
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


def pct_it(x):
    """percentuale con una cifra decimale e virgola, per il registro in locale it_IT"""
    return f"{100 * x:.1f}".replace(".", ",")


# ============================================================================
st.set_page_config(page_title="Put spread MIBO — v9", page_icon="📉", layout="centered")
st.title("📉 Put spread MIBO — v9")

c1, c2 = st.columns([3, 1])
with c2:
    if st.button("🔄 Aggiorna dati"):
        st.cache_data.clear()
        st.rerun()

# ---------------- orizzonte ----------------
# Le festivita' di Borsa Italiana sono nove e sono FISSE: 1 gennaio, venerdi'
# santo, lunedi' dell'Angelo, 1 maggio, 15 agosto, 24-25-26 dicembre, 31 dicembre.
# La borsa NON chiude nelle altre feste civili italiane.
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
    """(scadenza, giorno di settlement, H) per la settimanale aperta in data_op."""
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
        help="Calcolato dal calendario di Borsa Italiana. Resta modificabile.")

st.caption(f"Scadenza {_scad:%d/%m/%Y} · regolamento all'apertura del "
           f"{_gset:%d/%m/%Y} · **{_H_auto} sedute**"
           + ("" if int(H) == int(_H_auto) else
              f" — stai usando H={int(H)} invece di {_H_auto}"))

if int(H) < H_MIN:
    st.error(
        f"⛔ SALTA — solo {int(H)} sedut{'a' if int(H) == 1 else 'e'} fino al "
        f"regolamento. Sotto le {H_MIN} sedute il passo della griglia MIBO (100 punti) "
        f"vale il 72-96% della distanza dello strike. Registra la riga e passa la settimana.")
    st.stop()

M0 = calcola_modello(int(H))

capitale = st.number_input("Capitale (EUR)", min_value=1000, max_value=1_000_000,
                           value=10000, step=1000,
                           help="Da qui discende il numero di lotti: parte intera di "
                                "capitale diviso il margine per lotto.")
_p_live, _fonte_live = prezzo_corrente()
_p_auto = _p_live if _p_live else M0["px_last"]
_desc = (f"{_fonte_live}, aggiornato ogni minuto" if _p_live
         else f"ultima chiusura del {M0['data']} — prezzo live non disponibile")
pm = st.number_input("Prezzo FTSE MIB (lascia 0 per usare %s: %.0f)" % (_desc, _p_auto),
                     min_value=0.0, value=0.0, step=1.0, format="%.0f")
P0 = pm if pm > 0 else _p_auto
fonte = "MANUALE" if pm > 0 else (_fonte_live or "ultima chiusura")

_ultima = pd.Timestamp(M0["data"]).date()
if data_op == date.today() and _seduta(data_op) and _ultima < data_op and P0 > 0:
    r_oggi = round(float(100 * np.log(P0 / M0["px_last"])) / 0.05) * 0.05
else:
    r_oggi = None
M = calcola_modello(int(H), r_oggi)

# ---------------- regola del drift ----------------
# Se la seduta del venerdi' e' in corso l'ultima barra di px e' giovedi': il
# riferimento e' 252 sedute prima di oggi. Altrimenti lo spot E' l'ultima
# chiusura e il riferimento slitta di una barra.
ref_px, ref_data = M["ref_in_corso"] if r_oggi is not None else M["ref_chiuso"]
sopra = P0 > ref_px
SCELTO = "zero" if sopra else "drift"
ALTRO = "drift" if SCELTO == "zero" else "zero"
REGIME = "ZERO" if SCELTO == "zero" else "DRIFT"

with c1:
    st.caption(f"Ultimo dato: {M['data']} · H={int(H)} · struttura fissa "
               f"{int(PCT_SHORT)}/{int(DIST_ALA)} · pavimento "
               f"fair×{1 + MARGINE_PCT:.2f}+{N_GAMBE * COSTO_GAMBA:.0f}pt")
st.caption(f"Spot **{P0:,.0f}** ({fonte}) · GARCH stimato sulle sedute chiuse fino "
           f"al {M['data']}"
           + (f" · **seduta di oggi {r_oggi:+.2f}%** gia' dentro il modello (var0 e HAR)"
              if r_oggi is not None else
              " · nessuna seduta in corso: il modello e' condizionato all'ultima chiusura")
           + f" · orizzonte calcolato da {data_op:%d/%m/%Y}")

# distribuzioni delle due varianti, sullo spot di oggi
PTV = {k: P0 * np.exp(M["cum"][k]) for k in ("drift", "zero")}
rnd_giu = lambda x: np.floor(x / STEP) * STEP
rnd_su = lambda x: np.ceil(x / STEP) * STEP


def sigma_pt_di(k):
    return P0 * float(M["cum"][k].std())


REC = {}
for k in ("drift", "zero"):
    ps = float(rnd_giu(np.percentile(PTV[k], PCT_SHORT)))
    pw = ps - DIST_ALA
    fv = lambda K, k=k: float(np.mean(np.maximum(K - PTV[k], 0.0)))
    REC[k] = dict(ps=ps, pw=pw, sig=sigma_pt_di(k), equo=round(fv(ps)) - round(fv(pw)))

PT = PTV[SCELTO]
sigma_pt = REC[SCELTO]["sig"]
vol_ann = float(M["cum"][SCELTO].std()) * 100 * np.sqrt(252 / int(H))
rec_ps, rec_pw = REC[SCELTO]["ps"], REC[SCELTO]["pw"]
k_call_rif = float(rnd_su(np.percentile(PT, PCT_CALL_RIF)))   # solo diagnostica

# ---------------- pannello del motore ----------------
m1, m2, m3 = st.columns(3)
m1.metric("FTSE MIB", f"{P0:,.0f}", fonte)
m2.metric("Vol attesa (riscalata)", f"{vol_ann:.1f}%",
          f"GARCH {M['vol_garch'][SCELTO]:.1f}% · HAR {M['vol_har']:.1f}%")
m3.metric("Fattore di riscalatura", f"{M['fattore'][SCELTO]:.3f}",
          "cap attivo" if M["capped"][SCELTO] else f"sigma {sigma_pt:,.0f} pt")
if M["capped"][SCELTO]:
    st.warning(f"⚠️ Il fattore HAR/GARCH era fuori dall'intervallo "
               f"[{MIN_FATT:.2f}, {MAX_FATT:.2f}] ed e' stato limitato. Guarda i due "
               f"numeri di volatilita' prima di operare.")

# ---------------- strike suggeriti: le due varianti ----------------
st.subheader("Strike suggeriti — le due varianti")
_var_pct = 100 * (P0 / ref_px - 1)
st.caption(f"Regola del drift: spot {P0:,.0f} contro la chiusura di {SEDUTE_DRIFT} sedute "
           f"prima ({ref_px:,.0f} del {pd.Timestamp(ref_data):%d/%m/%Y}, {_var_pct:+.1f}%). "
           + ("**Indice sopra il livello di un anno fa → si opera SENZA drift.**" if sopra else
              "**Indice sotto il livello di un anno fa → si opera CON il drift (motore v8).**")
           + f" Drift attuale nel pool: {M['media_pool']:+.3f}.")


def _riga_html(k):
    r = REC[k]
    sel = (k == SCELTO)
    stile = ("border:3px solid #d32f2f;font-weight:700;"
             if sel else "border:1px solid #888;opacity:0.6;")
    tag = " ◀ DA OPERARE" if sel else ""
    return (f"<tr style='{stile}'>"
            f"<td style='padding:6px 10px'>{NOMI[k]}{tag}</td>"
            f"<td style='padding:6px 10px;text-align:right'>{r['ps']:,.0f}</td>"
            f"<td style='padding:6px 10px;text-align:right'>{r['pw']:,.0f}</td>"
            f"<td style='padding:6px 10px;text-align:right'>{P0 - r['ps']:,.0f} pt · {100*(P0 - r['ps'])/P0:.2f}%</td>"
            f"<td style='padding:6px 10px;text-align:right'>{(P0 - r['ps'])/r['sig']:.2f}σ</td>"
            f"<td style='padding:6px 10px;text-align:right'>{r['equo']:,.0f}</td></tr>")


st.markdown(
    "<table style='border-collapse:collapse;width:100%;font-size:0.95em'>"
    "<tr style='border-bottom:2px solid #999'><th style='text-align:left;padding:6px 10px'>variante</th>"
    "<th style='padding:6px 10px'>SHORT</th><th style='padding:6px 10px'>LONG</th>"
    "<th style='padding:6px 10px'>distanza short</th><th style='padding:6px 10px'>in σ</th>"
    "<th style='padding:6px 10px'>net equo</th></tr>"
    + _riga_html(SCELTO) + _riga_html(ALTRO) + "</table>", unsafe_allow_html=True)
#st.markdown(
#    f"<div style='border:3px solid #d32f2f;border-radius:8px;padding:12px 16px;margin:12px 0;"
#    f"font-size:1.15em'>🔴 <b>SHORT DA VENDERE: {rec_ps:,.0f}</b> · "
#    f"long {rec_pw:,.0f} · variante {NOMI[SCELTO]}</div>", unsafe_allow_html=True)

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
               "markup e numero di lotti si ricalcolano su questi.")

e1, e2 = st.columns(2)
Kps = e1.number_input("SHORT PUT operata", min_value=0.0, step=STEP, key="k_ps", format="%.0f")
Kpw = e2.number_input("LONG PUT operata", min_value=0.0, step=STEP, key="k_pw", format="%.0f")
if Kps != rec_ps:
    st.warning(f"⚠️ Lo short operato ({Kps:,.0f}) non e' quello indicato dalla regola "
               f"({rec_ps:,.0f}).")

ala = Kps - Kpw
if ala <= 0:
    st.error("La long put deve stare SOTTO la short put (strike piu' basso).")
    st.stop()

marg_lotto = ala * MOLT
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
    st.info(f"ℹ️ Ala ripiegata a {ala:,.0f} pt: la size dal margine sarebbe "
            f"**{lotti_liberi} lotti**. Si opera comunque a **{lotti}**, la size dell'ala "
            f"nominale da {DIST_ALA:,.0f} pt.")
elif abs(ala - DIST_ALA) >= 1:
    st.info(f"ℹ️ Ala operata {ala:,.0f} pt invece dei {DIST_ALA:,.0f} previsti; il numero "
            f"di lotti si e' adattato di conseguenza.")

ampiezza_pct = (k_call_rif - Kps) / P0 if P0 > 0 else float("nan")
st.caption(f"Ampiezza short (diagnostica): **{100*ampiezza_pct:.2f}%** dello spot · "
           f"riferimento storico {100*AMPIEZZA_RIF:.1f}% · ala operata {100*ala/P0:.2f}% dello spot.")


def fair(K):
    """valore atteso del payoff della put allo strike K, sulla variante OPERATIVA"""
    return float(np.mean(np.maximum(K - PT, 0.0)))


fv_p, fv_l = fair(Kps), fair(Kpw)

# ---------------- coda: motore operativo contro HAR corretto (informativo) ----------------
PTC = P0 * np.exp(M["cum_corr"][SCELTO])
p_maxl = float(np.mean(PT < Kpw))
p_maxl_c = float(np.mean(PTC < Kpw))
fvc = lambda K: float(np.mean(np.maximum(K - PTC, 0.0)))
net_equo_corr = round(fvc(Kps)) - round(fvc(Kpw))
q1, q2, q3 = st.columns(3)
q1.metric("Perdita massima: motore", f"{100*p_maxl:.1f}%", "probabilita' a settimana")
q2.metric("Perdita massima: HAR corretto", f"{100*p_maxl_c:.1f}%",
          f"circa una ogni {1/p_maxl_c:.0f} settimane" if p_maxl_c > 0 else "")
q3.metric("Net equo corretto", f"{net_equo_corr:.0f} pt",
          f"smearing {M['smear']:.3f} · vol {M['vol_har_corr']:.1f}%")
st.caption("Solo misura, non decide nulla: il pavimento resta quello del motore operativo. "
           "Il HAR corretto porta la volatilita' prevista dalla mediana alla media della "
           "varianza futura; nel 2024-2026 ha previsto la perdita massima al 5,3% contro il "
           "5,7% osservato, il motore operativo al 3,5%. I due numeri vanno nel registro "
           "(colonne M e N).")

# ---------------- ala comprata: l'unico book che si trascrive ---------------
st.subheader("1 · Ala comprata — bid/ask dal book")

c1, c2 = st.columns(2)
lb = c1.number_input("LONG PUT BID", min_value=0.0, value=0.0, step=1.0,
                     format="%.0f", key="LONGPUTb")
la = c2.number_input("LONG PUT ASK", min_value=0.0, value=0.0, step=1.0,
                     format="%.0f", key="LONGPUTa")
ml = (lb + la) / 2 if (lb > 0 and la > 0) else 0.0
if ml > 0:
    _semi = (la - lb) / 2
    st.caption(f"Mid long put **{ml:.0f} pt** · semispread {_semi:.0f} pt "
               f"({100 * (la - lb) / ml:.0f}% del mid) · fair del modello {fv_l:.0f} pt "
               f"({ml / fv_l:.2f}x)" if fv_l > 0 else
               f"Mid long put **{ml:.0f} pt** · semispread {_semi:.0f} pt")


def riga_registro(verdetto, le_, pe_, net_equo_, n_contr):
    """Quattordici colonne A..N del Diario (registro v6)."""
    return [data_op.strftime("%d/%m/%Y"),          # A  Data apertura
            f"{P0:.0f}",                           # B  Spot
            f"{Kps:.0f}",                          # C  Strike PUT (vendi)
            f"{Kpw:.0f}",                          # D  Strike LONG PUT (compri)
            le_, pe_,                              # E  F  eseguiti
            f"{net_equo_:.0f}",                    # G  Net equo (motore operativo)
            f"{sigma_pt:.0f}",                     # H  Sigma settimanale (punti)
            f"{n_contr:d}",                        # I  Lotti
            verdetto,                              # J  Verdetto
            REGIME,                                # K  Regime drift: ZERO / DRIFT
            f"{REC[ALTRO]['ps']:.0f}",             # L  Strike short dell'altra variante
            pct_it(p_maxl_c),                      # M  Prob. perdita max, HAR corretto (%)
            f"{net_equo_corr:.0f}"]                # N  Net equo, HAR corretto


nq = st.checkbox(
    "⛔ La catena non quota strike utili sotto lo spot",
    key="nq",
    help="Spuntalo se sul book non esiste nessuna put quotata sotto lo spot. Genera la "
         "riga NQ per il registro e chiude la settimana.")

if nq:
    st.error("⛔ SALTA — strumento non disponibile, non e' una decisione di prezzo. "
             "Non alzare la short per trovare una copertura.")
    st.subheader("Riga per il foglio")
    st.code("\t".join(riga_registro("NQ", "", "", round(fair(Kps)) - round(fair(Kpw)), 0)),
            language=None)
    st.caption("Quattordici colonne, **A→N**. Gli eseguiti restano vuoti e le colonne della "
               "scadenza (da O in poi) non si compilano.")
    st.stop()

if ml <= 0:
    st.info("Inserisci bid e ask della LONG PUT per calcolare il pavimento e il bid "
            "minimo accettabile sulla short.")
    st.stop()

# ---------------- 2 · pavimento e bid minimo sulla short --------------------
fpr, flr = round(fv_p), round(fv_l)
net_equo_r = fpr - flr

if net_equo_r <= 0:
    st.error("⛔ Fair value netto non positivo: controlla gli strike.")
    st.stop()

net_min_pav = net_equo_r * (1 + MARGINE_PCT) + N_GAMBE * COSTO_GAMBA
net_min_pav_c = net_equo_corr * (1 + MARGINE_PCT) + N_GAMBE * COSTO_GAMBA

st.subheader("2 · Compra l'ala, poi vendi la short")

le = st.number_input("LONG PUT eseguito (0 = non ancora comprata)",
                     min_value=0.0, value=0.0, step=1.0, format="%.0f", key="LONGPUTe")
costo_long = le if le > 0 else la
base_long = "eseguito" if le > 0 else "ask, caso peggiore"
bid_min_short = net_min_pav + costo_long

st.metric("Bid minimo accettabile sulla SHORT PUT", f"{bid_min_short:.0f} pt",
          f"pavimento {net_min_pav:.0f} + ala {costo_long:.0f} ({base_long})")
st.caption(f"Sotto **{bid_min_short:.0f} punti** la struttura non copre il pavimento "
           f"economico: non si vende. Fair netto del modello {net_equo_r:.0f} pt "
           f"(put {fpr:.0f} − ala {flr:.0f}). Con il HAR corretto il pavimento sarebbe "
           f"{net_min_pav_c:.0f} pt, cioe' bid {net_min_pav_c + costo_long:.0f}: solo informativo.")
if le <= 0:
    st.caption("⚠️ Valore provvisorio: calcolato sull'ASK della long.")

# ---------------- 3 · esecuzione della short ----------------
st.subheader("3 · Esito")
pe = st.number_input("SHORT PUT eseguito (0 = non ancora venduta)",
                     min_value=0.0, value=0.0, step=1.0, format="%.0f", key="PUTe")
saltata = st.checkbox("Il mercato non paga il pavimento: non ho operato", key="salta")

eseguito_completo = (pe > 0 and le > 0)
net_exe = (pe - le) if eseguito_completo else float("nan")
markup = (net_exe / net_equo_r) if eseguito_completo else float("nan")

if saltata:
    verdetto = "NEG"
elif not eseguito_completo:
    verdetto = "—"
elif margine > capitale:
    verdetto = "MARGINE"
elif net_exe < net_min_pav:
    verdetto = "NEG"
else:
    verdetto = "POS"

if eseguito_completo:
    premio_tot = net_exe * MOLT * lotti
    perdita_max = (ala - net_exe) * MOLT * lotti
    edge = net_exe - net_equo_r - N_GAMBE * COSTO_GAMBA
    d1, d2, d3 = st.columns(3)
    d1.metric("Net eseguito", f"{net_exe:.0f} pt", f"{net_exe - net_min_pav:+.0f} vs pavimento")
    d2.metric("Markup eseguito", f"{markup:.3f}x", "net eseguito / net equo")
    d3.metric("EDGE netto", f"{edge:+.0f} pt", f"{edge*MOLT*lotti:+,.0f} € su {lotti} "
              f"lott{'o' if lotti==1 else 'i'}")
    st.caption(f"Premio incassato **{premio_tot:,.0f} €** · perdita massima "
               f"**{perdita_max:,.0f} €** ({100*perdita_max/capitale:.0f}% del capitale) · "
               f"la long put entra a **{100*(Kpw/P0-1):+.2f}%** dallo spot · "
               f"pareggio a **{100*((Kps - net_exe)/P0 - 1):+.2f}%**")
    st.caption(f"Pavimento con HAR corretto (informativo): "
               f"**{'POS' if net_exe >= net_min_pav_c else 'NEG'}** "
               f"({net_exe:.0f} contro {net_min_pav_c:.0f} pt).")
    if verdetto == "POS":
        st.success(f"✅ OPERATA — {lotti} lott{'o' if lotti==1 else 'i'}, margine "
                   f"{margine:,.0f} €. Net {net_exe:.0f} pt contro un pavimento di "
                   f"{net_min_pav:.0f}.")
    elif verdetto == "MARGINE":
        st.error(f"⛔ margine {margine:,.0f} € sopra il capitale {capitale:,.0f} €.")
    else:
        st.error(f"⛔ Net eseguito {net_exe:.0f} pt SOTTO il pavimento ({net_min_pav:.0f} pt).")
elif saltata:
    st.warning(f"Settimana saltata: il bid minimo di {bid_min_short:.0f} pt non e' "
               f"stato raggiunto. La riga si registra comunque come NEG.")
else:
    st.info("Inserisci i due prezzi eseguiti — prima l'ala, poi la short — oppure spunta "
            "la casella qui sopra se il mercato non paga.")

ivw = iv_imp(le if le > 0 else ml, P0, Kpw, "p")
iv_mod = float(M["cum"][SCELTO].std()) * VRP_MARKUP
if ivw == ivw and iv_mod > 0:
    st.caption(f"Ala long put: IV implicita {ivw*100:.2f}% contro {iv_mod*100:.2f}% del "
               f"modello = {ivw/iv_mod:.2f}x — costo {ml:.0f} pt contro fair {flr:.0f} pt")

# Delta dello spread. BS qui e' unita' di misura, non modello di prezzo.
ivs = iv_imp(pe, P0, Kps, "p") if pe > 0 else float("nan")
if ivs == ivs and ivw == ivw and ivs > 0 and ivw > 0:
    d_s = -norm.cdf(-((np.log(P0 / Kps) + 0.5 * ivs * ivs) / ivs))
    d_l = -norm.cdf(-((np.log(P0 / Kpw) + 0.5 * ivw * ivw) / ivw))
    d_net = -(d_s - d_l)
    st.caption(f"Delta dello spread **{d_net:+.3f}** per lotto → "
               f"**{d_net * MOLT * lotti:+.1f} € per punto di indice** su {lotti} "
               f"lott{'o' if lotti == 1 else 'i'}. Storico: mediana 0,278 per lotto.")

# ---------------- riga per il foglio ----------------
st.subheader("Riga per il foglio")
if verdetto == "—":
    st.info("La riga si genera quando la settimana e' chiusa: o con i due prezzi "
            "eseguiti, o spuntando la casella della settimana saltata.")
    st.stop()

si_opera = (verdetto == "POS")
riga = riga_registro(verdetto, f"{le:.0f}" if si_opera else "", f"{pe:.0f}" if si_opera else "",
                     net_equo_r, lotti if si_opera else 0)
st.code("\t".join(riga), language=None)
if not si_opera:
    st.caption("⚠️ Settimana NON operata: eseguiti vuoti e lotti a zero. La riga si "
               "registra comunque.")
st.caption("Incolla nella cella **A** della prima riga vuota del foglio *Diario*: sono "
           "14 colonne, **A→N** (K regime del drift, L strike dell'altra variante, M "
           "probabilita' di perdita massima corretta, N net equo corretto). Il venerdi' di "
           "scadenza si compilano **O** (data), **P** (risultato della long put in €) e "
           "**Q** (risultato della short put in €) dalla schermata di riepilogo Directa. "
           "**R** serve solo quando entrambe le gambe scadono a zero. Da **S** in poi e' "
           "tutto calcolato.")
