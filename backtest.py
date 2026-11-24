"""
Backtest Serie A: un modello statistico semplice contro le quote dei bookmaker.

Modello: Poisson "attacco/difesa" (tipo Dixon-Coles semplificato).
 - Ogni squadra ha una forza in attacco e una in difesa, piu' il vantaggio casa.
 - Si stima SOLO con le partite giocate PRIMA della giornata da prevedere
   (niente sbirciate al futuro), dando piu' peso alle partite recenti.
 - Dai gol attesi si calcolano le probabilita' di 1 / X / 2 e Over/Under 2.5.

Poi si confronta con le quote:
 - Precisione: chi prevede meglio, il modello o il bookmaker?
 - Soldi: se avessi puntato 1 euro ogni volta che il modello vede "valore",
   quanto avrei guadagnato o perso?

Uso:  python3 backtest.py
Dati: cartella dati/ (file di football-data.co.uk)
"""
import glob, json, math, os
import numpy as np
import pandas as pd

QUI = os.path.dirname(os.path.abspath(__file__))
EMIVITA_GIORNI = 240      # una partita di 240 giorni fa pesa la meta'
FINESTRA_GIORNI = 730     # uso al massimo gli ultimi 2 anni
RIDGE = 2.0               # tira le squadre con pochi dati verso la media
MAXGOL = 10


# ---------------------------------------------------------------- dati
def carica():
    righe = []
    for f in sorted(glob.glob(os.path.join(QUI, "dati", "I1_*.csv"))):
        stagione = os.path.basename(f)[3:7]
        d = pd.read_csv(f, encoding="utf-8-sig")
        d = d.dropna(subset=["HomeTeam", "AwayTeam"])
        d["Date"] = pd.to_datetime(d["Date"], dayfirst=True, format="mixed")
        g = lambda *nomi: next((d[n] for n in nomi if n in d), pd.Series(np.nan, index=d.index))
        righe.append(pd.DataFrame({
            "stagione": stagione, "data": d["Date"],
            "casa": d["HomeTeam"], "ospite": d["AwayTeam"],
            "gc": d["FTHG"], "go": d["FTAG"],
            # quote 1X2 Bet365 (pre-partita, quelle che vede uno scommettitore italiano)
            "b365_1": d["B365H"], "b365_x": d["B365D"], "b365_2": d["B365A"],
            # migliore quota sul mercato (chi gira tra piu' bookmaker)
            "max_1": g("MaxH", "BbMxH"), "max_x": g("MaxD", "BbMxD"), "max_2": g("MaxA", "BbMxA"),
            # Pinnacle chiusura (il riferimento piu' "furbo" del mercato)
            "pin_1": g("PSCH", "PSH"), "pin_x": g("PSCD", "PSD"), "pin_2": g("PSCA", "PSA"),
            # Over/Under 2.5 (media e migliore del mercato)
            "avg_o": g("Avg>2.5", "BbAv>2.5"), "avg_u": g("Avg<2.5", "BbAv<2.5"),
            "max_o": g("Max>2.5", "BbMx>2.5"), "max_u": g("Max<2.5", "BbMx<2.5"),
        }))
    df = pd.concat(righe, ignore_index=True).sort_values("data").reset_index(drop=True)
    return df


# ---------------------------------------------------------------- modello
def stima(storico, oggi):
    """Regressione di Poisson pesata (IRLS) su attacco/difesa/vantaggio casa."""
    s = storico[(storico["data"] < oggi) & (storico["data"] >= oggi - pd.Timedelta(days=FINESTRA_GIORNI))]
    squadre = sorted(set(s["casa"]) | set(s["ospite"]))
    idx = {t: i for i, t in enumerate(squadre)}
    n, T = len(s), len(squadre)
    eta = (oggi - s["data"]).dt.days.values
    w = 0.5 ** (eta / EMIVITA_GIORNI)
    # due righe per partita: gol casa, gol ospite
    P = 2 + 2 * T  # intercetta, casa, attacchi, difese
    X = np.zeros((2 * n, P)); y = np.zeros(2 * n); W = np.concatenate([w, w])
    ic = s["casa"].map(idx).values; io = s["ospite"].map(idx).values
    r = np.arange(n)
    X[r, 0] = 1; X[r, 1] = 1; X[r, 2 + ic] = 1; X[r, 2 + T + io] = 1; y[:n] = s["gc"].values
    X[n + r, 0] = 1;          X[n + r, 2 + io] = 1; X[n + r, 2 + T + ic] = 1; y[n:] = s["go"].values
    pen = np.full(P, RIDGE); pen[:2] = 0
    beta = np.zeros(P); beta[0] = math.log(max(y.mean(), 0.1))
    for _ in range(30):
        mu = np.exp(X @ beta)
        z = X @ beta + (y - mu) / mu
        A = X.T @ (X * (W * mu)[:, None]) + np.diag(pen)
        b = X.T @ (W * mu * z)
        nuovo = np.linalg.solve(A, b)
        if np.max(np.abs(nuovo - beta)) < 1e-6:
            beta = nuovo; break
        beta = nuovo
    return beta, idx, T


def probabilita(beta, idx, T, casa, ospite):
    med_att = 0.0  # squadra nuova (neopromossa) = media, grazie al ridge
    a = lambda t: beta[2 + idx[t]] if t in idx else med_att
    d = lambda t: beta[2 + T + idx[t]] if t in idx else 0.0
    lc = math.exp(beta[0] + beta[1] + a(casa) + d(ospite))
    lo = math.exp(beta[0] + a(ospite) + d(casa))
    k = np.arange(MAXGOL + 1)
    fatt = np.array([math.factorial(i) for i in k])
    pc = np.exp(-lc) * lc ** k / fatt
    po = np.exp(-lo) * lo ** k / fatt
    M = np.outer(pc, po)
    p1, px, p2 = np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()
    tot = np.add.outer(k, k)
    pover = M[tot >= 3].sum()
    s = p1 + px + p2
    return dict(p1=p1 / s, px=px / s, p2=p2 / s, pover=pover, lc=lc, lo=lo)


# ---------------------------------------------------------------- backtest
def prevedi_tutto(df, inizio="1718"):
    out = []
    for giorno, gruppo in df[df["stagione"] >= inizio].groupby(df["data"].dt.date):
        oggi = pd.Timestamp(giorno)
        beta, idx, T = stima(df, oggi)
        for i, m in gruppo.iterrows():
            out.append({"i": i, **probabilita(beta, idx, T, m["casa"], m["ospite"])})
    p = pd.DataFrame(out).set_index("i")
    return df.join(p, how="inner")


def implicite(q1, qx, q2):
    inv = np.vstack([1 / q1, 1 / qx, 1 / q2])
    return inv / inv.sum(0), inv.sum(0) - 1  # probabilita' normalizzate, margine


def valuta(df):
    r = {}
    esito = np.select([df.gc > df.go, df.gc == df.go], [0, 1], 2)
    over = (df.gc + df.go >= 3).values
    P_mod = df[["p1", "px", "p2"]].values.T
    pb, marg_b365 = implicite(df.b365_1.values, df.b365_x.values, df.b365_2.values)
    pp, _ = implicite(df.pin_1.values, df.pin_x.values, df.pin_2.values)
    n = len(df)
    ll = lambda P: float(-np.mean(np.log(np.clip(P[esito, np.arange(n)], 1e-9, 1))))
    brier = lambda P: float(np.mean(((P - np.eye(3)[:, esito]) ** 2).sum(0)))
    acc = lambda P: float(np.mean(P.argmax(0) == esito))
    r["partite"] = n
    r["margine_medio_b365_1x2"] = float(np.nanmean(marg_b365))
    r["precisione"] = {
        nome: {"logloss": ll(P), "brier": brier(P), "azzecca_esito_piu_probabile": acc(P)}
        for nome, P in [("modello", P_mod), ("bet365", pb), ("pinnacle_chiusura", pp)]
    }
    # mix: quanto aiuta il modello se lo aggiungo al mercato?
    r["precisione"]["media_modello_e_bet365"] = {"logloss": ll(0.5 * P_mod + 0.5 * pb)}

    # strategie di scommessa (1 euro a puntata)
    def scommetti(nome, quote_cols, soglie=(0.0, 0.05, 0.10, 0.20)):
        Q = df[quote_cols].values.T
        res = {}
        for s in soglie:
            valore = P_mod * Q - 1 > s
            vinto = np.zeros_like(valore)
            vinto[esito, np.arange(n)] = True
            puntate = int(valore.sum())
            utile = float(np.nansum(np.where(valore, np.where(vinto, Q - 1, -1), 0)))
            res[f"valore>{int(s*100)}%"] = {"puntate": puntate, "utile": round(utile, 1),
                                           "roi_%": round(100 * utile / puntate, 1) if puntate else None}
        return res
    r["scommesse_1x2_bet365"] = scommetti("b365", ["b365_1", "b365_x", "b365_2"])
    r["scommesse_1x2_migliore_quota"] = scommetti("max", ["max_1", "max_x", "max_2"])

    # Over/Under 2.5 con quota media del mercato
    ok = df.avg_o.notna().values
    po = df.pover.values
    res = {}
    for s in (0.0, 0.05, 0.10):
        for lato, p, q, vince in [("over", po, df.avg_o.values, over), ("under", 1 - po, df.avg_u.values, ~over)]:
            v = ok & (p * q - 1 > s)
            utile = float(np.sum(np.where(vince[v], q[v] - 1, -1)))
            res[f"{lato}_valore>{int(s*100)}%"] = {"puntate": int(v.sum()), "utile": round(utile, 1),
                                                    "roi_%": round(100 * utile / v.sum(), 1) if v.sum() else None}
    r["scommesse_over_under_quota_media"] = res

    # strategie "da bar" per confronto
    Q = df[["b365_1", "b365_x", "b365_2"]].values.T
    fav = Q.argmin(0)
    utile_fav = float(np.sum(np.where(fav == esito, Q[fav, np.arange(n)] - 1, -1)))
    r["sempre_il_favorito_bet365"] = {"puntate": n, "utile": round(utile_fav, 1), "roi_%": round(100 * utile_fav / n, 1)}
    # puntare a caso 1 euro su tutti e tre gli esiti = costo puro del margine
    r["costo_del_margine_%"] = round(-100 * float(np.mean(marg_b365 / (1 + marg_b365))), 1)

    # schedine: ogni giornata, multipla con le N partite piu' "sicure" secondo il modello
    sched = {}
    df2 = df.assign(esito=esito, pmax=P_mod.max(0), scelta=P_mod.argmax(0))
    df2["quota_scelta"] = Q[df2.scelta.values, np.arange(n)]
    df2["giornata"] = df2["data"].dt.to_period("W")
    for N in (2, 3, 5):
        utile = 0; giocate = 0; vinte = 0
        for _, g in df2.groupby("giornata"):
            g = g.nlargest(N, "pmax")
            if len(g) < N:
                continue
            giocate += 1
            if (g.scelta == g.esito).all():
                vinte += 1; utile += float(np.prod(g.quota_scelta)) - 1
            else:
                utile -= 1
        sched[f"multipla_{N}_partite"] = {"schedine": giocate, "vinte": vinte, "utile": round(utile, 1),
                                          "roi_%": round(100 * utile / giocate, 1)}
    r["schedine_settimanali_bet365"] = sched

    # per stagione: ROI valore>5% su migliore quota
    per = {}
    Qm = df[["max_1", "max_x", "max_2"]].values.T
    for st in sorted(df.stagione.unique()):
        m = (df.stagione == st).values
        valore = (P_mod * Qm - 1 > 0.05) & m
        vinto = np.zeros_like(valore); vinto[esito, np.arange(n)] = True
        u = float(np.nansum(np.where(valore, np.where(vinto, Qm - 1, -1), 0)))
        per["20" + st[:2] + "/" + st[2:]] = {"puntate": int(valore.sum()), "utile": round(u, 1),
                                           "roi_%": round(100 * u / valore.sum(), 1) if valore.sum() else None}
    r["per_stagione_1x2_migliore_quota_valore>5%"] = per
    return r


if __name__ == "__main__":
    df = carica()
    df = df[df.gc.notna()].reset_index(drop=True)
    print(f"Partite caricate: {len(df)}")
    prev = prevedi_tutto(df)
    prev = prev.dropna(subset=["b365_1", "max_1"])
    prev[["pin_1", "pin_x", "pin_2"]] = prev[["pin_1", "pin_x", "pin_2"]].fillna(prev[["max_1", "max_x", "max_2"]].rename(columns=lambda c: c.replace("max", "pin")))
    prev.to_csv(os.path.join(QUI, "previsioni_backtest.csv"), index=False)
    ris = valuta(prev)
    with open(os.path.join(QUI, "risultati_backtest.json"), "w") as f:
        json.dump(ris, f, indent=2, ensure_ascii=False)
    print(json.dumps(ris, indent=2, ensure_ascii=False))
