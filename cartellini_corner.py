"""
Cartellini e corner in Serie A: quanto si possono prevedere?

Modello: per ogni squadra due "tendenze" (es. cartellini presi / cartellini fatti
prendere agli avversari), piu' il fattore casa. Stimato solo sul passato,
con piu' peso alle partite recenti. Il totale partita segue una binomiale
negativa (piu' variabile di una Poisson, come succede davvero).

Confronto con un riferimento "ingenuo": la media del campionato recente.
Se il modello non batte nemmeno quello, non ha senso confrontarlo coi bookmaker.

Limite: i dati gratuiti NON hanno le quote cartellini/corner, quindi qui non
c'e' un conto in euro. Lo script calcola pero' le "quote giuste" per le
prossime partite (dati/fixtures.csv) da confrontare a mano con quelle vere.

Uso: python3 cartellini_corner.py
"""
import glob, json, math, os
import numpy as np
import pandas as pd

QUI = os.path.dirname(os.path.abspath(__file__))
EMIVITA = 240
FINESTRA = 730
RIDGE = 3.0
LINEE = {"cartellini": [2.5, 3.5, 4.5, 5.5], "corner": [7.5, 8.5, 9.5, 10.5, 11.5]}


def carica():
    pezzi = []
    for f in sorted(glob.glob(os.path.join(QUI, "dati", "I1_*.csv"))):
        d = pd.read_csv(f, encoding="utf-8-sig").dropna(subset=["HomeTeam", "HY", "HC"])
        pezzi.append(pd.DataFrame({
            "stagione": os.path.basename(f)[3:7],
            "data": pd.to_datetime(d["Date"], dayfirst=True, format="mixed"),
            "casa": d["HomeTeam"], "ospite": d["AwayTeam"],
            "cartellini_c": d["HY"] + d["HR"], "cartellini_o": d["AY"] + d["AR"],
            "corner_c": d["HC"], "corner_o": d["AC"],
        }))
    return pd.concat(pezzi, ignore_index=True).sort_values("data").reset_index(drop=True)


def pesi(s, oggi):
    return 0.5 ** ((oggi - s["data"]).dt.days.values / EMIVITA)


def stima(storico, oggi, cosa):
    s = storico[(storico.data < oggi) & (storico.data >= oggi - pd.Timedelta(days=FINESTRA))]
    squadre = sorted(set(s.casa) | set(s.ospite)); idx = {t: i for i, t in enumerate(squadre)}
    n, T = len(s), len(squadre); P = 2 + 2 * T
    w = pesi(s, oggi); W = np.concatenate([w, w])
    X = np.zeros((2 * n, P)); y = np.concatenate([s[cosa + "_c"].values, s[cosa + "_o"].values])
    ic, io, r = s.casa.map(idx).values, s.ospite.map(idx).values, np.arange(n)
    X[r, 0] = 1; X[r, 1] = 1; X[r, 2 + ic] = 1; X[r, 2 + T + io] = 1
    X[n + r, 0] = 1; X[n + r, 2 + io] = 1; X[n + r, 2 + T + ic] = 1
    pen = np.full(P, RIDGE); pen[:2] = 0
    beta = np.zeros(P); beta[0] = math.log(max(y.mean(), .1))
    for _ in range(30):
        mu = np.exp(X @ beta); z = X @ beta + (y - mu) / mu
        nuovo = np.linalg.solve(X.T @ (X * (W * mu)[:, None]) + np.diag(pen), X.T @ (W * mu * z))
        fatto = np.max(np.abs(nuovo - beta)) < 1e-6; beta = nuovo
        if fatto: break
    # dispersione del totale partita (varianza / media) sui residui pesati
    tot = s[cosa + "_c"].values + s[cosa + "_o"].values
    mu_tot = np.exp(X[:n] @ beta) + np.exp(X[n:] @ beta)
    disp = max(1.0, float(np.sum(w * (tot - mu_tot) ** 2 / mu_tot) / np.sum(w)))
    media_lega = float(np.sum(w * tot) / np.sum(w))
    disp_lega = max(1.0, float(np.sum(w * (tot - media_lega) ** 2) / np.sum(w) / media_lega))
    return dict(beta=beta, idx=idx, T=T, disp=disp, media_lega=media_lega, disp_lega=disp_lega)


def atteso(m, casa, ospite):
    b, idx, T = m["beta"], m["idx"], m["T"]
    a = lambda t: b[2 + idx[t]] if t in idx else 0.0
    d = lambda t: b[2 + T + idx[t]] if t in idx else 0.0
    return math.exp(b[0] + b[1] + a(casa) + d(ospite)) + math.exp(b[0] + a(ospite) + d(casa))


def prob_over(mu, disp, linea):
    """P(totale > linea) con binomiale negativa media mu, varianza disp*mu."""
    k = np.arange(0, int(linea) + 1)
    if disp <= 1.0001:
        logp = -mu + k * math.log(mu) - np.array([math.lgamma(i + 1) for i in k])
    else:
        rr = mu / (disp - 1); p = 1 / disp
        logp = np.array([math.lgamma(i + rr) - math.lgamma(rr) - math.lgamma(i + 1) for i in k]) \
            + rr * math.log(p) + k * math.log(1 - p)
    return float(1 - np.exp(logp).sum())


def backtest(df, cosa, inizio="1718"):
    righe = []
    for giorno, g in df[df.stagione >= inizio].groupby(df.data.dt.date):
        m = stima(df, pd.Timestamp(giorno), cosa)
        for _, p in g.iterrows():
            mu = atteso(m, p.casa, p.ospite)
            riga = dict(stagione=p.stagione, data=p.data, casa=p.casa, ospite=p.ospite,
                        reale=p[cosa + "_c"] + p[cosa + "_o"], atteso=mu, media_lega=m["media_lega"])
            for L in LINEE[cosa]:
                riga[f"mod_{L}"] = prob_over(mu, m["disp"], L)
                riga[f"base_{L}"] = prob_over(m["media_lega"], m["disp_lega"], L)
            righe.append(riga)
    return pd.DataFrame(righe)


def valuta(b, cosa):
    r = {"partite": len(b), "media_reale": round(b.reale.mean(), 2)}
    err = lambda col: float(np.mean((b.reale - b[col]) ** 2))
    r["errore_quadratico"] = {"modello": round(err("atteso"), 2), "media_campionato": round(err("media_lega"), 2)}
    r["varianza_spiegata_%"] = round(100 * (1 - err("atteso") / err("media_lega")), 1)
    r["linee"] = {}
    for L in LINEE[cosa]:
        y = (b.reale > L).values
        ll = lambda p: float(-np.mean(np.where(y, np.log(np.clip(p, 1e-9, 1)), np.log(np.clip(1 - p, 1e-9, 1)))))
        pm, pb = b[f"mod_{L}"].values, b[f"base_{L}"].values
        # calibrazione: quando il modello dice "molto probabile over", succede davvero?
        alto = pm >= 0.65; basso = pm <= 0.35
        r["linee"][f"over_{L}"] = {
            "frequenza_reale_%": round(100 * y.mean(), 1),
            "logloss_modello": round(ll(pm), 4), "logloss_media": round(ll(pb), 4),
            "miglioramento_%": round(100 * (1 - ll(pm) / ll(pb)), 2),
            "quando_modello_dice_over>=65%": {"partite": int(alto.sum()),
                                             "prob_media_modello_%": round(100 * pm[alto].mean(), 1) if alto.any() else None,
                                             "successo_reale_%": round(100 * y[alto].mean(), 1) if alto.any() else None},
            "quando_modello_dice_over<=35%": {"partite": int(basso.sum()),
                                             "prob_media_modello_%": round(100 * pm[basso].mean(), 1) if basso.any() else None,
                                             "successo_reale_%": round(100 * y[basso].mean(), 1) if basso.any() else None},
        }
    return r


def quote_giuste(df, cosa):
    fx = os.path.join(QUI, "dati", "fixtures.csv")
    if not os.path.exists(fx):
        return []
    f = pd.read_csv(fx, encoding="utf-8-sig"); f = f[f.Div == "I1"]
    oggi = pd.to_datetime(f.Date, dayfirst=True).min()
    m = stima(df, oggi, cosa); out = []
    for _, p in f.iterrows():
        mu = atteso(m, p.HomeTeam, p.AwayTeam)
        riga = {"data": p.Date, "partita": f"{p.HomeTeam} - {p.AwayTeam}", "attesi": round(mu, 1)}
        for L in LINEE[cosa]:
            po = prob_over(mu, m["disp"], L)
            riga[f"over {L}"] = round(1 / po, 2); riga[f"under {L}"] = round(1 / (1 - po), 2)
        out.append(riga)
    return out


if __name__ == "__main__":
    df = carica()
    ris, giuste = {}, {}
    for cosa in ("cartellini", "corner"):
        b = backtest(df, cosa)
        b.to_csv(os.path.join(QUI, f"previsioni_{cosa}.csv"), index=False)
        ris[cosa] = valuta(b, cosa)
        ris[cosa]["media_per_stagione"] = {"20" + s[:2] + "/" + s[2:]: round(v, 2)
                                           for s, v in b.groupby("stagione").reale.mean().items()}
        giuste[cosa] = quote_giuste(df, cosa)
    json.dump(ris, open(os.path.join(QUI, "risultati_cartellini_corner.json"), "w"), indent=2, ensure_ascii=False)
    for cosa, righe in giuste.items():
        pd.DataFrame(righe).to_csv(os.path.join(QUI, f"quote_giuste_{cosa}.csv"), index=False)
    print(json.dumps(ris, indent=2, ensure_ascii=False))
    for cosa, righe in giuste.items():
        print(f"\nQuote giuste {cosa} (senza margine):")
        print(pd.DataFrame(righe).to_string(index=False))
