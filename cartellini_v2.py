"""
Cartellini v2: tre correzioni al modello squadre + arbitro, verificate su partite mai viste.

1. Tendenza di stagione: i cartellini stanno calando, il modello resta indietro.
   Correzione = cartellini veri / previsti nelle ultime N partite (solo passato).
2. Equilibrio partita: dalle quote 1X2 (Bet365) si sa se la partita e' equilibrata.
   Ipotesi: partite equilibrate -> piu' cartellini. Piu' flag derby.
3. Peso dell'arbitro (prudenza K ed emivita) scelto sulle stagioni vecchie.

Impara su 2018/19 - 2022/23, verifica su 2023/24 - 2026/27.
Richiede: previsioni_cartellini_arbitro.csv (da arbitri_cartellini.py).
Uso: python3 cartellini_v2.py
"""
import json, math, os
from collections import defaultdict
import numpy as np
import pandas as pd
from cartellini_corner import prob_over, LINEE
import backtest

QUI = os.path.dirname(os.path.abspath(__file__))
TRAIN = ("1819", "1920", "2021", "2122", "2223")
TEST = ("2324", "2425", "2526", "2627")
N_TREND, K_TREND = 120, 40.0
DERBY = [{"Inter", "Milan"}, {"Roma", "Lazio"}, {"Genoa", "Sampdoria"}, {"Juventus", "Torino"},
         {"Verona", "Chievo"}, {"Fiorentina", "Juventus"}, {"Inter", "Juventus"}, {"Napoli", "Roma"},
         {"Atalanta", "Brescia"}, {"Cagliari", "Sassuolo"}]  # rivalita' storiche (l'ultima e' un falso controllo)
DERBY = DERBY[:-1]


def fattore_arbitro(b, K, H):
    storia = defaultdict(list); out = np.ones(len(b))
    for i, r in enumerate(b.itertuples()):
        h = storia.get(r.arbitro)
        if h:
            a = np.array(h); w = 0.5 ** ((r.data - pd.to_datetime(a[:, 0])).days.values / H) if H else 1.0
            out[i] = (np.sum(w * a[:, 1].astype(float)) + K) / (np.sum(w * a[:, 2].astype(float)) + K)
        if isinstance(r.arbitro, str):
            storia[r.arbitro].append((r.data, r.reale, r.atteso))
    return out


def tendenza(b, col):
    """Rapporto veri/previsti sulle ultime N partite giocate in date precedenti."""
    out = np.ones(len(b)); date = b.data.values
    reale, prev = b.reale.values, b[col].values
    for i in range(len(b)):
        j = np.searchsorted(date, date[i])  # partite con data < oggi
        a = max(0, j - N_TREND)
        out[i] = (reale[a:j].sum() + K_TREND) / (prev[a:j].sum() + K_TREND)
    return out


def poisson_offset(X, y, off, ridge=1.0):
    beta = np.zeros(X.shape[1])
    for _ in range(50):
        mu = np.exp(off + X @ beta); z = X @ beta + (y - mu) / mu
        nuovo = np.linalg.solve(X.T @ (X * mu[:, None]) + ridge * np.eye(X.shape[1]), X.T @ (mu * z))
        if np.max(np.abs(nuovo - beta)) < 1e-8:
            return nuovo
        beta = nuovo
    return beta


def logloss_linee(y_tot, mu, disp):
    out = {}
    for L in LINEE["cartellini"]:
        p = np.array([prob_over(m, disp, L) for m in mu]); y = y_tot > L
        out[L] = float(-np.mean(np.where(y, np.log(np.clip(p, 1e-9, 1)), np.log(np.clip(1 - p, 1e-9, 1)))))
    return out


def prepara():
    b = pd.read_csv(os.path.join(QUI, "previsioni_cartellini_arbitro.csv"), dtype={"stagione": str}, parse_dates=["data"])
    b = b.sort_values("data").reset_index(drop=True)
    q = backtest.carica()[["stagione", "casa", "ospite", "b365_1", "b365_x", "b365_2"]]
    b = b.merge(q, on=["stagione", "casa", "ospite"], how="left")
    inv = 1 / b[["b365_1", "b365_x", "b365_2"]].values
    p = inv / inv.sum(1, keepdims=True)
    b["squilibrio"] = np.abs(p[:, 0] - p[:, 2])
    b["squilibrio"] = b.squilibrio.fillna(b.squilibrio.median())
    b["derby"] = [int({c, o} in DERBY) for c, o in zip(b.casa, b.ospite)]
    return b


def costruisci(b, K, H, coef=None):
    b = b.copy()
    b["f_arb2"] = fattore_arbitro(b, K, H)
    b["mu_arb"] = b.atteso * b.f_arb2
    b["trend"] = tendenza(b, "mu_arb")
    b["mu_trend"] = b.mu_arb * b.trend
    return b


if __name__ == "__main__":
    b = prepara()
    tr_mask, te_mask = b.stagione.isin(TRAIN).values, b.stagione.isin(TEST).values

    # 3. scelta prudenza/emivita arbitro SOLO su train (devianza Poisson)
    griglia = {}
    for K in (10, 25, 50, 100):
        for H in (270, 540, 1080, None):
            mu = costruisci(b, K, H).mu_arb.values[tr_mask]; y = b.reale.values[tr_mask]
            griglia[(K, H)] = float(np.mean(mu - y * np.log(mu)))
    K, H = min(griglia, key=griglia.get)
    b = costruisci(b, K, H)

    # 2. equilibrio + derby, pesi imparati su train
    feats = ["squilibrio", "derby"]
    X = b[feats].values.astype(float)
    coef = poisson_offset(X[tr_mask], b.reale.values[tr_mask], np.log(b.mu_trend.values[tr_mask]))
    b["mu_v2"] = b.mu_trend * np.exp(X @ coef)

    disp = lambda col: max(1.0, float(np.mean((b.reale[tr_mask] - b[col][tr_mask]) ** 2 / b[col][tr_mask])))
    te = b[te_mask]
    modelli = {"v1_squadre_arbitro": ("atteso_arb", disp("atteso_arb")),
               "+_arbitro_tarato": ("mu_arb", disp("mu_arb")),
               "+_tendenza_stagione": ("mu_trend", disp("mu_trend")),
               "v2_+_equilibrio_derby": ("mu_v2", disp("mu_v2"))}
    base = logloss_linee(te.reale.values, te.media_lega.values, disp("media_lega"))
    r = {"scelta_arbitro": {"prudenza_K": K, "emivita_giorni": H},
         "pesi": {"squilibrio_da_0_a_1": f"{100 * (math.exp(coef[0]) - 1):+.1f}% cartellini",
                  "derby": f"{100 * (math.exp(coef[1]) - 1):+.1f}% cartellini"},
         "test_partite": int(te_mask.sum()), "test": {}}
    for nome, (col, d) in modelli.items():
        ll = logloss_linee(te.reale.values, te[col].values, d)
        r["test"][nome] = {
            "media_prevista_vs_reale": f"{te[col].mean():.2f} vs {te.reale.mean():.2f}",
            "errore_quadratico": round(float(np.mean((te.reale - te[col]) ** 2)), 3),
            **{f"miglioramento_vs_media_over_{L}_%": round(100 * (1 - ll[L] / base[L]), 2) for L in ll}}
    r["test_per_stagione_media_prevista_v1_v2_reale"] = {
        s: [round(g.atteso_arb.mean(), 2), round(g.mu_v2.mean(), 2), round(g.reale.mean(), 2)] for s, g in te.groupby("stagione")}
    # squilibrio: cartellini veri in partite equilibrate vs sbilanciate (tutto il periodo)
    b["fascia"] = pd.qcut(b.squilibrio, 3, labels=["equilibrate", "medie", "sbilanciate"])
    r["cartellini_reali_su_previsti_v1_per_equilibrio"] = {
        str(k): round(float(g.reale.sum() / g.atteso_arb.sum()), 3) for k, g in b.groupby("fascia", observed=True)}
    json.dump(r, open(os.path.join(QUI, "risultati_cartellini_v2.json"), "w"), indent=2, ensure_ascii=False)
    params = {"K": K, "H": H, "coef": coef.tolist(), "feats": feats, "disp": disp("mu_v2"),
              "N_TREND": N_TREND, "K_TREND": K_TREND}
    json.dump(params, open(os.path.join(QUI, "parametri_cartellini_v2.json"), "w"), indent=2)
    b.to_csv(os.path.join(QUI, "previsioni_cartellini_v2.csv"), index=False)
    print(json.dumps(r, indent=2, ensure_ascii=False))
