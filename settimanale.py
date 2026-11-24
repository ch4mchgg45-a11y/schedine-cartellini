"""
Analisi automatica della giornata (gira su GitHub giovedi', venerdi', sabato).

1. Scarica i risultati aggiornati della stagione (football-data).
2. Legge le designazioni dal sito ufficiale AIA (con le variazioni).
3. Aggiorna lo storico arbitri con le partite appena giocate.
4. Calcola per ogni partita: cartellini attesi, probabilita', quota giusta
   e QUOTA MINIMA per giocare (giusta + 5%).
5. Scrive docs/dati.json, letto dalla pagina docs/index.html.

Modello: squadre + arbitro + equilibrio partita + derby (vedi LEGGIMI.md).
Uso: python3 settimanale.py
"""
import json, math, os, subprocess
from collections import defaultdict
from datetime import date, datetime, timezone
import numpy as np
import pandas as pd
import aia
import backtest
from cartellini_corner import carica, stima, atteso, prob_over
from cartellini_v2 import DERBY, poisson_offset

QUI = os.path.dirname(os.path.abspath(__file__))
DATI = os.path.join(QUI, "dati")
STORICO = os.path.join(DATI, "storico_cartellini.csv")
DESIGNAZIONI = os.path.join(DATI, "designazioni_aia.csv")
PUBBLICATE = os.path.join(DATI, "previsioni_pubblicate.csv")
K_ARB, H_ARB = 25.0, 540.0
MARGINE_SICUREZZA = 0.05
LINEE = [2.5, 3.5, 4.5, 5.5]


def stagione_corrente(oggi):
    a = oggi.year if oggi.month >= 7 else oggi.year - 1
    return f"{a % 100:02d}{(a + 1) % 100:02d}"


def scarica_stagione(stag):
    dest = os.path.join(DATI, f"I1_{stag}.csv")
    tmp = dest + ".tmp"
    r = subprocess.run(["curl", "-sfL", "--max-time", "60", "-o", tmp,
                        f"https://www.football-data.co.uk/mmz4281/{stag}/I1.csv"])
    if r.returncode == 0 and os.path.getsize(tmp) > 1000:
        os.replace(tmp, dest)
    elif os.path.exists(tmp):
        os.remove(tmp)
    subprocess.run(["curl", "-sfL", "--max-time", "60", "-o", os.path.join(DATI, "fixtures.csv"),
                    "https://www.football-data.co.uk/fixtures.csv"])


def norm(s):
    return "".join(c for c in s.lower() if c.isalpha())


def mappa_squadra(nome_aia, squadre_fd):
    n = norm(nome_aia)
    for s in squadre_fd:
        if norm(s) == n:
            return s
    for s in squadre_fd:
        if norm(s) in n or n in norm(s):
            return s
    return nome_aia.title()


def mappa_arbitro(nome_aia, conosciuti):
    """'FERRIERI CAPUTI' -> 'Maria Sole Ferrieri Caputi' (dallo storico)."""
    pulito = " ".join(t for t in nome_aia.split() if "." not in t).strip().upper()
    cand = [c for c in conosciuti if c.upper() == pulito or c.upper().endswith(" " + pulito)]
    return cand[-1] if cand else pulito.title()


def squilibrio_da_quote(q1, qx, q2):
    inv = np.array([1 / q1, 1 / qx, 1 / q2]); p = inv / inv.sum()
    return float(abs(p[0] - p[2]))


def carica_storico():
    if os.path.exists(STORICO):
        s = pd.read_csv(STORICO, dtype={"stagione": str}, parse_dates=["data"])
    else:  # prima volta: dal lavoro di backtest
        b = pd.read_csv(os.path.join(QUI, "previsioni_cartellini_v2.csv"), dtype={"stagione": str}, parse_dates=["data"])
        s = b[["stagione", "data", "casa", "ospite", "arbitro", "reale", "atteso", "squilibrio", "derby"]].copy()
    return s


def main():
    oggi = date.today()
    stag = stagione_corrente(oggi)
    scarica_stagione(stag)
    df = carica()  # cartellini/corner di tutte le stagioni
    goals = backtest.carica(); goals = goals[goals.gc.notna()].reset_index(drop=True)
    storico = carica_storico()
    conosciuti = list(dict.fromkeys(storico.arbitro.dropna().tolist()))
    squadre_fd = sorted(set(df[df.stagione == stag].casa) | set(df[df.stagione == stag].ospite)) or \
        sorted(set(df.casa))

    # --- designazioni AIA (giornata piu' recente, variazioni applicate)
    errore_aia = None
    try:
        g = aia.giornata_corrente()
    except Exception as e:  # sito non raggiungibile: si lavora con l'archivio
        g, errore_aia = None, str(e)
    arch = pd.read_csv(DESIGNAZIONI, dtype={"stagione": str}) if os.path.exists(DESIGNAZIONI) else \
        pd.DataFrame(columns=["stagione", "giornata", "data", "ora", "casa", "ospite", "arbitro", "arbitro_aia"])
    if g:
        nuove = pd.DataFrame([{
            "stagione": stag, "giornata": g["giornata"], "data": p["data"], "ora": p["ora"],
            "casa": mappa_squadra(p["casa_aia"], squadre_fd), "ospite": mappa_squadra(p["ospite_aia"], squadre_fd),
            "arbitro": mappa_arbitro(p["arbitro_aia"], conosciuti), "arbitro_aia": p["arbitro_aia"]} for p in g["partite"]])
        arch = pd.concat([arch[~((arch.stagione == stag) & (arch.giornata == g["giornata"]))], nuove], ignore_index=True)
        arch.to_csv(DESIGNAZIONI, index=False)

    # --- aggiorno lo storico con le partite giocate che hanno un arbitro AIA
    giocate = df[df.stagione == stag]
    q = goals[goals.stagione == stag][["casa", "ospite", "b365_1", "b365_x", "b365_2"]]
    giocate = giocate.merge(q, on=["casa", "ospite"], how="left")
    gia = set(zip(storico.stagione, storico.casa, storico.ospite))
    aggiunte = 0
    for _, r in giocate.iterrows():
        if (stag, r.casa, r.ospite) in gia:
            continue
        d = arch[(arch.stagione == stag) & (arch.casa == r.casa) & (arch.ospite == r.ospite)]
        if d.empty:
            continue
        m = stima(df, r.data, "cartellini")
        sq = squilibrio_da_quote(r.b365_1, r.b365_x, r.b365_2) if pd.notna(r.b365_1) else float(storico.squilibrio.median())
        storico.loc[len(storico)] = [stag, r.data, r.casa, r.ospite, d.iloc[-1].arbitro,
                                     r.cartellini_c + r.cartellini_o, atteso(m, r.casa, r.ospite), sq,
                                     int({r.casa, r.ospite} in DERBY)]
        aggiunte += 1
    storico = storico.sort_values("data").reset_index(drop=True)
    storico.to_csv(STORICO, index=False)

    # --- modello: pesi equilibrio/derby e dispersione da tutto lo storico
    fa_hist = fattori_arbitro_passati(storico)
    mu_arb = storico.atteso.values * fa_hist
    X = storico[["squilibrio", "derby"]].values.astype(float)
    coef = poisson_offset(X, storico.reale.values.astype(float), np.log(mu_arb))
    mu_all = mu_arb * np.exp(X @ coef)
    disp = max(1.0, float(np.mean((storico.reale - mu_all) ** 2 / mu_all)))
    storia = defaultdict(list)
    for r in storico.itertuples():
        if isinstance(r.arbitro, str):
            storia[r.arbitro].append((r.data, r.reale, r.atteso))

    # --- previsioni per la giornata designata
    giornata = int(arch[arch.stagione == stag].giornata.max()) if (arch.stagione == stag).any() else None
    partite_out = []
    fixtures = leggi_fixtures()
    if giornata:
        des = arch[(arch.stagione == stag) & (arch.giornata == giornata)].sort_values(["data", "ora"])
        giocate_set = set(zip(giocate.casa, giocate.ospite))
        pubblicate = {}
        if os.path.exists(PUBBLICATE):
            pp = pd.read_csv(PUBBLICATE, dtype={"stagione": str})
            pubblicate = {(r.stagione, r.casa, r.ospite): float(r.attesi) for r in pp.itertuples()}
        m = stima(df, pd.Timestamp(oggi), "cartellini")
        mg = backtest.stima(goals, pd.Timestamp(oggi))
        for _, p in des.iterrows():
            base = atteso(m, p.casa, p.ospite)
            h = storia.get(p.arbitro, [])
            if h:
                w = np.array([0.5 ** ((pd.Timestamp(oggi) - x[0]).days / H_ARB) for x in h])
                fa = float((np.sum(w * [x[1] for x in h]) + K_ARB) / (np.sum(w * [x[2] for x in h]) + K_ARB))
                ultime = [x for x in h if x[0] >= pd.Timestamp(oggi) - pd.Timedelta(days=730)]
                media_arb = float(np.mean([x[1] for x in ultime])) if ultime else None
            else:
                fa, media_arb = 1.0, None
            fx = fixtures.get((p.casa, p.ospite))
            if fx:
                sq, fonte_sq = squilibrio_da_quote(*fx), "quote 1X2"
            else:
                pr = backtest.probabilita(*mg, p.casa, p.ospite)
                sq, fonte_sq = float(abs(pr["p1"] - pr["p2"])), "modello gol"
            derby = int({p.casa, p.ospite} in DERBY)
            mu = base * fa * math.exp(coef[0] * sq + coef[1] * derby)
            linee = []
            for L in LINEE:
                po = prob_over(mu, disp, L)
                linee.append({"linea": L,
                              "over": {"prob": round(100 * po, 1), "giusta": round(1 / po, 2),
                                       "minima": round((1 + MARGINE_SICUREZZA) / po, 2)},
                              "under": {"prob": round(100 * (1 - po), 1), "giusta": round(1 / (1 - po), 2),
                                        "minima": round((1 + MARGINE_SICUREZZA) / (1 - po), 2)}})
            reali, attesi_pubbl = None, None
            if (p.casa, p.ospite) in giocate_set:
                rr = giocate[(giocate.casa == p.casa) & (giocate.ospite == p.ospite)].iloc[0]
                reali = int(rr.cartellini_c + rr.cartellini_o)
                attesi_pubbl = pubblicate.get((stag, p.casa, p.ospite))
            partite_out.append({
                "data": p.data, "ora": p.ora, "casa": p.casa, "ospite": p.ospite,
                "arbitro": p.arbitro, "arbitro_partite": len(h), "arbitro_fattore": round(fa, 2),
                "arbitro_media_cartellini": round(media_arb, 2) if media_arb is not None else None,
                "attesi_squadre": round(base, 2), "attesi": round(mu, 2), "squilibrio": round(sq, 2),
                "fonte_squilibrio": fonte_sq, "derby": bool(derby), "linee": linee,
                "giocata": reali is not None, "cartellini_reali": reali, "attesi_pubblicati": attesi_pubbl})
        salva_pubblicate(stag, giornata, partite_out)

    out = {
        "aggiornato": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "stagione": f"20{stag[:2]}/{stag[2:]}", "giornata": giornata,
        "designazioni_pubblicate": g["pubblicata"] if g else None,
        "variazioni": g["variazioni"] if g else [],
        "errore_aia": errore_aia, "margine_sicurezza_%": int(100 * MARGINE_SICUREZZA),
        "partite": partite_out,
        "verifica": verifica(stag),
        "media_cartellini_stagione": round(float(giocate.eval("cartellini_c + cartellini_o").mean()), 2) if len(giocate) else None,
        "partite_storico_aggiunte": aggiunte,
    }
    os.makedirs(os.path.join(QUI, "docs"), exist_ok=True)
    with open(os.path.join(QUI, "docs", "dati.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)
    print(f"Giornata {giornata}: {len(partite_out)} partite, storico +{aggiunte}, errore AIA: {errore_aia}")


def fattori_arbitro_passati(storico):
    """Fattore arbitro per ogni partita dello storico, usando solo le partite precedenti."""
    st = defaultdict(list); out = np.ones(len(storico))
    for i, r in enumerate(storico.itertuples()):
        h = st.get(r.arbitro)
        if h:
            w = np.array([0.5 ** ((r.data - x[0]).days / H_ARB) for x in h])
            out[i] = (np.sum(w * [x[1] for x in h]) + K_ARB) / (np.sum(w * [x[2] for x in h]) + K_ARB)
        if isinstance(r.arbitro, str):
            st[r.arbitro].append((r.data, r.reale, r.atteso))
    return out


def leggi_fixtures():
    f = os.path.join(DATI, "fixtures.csv")
    if not os.path.exists(f):
        return {}
    try:
        x = pd.read_csv(f, encoding="utf-8-sig"); x = x[x.Div == "I1"]
        return {(r.HomeTeam, r.AwayTeam): (r.B365H, r.B365D, r.B365A) for r in x.itertuples() if pd.notna(r.B365H)}
    except Exception:
        return {}


def salva_pubblicate(stag, giornata, partite):
    """Tiene traccia di cosa abbiamo previsto PRIMA della partita (ultima versione prima del fischio)."""
    old = pd.read_csv(PUBBLICATE, dtype={"stagione": str}) if os.path.exists(PUBBLICATE) else pd.DataFrame()
    nuove = pd.DataFrame([{"stagione": stag, "giornata": giornata, "data": p["data"], "casa": p["casa"],
                           "ospite": p["ospite"], "arbitro": p["arbitro"], "attesi": p["attesi"],
                           **{f"p_over_{l['linea']}": l["over"]["prob"] for l in p["linee"]}}
                          for p in partite if not p["giocata"]])
    if nuove.empty:
        return
    if not old.empty:
        chiave = set(zip(nuove.stagione, nuove.casa, nuove.ospite))
        old = old[[k not in chiave for k in zip(old.stagione, old.casa, old.ospite)]]
    pd.concat([old, nuove], ignore_index=True).to_csv(PUBBLICATE, index=False)


def verifica(stag):
    """Previsioni pubblicate vs cartellini veri, per tutte le partite gia' giocate."""
    if not os.path.exists(PUBBLICATE):
        return None
    p = pd.read_csv(PUBBLICATE, dtype={"stagione": str})
    s = pd.read_csv(STORICO, dtype={"stagione": str})
    v = p.merge(s[["stagione", "casa", "ospite", "reale"]], on=["stagione", "casa", "ospite"])
    if v.empty:
        return {"partite": 0}
    righe = []
    for L in LINEE:
        col = f"p_over_{L}"
        if col in v:
            righe.append({"linea": L, "prob_media_over": round(float(v[col].mean()), 1),
                          "over_reali_%": round(float(100 * (v.reale > L).mean()), 1)})
    ult = v[v.giornata == v.giornata.max()]
    return {"partite": int(len(v)), "attesi_media": round(float(v.attesi.mean()), 2),
            "reali_media": round(float(v.reale.mean()), 2), "linee": righe,
            "ultima_giornata": int(ult.giornata.iloc[0]),
            "ultima": [{"casa": r.casa, "ospite": r.ospite, "arbitro": r.arbitro, "attesi": r.attesi, "reali": int(r.reale)}
                       for r in ult.itertuples()]}


if __name__ == "__main__":
    main()
