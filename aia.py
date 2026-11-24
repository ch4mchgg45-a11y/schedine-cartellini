"""
Designazioni arbitrali Serie A dal sito ufficiale AIA (aia-figc.it).

lista_serie_a()  -> pubblicazioni Serie A (designazioni + variazioni), le piu' recenti prima
leggi(id)        -> partite con data/ora e arbitro
giornata_corrente() -> designazioni dell'ultima giornata pubblicata, variazioni applicate
"""
import html, re, subprocess
from datetime import date

BASE = "https://www.aia-figc.it"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"
GIORNI = ("lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica",
          "lunedi", "martedi", "mercoledi", "giovedi", "venerdi")


def scarica(url):
    r = subprocess.run(["curl", "-sfL", "--max-time", "60", "-A", UA, url], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"AIA non raggiungibile: {url}")
    return r.stdout.decode("utf-8", errors="ignore")


def testo(pagina):
    t = html.unescape(re.sub(r"<[^>]+>", "\n", pagina)).replace("﻿", "")
    t = t.replace("\xa0", " ")
    out = []
    for l in t.splitlines():
        l = l.strip()
        # "NAPOLI – TORINO    Sabato 21/11 h. 20.45" -> due righe
        m = re.match(r"^(.+?[–—-].+?)\s{2,}(.*(?:\d{1,2}/\d{1,2}|h\.).*)$", l)
        if m and RIGA_PARTITA.match(m.group(1).strip()):
            out += [m.group(1).strip(), m.group(2).strip()]
        elif l:
            out.append(l)
    return out


def lista_serie_a():
    p = scarica(BASE + "/designazioni/can/")
    out = []
    for id_, titolo in re.findall(r'href="/dettaglio\.asp\?ID=(\d+)">([^<]+)</a>', p):
        t = html.unescape(titolo).strip().upper()
        if "SERIE A" not in t or "DESIGNAZION" not in t:
            continue
        g = re.search(r"(\d+)\s*[AaªÂ°]*\s*GIORNATA", t)
        out.append({"id": int(id_), "titolo": t, "variazione": "VARIAZION" in t,
                    "giornata": int(g.group(1)) if g else None})
    return sorted(out, key=lambda x: -x["id"])


RIGA_PARTITA = re.compile(r"^([A-ZÀ-Ü'\. ]{3,})\s+[–—-]\s+([A-ZÀ-Ü'\. ]{3,})$")
RIGA_DATA = re.compile(r"(?i)(\d{1,2})/(\d{1,2})")
RIGA_ORA = re.compile(r"(?i)h\.?\s*(\d{1,2})[.:](\d{2})")


def leggi(id_, anno_inizio=None):
    righe = testo(scarica(f"{BASE}/dettaglio.asp?ID={id_}"))
    pubblicata = next((l for l in righe if re.fullmatch(r"\d{2}/\d{2}/\d{4}", l)), None)
    oggi = date.today()
    # ogni partita: riga "CASA – OSPITE", [data/ora], ARBITRO [(foto)], assistenti, "IV: ..."
    blocchi = []
    for k, l in enumerate(righe):
        if not re.match(r"^IV\s*:", l):
            continue
        j = k - 2  # salto la riga assistenti
        while j > 0 and re.fullmatch(r"[()\s]*(foto)?[()\s]*", righe[j], re.I):
            j -= 1
        arbitro = re.sub(r"\(.*", "", righe[j]).strip()
        data_txt = []
        j -= 1
        while j >= 0 and not RIGA_PARTITA.match(righe[j]):
            data_txt.insert(0, righe[j]); j -= 1
        if j < 0:
            continue
        m = RIGA_PARTITA.match(righe[j])
        blocchi.append((m.group(1).strip(), m.group(2).strip(), " ".join(data_txt), arbitro))
    partite, data_corr, ora_prec = [], None, None
    for casa, ospite, dt, arbitro in blocchi:
        d = RIGA_DATA.search(dt)
        o = RIGA_ORA.search(dt)
        ora = f"{int(o.group(1)):02d}:{o.group(2)}" if o else None
        if not d and data_corr and ora and ora_prec and ora < ora_prec:
            data_corr = date.fromordinal(data_corr.toordinal() + 1)  # senza data e orario "indietro" = giorno dopo
        ora_prec = ora or ora_prec
        if d:
            gg, mm = int(d.group(1)), int(d.group(2))
            yy = oggi.year + 1 if (oggi.month >= 7 and mm <= 6) else (oggi.year - 1 if (oggi.month <= 6 and mm >= 7) else oggi.year)
            data_corr = date(yy, mm, gg)
        partite.append({"casa_aia": casa, "ospite_aia": ospite, "data": data_corr.isoformat() if data_corr else None,
                        "ora": ora, "arbitro_aia": arbitro})
    return {"pubblicata": pubblicata, "partite": partite}


def giornata_corrente():
    pubb = lista_serie_a()
    designazioni = [p for p in pubb if not p["variazione"] and p["giornata"]]
    if not designazioni:
        return None
    ult = designazioni[0]
    base = leggi(ult["id"])
    variazioni = []
    for v in sorted([p for p in pubb if p["variazione"] and p["giornata"] == ult["giornata"]], key=lambda x: x["id"]):
        lv = leggi(v["id"])
        for pv in lv["partite"]:
            for p in base["partite"]:
                if (p["casa_aia"], p["ospite_aia"]) == (pv["casa_aia"], pv["ospite_aia"]) and pv["arbitro_aia"] \
                        and pv["arbitro_aia"] != p["arbitro_aia"]:
                    variazioni.append({"partita": f'{p["casa_aia"]} - {p["ospite_aia"]}', "prima": p["arbitro_aia"],
                                       "dopo": pv["arbitro_aia"], "pubblicata": lv["pubblicata"]})
                    p["arbitro_aia"] = pv["arbitro_aia"]
    return {"giornata": ult["giornata"], "id": ult["id"], "pubblicata": base["pubblicata"],
            "partite": base["partite"], "variazioni": variazioni}


if __name__ == "__main__":
    import json
    print(json.dumps(giornata_corrente(), indent=1, ensure_ascii=False))
