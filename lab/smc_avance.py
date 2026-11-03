"""Smart Money / ICT avance (demande du trader) : les modeles complets des traders institutionnels, reecrits a partir
de leurs descriptions publiques (ICT, SMC), sans code copie. Tout est causal : un swing n'existe qu'une fois confirme
k bougies plus tard, un FVG qu'a la cloture de sa 3e bougie, un order block qu'a la cassure de structure qui le valide.
   - bougies de deplacement (impulsion institutionnelle) ;
   - FVG avec leur etat : frais / comble, taille, age ; FVG inverse (iFVG) quand un FVG est casse ;
   - order blocks VALIDES : derniere bougie opposee avant un deplacement qui casse la structure (BOS) en laissant un
     FVG ; etat frais / touche, premier contact, test du seuil moyen (50 %), age, taille ;
   - breaker blocks : order block casse qui change de camp ;
   - modele ICT complet : chasse de liquidite (balayage d'un creux) -> changement de structure (MSS) -> retour en zone
     OTE (62 a 79 % de la jambe d'impulsion) ; et le miroir baissier ;
   - killzones Londres / New York."""
import numpy as np

NAN = np.nan


def _div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.asarray(a, dtype=float) / b
    r[~np.isfinite(r)] = NAN
    return r


def compute_smc_avance(ts, o, h, l, c, atr, k=3, life=300):
    n = len(c)
    out = {}
    rng = h - l
    body = np.abs(c - o)
    ok_atr = np.isfinite(atr) & (atr > 0)
    disp = np.where(ok_atr & (body >= 1.2 * np.where(ok_atr, atr, 1)) & (body >= 0.6 * np.where(rng > 0, rng, 1)),
                    np.sign(c - o), 0.0)
    out["deplacement"] = disp
    f = {nm: np.full(n, NAN) for nm in (
        "fvg_haussier_dist", "fvg_haussier_taille", "fvg_haussier_age", "fvg_baissier_dist", "fvg_baissier_taille",
        "fvg_baissier_age", "ifvg_haussier_dist", "ifvg_baissier_dist",
        "ob_haussier_dist", "ob_haussier_taille", "ob_haussier_age", "ob_baissier_dist", "ob_baissier_taille",
        "ob_baissier_age", "breaker_haussier_dist", "breaker_baissier_dist",
        "ict_haussier_ote", "ict_baissier_ote", "ict_haussier_bougies", "ict_baissier_bougies")}
    flags = {nm: np.zeros(n) for nm in (
        "dans_fvg_haussier", "dans_fvg_baissier", "dans_ob_haussier", "dans_ob_baissier", "ob_haussier_premier_contact",
        "ob_baissier_premier_contact", "ob_haussier_seuil_moyen", "ob_baissier_seuil_moyen", "dans_breaker_haussier",
        "dans_breaker_baissier", "modele_ict", "chasse_puis_mss")}
    fvgs, obs = [], []            # zones actives : dict(dir, top, bot, born, state)
    sh = sl = None                # dernier sommet / creux confirmes : (indice, prix)
    bos_up_done = bos_dn_done = False
    sweep = {1: None, -1: None}   # chasse de liquidite recente : (indice, prix extreme)
    model = {1: None, -1: None}   # modele ICT : dict(mss, lo, hi)
    for i in range(n):
        a_ = atr[i] if ok_atr[i] else NAN
        # --- swings confirmes k bougies apres
        if i >= 2 * k:
            j = i - k
            if h[j] == h[j - k:i + 1].max():
                sh, bos_up_done = (j, h[j]), False
            if l[j] == l[j - k:i + 1].min():
                sl, bos_dn_done = (j, l[j]), False
        # --- nouveaux FVG (3 bougies) a la cloture de la 3e
        if i >= 2:
            if l[i] > h[i - 2]:
                fvgs.append({"dir": 1, "top": l[i], "bot": h[i - 2], "born": i, "state": 0})
            if h[i] < l[i - 2]:
                fvgs.append({"dir": -1, "top": l[i - 2], "bot": h[i], "born": i, "state": 0})
        # --- chasse de liquidite : balayage du dernier creux / sommet puis cloture de retour
        if sl is not None and l[i] < sl[1] and c[i] > sl[1]:
            sweep[1] = (i, l[i])
        if sh is not None and h[i] > sh[1] and c[i] < sh[1]:
            sweep[-1] = (i, h[i])
        # --- cassure de structure (BOS) -> order block valide (derniere bougie opposee + deplacement + FVG)
        for d, sw, done in ((1, sh, bos_up_done), (-1, sl, bos_dn_done)):
            if sw is None or done:
                continue
            broke = c[i] > sw[1] if d > 0 else c[i] < sw[1]
            if not broke:
                continue
            if d > 0:
                bos_up_done = True
            else:
                bos_dn_done = True
            start = max(0, sw[0] - 10)
            leg = range(start, i + 1)
            if not any(disp[x] == d for x in leg):
                continue
            if not any(z["dir"] == d and start <= z["born"] <= i for z in fvgs):
                continue
            opp = [x for x in range(i, start - 1, -1) if (c[x] - o[x]) * d < 0]
            if opp:
                x = opp[0]
                obs.append({"dir": d, "top": h[x], "bot": l[x], "born": i, "touch": 0, "broken": False})
            # changement de structure apres une chasse de liquidite : modele ICT
            sw_ = sweep[d]
            if sw_ is not None and i - sw_[0] <= 20:
                model[d] = {"mss": i, "lo": sw_[1], "hi": h[i] if d > 0 else l[i]}
                flags["chasse_puis_mss"][i] = d
        # --- etat des FVG
        best = {1: None, -1: None}
        keep = []
        for z in fvgs:
            if i - z["born"] > life:
                continue
            d = z["dir"]
            if z["state"] == 0:                       # FVG frais
                if (d > 0 and l[i] <= z["bot"]) or (d < 0 and h[i] >= z["top"]):
                    z["state"] = 1                    # comble
                    if (d > 0 and c[i] < z["bot"]) or (d < 0 and c[i] > z["top"]):
                        z["state"] = 2                # casse en cloture : devient FVG inverse
            if z["state"] == 0 and i > z["born"] and np.isfinite(a_):
                dist = ((c[i] - z["top"]) if d > 0 else (z["bot"] - c[i])) / a_
                inside = z["bot"] <= c[i] <= z["top"]
                if inside:
                    flags["dans_fvg_haussier" if d > 0 else "dans_fvg_baissier"][i] = 1.0
                if dist >= -((z["top"] - z["bot"]) / a_) and (best[d] is None or abs(dist) < abs(best[d][0])):
                    best[d] = (max(dist, 0.0), z)
            if z["state"] == 2 and np.isfinite(a_):   # iFVG : zone qui change de camp
                nm = "ifvg_baissier_dist" if d > 0 else "ifvg_haussier_dist"
                dist = (z["bot"] - c[i]) / a_ if d > 0 else (c[i] - z["top"]) / a_
                if not np.isfinite(f[nm][i]) or abs(dist) < abs(f[nm][i]):
                    f[nm][i] = dist
            keep.append(z)
        fvgs = keep[-80:]
        for d, nm in ((1, "haussier"), (-1, "baissier")):
            if best[d] is not None:
                dist, z = best[d]
                f["fvg_%s_dist" % nm][i] = dist
                f["fvg_%s_taille" % nm][i] = (z["top"] - z["bot"]) / a_
                f["fvg_%s_age" % nm][i] = i - z["born"]
        # --- etat des order blocks / breakers
        bestob = {1: None, -1: None}
        keep = []
        for z in obs:
            if i - z["born"] > life:
                continue
            d = z["dir"]
            if i > z["born"]:
                in_zone = (l[i] <= z["top"]) if d > 0 else (h[i] >= z["bot"])
                if not z["broken"] and ((d > 0 and c[i] < z["bot"]) or (d < 0 and c[i] > z["top"])):
                    z["broken"] = True                # order block casse : breaker block (camp oppose)
                elif not z["broken"] and in_zone:
                    if z["touch"] == 0:
                        flags["ob_%s_premier_contact" % ("haussier" if d > 0 else "baissier")][i] = 1.0
                    mid = (z["top"] + z["bot"]) / 2
                    if (d > 0 and l[i] <= mid) or (d < 0 and h[i] >= mid):
                        flags["ob_%s_seuil_moyen" % ("haussier" if d > 0 else "baissier")][i] = 1.0
                    z["touch"] += 1
            if not np.isfinite(a_):
                keep.append(z)
                continue
            if not z["broken"]:
                dist = ((c[i] - z["top"]) if d > 0 else (z["bot"] - c[i])) / a_
                if z["bot"] <= c[i] <= z["top"]:
                    flags["dans_ob_haussier" if d > 0 else "dans_ob_baissier"][i] = 1.0
                if bestob[d] is None or abs(max(dist, 0)) < abs(bestob[d][0]):
                    bestob[d] = (max(dist, 0.0), z)
            else:
                nm = "breaker_baissier" if d > 0 else "breaker_haussier"
                dist = (z["bot"] - c[i]) / a_ if d > 0 else (c[i] - z["top"]) / a_
                if z["bot"] <= c[i] <= z["top"]:
                    flags["dans_" + nm][i] = 1.0
                if not np.isfinite(f[nm + "_dist"][i]) or abs(dist) < abs(f[nm + "_dist"][i]):
                    f[nm + "_dist"][i] = dist
            keep.append(z)
        obs = keep[-60:]
        for d, nm in ((1, "haussier"), (-1, "baissier")):
            if bestob[d] is not None:
                dist, z = bestob[d]
                f["ob_%s_dist" % nm][i] = dist
                f["ob_%s_taille" % nm][i] = (z["top"] - z["bot"]) / a_
                f["ob_%s_age" % nm][i] = i - z["born"]
        # --- modele ICT : retour en zone OTE apres chasse + MSS (actif 30 bougies)
        for d, nm in ((1, "haussier"), (-1, "baissier")):
            md = model[d]
            if md is None:
                continue
            if i - md["mss"] > 30:
                model[d] = None
                continue
            md["hi"] = max(md["hi"], h[i]) if d > 0 else min(md["hi"], l[i])
            leg = md["hi"] - md["lo"]
            if leg == 0:
                continue
            ret = (md["hi"] - c[i]) / leg                 # 0 = au sommet de l'impulsion, 1 = revenu au creux chasse
            if ret > 1:
                model[d] = None                           # le creux chasse est perdu : modele invalide
                continue
            f["ict_%s_ote" % nm][i] = ret
            f["ict_%s_bougies" % nm][i] = i - md["mss"]
            if 0.62 <= ret <= 0.79 and i > md["mss"]:
                flags["modele_ict"][i] = d
    hour = ((ts // 3600) % 24).astype(int)
    out["killzone_londres"] = ((hour >= 7) & (hour < 10)).astype(float)
    out["killzone_new_york"] = ((hour >= 12) & (hour < 15)).astype(float)
    out.update(f)
    out.update(flags)
    return out
