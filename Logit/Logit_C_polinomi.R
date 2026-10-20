# ============================================================================
# LOGIT CON POTENZE DELLE VARIABILI NUMERICHE (fino al grado 10) + REGOLARIZZAZIONE
# (Rubik, palette UniBG, titoli in inglese)
# ============================================================================
################################################################################
#  LOGIT POLINOMIALE (SOLO POTENZE) REGOLARIZZATA - dropout al primo anno (UniBG)
#
#  Stessi dati, scenari e metriche di Logit_C.R / Logit_C_elasticnet_unibg.R. Si parte dalla logit con
#  tutte le variabili dello scenario (termini lineari + dummy) e si AGGIUNGONO le potenze x^2 ... x^d delle
#  sole variabili NUMERICHE (le dummy elevate a potenza non aggiungono nulla), per d da 2 a 10.
#  NESSUN termine misto (prodotti tra variabili): solo potenze pure, una variabile alla volta.
#
#  Base dei polinomi (POLY_ORTOGONALE):
#     TRUE  (default) polinomi ortogonali di poly() calcolati sul TRAIN e valutati sugli altri set:
#           generano lo stesso spazio delle potenze x^k ma evitano la collinearita' quasi perfetta tra
#           x^8, x^9, x^10 (con le potenze grezze i coefficienti sarebbero numericamente instabili)
#     FALSE potenze grezze di z = (x - media_train) / sd_train
#  Se una variabile ha pochi valori distinti (es. punteggi imputati) il suo grado massimo e' limitato
#  a (valori distinti - 1).
#
#  La regolarizzazione (glmnet: ridge, elastic net, lasso) e' quello che tiene a bada l'overfitting al
#  crescere del grado. Per ogni grado si prova alpha in {0, 0.5, 1} e lambda lungo tutto il percorso e si tiene
#  la coppia (alpha, lambda) con ASE VALIDATION minimo. Il test non entra mai nelle scelte. Il modello
#  "scelto" per scenario e' quello con ASE validation minimo tra i gradi 1..10 (grado 1 = logit lineare
#  penalizzata, riferimento).
#
#  Output (cartella logit_poly_v2):
#    metriche_confronto.csv         SOLO il modello scelto, per scenario (train / validation / test, soglia 0.5)
#    metriche_tutti_i_modelli.csv   una riga per scenario x grado (alpha e lambda ottimi sul validation)
#    griglia_completa.csv           tutte le combinazioni scenario x grado x alpha (ASE validation)
#    coefficienti.csv               coefficienti diversi da 0 dei modelli scelti
#    <scenario>/C_LogitPoly_d<grado>_previsioni_<train|validation|test>.csv   (riga, y, p)
#    grafici logit_poly_*_unibg.png
#  I file si chiamano C_LogitPoly_*: NON vengono letti da assessment_modelli_C_v2.R (cerca C_Logit_*).
################################################################################
rm(list = ls())
library(arrow)     # serve solo per leggere i file .parquet
library(ggplot2)
library(ragg)      # device agg_png (font Rubik nei png)
library(glmnet)    # logit regolarizzata (ridge / elastic net / lasso)

#### 0. Percorsi e impostazioni ####
## PERCORSI (stesso approccio degli altri script)
DATA_DIR <- "C:\\Users\\Dario\\Downloads\\Grafici sistemati\\dataset_C"                  # parquet e variabili_scenari_C_v2.csv
BASE     <- "C:\\Users\\Dario\\Downloads\\Grafici sistemati\\Logit solo C"
OUT_DIR  <- file.path(BASE, "logit_poly_v2")        # previsioni, metriche, grafici (creata se non esiste)
CARTELLA_DATI   <- DATA_DIR                       # nomi usati nel resto dello script
CARTELLA_OUTPUT <- OUT_DIR

file_necessari <- c("C_v2_train.parquet", "C_v2_valid.parquet", "C_v2_test.parquet",
                    "variabili_scenari_C_v2.csv")
mancanti <- file_necessari[!file.exists(file.path(DATA_DIR, file_necessari))]
if (length(mancanti) > 0)
  stop("Nella cartella '", DATA_DIR, "' mancano: ", paste(mancanti, collapse = ", "))
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
cat("Dati letti da:", DATA_DIR, "\n")

unibg_palette <- c("#111F37", "#1C355D", "#284C85", "#3462AC", "#3977DA",
                   "#6D9BE4", "#A0BEED", "#E5EDFA")

#### Graphic identity (same as assessment_modelli_C_v2.R): Rubik font, UniBG palette, English titles/labels ####
# The Rubik font must be installed on the system (same requirement as assessment_modelli_C_v2.R)
custom_theme <- theme_bw(base_family = "Rubik") +
  theme(axis.text  = element_text(size = 10),
        axis.title = element_text(size = 12),
        legend.text = element_text(size = 12),
        legend.title = element_text(size = 14),
        plot.title  = element_text(size = 16, hjust = 0.5))
tema <- custom_theme + theme(legend.position = "bottom", legend.title = element_blank(),
                             plot.subtitle = element_text(size = 11, hjust = 0.5))
# threshold plots: fixed threshold + measure colours exactly as in assessment_5_soglie
# (there ggplot maps the unnamed colours alphabetically: accuracy, precision, sensitivity, specificity -> palette[c(6, 1, 3, 5)])
SOGLIA_FISSA <- 0.31
MIS_COL <- c(accuracy = unibg_palette[6], precision = unibg_palette[1],
             sensitivity = unibg_palette[3], specificity = unibg_palette[5])

DATASET     <- "C"
DESCRIZIONE <- "C v2 - con TOLC, coorti dal 2020"
TARGET      <- "DROPOUT"                     # 1 = abbandono, 0 = prosegue
SOGLIA_CLASSIFICAZIONE <- 0.5                # serve solo per sensibilità, specificità,
# precisione e F1 (non per AUC e ASE)

# nomi degli scenari come nel modello XGBoost; 'colonna' = colonna del csv con le variabili
SCENARI <- list(
  list(nome = "1_early_isee", colonna = "EARLY_ISEE", etichetta = "Early"),
  list(nome = "2_mid_sem1",   colonna = "MID",        etichetta = "Mid (after 1st semester)"))

ELENCO_VARIABILI <- read.csv(file.path(CARTELLA_DATI, "variabili_scenari_C_v2.csv"),
                             stringsAsFactors = FALSE)
variabili_scenario <- function(colonna) ELENCO_VARIABILI$variabile[ELENCO_VARIABILI[[colonna]] == 1]

#### 1. Funzioni delle metriche (le stesse del modello XGBoost) ####
# AUC con la formula di Mann-Whitney (nessun pacchetto aggiuntivo)
auc_mw <- function(y, p) {
  r <- rank(p); n1 <- sum(y == 1); n0 <- sum(y == 0)
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

# AUC e ASE non usano la soglia; sensibilità, specificità, precisione e F1 sì
metriche <- function(y, p, soglia) {
  prev <- as.integer(p >= soglia)
  vp <- sum(prev == 1 & y == 1); fp <- sum(prev == 1 & y == 0)
  fn <- sum(prev == 0 & y == 1); vn <- sum(prev == 0 & y == 0)
  sens <- vp / (vp + fn)
  prec <- if (vp + fp > 0) vp / (vp + fp) else NA
  c(AUC         = auc_mw(y, p),
    ASE         = mean((p - y)^2),
    sensibilita = sens,
    specificita = vn / (vn + fp),
    precisione  = prec,
    recall      = sens,
    F1          = if (is.na(prec) || prec + sens == 0) NA else 2 * prec * sens / (prec + sens))
}

# una riga di metriche (train, validation, test) nel formato del modello XGBoost
riga_metriche <- function(info, y, p) {
  met <- lapply(c("train", "valid", "test"),
                function(s) metriche(y[[s]], p[[s]], SOGLIA_CLASSIFICAZIONE))
  names(met) <- c("train", "valid", "test")
  as.data.frame(c(info, list(valid_prevalenza = mean(y$valid)),
                  setNames(as.list(met$train), paste0("train_", names(met$train))),
                  setNames(as.list(met$valid), paste0("valid_", names(met$valid))),
                  setNames(as.list(met$test),  paste0("test_",  names(met$test)))),
                check.names = FALSE)
}

# punti della curva ROC (un punto per soglia distinta)
punti_roc <- function(y, p) {
  o  <- order(p, decreasing = TRUE); y <- y[o]; p <- p[o]
  tp <- cumsum(y == 1) / sum(y == 1); fp <- cumsum(y == 0) / sum(y == 0)
  ult <- c(diff(p) != 0, TRUE)
  r <- data.frame(fpr = c(0, fp[ult]), tpr = c(0, tp[ult]))
  r[!duplicated(round(r$fpr, 3), fromLast = TRUE) | r$fpr %in% c(0, 1), ]
}


#### 2. Lettura dei dati ####
leggi <- function(file) as.data.frame(read_parquet(file.path(CARTELLA_DATI, file)))
dati <- list(train = leggi("C_v2_train.parquet"),
             valid = leggi("C_v2_valid.parquet"),
             test  = leggi("C_v2_test.parquet"))
cat("Righe train:", nrow(dati$train), "| validation:", nrow(dati$valid),
    "| test:", nrow(dati$test), "\n")
# NB: glmnet standardizza da solo le colonne; i coefficienti salvati sono sulla scala delle colonne
# (termini lineari nelle unita' originali, termini polinomiali sulla base scelta in POLY_ORTOGONALE)


#### 3. Logit con potenze fino al grado 10, regolarizzata: grado x alpha, lambda sul validation ####
ALPHAS    <- c(ridge = 0, elnet = 0.5, lasso = 1)
GRADO_MAX <- 10
GRADI     <- 1:GRADO_MAX                    # grado 1 = logit lineare penalizzata (riferimento)
POLY_ORTOGONALE <- TRUE                     # TRUE: poly() ortogonale sul train; FALSE: potenze grezze di z
THR_GRID  <- seq(0.05, 0.80, by = 0.05)     # soglie del grafico (come negli altri script)
nome_modello <- function(g) if (g == 1) "LogitPoly_lineare" else paste0("LogitPoly_d", g)

elenco     <- data.frame()   # una riga per scenario x grado (alpha e lambda ottimi sul validation)
griglia    <- data.frame()   # tutte le combinazioni scenario x grado x alpha
coef_list  <- list()         # coefficienti diversi da 0 (lambda ottimo)
previsioni <- list()         # probabilita' previste, per scenario x modello

for (sc in SCENARI) {
  cat("\n=====", sc$etichetta, "=====\n")
  usa <- variabili_scenario(sc$colonna)
  stopifnot("variabili del csv assenti nei dati" = all(usa %in% names(dati$train)))
  d <- lapply(dati, function(x) x[, c(usa, TARGET)])
  if (any(sapply(d, function(x) anyNA(x)))) stop("Ci sono NA nelle variabili usate")
  y <- lapply(d, function(x) x[[TARGET]])

  # matrice di base: lineari + dummy (come nella logit penalizzata)
  tutto <- do.call(rbind, lapply(d, function(x) x[, usa, drop = FALSE]))
  X  <- model.matrix(~ ., data = tutto)[, -1, drop = FALSE]
  n1 <- nrow(d$train); n2 <- nrow(d$valid); n3 <- nrow(d$test)
  idx <- list(train = seq_len(n1), valid = n1 + seq_len(n2), test = n1 + n2 + seq_len(n3))
  Xs <- lapply(idx, function(i) X[i, , drop = FALSE])

  # potenze 2..GRADO_MAX delle variabili NUMERICHE (tipo "numerica" nel csv); base calcolata sul train
  tipo_var <- ELENCO_VARIABILI$tipo[match(usa, ELENCO_VARIABILI$variabile)]
  num <- usa[tipo_var == "numerica"]
  num <- num[sapply(tutto[num], is.numeric)]
  P <- list(train = NULL, valid = NULL, test = NULL); info_col <- data.frame()
  for (v in num) {
    gmax_v <- min(GRADO_MAX, length(unique(d$train[[v]])) - 1)      # grado massimo ammesso dai valori distinti
    if (gmax_v < 2) next
    if (POLY_ORTOGONALE) {
      pj <- poly(d$train[[v]], degree = gmax_v)
      blocchi <- lapply(d, function(x) predict(pj, newdata = x[[v]])[, 2:gmax_v, drop = FALSE])
    } else {
      m <- mean(d$train[[v]]); sdv <- sd(d$train[[v]]); if (sdv == 0) sdv <- 1
      blocchi <- lapply(d, function(x) sapply(2:gmax_v, function(k) ((x[[v]] - m) / sdv)^k))
    }
    for (s in names(P)) {
      b <- matrix(blocchi[[s]], nrow = nrow(d[[s]]))
      colnames(b) <- paste0(v, "^", 2:gmax_v)
      P[[s]] <- cbind(P[[s]], b)
    }
    info_col <- rbind(info_col, data.frame(variabile = v, k = 2:gmax_v))
  }
  cat("Variabili:", length(usa), "| numeriche con potenze:", length(unique(info_col$variabile)),
      "| colonne di base:", ncol(X), "| colonne polinomiali (grado 2..", GRADO_MAX, "):", nrow(info_col),
      "| quota dropout train:", round(mean(y$train), 3), "| validation:", round(mean(y$valid), 3), "\n")

  for (g in GRADI) {
    sel <- if (g == 1) integer(0) else which(info_col$k <= g)
    Xg <- if (length(sel) == 0) Xs else
      setNames(lapply(names(Xs), function(s) cbind(Xs[[s]], P[[s]][, sel, drop = FALSE])), names(Xs))

    migliore <- NULL
    for (a in names(ALPHAS)) {
      fit <- glmnet(Xg$train, factor(y$train, levels = c(0, 1)), family = "binomial", alpha = ALPHAS[[a]])
      pm  <- lapply(Xg, function(m) predict(fit, newx = m, type = "response"))   # righe = studenti, colonne = lambda
      ase_tr <- colMeans((pm$train - y$train)^2)
      ase_va <- colMeans((pm$valid - y$valid)^2)
      k <- which.min(ase_va)                                                      # lambda scelto: ASE minimo sul validation
      griglia <- rbind(griglia, data.frame(scenario = sc$nome, grado = g, alpha_nome = a,
                                           alpha = ALPHAS[[a]], lambda = fit$lambda[k], n_coef = fit$df[k],
                                           train_ASE = ase_tr[k], valid_ASE = ase_va[k]))
      if (is.null(migliore) || ase_va[k] < migliore$valid_ASE) {
        cf <- as.matrix(coef(fit, s = fit$lambda[k]))[, 1]
        migliore <- list(valid_ASE = ase_va[k], alpha_nome = a, lambda = fit$lambda[k], n_coef = fit$df[k],
                         p = lapply(pm, function(m) as.numeric(m[, k])), coef = cf[cf != 0])
      }
    }
    mod_nome <- nome_modello(g)
    elenco <- rbind(elenco, riga_metriche(
      info = list(scenario = sc$nome, dataset = DATASET, descrizione = DESCRIZIONE, modello = mod_nome,
                  grado = g, alpha_nome = migliore$alpha_nome, alpha = ALPHAS[[migliore$alpha_nome]],
                  lambda = migliore$lambda, n_coef = migliore$n_coef, colonne = ncol(Xg$train)),
      y, migliore$p))
    rr <- elenco[nrow(elenco), ]
    cat(sprintf("  grado %2d | colonne %4d | alpha %-5s lambda %.5f | coef != 0: %4d | AUC train %.3f valid %.3f | ASE train %.4f valid %.4f\n",
                g, ncol(Xg$train), migliore$alpha_nome, migliore$lambda, migliore$n_coef,
                rr$train_AUC, rr$valid_AUC, rr$train_ASE, rr$valid_ASE))

    # previsioni: riga = posizione nel file parquet
    cartella <- file.path(CARTELLA_OUTPUT, sc$nome)
    dir.create(cartella, showWarnings = FALSE, recursive = TRUE)
    for (s in c("train", "valid", "test")) {
      nome.split <- c(train = "train", valid = "validation", test = "test")[[s]]
      write.csv(data.frame(riga = seq_along(migliore$p[[s]]), y = y[[s]], p = migliore$p[[s]]),
                file.path(cartella, paste0(DATASET, "_", mod_nome, "_previsioni_", nome.split, ".csv")),
                row.names = FALSE)
    }
    previsioni[[paste(sc$nome, mod_nome)]] <- list(scenario = sc$nome, modello = mod_nome, y = y, p = migliore$p)
    coef_list[[paste(sc$nome, mod_nome)]] <- data.frame(scenario = sc$nome, modello = mod_nome,
                                                        variabile = names(migliore$coef), coef = as.numeric(migliore$coef),
                                                        stringsAsFactors = FALSE)
  }
}

# modello scelto per scenario: ASE minimo sul validation (il test non entra nella scelta)
elenco$scelto <- ""
for (sc in SCENARI) {
  i <- which(elenco$scenario == sc$nome)
  elenco$scelto[i[which.min(elenco$valid_ASE[i])]] <- "scelto"
}
riepilogo <- elenco[elenco$scelto == "scelto", setdiff(names(elenco), "scelto")]
rownames(riepilogo) <- NULL
coef_tab <- do.call(rbind, coef_list)
coef_tab <- coef_tab[paste(coef_tab$scenario, coef_tab$modello) %in% paste(riepilogo$scenario, riepilogo$modello), ]
rownames(coef_tab) <- NULL
write.csv(riepilogo, file.path(CARTELLA_OUTPUT, "metriche_confronto.csv"), row.names = FALSE)
write.csv(elenco, file.path(CARTELLA_OUTPUT, "metriche_tutti_i_modelli.csv"), row.names = FALSE)
write.csv(griglia, file.path(CARTELLA_OUTPUT, "griglia_completa.csv"), row.names = FALSE)
write.csv(coef_tab, file.path(CARTELLA_OUTPUT, "coefficienti.csv"), row.names = FALSE)

cat("\n=================== RIEPILOGO (modello scelto) ===================\n")
print(format(riepilogo[, c("scenario", "modello", "alpha_nome", "lambda", "n_coef", "colonne", "train_AUC",
                           "valid_AUC", "test_AUC", "valid_ASE", "test_ASE")], digits = 3), row.names = FALSE)
cat("\n=================== TUTTI I GRADI ===================\n")
print(elenco[, c("scenario", "grado", "alpha_nome", "n_coef", "colonne", "train_ASE", "valid_ASE",
                 "valid_AUC", "scelto")], row.names = FALSE)


#### 4. Grafici: stessa serie degli altri modelli ####
th <- tema
SC_LAB   <- c("1_early_isee" = "Early", "2_mid_sem1" = "Mid (after 1st semester)")
SETS     <- c("train", "validation", "test")
nomi.split <- c(train = "train", valid = "validation", test = "test")

# previsioni di tutti i modelli: una riga per studente, set e modello
prev_long <- do.call(rbind, lapply(previsioni, function(pr)
  do.call(rbind, lapply(names(nomi.split), function(s)
    data.frame(scenario = pr$scenario, modello = pr$modello, set = nomi.split[[s]], y = pr$y[[s]], prob = pr$p[[s]])))))
ps <- prev_long[paste(prev_long$scenario, prev_long$modello) %in%
                  paste(riepilogo$scenario, riepilogo$modello), ]      # solo il modello scelto

# 1. ASE al variare del grado (train e validation, linee continue); rombo = grado scelto
#    (stesso grafico di "model complexity vs ASE" della logit a blocchi)
ase_long <- rbind(data.frame(elenco[, c("scenario", "grado")], campione = "train",      ASE = elenco$train_ASE),
                  data.frame(elenco[, c("scenario", "grado")], campione = "validation", ASE = elenco$valid_ASE))
ase_long$pannello <- factor(SC_LAB[ase_long$scenario], levels = SC_LAB)
scelta <- elenco[elenco$scelto == "scelto", ]
scelta$pannello <- factor(SC_LAB[scelta$scenario], levels = SC_LAB)
g1 <- ggplot(ase_long, aes(x = grado, y = ASE, color = campione)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  geom_point(data = scelta, aes(x = grado, y = valid_ASE), color = unibg_palette[1], size = 5, shape = 18, inherit.aes = FALSE) +
  facet_wrap(~ pannello, scales = "free", ncol = 2) +
  scale_color_manual(values = unibg_palette[c(6, 3)]) +
  scale_x_continuous(breaks = GRADI) +
  labs(title = "Polynomial logit: model complexity vs ASE",
       x = "Polynomial degree", y = "ASE", color = NULL) + th
ggsave(file.path(OUT_DIR, "logit_poly_complessita_vs_ase_unibg.png"), g1, width = 11, height = 5, dpi = 300, device = agg_png, bg = "white")

# 2. performance sul test: riferimento lineare e grado scelto (come le due basi di RF)
BASE_EN  <- c(lineare = "Linear (degree 1)", scelto = "Selected degree")
COL_BASE <- setNames(unibg_palette[c(1, 5)], names(BASE_EN))
rappr <- do.call(rbind, lapply(riepilogo$scenario, function(sc) rbind(
  data.frame(scenario = sc, modello = nome_modello(1), base = "lineare"),
  data.frame(scenario = sc, modello = riepilogo$modello[riepilogo$scenario == sc], base = "scelto"))))
bar <- merge(rappr, elenco, by = c("scenario", "modello"))
bar$base <- factor(bar$base, levels = names(BASE_EN))
bar_long <- rbind(data.frame(bar[, c("scenario", "base")], valore = bar$test_AUC, metrica = "AUC"),
                  data.frame(bar[, c("scenario", "base")], valore = bar$test_ASE, metrica = "ASE"))
g2 <- ggplot(bar_long, aes(base, valore, fill = base)) +
  geom_col(position = position_dodge(width = 0.85), width = 0.8) +
  geom_text(aes(label = sprintf("%.3f", valore)), position = position_dodge(width = 0.85), vjust = -0.4, size = 2.8) +
  facet_grid(metrica ~ scenario, scales = "free_y", labeller = labeller(scenario = SC_LAB)) +
  scale_fill_manual(values = COL_BASE, guide = "none") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  scale_x_discrete(labels = BASE_EN) +
  labs(title = "Polynomial logit: test performance", x = NULL, y = NULL) + th +
  theme(axis.text.x = element_text(angle = 0))
ggsave(file.path(OUT_DIR, "logit_poly_test_performance_unibg.png"), g2, width = 12, height = 7, dpi = 300, device = agg_png, bg = "white")

# 3. misure di classificazione al variare della soglia (modello scelto, validation)
sw <- do.call(rbind, lapply(split(ps[ps$set == "validation", ], ps$scenario[ps$set == "validation"]), function(dd) {
  do.call(rbind, lapply(THR_GRID, function(t) {
    pred <- dd$prob >= t
    TP <- sum(pred & dd$y == 1); FP <- sum(pred & dd$y == 0)
    FN <- sum(!pred & dd$y == 1); TN <- sum(!pred & dd$y == 0)
    data.frame(scenario = dd$scenario[1], soglia = t, accuracy = (TP + TN) / nrow(dd),
               precisione = if (TP + FP > 0) TP / (TP + FP) else NA_real_,
               sensibilita = TP / (TP + FN), specificita = TN / (TN + FP))
  }))
}))
sw_long <- rbind(data.frame(sw[, c("scenario", "soglia")], valore = sw$accuracy,    misura = "accuracy"),
                 data.frame(sw[, c("scenario", "soglia")], valore = sw$precisione,  misura = "precision"),
                 data.frame(sw[, c("scenario", "soglia")], valore = sw$sensibilita, misura = "sensitivity"),
                 data.frame(sw[, c("scenario", "soglia")], valore = sw$specificita, misura = "specificity"))
g3 <- ggplot(sw_long, aes(soglia, valore, colour = misura)) +
  geom_vline(xintercept = SOGLIA_FISSA, linetype = "dashed", colour = "grey40") +
  geom_line(linewidth = 1, na.rm = TRUE) +
  facet_wrap(~ scenario, labeller = as_labeller(SC_LAB)) +
  scale_colour_manual(values = MIS_COL) +
  labs(title = "Polynomial logit: classification measurements varying threshold",
       x = "Dropout probability threshold", y = NULL, colour = NULL) + th
ggsave(file.path(OUT_DIR, "logit_poly_effetto_soglia_unibg.png"), g3, width = 11, height = 5.5, dpi = 300, device = agg_png, bg = "white")

# 4. lift cumulato del modello scelto (validation e test)
lift_decili <- function(y, p) {
  dec <- ceiling(10 * rank(-p, ties.method = "first") / length(p))
  ev  <- as.numeric(tapply(y, dec, sum)); nn <- as.numeric(tapply(y, dec, length))
  data.frame(percentile = 1:10 * 10, lift = (cumsum(ev) / cumsum(nn)) / mean(y))
}
lift <- do.call(rbind, lapply(riepilogo$scenario, function(sc) do.call(rbind, lapply(c("validation", "test"), function(s) {
  dd <- ps[ps$scenario == sc & ps$set == s, ]
  cbind(scenario = sc, set = s, lift_decili(dd$y, dd$prob))
}))))
g4 <- ggplot(lift, aes(percentile, lift, colour = set)) +
  geom_hline(yintercept = 1, linetype = 2, colour = "grey60") +
  geom_line(linewidth = 1) + geom_point(size = 2) +
  facet_wrap(~ scenario, labeller = as_labeller(SC_LAB)) +
  scale_x_continuous(breaks = 1:10 * 10) +
  scale_colour_manual(values = c(validation = unibg_palette[3], test = unibg_palette[6])) +
  labs(title = "Polynomial logit: cumulative lift of the selected model", x = "% of students selected (deciles)",
       y = "Cumulative lift", colour = NULL) + th
ggsave(file.path(OUT_DIR, "logit_poly_lift_cumulato_unibg.png"), g4, width = 11, height = 5.5, dpi = 300, device = agg_png, bg = "white")

# 5. curve ROC (train, validation, test): righe = scenario, colonne = riferimento lineare e modello scelto
pl_roc <- merge(prev_long, transform(rappr, colonna = base)[, c("scenario", "modello", "colonna")], by = c("scenario", "modello"))
roc_df <- do.call(rbind, lapply(split(pl_roc, list(pl_roc$scenario, pl_roc$colonna, pl_roc$set), drop = TRUE), function(dd)
  cbind(punti_roc(dd$y, dd$prob), scenario = dd$scenario[1], colonna = dd$colonna[1], set = dd$set[1],
        auc = auc_mw(dd$y, dd$prob))))
roc_df$set      <- factor(roc_df$set, levels = SETS)
roc_df$colonna  <- factor(roc_df$colonna, levels = c("lineare", "scelto"))
roc_df$scenario <- factor(roc_df$scenario, levels = names(SC_LAB), labels = unname(SC_LAB))
auc_lab <- unique(roc_df[, c("scenario", "colonna", "set", "auc")])
auc_lab$y <- c(train = 0.25, validation = 0.15, test = 0.05)[as.character(auc_lab$set)]
auc_lab$txt <- sprintf("AUC %s = %.3f", as.character(auc_lab$set), auc_lab$auc)
COL_SET <- c(train = unibg_palette[6], validation = unibg_palette[1], test = unibg_palette[5])
g5 <- ggplot(roc_df, aes(fpr, tpr, colour = set)) +
  geom_abline(intercept = 0, slope = 1, linetype = 2, colour = "grey60") +
  geom_path(linewidth = 0.9) +
  geom_text(data = auc_lab, aes(x = 1, y = y, label = txt, colour = set), hjust = 1, size = 3.4, show.legend = FALSE) +
  facet_grid(scenario ~ colonna, labeller = labeller(colonna = BASE_EN)) +
  scale_colour_manual(values = COL_SET) +
  coord_equal() +
  labs(title = "Polynomial logit: ROC curves by scenario (rows) and model (columns)",
       x = "1 - specificity (FPR)", y = "Sensitivity (TPR)", colour = NULL) + th
ggsave(file.path(OUT_DIR, "logit_poly_roc_unibg.png"), g5, width = 12, height = 8.5, dpi = 300, device = agg_png, bg = "white")
write.csv(unique(roc_df[, c("scenario", "colonna", "set", "auc")]), file.path(OUT_DIR, "auc_roc.csv"), row.names = FALSE)

cat("\nOutput salvati in:", normalizePath(CARTELLA_OUTPUT), "\n")
