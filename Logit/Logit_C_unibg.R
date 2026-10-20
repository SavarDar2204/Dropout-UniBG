# ============================================================================
# GRAPHIC-STYLE COPY of Logit_C.R
# Same analysis as the original; only the plots changed (Rubik font, UniBG palette,
# English titles/labels, as in assessment_modelli_C_v2.R). PNG files get the suffix _unibg.
# ============================================================================
################################################################################
#  REGRESSIONE LOGISTICA (LOGIT) - Previsione del dropout al primo anno (UniBG)
#
#  Dataset: C v2 (coorti dal 2020 con TOLC), cartella "dataset_C":
#           C_v2_train / C_v2_valid / C_v2_test .parquet
#           variabili_scenari_C_v2.csv, LEGGIMI_C_v2.txt
#
#  Due scenari (stesso outcome: DROPOUT = abbandono entro il 1° anno):
#    1_early_isee  ottobre, what-if con ISEE raccolto all'immatricolazione
#                  (colonna EARLY_ISEE del csv, 18 variabili)
#    2_mid_sem1    dopo il 1° semestre: + esiti del 1° semestre e borsa
#                  (colonna MID del csv, 25 variabili)
#  Le variabili di ogni scenario NON sono scritte qui: si leggono da
#  variabili_scenari_C_v2.csv, lo stesso file usato dagli altri modelli del gruppo.
#
#  Scelta del modello (come a lezione): modelli ANNIDATI, aggiungendo un blocco di
#  variabili alla volta (modello 1 dentro modello 2 dentro modello 3 ...).
#  Complessità = numero di blocchi. Si calcola l'ASE sul validation e si sceglie
#  il modello con ASE minimo. Il test non entra nella scelta.
#
#  Output (cartella output_logit), stesso formato del modello XGBoost:
#    metriche_confronto.csv      SOLO il modello scelto, per scenario, su train /
#                                validation / test (soglia 0.5): è il file per il confronto
#    metriche_tutti_i_modelli.csv  TUTTI i modelli annidati provati, stesse colonne
#                                (+ blocco aggiunto e colonna "scelto")
#    <scenario>/C_<modello>_previsioni_<train|validation|test>.csv   (riga, y, p)
#    logit_complessita_vs_ase.png, logit_roc.png
#
#  ASE (Average Squared Error) = media di (y - probabilità prevista)^2
################################################################################
rm(list = ls())
library(arrow)     # serve solo per leggere i file .parquet
library(ggplot2)
library(ragg)      # device agg_png (font Rubik nei png)
library(car)       # vif

#### 0. Percorsi e impostazioni ####
## PERCORSI (stesso approccio degli script kNN / RF / Naive Bayes)
DATA_DIR <- "C:\\Users\\Dario\\Downloads\\Grafici sistemati\\dataset_C"                  # parquet e variabili_scenari_C_v2.csv
BASE     <- "C:\\Users\\Dario\\Downloads\\Grafici sistemati\\Logit solo C"
OUT_DIR  <- file.path(BASE, "logit_v2")          # previsioni, metriche, grafici (creata se non esiste)
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

# Blocchi di variabili, dal più semplice al più completo. In ogni scenario entrano
# solo le variabili che il csv assegna a quello scenario: nell'early i blocchi
# "semestre1" restano vuoti e vengono saltati.
blocchi <- list(
  anagrafica  = c("GENERE", "CITTADINANZA_ITA", "AREA_RESIDENZA", "FUORI_SEDE_FLG",
                  "REF_CAR_MIN_CAP", "res_car_missing"),
  corso       = c("CORSODISTUDIO"),
  maturita    = c("TIPO_MATURITA", "VOTOMATURITA_PCT", "ANNI_DA_MATURITA_CL"),
  iscrizione  = c("ISCRIZIONE_PERIODO"),
  tolc        = c("TOLC_STATO", "TOLC_PCT", "TOLC_PCT_REL_CORSO", "TOLC_INGLESE"),
  ofa_reddito = c("OFA_OBBLIGO", "ISEE_CLASSE_EARLY", "ISEE_CLASSE_MID", "ISPE_LOG1P",
                  "BORSA_STUDIO"),
  semestre1   = c("CFU_PREVISTI_SEM1", "PCT_N_ESAMI_PASSATI_SEM1", "PCT_SUCCESSO_SEM1",
                  "MEDIA_VOTO_DATI_SEM1", "CFU_ACQUISITI_SEM1_STIMA", "stato_sem1"))

# Variabile lasciata fuori dalla logit per collinearità (vedi vif nella sezione 5):
# res_car_missing vale 1 quasi solo per chi risiede all'estero, che è già una
# categoria di AREA_RESIDENZA (GVIF di centinaia di migliaia). Nei modelli ad alberi
# e nella rete neurale resta, come da csv.
var.escluse.logit <- c("res_car_missing")


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
# NB: la logit non ha bisogno di standardizzare le variabili,
# così i coefficienti restano interpretabili nelle unità originali


#### 3. Un ciclo per scenario: modelli annidati, scelta, metriche ####
elenco     <- data.frame()   # TUTTI i modelli annidati provati, con le metriche
scelti     <- data.frame()   # modello scelto per ogni scenario
previsioni <- list()         # probabilità previste dal modello scelto
modelli    <- list()         # logit scelta per ogni scenario

for (sc in SCENARI) {
  cat("\n=====", sc$etichetta, "=====\n")
  usa <- setdiff(variabili_scenario(sc$colonna), var.escluse.logit)
  stopifnot("variabili del csv assenti nei dati" = all(usa %in% names(dati$train)),
            "variabili del csv non assegnate a nessun blocco" = all(usa %in% unlist(blocchi)))
  d <- lapply(dati, function(x) x[, c(usa, TARGET)])
  if (any(sapply(d, function(x) anyNA(x)))) stop("Ci sono NA nelle variabili usate")
  y <- lapply(d, function(x) x[[TARGET]])
  cat("Variabili:", length(usa), "| quota dropout train:", round(mean(y$train), 3),
      "| validation:", round(mean(y$valid), 3), "\n")
  
  #### 3a. Modelli annidati: un blocco di variabili in più a ogni passo ####
  variabili <- c()
  fit       <- list()   # modelli stimati
  pk        <- list()   # previsioni di ogni modello (train, validation, test)
  k <- 0
  for (b in names(blocchi)) {
    nuove <- intersect(blocchi[[b]], usa)
    if (length(nuove) == 0) next                 # blocco vuoto in questo scenario
    k <- k + 1
    variabili <- c(variabili, nuove)
    fit[[k]] <- glm(as.formula(paste(TARGET, "~", paste(variabili, collapse = " + "))),
                    data = d$train, family = "binomial")
    pk[[k]]  <- lapply(d, function(x) as.numeric(predict(fit[[k]], newdata = x, type = "response")))
    elenco <- rbind(elenco, riga_metriche(
      info = list(scenario = sc$nome, dataset = DATASET, descrizione = DESCRIZIONE,
                  modello = sprintf("Logit_%dblocchi", k), blocchi = k, blocco.aggiunto = b,
                  variabili.aggiunte = paste(nuove, collapse = ", "),
                  n.variabili = length(variabili), colonne = length(coef(fit[[k]])) - 1),
      y, pk[[k]]))
    cat(sprintf("  blocchi %d (+ %-11s) | ASE validation %.4f | AUC validation %.3f\n",
                k, b, tail(elenco$valid_ASE, 1), tail(elenco$valid_AUC, 1)))
  }
  
  #### 3b. Scelta del modello: ASE minimo sul validation ####
  el.sc  <- elenco[elenco$scenario == sc$nome, ]
  k.best <- el.sc$blocchi[which.min(el.sc$valid_ASE)]
  best   <- fit[[k.best]]
  p      <- pk[[k.best]]
  mod_nome <- sprintf("Logit_%dblocchi", k.best)
  modelli[[sc$nome]] <- best
  scelti <- rbind(scelti, data.frame(scenario = sc$nome, blocchi = k.best))
  r <- el.sc[k.best, ]
  cat(sprintf("Scelto: %s | AUC train %.3f valid %.3f test %.3f | ASE valid %.3f | sens %.2f spec %.2f prec %.2f F1 %.2f\n",
              mod_nome, r$train_AUC, r$valid_AUC, r$test_AUC, r$valid_ASE,
              r$valid_sensibilita, r$valid_specificita, r$valid_precisione, r$valid_F1))
  
  # previsioni del modello scelto: riga = posizione nel file parquet
  cartella <- file.path(CARTELLA_OUTPUT, sc$nome)
  dir.create(cartella, showWarnings = FALSE, recursive = TRUE)
  for (s in c("train", "valid", "test")) {
    nome.split <- c(train = "train", valid = "validation", test = "test")[[s]]
    write.csv(data.frame(riga = seq_along(p[[s]]), y = y[[s]], p = p[[s]]),
              file.path(cartella, paste0(DATASET, "_", mod_nome, "_previsioni_", nome.split, ".csv")),
              row.names = FALSE)
  }
  previsioni[[sc$nome]] <- list(etichetta = sc$etichetta, y = y, p = p)
}

elenco$scelto <- ifelse(paste(elenco$scenario, elenco$blocchi) %in%
                          paste(scelti$scenario, scelti$blocchi), "scelto", "")
# file 1: SOLO il modello scelto per ogni scenario (quello che entra nel confronto di gruppo)
riepilogo <- elenco[elenco$scelto == "scelto", setdiff(names(elenco), c("blocco.aggiunto", "variabili.aggiunte", "n.variabili", "scelto"))]
rownames(riepilogo) <- NULL
write.csv(riepilogo, file.path(CARTELLA_OUTPUT, "metriche_confronto.csv"), row.names = FALSE)
# file 2: TUTTI i modelli provati, con le stesse metriche (per analisi e grafici).
# Attenzione: contiene anche le metriche sul test di tutti i modelli, ma la scelta
# si fa sul validation (colonna "scelto"), mai sul test.
write.csv(elenco, file.path(CARTELLA_OUTPUT, "metriche_tutti_i_modelli.csv"), row.names = FALSE)
saveRDS(modelli, file.path(CARTELLA_OUTPUT, "modelli_logit.rds"))

cat("\n=================== RIEPILOGO ===================\n")
print(format(riepilogo[, c("scenario", "modello", "train_AUC", "valid_AUC", "test_AUC",
                           "valid_ASE", "test_ASE", "valid_sensibilita", "valid_specificita",
                           "valid_precisione", "valid_F1")], digits = 3), row.names = FALSE)


#### 4. Com'è fatto ogni modello annidato ####
# modello k = modello k-1 + il blocco indicato; "scelto" = ASE minimo sul validation
print(elenco[, c("scenario", "blocchi", "blocco.aggiunto", "n.variabili", "colonne",
                 "valid_ASE", "valid_AUC", "scelto")], row.names = FALSE)
# per vedere UN modello in dettaglio, per esempio:
#   formula(modelli[["1_early_isee"]])


#### 5. Interpretazione di un modello scelto (come nella lezione) ####
# cambia il nome per vedere l'altro scenario: "2_mid_sem1"
m <- modelli[["1_early_isee"]]
summary(m)
exp(m$coefficients)   # odds ratio
vif(m)                # multicollinearità (GVIF)
# Nel mid alcune variabili di carriera sono per costruzione molto correlate
# (CFU_ACQUISITI_SEM1_STIMA = PCT_N_ESAMI_PASSATI_SEM1 x CFU_PREVISTI_SEM1): per la
# previsione non è un problema, per leggere i coefficienti sì.


#### 6. Grafico: complessità vs ASE (un pannello per scenario) ####
tab.ase <- rbind(
  data.frame(elenco[, c("scenario", "blocchi")], campione = "train",      ASE = elenco$train_ASE),
  data.frame(elenco[, c("scenario", "blocchi")], campione = "validation", ASE = elenco$valid_ASE))
scelta <- elenco[elenco$scelto == "scelto", ]
etichette.sc <- setNames(sapply(SCENARI, `[[`, "etichetta"), sapply(SCENARI, `[[`, "nome"))
tab.ase$pannello <- factor(etichette.sc[tab.ase$scenario], levels = etichette.sc)
scelta$pannello  <- factor(etichette.sc[scelta$scenario],  levels = etichette.sc)

g1 <- ggplot(tab.ase, aes(x = blocchi, y = ASE, color = campione)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  geom_point(data = scelta, aes(x = blocchi, y = valid_ASE),
             color = unibg_palette[1], size = 5, shape = 18, inherit.aes = FALSE) +
  facet_wrap(~ pannello, scales = "free", ncol = 2) +
  scale_color_manual(values = unibg_palette[c(6, 3)]) +
  scale_x_continuous(breaks = 1:9) +
  labs(title = "Logit: model complexity vs ASE",
       x = "Number of variable blocks in the model", y = "ASE", color = NULL) +
  tema
ggsave(file.path(OUT_DIR, "logit_complessita_vs_ase_unibg.png"), g1,
       width = 11, height = 5, dpi = 300, device = agg_png, bg = "white")


#### 7. Graphs: same set as the kNN / Random Forest / Naive Bayes scripts ####
th <- tema
SC_LAB   <- c("1_early_isee" = "Early", "2_mid_sem1" = "Mid (after 1st semester)")
BASE_EN  <- c("Logit" = "Logit")
COL_BASE <- c("Logit" = unibg_palette[1])
THR_GRID <- seq(0, 1, by = 0.02)     # thresholds of the threshold plot (as in the other scripts)
nomi.split <- c(train = "train", valid = "validation", test = "test")

# predictions of the selected model: one row per student and set
prev_long <- do.call(rbind, lapply(SCENARI, function(sc) {
  pr <- previsioni[[sc$nome]]
  do.call(rbind, lapply(names(nomi.split), function(s)
    data.frame(scenario = sc$nome, set = nomi.split[[s]], y = pr$y[[s]], prob = pr$p[[s]])))
}))

# 1. test performance (selected model)
bar_long <- rbind(data.frame(riepilogo[, "scenario", drop = FALSE], base = "Logit", valore = riepilogo$test_AUC, metrica = "AUC"),
                  data.frame(riepilogo[, "scenario", drop = FALSE], base = "Logit", valore = riepilogo$test_ASE, metrica = "ASE"))
g2 <- ggplot(bar_long, aes(base, valore, fill = base)) +
  geom_col(position = position_dodge(width = 0.85), width = 0.8) +
  geom_text(aes(label = sprintf("%.3f", valore)), position = position_dodge(width = 0.85), vjust = -0.4, size = 2.8) +
  facet_grid(metrica ~ scenario, scales = "free_y", labeller = labeller(scenario = SC_LAB)) +
  scale_fill_manual(values = COL_BASE, guide = "none") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  scale_x_discrete(labels = BASE_EN) +
  labs(title = "Logit: test performance", x = NULL, y = NULL) + th +
  theme(axis.text.x = element_text(angle = 0))
ggsave(file.path(OUT_DIR, "logit_test_performance_unibg.png"), g2, width = 12, height = 7, dpi = 300, device = agg_png, bg = "white")

# 2. classification measures varying the threshold (selected model, validation)
sw <- do.call(rbind, lapply(names(SC_LAB), function(sc) {
  d <- prev_long[prev_long$scenario == sc & prev_long$set == "validation", ]
  do.call(rbind, lapply(THR_GRID, function(t) {
    pred <- d$prob >= t
    TP <- sum(pred & d$y == 1); FP <- sum(pred & d$y == 0)
    FN <- sum(!pred & d$y == 1); TN <- sum(!pred & d$y == 0)
    data.frame(scenario = sc, soglia = t, accuracy = (TP + TN) / nrow(d),
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
  labs(title = "Logit: classification measurements varying threshold",
       x = "Dropout probability threshold", y = NULL, colour = NULL) + th
ggsave(file.path(OUT_DIR, "logit_effetto_soglia_unibg.png"), g3, width = 11, height = 5.5, dpi = 300, device = agg_png, bg = "white")

# 3. cumulative lift of the selected model (validation and test)
lift_decili <- function(y, p) {
  dec <- ceiling(10 * rank(-p, ties.method = "first") / length(p))
  ev  <- as.numeric(tapply(y, dec, sum)); nn <- as.numeric(tapply(y, dec, length))
  data.frame(percentile = 1:10 * 10, lift = (cumsum(ev) / cumsum(nn)) / mean(y))
}
lift <- do.call(rbind, lapply(names(SC_LAB), function(sc) do.call(rbind, lapply(c("validation", "test"), function(s) {
  d <- prev_long[prev_long$scenario == sc & prev_long$set == s, ]
  cbind(scenario = sc, set = s, lift_decili(d$y, d$prob))
}))))
g4 <- ggplot(lift, aes(percentile, lift, colour = set)) +
  geom_hline(yintercept = 1, linetype = 2, colour = "grey60") +
  geom_line(linewidth = 1) + geom_point(size = 2) +
  facet_wrap(~ scenario, labeller = as_labeller(SC_LAB)) +
  scale_x_continuous(breaks = 1:10 * 10) +
  scale_colour_manual(values = c(validation = unibg_palette[3], test = unibg_palette[6])) +
  labs(title = "Logit: cumulative lift of the selected model", x = "% of students selected (deciles)",
       y = "Cumulative lift", colour = NULL) + th
ggsave(file.path(OUT_DIR, "logit_lift_cumulato_unibg.png"), g4, width = 11, height = 5.5, dpi = 300, device = agg_png, bg = "white")

# 4. ROC curves (train, validation, test): rows = scenario, columns = model
roc_df <- do.call(rbind, lapply(names(SC_LAB), function(sc) do.call(rbind, lapply(unname(nomi.split), function(s) {
  d <- prev_long[prev_long$scenario == sc & prev_long$set == s, ]
  cbind(punti_roc(d$y, d$prob), scenario = sc, base = "Logit", set = s, auc = auc_mw(d$y, d$prob))
}))))
roc_df$set      <- factor(roc_df$set, levels = c("train", "validation", "test"))
roc_df$scenario <- factor(roc_df$scenario, levels = names(SC_LAB), labels = unname(SC_LAB))
auc_lab <- unique(roc_df[, c("scenario", "base", "set", "auc")])
auc_lab$y <- c(train = 0.25, validation = 0.15, test = 0.05)[as.character(auc_lab$set)]
auc_lab$txt <- sprintf("AUC %s = %.3f", as.character(auc_lab$set), auc_lab$auc)
COL_SET <- c(train = unibg_palette[6], validation = unibg_palette[1], test = unibg_palette[5])
g5 <- ggplot(roc_df, aes(fpr, tpr, colour = set)) +
  geom_abline(intercept = 0, slope = 1, linetype = 2, colour = "grey60") +
  geom_path(linewidth = 0.9) +
  geom_text(data = auc_lab, aes(x = 1, y = y, label = txt, colour = set), hjust = 1, size = 3.4, show.legend = FALSE) +
  facet_grid(scenario ~ base, labeller = labeller(base = BASE_EN)) +
  scale_colour_manual(values = COL_SET) +
  coord_equal() +
  labs(title = "Logit: ROC curves by scenario",
       x = "1 - specificity (FPR)", y = "Sensitivity (TPR)", colour = NULL) + th
ggsave(file.path(OUT_DIR, "logit_roc_unibg.png"), g5, width = 12, height = 8.5, dpi = 300, device = agg_png, bg = "white")
write.csv(unique(roc_df[, c("scenario", "base", "set", "auc")]), file.path(OUT_DIR, "auc_roc.csv"), row.names = FALSE)

cat("\nOutput salvati in:", normalizePath(CARTELLA_OUTPUT), "\n")