# ============================================================================
# GRAPHIC-STYLE COPY of xgboost_dropout_FINALE.R
# Same analysis as the original; only the plots changed (Rubik font, UniBG palette,
# English titles/labels, as in assessment_modelli_C_v2.R). PNG files get the suffix _unibg.
# ============================================================================
# =============================================================================
#  VERSIONE FINALE – XGBoost – Previsione del dropout al primo anno (UniBG)
#  File: xgboost_dropout_FINALE.R
#
#  Dataset: C v2 (coorti dal 2020 con TOLC), cartella "dataset_C_v2":
#           C_v2_train / C_v2_valid / C_v2_test .parquet
#           Trasformazioni delle variabili: vedi LEGGIMI_C_v2.txt
#
#  3 configurazioni (M1 conservativo, M2 base, M3 profondo)  x  2 scenari
#  = 6 modelli
#
#  Scenari (stesso outcome: DROPOUT = abbandono entro il 1° anno):
#    1_early_isee  ottobre, what-if con ISEE raccolto all'immatricolazione:
#                  variabili d'ingresso + ISEE_CLASSE_EARLY (18 variabili)
#    2_mid_sem1    dopo il 1° semestre: + esiti del 1° semestre, borsa,
#                  ISEE_CLASSE_MID (25 variabili)
#  Le variabili di ciascuno scenario NON sono scritte nello script: vengono
#  lette da variabili_scenari_C_v2.csv (colonne EARLY_ISEE e MID, 1 = usare),
#  lo stesso file usato dagli altri modelli del gruppo.
#
#  Per ogni modello:
#    1. addestra su TRAIN, usa VALIDATION per l'early stopping (scelta del
#       numero di alberi);
#    2. ristima il modello finale con il numero di alberi scelto;
#    3. calcola su TRAIN, VALIDATION e TEST: AUC, ASE (errore quadratico
#       medio), sensibilità (= recall), specificità, precisione, F1;
#    4. salva modello, previsioni, importanza delle variabili.
#  Output principali (cartella output_xgboost_FINALE):
#    - metriche_confronto.csv  metriche dei 6 modelli su train/validation/test
#    - xgboost_roc_unibg.png curve ROC (righe = scenari, colonne = modelli) + grafici comuni agli altri modelli
#  Output secondari: grafici di confronto per scenario, curve di
#  apprendimento, importanza delle variabili, previsioni.
#
#  Uso del TEST: solo per le previsioni finali e le curve ROC; non entra nella
#  stima. La scelta del modello si fa sul validation, il test è la verifica.
#
#  Compatibile con xgboost 1.7.x e 2.x / 3.x (CRAN).
# =============================================================================


# -----------------------------------------------------------------------------
# 0. INSTALLAZIONE E CARICAMENTO DELLE LIBRERIE
# -----------------------------------------------------------------------------
#   arrow    -> lettura dei file .parquet
#   xgboost  -> stima dei modelli
#   ggplot2  -> pannello con le curve ROC
#   parallel -> conta dei core del PC (inclusa in R, non va installata)
# Le librerie vengono installate solo se mancano.

if (!requireNamespace("arrow", quietly = TRUE))   install.packages("arrow")
if (!requireNamespace("xgboost", quietly = TRUE)) install.packages("xgboost")
if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2")

library(arrow)
library(xgboost)
library(ggplot2)
library(ragg)      # device agg_png (font Rubik nei png)
library(parallel)

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

cat("\n=== SCRIPT FINALE: dataset C v2, 2 scenari (early + ISEE, mid), M1-M2-M3 ===\n\n")
cat("Versione R:", R.version.string, "\n")
cat("Versione arrow:", as.character(packageVersion("arrow")), "\n")
cat("Versione xgboost:", as.character(packageVersion("xgboost")), "\n")


# -----------------------------------------------------------------------------
# 1. PERCORSI E IMPOSTAZIONI GENERALI
# -----------------------------------------------------------------------------
## PERCORSI (stesso approccio degli script kNN / RF / Naive Bayes)
DATA_DIR <- "C:\\Users\\Dario\\Downloads\\Grafici sistemati\\dataset_C"                  # parquet e CSV delle variabili
BASE     <- "C:\\Users\\Dario\\Downloads\\Grafici sistemati\\XGBoost"
OUT_DIR  <- file.path(BASE, "xgboost_v2")        # previsioni, metriche, grafici (creata se non esiste)
CARTELLA_BASE   <- BASE                          # nomi usati nel resto dello script
CARTELLA_DATI   <- DATA_DIR
CARTELLA_OUTPUT <- OUT_DIR

DATASET <- list(
  C = list(descrizione = "C v2 - con TOLC, coorti dal 2020",
           train = "C_v2_train.parquet",
           valid = "C_v2_valid.parquet",
           test  = "C_v2_test.parquet")
)

TARGET         <- "DROPOUT"                     # 1 = abbandono, 0 = prosegue
FILE_VARIABILI <- "variabili_scenari_C_v2.csv"  # elenco variabili per scenario

# Gli scenari: 'colonna' è la colonna del CSV che indica le variabili da usare.
# Per saltarne uno: esegui = FALSE.
SCENARI <- list(
  list(nome = "1_early_isee", esegui = TRUE, colonna = "EARLY_ISEE",
       etichetta = "Early"),
  list(nome = "2_mid_sem1",   esegui = TRUE, colonna = "MID",
       etichetta = "Mid (after 1st semester)")
)

SOGLIA_CLASSIFICAZIONE <- 0.5   # probabilità oltre la quale uno studente è
                                # classificato come dropout; serve solo per
                                # sensibilità, specificità, precisione e F1
                                # (non per AUC e ASE, che non usano soglie).
SEME                   <- 2026  # riproducibilità

# Lettura dell'elenco delle variabili e controllo di coerenza con i dati
ELENCO_VARIABILI <- read.csv(file.path(CARTELLA_DATI, FILE_VARIABILI), stringsAsFactors = FALSE)
variabili_scenario <- function(colonna) ELENCO_VARIABILI$variabile[ELENCO_VARIABILI[[colonna]] == 1]
for (sc in SCENARI) cat("Variabili", sc$nome, ":", length(variabili_scenario(sc$colonna)), "\n")


# -----------------------------------------------------------------------------
# 2. LE 3 CONFIGURAZIONI DI PARAMETRI                <<< MODIFICABILI >>>
# -----------------------------------------------------------------------------
# Promemoria su cosa controlla ogni parametro:
#
#  eta              learning rate: quanto pesa ogni nuovo albero (0.01–0.3).
#                   Più basso = apprendimento lento e stabile, servono più alberi.
#  max_depth        profondità degli alberi (3–8): quante variabili possono
#                   interagire. Più alto = modello più flessibile, più overfitting.
#  min_child_weight osservazioni minime in una foglia (1–20). Più alto = regole
#                   più robuste, modello più conservativo.
#  gamma            miglioramento minimo per accettare uno split (0–5).
#                   Più alto = alberi più "potati".
#  subsample        quota di studenti usata per ogni albero (0.5–1).
#  colsample_bytree quota di variabili usata per ogni albero (0.5–1).
#                   Valori < 1 riducono l'overfitting introducendo casualità.
#  lambda / alpha   penalità L2 / L1 sui pesi delle foglie. Più alte = previsioni
#                   più prudenti.
#  scale_pos_weight peso dei dropout rispetto ai non-dropout. 1 = probabilità
#                   calibrate; "auto" = rapporto non-dropout/dropout del train
#                   (circa 3): aumenta la sensibilità ma gonfia le probabilità.
#
# Nel primo giro erano 5 configurazioni (M1-M5). Ne restano 3, in ordine di
# complessità crescente: M1 conservativo, M2 base, M3 profondo. M3 è tenuto
# apposta per mostrare l'overfitting (AUC train molto sopra la validation) e
# motivarne lo scarto. Scartati dopo il primo giro (vedi in fondo alla lista):
#  - M4 regolarizzato: AUC e ASE praticamente uguali a M1, quindi un doppione;
#  - M5 bilanciato: AUC in linea, ma ASE molto peggiore (circa 0,22 contro 0,17
#    nell'early) perché il peso dei dropout gonfia le probabilità.
# -----------------------------------------------------------------------------

MODELLI <- list(

  # M1 – CONSERVATIVO
  # Motivazione: è il punto di riferimento "difficile da sbagliare". Con circa
  #   25% di dropout e molte dummy (corso, tipo di maturità, regione), un modello
  #   semplice evita di inseguire combinazioni rare che non si ripetono.
  # Cosa cambia nella stima:
  #   - max_depth 3: ogni albero usa al massimo 3 variabili in sequenza, quindi
  #     coglie effetti principali e interazioni semplici (es. corso x esami);
  #   - min_child_weight 10 e gamma 1: una regola viene creata solo se riguarda
  #     abbastanza studenti e migliora davvero la previsione;
  #   - eta 0,03: ogni albero corregge poco gli errori precedenti, servono più
  #     alberi ma la curva è più stabile;
  #   - lambda 5: le probabilità delle foglie vengono "tirate" verso la media,
  #     previsioni meno estreme.
  #   Atteso: AUC train e validation vicine; AUC forse leggermente più bassa.
  M1_conservativo = list(eta = 0.03, max_depth = 3, min_child_weight = 10, gamma = 1,
                         subsample = 0.8, colsample_bytree = 0.8,
                         lambda = 5, alpha = 0, scale_pos_weight = 1),

  # M2 – BASE
  # Motivazione: impostazione standard consigliata per dati tabellari di queste
  #   dimensioni; fa da termine di paragone per gli altri.
  # Cosa cambia nella stima rispetto a M1:
  #   - profondità 4 e foglie più piccole (min 5): interazioni fino a 4
  #     variabili, regole su gruppi più piccoli;
  #   - eta 0,05 e lambda 1: apprendimento più rapido, meno penalità.
  #   Atteso: AUC un po' più alta di M1, con un po' più di distanza train-valid.
  M2_base        = list(eta = 0.05, max_depth = 4, min_child_weight = 5, gamma = 0,
                        subsample = 0.8, colsample_bytree = 0.8,
                        lambda = 1, alpha = 0, scale_pos_weight = 1),

  # M3 – PROFONDO
  # Motivazione: il dropout può dipendere da combinazioni di più fattori (es.
  #   un certo corso + scuola tecnica + nessun esame nel primo semestre).
  #   Alberi più profondi possono cogliere queste interazioni.
  # Cosa cambia nella stima:
  #   - max_depth 6: fino a 6 variabili per regola, molte più foglie per albero;
  #   - min_child_weight 3: regole anche su gruppi piccoli (rischio di imparare
  #     casi particolari del train);
  #   - colsample 0,7: ogni albero vede solo il 70% delle variabili, per
  #     limitare l'overfitting dovuto alla maggiore profondità.
  #   Atteso: AUC train più alta; se l'AUC validation non sale, le interazioni
  #   complesse non servono e conviene un modello più semplice.
  M3_profondo    = list(eta = 0.05, max_depth = 6, min_child_weight = 3, gamma = 0,
                        subsample = 0.8, colsample_bytree = 0.7,
                        lambda = 1, alpha = 0, scale_pos_weight = 1)
)

# Configurazioni da stimare (versione finale: tutte e 3).
MODELLI_DA_STIMARE <- names(MODELLI)   # tutte e 3; per una sola: c("M1_conservativo")
MODELLI <- MODELLI[MODELLI_DA_STIMARE]

# Configurazioni scartate dopo il primo giro (documentazione, non vengono stimate).
# Per rimetterle basta riportarle dentro la lista MODELLI.
# # M4 – REGOLARIZZATO
# # Motivazione: dopo la codifica ci sono 90–150 colonne, molte dummy poco
# #   frequenti e rumorose. Un modello con molta casualità e penalità tende a
# #   generalizzare meglio quando il segnale è distribuito su poche variabili
# #   forti e molte deboli.
# # Cosa cambia nella stima:
# #   - subsample 0,6 e colsample 0,5: ogni albero usa il 60% degli studenti e
# #     metà delle variabili: alberi più diversi tra loro, effetto "media";
# #   - min_child_weight 20 e gamma 2: solo regole su gruppi ampi e utili;
# #   - lambda 10 e alpha 1: forte penalità; alpha (L1) può azzerare il
# #     contributo di foglie poco utili, simile a una selezione di variabili.
# #   Atteso: AUC simile o di poco inferiore, minima distanza train-valid,
# #   importanza delle variabili più "pulita".
#   M4_regolarizzato = list(eta = 0.05, max_depth = 5, min_child_weight = 20, gamma = 2,
#                           subsample = 0.6, colsample_bytree = 0.5,
#                           lambda = 10, alpha = 1, scale_pos_weight = 1),
#
# # M5 – BILANCIATO
# # Motivazione: i dropout sono la classe minoritaria (circa 1 su 4); con la
# #   soglia 0,5 un modello standard ne riconosce pochi. Pesandoli di più si
# #   vede l'effetto della correzione dello sbilanciamento.
# # Cosa cambia nella stima:
# #   - scale_pos_weight = rapporto non-dropout/dropout del train (circa 3):
# #     nella funzione di perdita ogni dropout conta circa come 3 studenti;
# #   - il resto è identico a M2, quindi la differenza è dovuta solo al peso.
# #   Atteso: AUC quasi uguale a M2 (l'ordinamento cambia poco), sensibilità
# #   più alta e precisione più bassa alla soglia 0,5, ASE peggiore perché le
# #   probabilità sono gonfiate. Non adatto alla matrice di payoff senza
# #   ricalibrazione.
#   M5_bilanciato  = list(eta = 0.05, max_depth = 4, min_child_weight = 5, gamma = 0,
#                         subsample = 0.8, colsample_bytree = 0.8,
#                         lambda = 1, alpha = 0, scale_pos_weight = "auto")

# Numero di alberi ed early stopping (uguali per tutti i modelli)
NROUNDS_MAX    <- 3000   # tetto massimo di alberi
EARLY_STOPPING <- 100    # si ferma se la metrica sul validation non migliora per
                         # N alberi consecutivi

# Metrica usata per l'early stopping:
#  "auc"     = capacità di ordinare gli studenti per rischio (consigliata);
#  "logloss" = qualità delle probabilità (premia anche la calibrazione);
#  "aucpr"   = area sotto la curva precision-recall.
METRICA <- "auc"

NTHREAD <- max(1, parallel::detectCores() - 1)   # core usati


# -----------------------------------------------------------------------------
# 3. FUNZIONI (non serve modificarle)
# -----------------------------------------------------------------------------

leggi <- function(file) as.data.frame(arrow::read_parquet(file.path(CARTELLA_DATI, file)))

# Codifica le variabili e restituisce matrici con le stesse colonne per tutti
# gli split: variabili logiche -> 0/1, categoriche -> dummy (una colonna per
# livello). I livelli sono l'unione di quelli presenti negli split: è solo una
# questione di codifica, non introduce informazione dal validation.
prepara_matrici <- function(lista_df, target, escludi) {
  nomi_split <- names(lista_df)
  y <- lapply(lista_df, function(d) as.numeric(d[[target]]))
  X <- lapply(lista_df, function(d) d[, setdiff(names(d), c(target, escludi)), drop = FALSE])

  tutte <- do.call(rbind, lapply(nomi_split, function(s) cbind(X[[s]], .split = s)))
  for (v in setdiff(names(tutte), ".split")) {
    if (is.logical(tutte[[v]]))   tutte[[v]] <- as.integer(tutte[[v]])
    if (is.character(tutte[[v]])) tutte[[v]] <- factor(tutte[[v]])
    if (is.factor(tutte[[v]]))    tutte[[v]] <- droplevels(tutte[[v]])
  }
  fattori <- names(tutte)[vapply(tutte, is.factor, logical(1))]
  costanti <- fattori[vapply(tutte[fattori], nlevels, integer(1)) < 2]
  if (length(costanti) > 0) {
    tutte[costanti] <- NULL
    fattori <- setdiff(fattori, costanti)
  }
  split_col <- tutte$.split
  tutte$.split <- NULL

  mm <- model.matrix(~ . - 1, data = tutte,
                     contrasts.arg = lapply(tutte[fattori], contrasts, contrasts = FALSE))
  colnames(mm) <- make.names(colnames(mm), unique = TRUE)

  out <- lapply(nomi_split, function(s) list(X = mm[split_col == s, , drop = FALSE], y = y[[s]]))
  names(out) <- nomi_split
  out
}

# Chiamata a xgb.train compatibile con le diverse versioni del pacchetto
# (1.7 usa 'watchlist', 2.x/3.x usano 'evals').
addestra <- function(params, dtrain, dvalid, nrounds, early = NULL) {
  args <- list(params = params, data = dtrain, nrounds = nrounds, verbose = 0)
  lista_eval <- list(train = dtrain, valid = dvalid)
  if (packageVersion("xgboost") >= "2.0.0") args$evals <- lista_eval else args$watchlist <- lista_eval
  if (!is.null(early)) args$early_stopping_rounds <- early
  set.seed(SEME)
  do.call(xgb.train, args)
}

log_valutazione <- function(modello) {
  el <- tryCatch(attributes(modello)$evaluation_log, error = function(e) NULL)
  if (is.null(el)) el <- tryCatch(modello$evaluation_log, error = function(e) NULL)
  as.data.frame(el)
}

miglior_iterazione <- function(modello, metrica) {
  el  <- log_valutazione(modello)
  col <- grep("^valid_", names(el), value = TRUE)[1]
  if (metrica %in% c("auc", "aucpr")) which.max(el[[col]]) else which.min(el[[col]])
}

# AUC con la formula di Mann-Whitney (nessun pacchetto aggiuntivo)
auc_mw <- function(y, p) {
  r <- rank(p); n1 <- sum(y == 1); n0 <- sum(y == 0)
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

# Metriche di confronto:
#  AUC          capacità di ordinare gli studenti per rischio (0,5 = a caso, 1 = perfetto)
#  ASE          errore quadratico medio delle probabilità, mean((p - y)^2),
#               detto anche Brier score; riferimento "senza modello" = q(1 - q),
#               con q quota di dropout (circa 0,19 con q = 0,25)
#  sensibilita  quota di dropout veri classificati come dropout (= recall)
#  specificita  quota di non-dropout classificati come non-dropout
#  precisione   quota di dropout veri tra i classificati come dropout
#  F1           media armonica di precisione e recall
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

# Etichette e colori usati nei grafici (sezione 8)
COLORI_MODELLI <- unibg_palette[c(1, 3, 5, 6, 7)]
MOD_EN <- c(M1_conservativo = "M1 Conservative", M2_base = "M2 Base", M3_profondo = "M3 Deep")
SC_EN  <- c("1_early_isee" = "Early", "2_mid_sem1" = "Mid (after 1st semester)")


# Curva di apprendimento di ogni configurazione (train e validation vs numero di alberi)
grafico_apprendimento <- function(etichetta, el, file) {
  col_tr <- grep("^train_", names(el), value = TRUE)[1]
  col_va <- grep("^valid_", names(el), value = TRUE)[1]
  d <- rbind(data.frame(iter = el$iter, valore = el[[col_tr]], campione = "train"),
             data.frame(iter = el$iter, valore = el[[col_va]], campione = "validation"))
  d$campione <- factor(d$campione, levels = c("train", "validation"))
  g <- ggplot(d, aes(iter, valore, colour = campione)) +
    geom_line(linewidth = 0.9) +
    scale_colour_manual(values = unibg_palette[c(6, 1)]) +
    labs(title = paste("Learning curve -", etichetta),
         x = "Number of trees", y = toupper(METRICA)) + tema
  ggsave(file, g, width = 9, height = 5.5, dpi = 300, device = agg_png, bg = "white")
}

# Grafici di confronto tra le configurazioni, un pannello per dataset
grafici_confronto <- function(ris, scenario, cartella) {
  ris <- ris[ris$scenario == scenario, ]
  dataset <- intersect(names(DATASET), unique(ris$dataset))
  modelli <- names(MODELLI)
  col <- setNames(COLORI_MODELLI[seq_along(modelli)], modelli)
  etichette <- unname(ifelse(is.na(MOD_EN[modelli]), modelli, MOD_EN[modelli]))
  lab_mod <- setNames(etichette, modelli)
  sc_lab <- if (scenario %in% names(SC_EN)) SC_EN[[scenario]] else scenario

  # one row per (dataset, model), in the order of MODELLI
  rr <- do.call(rbind, lapply(dataset, function(d) {
    r <- ris[ris$dataset == d, ][match(modelli, ris$modello[ris$dataset == d]), ]
    r$mod_f <- factor(modelli, levels = modelli)
    r$pannello <- paste("Dataset", d)
    r
  }))

  # (1) AUC: bars = validation, circles = train (gap = overfitting)
  lo <- max(0, floor(min(rr$valid_AUC, rr$train_AUC) * 20) / 20 - 0.05)
  hi <- min(1, ceiling(max(rr$valid_AUC, rr$train_AUC) * 20) / 20 + 0.02)
  g1 <- ggplot(rr, aes(mod_f, valid_AUC, fill = mod_f)) +
    geom_col(width = 0.75) +
    geom_point(aes(y = train_AUC), shape = 21, fill = "white", colour = unibg_palette[1], size = 3.5, stroke = 1) +
    geom_text(aes(y = pmax(valid_AUC, train_AUC), label = sprintf("%.3f", valid_AUC)), vjust = -0.9, size = 3) +
    facet_wrap(~ pannello) +
    scale_fill_manual(values = col, guide = "none") +
    scale_x_discrete(labels = lab_mod) +
    coord_cartesian(ylim = c(lo, hi)) +
    labs(title = paste("AUC by configuration -", sc_lab),
         x = NULL, y = "AUC (bars = validation, circles = train)") + tema
  ggsave(file.path(cartella, paste0("confronto_AUC_", scenario, "_unibg.png")), g1,
         width = 10, height = 5.5, dpi = 300, device = agg_png, bg = "white")

  # (2) ASE: bars = validation, dashed line = no-model reference
  rr$rif <- rr$valid_prevalenza * (1 - rr$valid_prevalenza)
  g2 <- ggplot(rr, aes(mod_f, valid_ASE, fill = mod_f)) +
    geom_col(width = 0.75) +
    geom_hline(aes(yintercept = rif), linetype = "dashed", colour = "grey40") +
    geom_text(aes(label = sprintf("%.3f", valid_ASE)), vjust = -0.4, size = 3) +
    facet_wrap(~ pannello) +
    scale_fill_manual(values = col, guide = "none") +
    scale_x_discrete(labels = lab_mod) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
    labs(title = paste("Validation ASE -", sc_lab),
         x = NULL, y = "ASE (dashed line = no-model reference)") + tema
  ggsave(file.path(cartella, paste0("confronto_ASE_", scenario, "_unibg.png")), g2,
         width = 10, height = 5.5, dpi = 300, device = agg_png, bg = "white")

  # (3) Classification at threshold: sensitivity, specificity, precision, F1
  metr <- c("valid_sensibilita", "valid_specificita", "valid_precisione", "valid_F1")
  nomi_metr <- c("Sensitivity", "Specificity", "Precision", "F1")
  cl <- do.call(rbind, lapply(seq_along(metr), function(j)
    data.frame(pannello = rr$pannello, mod_f = rr$mod_f, metrica = nomi_metr[j], valore = rr[[metr[j]]])))
  cl$metrica <- factor(cl$metrica, levels = nomi_metr)
  g3 <- ggplot(cl, aes(metrica, valore, fill = mod_f)) +
    geom_col(position = position_dodge(width = 0.85), width = 0.8) +
    geom_text(aes(label = sprintf("%.2f", valore)), position = position_dodge(width = 0.85),
              vjust = -0.4, size = 2.8, na.rm = TRUE) +
    facet_wrap(~ pannello) +
    scale_fill_manual(values = col, labels = lab_mod) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
    coord_cartesian(ylim = c(0, 1.05)) +
    labs(title = paste("Classification at threshold", SOGLIA_CLASSIFICAZIONE, "-", sc_lab),
         x = NULL, y = NULL) + tema
  ggsave(file.path(cartella, paste0("confronto_classificazione_", scenario, "_unibg.png")), g3,
         width = 10, height = 5.5, dpi = 300, device = agg_png, bg = "white")
}


# -----------------------------------------------------------------------------
# 4. FUNZIONE CHE STIMA UNO SCENARIO: per ogni dataset, le 3 configurazioni
# -----------------------------------------------------------------------------
esegui_scenario <- function(scenario, etichetta, usa) {
  cartella <- file.path(CARTELLA_OUTPUT, scenario)
  dir.create(cartella, showWarnings = FALSE, recursive = TRUE)
  righe <- list()

  for (nome in names(DATASET)) {
    cfg <- DATASET[[nome]]
    cat("\n==============================================================\n")
    cat(" ", etichetta, "| Dataset", nome, "-", cfg$descrizione, "\n")
    cat("==============================================================\n")

    # --- Lettura e preparazione (una volta per dataset) ----------------------
    dati <- list(train = leggi(cfg$train), valid = leggi(cfg$valid), test = leggi(cfg$test))
    mancanti <- setdiff(usa, names(dati$train))
    if (length(mancanti)) stop("Variabili del CSV assenti nel dataset: ", paste(mancanti, collapse = ", "))
    na <- vapply(dati, function(d) sum(is.na(d[, usa])), numeric(1))
    if (any(na > 0)) stop("Ci sono NA nelle variabili usate: model.matrix scarterebbe righe")
    escluse <- setdiff(names(dati$train), c(usa, TARGET))
    m <- prepara_matrici(dati, TARGET, escluse)
    cat("Variabili escluse:", if (length(escluse)) paste(escluse, collapse = ", ") else "nessuna", "\n")
    cat("Righe train:", nrow(m$train$X), "| validation:", nrow(m$valid$X),
        "| test:", nrow(m$test$X), "| colonne dopo la codifica:", ncol(m$train$X), "\n")
    cat("Quota dropout train:", round(mean(m$train$y), 3),
        "| validation:", round(mean(m$valid$y), 3), "\n")

    dtrain <- xgb.DMatrix(m$train$X, label = m$train$y)
    dvalid <- xgb.DMatrix(m$valid$X, label = m$valid$y)
    dtest  <- xgb.DMatrix(m$test$X,  label = m$test$y)   # solo per prevedere
    rapporto_classi <- sum(m$train$y == 0) / sum(m$train$y == 1)

    # --- Le 3 configurazioni -------------------------------------------------
    for (mod_nome in names(MODELLI)) {
      pr <- MODELLI[[mod_nome]]
      if (identical(pr$scale_pos_weight, "auto")) pr$scale_pos_weight <- rapporto_classi
      params <- c(list(objective = "binary:logistic", eval_metric = METRICA,
                       nthread = NTHREAD, seed = SEME), pr)

      # 1° passaggio: early stopping sul validation
      mod_es   <- addestra(params, dtrain, dvalid, NROUNDS_MAX, EARLY_STOPPING)
      n_alberi <- miglior_iterazione(mod_es, METRICA)

      # 2° passaggio: modello finale con il numero di alberi scelto
      modello <- addestra(params, dtrain, dvalid, n_alberi, early = NULL)
      el <- log_valutazione(modello)

      p_train   <- predict(modello, dtrain)
      p_valid   <- predict(modello, dvalid)
      p_test    <- predict(modello, dtest)
      met_train <- metriche(m$train$y, p_train, SOGLIA_CLASSIFICAZIONE)
      met_valid <- metriche(m$valid$y, p_valid, SOGLIA_CLASSIFICAZIONE)
      met_test  <- metriche(m$test$y,  p_test,  SOGLIA_CLASSIFICAZIONE)

      cat(sprintf("  %-18s alberi %4d | AUC train %.3f valid %.3f test %.3f | ASE valid %.3f | sens %.2f spec %.2f prec %.2f F1 %.2f\n",
                  mod_nome, n_alberi, met_train["AUC"], met_valid["AUC"], met_test["AUC"], met_valid["ASE"],
                  met_valid["sensibilita"], met_valid["specificita"],
                  met_valid["precisione"], met_valid["F1"]))

      # Salvataggi
      pref <- file.path(cartella, paste0(nome, "_", mod_nome))
      imp  <- as.data.frame(xgb.importance(feature_names = colnames(m$train$X), model = modello))
      xgb.save(modello, paste0(pref, "_modello.json"))
      write.csv(data.frame(riga = seq_along(p_train), y = m$train$y, p = p_train),
                paste0(pref, "_previsioni_train.csv"), row.names = FALSE)
      write.csv(data.frame(riga = seq_along(p_valid), y = m$valid$y, p = p_valid),
                paste0(pref, "_previsioni_validation.csv"), row.names = FALSE)
      write.csv(data.frame(riga = seq_along(p_test), y = m$test$y, p = p_test),
                paste0(pref, "_previsioni_test.csv"), row.names = FALSE)
      write.csv(imp, paste0(pref, "_importanza.csv"), row.names = FALSE)
      write.csv(el,  paste0(pref, "_log_alberi.csv"), row.names = FALSE)
      write.csv(log_valutazione(mod_es), paste0(pref, "_log_alberi_early_stopping.csv"), row.names = FALSE)
      grafico_apprendimento(paste(nome, mod_nome, "-", scenario), el,
                            paste0(pref, "_curva_apprendimento_unibg.png"))

      righe[[paste(nome, mod_nome)]] <- as.data.frame(c(
        list(scenario = scenario, dataset = nome, descrizione = cfg$descrizione,
             modello = mod_nome, n_alberi = n_alberi, colonne = ncol(m$train$X),
             valid_prevalenza = mean(m$valid$y)),
        pr,
        setNames(as.list(met_train), paste0("train_", names(met_train))),
        setNames(as.list(met_valid), paste0("valid_", names(met_valid))),
        setNames(as.list(met_test),  paste0("test_",  names(met_test)))),
        check.names = FALSE)
    }
  }
  ris <- do.call(rbind, righe)
  grafici_confronto(ris, scenario, cartella)
  ris
}


# -----------------------------------------------------------------------------
# 5. ESECUZIONE
# -----------------------------------------------------------------------------
dir.create(CARTELLA_OUTPUT, showWarnings = FALSE, recursive = TRUE)
risultati <- list()

for (sc in SCENARI) {
  if (!isTRUE(sc$esegui)) next
  risultati[[sc$nome]] <- esegui_scenario(
    scenario  = sc$nome,
    etichetta = sc$etichetta,
    usa       = variabili_scenario(sc$colonna))
}


# -----------------------------------------------------------------------------
# 6. RIEPILOGO FINALE DEI MODELLI
# -----------------------------------------------------------------------------
# Come leggere il confronto:
#  - tra configurazioni (stesso dataset e scenario): differenze di AUC sotto
#    0,005 sono rumore; a parità di AUC preferire il modello più semplice e
#    con meno distanza tra AUC train e validation;
#  - tra scenari (stessi studenti): la differenza di AUC misura quanto si
#    guadagna aspettando gli esiti del 1° semestre;
#  - test_AUC e test_ASE sono la verifica finale: non usarli per scegliere tra
#    le configurazioni (altrimenti il test diventa un secondo validation).
riepilogo <- do.call(rbind, risultati)
rownames(riepilogo) <- NULL
write.csv(riepilogo, file.path(CARTELLA_OUTPUT, "metriche_confronto.csv"), row.names = FALSE)

colonne_chiave <- c("scenario", "dataset", "modello", "n_alberi",
                    "train_AUC", "valid_AUC", "test_AUC", "valid_ASE", "test_ASE",
                    "valid_sensibilita", "valid_specificita", "valid_precisione", "valid_F1")
cat("\n=================== RIEPILOGO (validation) ===================\n")
print(format(riepilogo[, colonne_chiave], digits = 3), row.names = FALSE)

cat("\nMiglior modello per AUC in ogni scenario e dataset:\n")
migliori <- do.call(rbind, lapply(split(riepilogo, list(riepilogo$scenario, riepilogo$dataset), drop = TRUE),
                                  function(d) d[which.max(d$valid_AUC), colonne_chiave]))
print(format(migliori, digits = 3), row.names = FALSE)
write.csv(migliori, file.path(CARTELLA_OUTPUT, "migliori_per_AUC.csv"), row.names = FALSE)

cat("\nOutput salvati in:", normalizePath(CARTELLA_OUTPUT), "\n")


# -----------------------------------------------------------------------------
# 7. GRAFICI (stessa serie degli altri modelli)
# -----------------------------------------------------------------------------
# Legge le previsioni salvate sopra. Per effetto soglia e lift si usa, in ogni scenario,
# la configurazione con la distanza minima tra AUC train e AUC validation (stessa regola dell'assessment:
# sul validation le tre configurazioni sono praticamente alla pari, quindi si sceglie la meno overfittata).
attivi   <- Filter(function(sc) isTRUE(sc$esegui), SCENARI)
th <- tema
SC_LAB   <- SC_EN
modelli  <- names(MODELLI)
lab_mod  <- setNames(unname(ifelse(is.na(MOD_EN[modelli]), modelli, MOD_EN[modelli])), modelli)
COL_BASE <- setNames(COLORI_MODELLI[seq_along(modelli)], modelli)
SETS     <- c("train", "validation", "test")

prev_long <- do.call(rbind, lapply(attivi, function(sc) do.call(rbind, lapply(modelli, function(mod)
  do.call(rbind, lapply(SETS, function(s) {
    d <- read.csv(file.path(CARTELLA_OUTPUT, sc$nome, paste0("C_", mod, "_previsioni_", s, ".csv")))
    data.frame(scenario = sc$nome, modello = mod, set = s, y = d$y, prob = d$p)
  }))))))
scelti_x <- do.call(rbind, lapply(split(riepilogo, riepilogo$scenario),
                                  function(d) d[which.min(abs(d$train_AUC - d$valid_AUC)), c("scenario", "modello")]))
ps <- prev_long[paste(prev_long$scenario, prev_long$modello) %in% paste(scelti_x$scenario, scelti_x$modello), ]

# 0. ASE train e validation per configurazione (M1 -> M3: complessita' crescente; cambiano insieme profondita',
#    learning rate, regolarizzazione e colsample, non un solo parametro); rombo = configurazione scelta
ase_cfg <- rbind(data.frame(riepilogo[, c("scenario", "modello")], campione = "train",      ASE = riepilogo$train_ASE),
                 data.frame(riepilogo[, c("scenario", "modello")], campione = "validation", ASE = riepilogo$valid_ASE))
ase_cfg$x <- match(ase_cfg$modello, modelli)
ase_cfg$pannello <- factor(SC_LAB[ase_cfg$scenario], levels = SC_LAB)
scelta_cfg <- merge(riepilogo[, c("scenario", "modello", "valid_ASE")], scelti_x, by = c("scenario", "modello"))
scelta_cfg$x <- match(scelta_cfg$modello, modelli)
scelta_cfg$pannello <- factor(SC_LAB[scelta_cfg$scenario], levels = SC_LAB)
g0 <- ggplot(ase_cfg, aes(x = x, y = ASE, color = campione)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  geom_point(data = scelta_cfg, aes(x = x, y = valid_ASE), color = unibg_palette[1], size = 5, shape = 18, inherit.aes = FALSE) +
  facet_wrap(~ pannello, scales = "free", ncol = 2) +
  scale_color_manual(values = unibg_palette[c(6, 3)]) +
  scale_x_continuous(breaks = seq_along(modelli), labels = unname(lab_mod)) +
  labs(title = "XGBoost: model complexity vs ASE",
       x = "Configuration (increasing complexity)", y = "ASE", color = NULL) + th
ggsave(file.path(OUT_DIR, "xgboost_complessita_vs_ase_unibg.png"), g0,
       width = 11, height = 5, dpi = 300, device = agg_png, bg = "white")

# 1. AUC sul validation al variare del numero di alberi (percorso dell'early stopping); rombo = numero di alberi scelto
curve_trees <- do.call(rbind, lapply(attivi, function(sc) do.call(rbind, lapply(modelli, function(mod) {
  el <- read.csv(file.path(CARTELLA_OUTPUT, sc$nome, paste0("C_", mod, "_log_alberi_early_stopping.csv")))
  data.frame(scenario = sc$nome, modello = mod, iter = el$iter,
             valore = el[[grep("^valid_", names(el), value = TRUE)[1]]])
}))))
curve_trees$modello <- factor(curve_trees$modello, levels = modelli)
best_pts <- riepilogo[, c("scenario", "modello", "n_alberi", "valid_AUC")]
best_pts$modello <- factor(best_pts$modello, levels = modelli)
g1 <- ggplot(curve_trees, aes(iter, valore, colour = modello)) +
  geom_line(linewidth = 1) +
  geom_point(data = best_pts, aes(n_alberi, valid_AUC, colour = modello), inherit.aes = FALSE, shape = 18, size = 5) +
  facet_wrap(~ scenario, scales = "free_y", labeller = as_labeller(SC_LAB)) +
  scale_colour_manual(values = COL_BASE, labels = lab_mod) +
  labs(title = "XGBoost: validation AUC by number of trees",
       x = "Number of trees", y = "AUC (validation)", colour = NULL) + th
ggsave(file.path(OUT_DIR, "xgboost_auc_validation_vs_trees_unibg.png"), g1, width = 12, height = 6, dpi = 300, device = agg_png, bg = "white")

# 2. test performance per configurazione
bar_long <- rbind(data.frame(riepilogo[, c("scenario", "modello")], valore = riepilogo$test_AUC, metrica = "AUC"),
                  data.frame(riepilogo[, c("scenario", "modello")], valore = riepilogo$test_ASE, metrica = "ASE"))
bar_long$modello <- factor(bar_long$modello, levels = modelli)
g2 <- ggplot(bar_long, aes(modello, valore, fill = modello)) +
  geom_col(position = position_dodge(width = 0.85), width = 0.8) +
  geom_text(aes(label = sprintf("%.3f", valore)), position = position_dodge(width = 0.85), vjust = -0.4, size = 2.8) +
  facet_grid(metrica ~ scenario, scales = "free_y", labeller = labeller(scenario = SC_LAB)) +
  scale_fill_manual(values = COL_BASE, guide = "none") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  scale_x_discrete(labels = lab_mod) +
  labs(title = "XGBoost: test performance", x = NULL, y = NULL) + th +
  theme(axis.text.x = element_text(angle = 0))
ggsave(file.path(OUT_DIR, "xgboost_test_performance_unibg.png"), g2, width = 12, height = 7, dpi = 300, device = agg_png, bg = "white")

# 3. misure di classificazione al variare della soglia (configurazione scelta, validation)
THR_GRID <- seq(0, 1, by = 0.02)
sw <- do.call(rbind, lapply(split(ps[ps$set == "validation", ], ps$scenario[ps$set == "validation"]), function(d) {
  do.call(rbind, lapply(THR_GRID, function(t) {
    pred <- d$prob >= t
    TP <- sum(pred & d$y == 1); FP <- sum(pred & d$y == 0)
    FN <- sum(!pred & d$y == 1); TN <- sum(!pred & d$y == 0)
    data.frame(scenario = d$scenario[1], soglia = t, accuracy = (TP + TN) / nrow(d),
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
  labs(title = "XGBoost: classification measurements varying threshold",
       x = "Dropout probability threshold", y = NULL, colour = NULL) + th
ggsave(file.path(OUT_DIR, "xgboost_effetto_soglia_unibg.png"), g3, width = 11, height = 5.5, dpi = 300, device = agg_png, bg = "white")

# 4. lift cumulato della configurazione scelta (validation e test)
lift_decili <- function(y, p) {
  dec <- ceiling(10 * rank(-p, ties.method = "first") / length(p))
  ev  <- as.numeric(tapply(y, dec, sum)); nn <- as.numeric(tapply(y, dec, length))
  data.frame(percentile = 1:10 * 10, lift = (cumsum(ev) / cumsum(nn)) / mean(y))
}
lift <- do.call(rbind, lapply(scelti_x$scenario, function(sc) do.call(rbind, lapply(c("validation", "test"), function(s) {
  d <- ps[ps$scenario == sc & ps$set == s, ]
  cbind(scenario = sc, set = s, lift_decili(d$y, d$prob))
}))))
g4 <- ggplot(lift, aes(percentile, lift, colour = set)) +
  geom_hline(yintercept = 1, linetype = 2, colour = "grey60") +
  geom_line(linewidth = 1) + geom_point(size = 2) +
  facet_wrap(~ scenario, labeller = as_labeller(SC_LAB)) +
  scale_x_continuous(breaks = 1:10 * 10) +
  scale_colour_manual(values = c(validation = unibg_palette[3], test = unibg_palette[6])) +
  labs(title = "XGBoost: cumulative lift of the selected model", x = "% of students selected (deciles)",
       y = "Cumulative lift", colour = NULL) + th
ggsave(file.path(OUT_DIR, "xgboost_lift_cumulato_unibg.png"), g4, width = 11, height = 5.5, dpi = 300, device = agg_png, bg = "white")

# 5. curve ROC (train, validation, test): righe = scenario, colonne = modello
punti_roc <- function(y, p) {
  o  <- order(p, decreasing = TRUE); y <- y[o]; p <- p[o]
  tp <- cumsum(y == 1) / sum(y == 1); fp <- cumsum(y == 0) / sum(y == 0)
  ult <- c(diff(p) != 0, TRUE)                 # un punto per soglia distinta
  r <- data.frame(fpr = c(0, fp[ult]), tpr = c(0, tp[ult]))
  r[!duplicated(round(r$fpr, 3), fromLast = TRUE) | r$fpr %in% c(0, 1), ]
}
roc_df <- do.call(rbind, lapply(split(prev_long, list(prev_long$scenario, prev_long$modello, prev_long$set), drop = TRUE), function(d)
  cbind(punti_roc(d$y, d$prob), scenario = d$scenario[1], modello = d$modello[1], set = d$set[1], auc = auc_mw(d$y, d$prob))))
roc_df$set      <- factor(roc_df$set, levels = SETS)
roc_df$modello  <- factor(roc_df$modello, levels = modelli)
roc_df$scenario <- factor(roc_df$scenario, levels = names(SC_LAB), labels = unname(SC_LAB))
auc_lab <- unique(roc_df[, c("scenario", "modello", "set", "auc")])
auc_lab$y <- c(train = 0.25, validation = 0.15, test = 0.05)[as.character(auc_lab$set)]
auc_lab$txt <- sprintf("AUC %s = %.3f", as.character(auc_lab$set), auc_lab$auc)
COL_SET <- c(train = unibg_palette[6], validation = unibg_palette[1], test = unibg_palette[5])
g5 <- ggplot(roc_df, aes(fpr, tpr, colour = set)) +
  geom_abline(intercept = 0, slope = 1, linetype = 2, colour = "grey60") +
  geom_path(linewidth = 0.9) +
  geom_text(data = auc_lab, aes(x = 1, y = y, label = txt, colour = set), hjust = 1, size = 3.4, show.legend = FALSE) +
  facet_grid(scenario ~ modello, labeller = labeller(modello = lab_mod)) +
  scale_colour_manual(values = COL_SET) +
  coord_equal() +
  labs(title = "XGBoost: ROC curves by scenario (rows) and model (columns)",
       x = "1 - specificity (FPR)", y = "Sensitivity (TPR)", colour = NULL) + th
ggsave(file.path(OUT_DIR, "xgboost_roc_unibg.png"), g5, width = 12, height = 8.5, dpi = 300, device = agg_png, bg = "white")
write.csv(unique(roc_df[, c("scenario", "modello", "set", "auc")]), file.path(OUT_DIR, "auc_roc.csv"), row.names = FALSE)

cat("\nGrafici salvati in:", normalizePath(OUT_DIR), "\n")
