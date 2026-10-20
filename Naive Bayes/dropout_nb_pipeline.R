# ============================================================================
# GRAPHIC-STYLE COPY of dropout_nb_pipeline.R
# Same analysis as the original; only the plots changed (Rubik font, UniBG palette,
# English titles/labels, as in assessment_modelli_C_v2.R). PNG files get the suffix _unibg.
# ============================================================================
rm(list = ls())

################################################################################
#  Naive Bayes per il DROPOUT - dataset C_v2 - scenari EARLY_ISEE e MID  (script unico)
#
#  Stessa struttura, stessi dati, stesse metriche e stessi grafici di knn_dropout_v2.R,
#  cosi' i due modelli sono direttamente confrontabili. NESSUN tuning degli iperparametri:
#  numero di classi (N_BINS) e smoothing di Laplace sono fissati a priori.
#
#  Cosa fa, in un'unica esecuzione:
#   1. legge C_v2 (train / valid / test gia' divisi) e le variabili di ogni scenario dal file
#      variabili_scenari_C_v2.csv (colonne EARLY_ISEE e MID: 1 = usare, 0 = escludere)
#   2. prepara le variabili usando SOLO il train (livelli rari -> "Altro", NA -> "ND",
#      numeriche molto correlate eliminate, numeriche discretizzate in classi)
#   3. stima il Naive Bayes sul TRAIN (nessuna scelta di iperparametri, quindi il validation
#      non serve per selezionare: e' solo un secondo insieme di verifica, come per il kNN)
#   4. valuta con due soglie: 0.5 e il tasso di dropout nel train
#   5. scrive TUTTE le metriche in un unico file, metriche_confronto_nb.csv (stesse colonne di
#      metriche_confronto_knn.csv), piu' previsioni, soglie, lift e grafici
#
#  Le metriche "train" sono in risostituzione (il modello e' valutato sui dati con cui e' stimato):
#  per il kNN sono leave-one-out, qui non esiste un equivalente immediato, quindi sono ottimistiche.
#  Non c'e' il grafico "ASE vs k": il Naive Bayes non ha k da scegliere.
################################################################################

library(e1071)      # naiveBayes
library(caret)      # nearZeroVar, findCorrelation, confusionMatrix
library(pROC)       # curve ROC
library(ggplot2)
library(arrow)      # lettura dei .parquet
library(ragg)

## PERCORSI
DATA_DIR <- "C:\\Users\\Dario\\Downloads\\Grafici sistemati\\dataset_C"
BASE     <- "C:\\Users\\Dario\\Downloads\\Grafici sistemati\\Naive Bayes Nuovo"
OUT_DIR  <- file.path(BASE, "nb_v2")                       # previsioni, grafici
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
FILE_PREFIX  <- "C_v2"                                     # C_v2_train.parquet, C_v2_valid.parquet, C_v2_test.parquet
VARS_FILE    <- file.path(DATA_DIR, "variabili_scenari_C_v2.csv")
OUT_METRICHE <- file.path(BASE, "metriche_confronto_nb.csv")   # il file con tutte le metriche

# scenario (nome come nel file degli XGBoost e del kNN) -> colonna di variabili_scenari_C_v2.csv
SCEN <- data.frame(
  scenario = c("1_early_isee", "2_mid_sem1"),
  col_var  = c("EARLY_ISEE", "MID"),
  stringsAsFactors = FALSE)
DATASET_NOME <- "C"
DESCRIZIONE  <- "C v2 - con TOLC, coorti dal 2020"

## PARAMETRI (fissi, nessun tuning)
TARGET      <- "DROPOUT"
SEED        <- 1
N_BINS      <- 10           # classi (quantili) in cui si discretizzano le variabili numeriche
LAPLACE     <- 1           # smoothing di Laplace
CORR_CUTOFF <- 0.9         # fra due numeriche con |correlazione di Spearman| > soglia ne resta una sola
MAX_LEVELS  <- 10          # al massimo i top-N livelli per variabile categorica, gli altri -> "Altro" (come nel kNN)
IMP_REP     <- 3           # ripetizioni della permutazione per l'importanza
THR_GRID    <- seq(0, 1, by = 0.02)   # soglie del grafico/tabella di sensibilita' alla soglia

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
options(scipen = 999, digits = 4)

# le configurazioni (fisse, non selezionate con un tuning). Per il confronto col kNN ne basta una;
# aggiungendo righe si ottengono automaticamente piu' colonne nei grafici.
CONF <- data.frame(modello = "NB_discretizzato", base = "naive bayes",
                   n_bins = N_BINS, laplace = LAPLACE, stringsAsFactors = FALSE)

## LETTURA
read_set <- function(f) {
  d <- as.data.frame(read_parquet(f))
  d$y <- factor(d[[TARGET]], levels = c(0, 1), labels = c("No", "Yes"))
  d
}

## FUNZIONI DI SUPPORTO (identiche al kNN)
fast_auc <- function(p, y01) {              # AUC basata sui ranghi (ranghi medi per i pareggi)
  n1 <- sum(y01 == 1); n0 <- sum(y01 == 0)
  (sum(rank(p)[y01 == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}
fast_ase <- function(p, y01) mean((y01 - p)^2)   # ASE = Brier score

# metriche a una soglia (calcolate a mano: veloci e senza NaN nascosti)
metr <- function(prob, y, thr) {
  y01  <- as.integer(y == "Yes")
  pred <- as.integer(prob >= thr)
  TP <- sum(pred == 1 & y01 == 1); FP <- sum(pred == 1 & y01 == 0)
  FN <- sum(pred == 0 & y01 == 1); TN <- sum(pred == 0 & y01 == 0)
  sens <- TP / (TP + FN); spec <- TN / (TN + FP)
  prec <- if (TP + FP > 0) TP / (TP + FP) else NA_real_
  data.frame(AUC = fast_auc(prob, y01), ASE = fast_ase(prob, y01),
             sensibilita = sens, specificita = spec, precisione = prec, recall = sens,
             F1 = 2 * TP / (2 * TP + FP + FN),
             accuracy = (TP + TN) / length(y01), bal_acc = (sens + spec) / 2)
}
MET_COLS <- c("AUC", "ASE", "sensibilita", "specificita", "precisione", "recall", "F1", "accuracy", "bal_acc")

# lift per decili (decile 1 = studenti con probabilita' piu' alta)
lift_decili <- function(prob, y) {
  y01 <- as.integer(y == "Yes")
  dec <- ceiling(10 * rank(-prob, ties.method = "first") / length(prob))
  ev  <- as.numeric(tapply(y01, dec, sum)); nn <- as.numeric(tapply(y01, dec, length))
  data.frame(percentile = 1:10 * 10, captured = cumsum(ev) / sum(ev) * 100,
             response = cumsum(ev) / cumsum(nn) * 100, lift = (cumsum(ev) / cumsum(nn)) / mean(y01))
}

## FUNZIONI DEL NAIVE BAYES
# preparazione: tutto cio' che si impara (livelli) e' stimato SOLO sul train
prepare_data <- function(tr_raw, others) {
  is_cat   <- sapply(tr_raw, function(v) is.character(v) || is.logical(v) || is.factor(v))
  cat_cols <- names(tr_raw)[is_cat]
  levels_ok <- lapply(tr_raw[cat_cols], function(v) {
    v <- as.character(v); v[is.na(v)] <- "ND"
    names(sort(table(v), decreasing = TRUE))[seq_len(min(MAX_LEVELS, length(unique(v))))]
  })
  prep <- function(d) {
    for (nm in cat_cols) {
      v <- as.character(d[[nm]]); v[is.na(v)] <- "ND"
      v[!v %in% levels_ok[[nm]]] <- "Altro"
      d[[nm]] <- factor(v, levels = union(levels_ok[[nm]], "Altro"))
    }
    d
  }
  list(tr = prep(tr_raw), others = lapply(others, prep), levels_ok = levels_ok, is_cat = is_cat)
}

# elimina le numeriche molto correlate (Spearman), tenendo di ciascuna coppia quella meno correlata con le altre
drop_correlated <- function(tr) {
  num <- names(tr)[sapply(tr, function(x) is.numeric(x) && isTRUE(sd(x, na.rm = TRUE) > 0))]
  if (length(num) < 2) return(character(0))
  cm <- cor(tr[num], method = "spearman", use = "pairwise.complete.obs")
  cm[is.na(cm)] <- 0
  num[findCorrelation(cm, cutoff = CORR_CUTOFF)]
}

# discretizzazione (i tagli si imparano solo sul train); gli NA diventano una classe "ND" solo se nel train ce ne sono
make_bins <- function(tr, n_bins) {
  num <- names(tr)[sapply(tr, is.numeric)]
  setNames(lapply(num, function(nm) {
    x <- tr[[nm]]
    u <- sort(unique(x))                                                        # sort() toglie gli NA
    if (length(u) <= n_bins) {                                                  # pochi valori distinti (es. 0/1): un taglio tra valori consecutivi
      inner <- (head(u, -1) + tail(u, -1)) / 2
    } else {                                                                    # altrimenti tagli ai quantili interni (i duplicati collassano)
      inner <- unique(quantile(x, seq(0, 1, length.out = n_bins + 1)[-c(1, n_bins + 1)], names = FALSE, na.rm = TRUE))
    }
    list(cuts = c(-Inf, inner, Inf), na = anyNA(x))
  }), num)
}

apply_bins <- function(d, bins) {
  for (nm in names(bins)) {
    b <- bins[[nm]]
    lev <- as.character(seq_len(length(b$cuts) - 1))
    v <- as.character(cut(d[[nm]], b$cuts, labels = FALSE))
    if (b$na) { v[is.na(d[[nm]])] <- "ND"; lev <- c(lev, "ND") }
    d[[nm]] <- factor(v, levels = lev)
  }
  d
}

fit_nb <- function(tr, y, n_bins, laplace) {
  bins <- make_bins(tr, n_bins)
  structure(list(nb = naiveBayes(apply_bins(tr, bins), y, laplace = laplace), bins = bins),
            class = "nbfit")
}

# P(dropout). predict.naiveBayes di e1071 cicla riga per riga in R ed e' lentissimo: poiche' tutte le variabili
# sono categoriche (dopo la discretizzazione) il posterior si calcola in modo vettoriale dalle tabelle del modello.
pred_nb <- function(m, newdata) {
  d <- apply_bins(newdata, m$bins)
  lp <- matrix(log(m$nb$apriori / sum(m$nb$apriori)), nrow(d), 2, byrow = TRUE)
  for (v in names(d)) {
    tab <- unclass(m$nb$tables[[v]])
    tab[tab <= 0] <- 0.001
    idx <- match(as.character(d[[v]]), colnames(tab))
    ok <- !is.na(idx)                                                           # livello mai visto -> variabile ignorata
    lp[ok, ] <- lp[ok, ] + t(log(tab))[idx[ok], , drop = FALSE]
  }
  1 / (1 + exp(lp[, 1] - lp[, 2]))
}

# importanza per permutazione: aumento dell'ASE quando si mescola una colonna
perm_importance <- function(m, d, y) {
  y01 <- as.integer(y == "Yes")
  base <- fast_ase(pred_nb(m, d), y01)
  set.seed(SEED)
  imp <- sapply(names(d), function(v) {
    mean(replicate(IMP_REP, {
      dd <- d
      dd[[v]] <- sample(dd[[v]])
      fast_ase(pred_nb(m, dd), y01)
    })) - base
  })
  imp <- data.frame(variable = names(imp), ASE_increase = imp, row.names = NULL)
  imp[order(imp$ASE_increase, decreasing = TRUE), ]
}

################################################################################
## PIPELINE PER UNO SCENARIO
################################################################################
run_scenario <- function(sc, col_var, tr_df, va_df, te_df, vars_tab) {
  set.seed(SEED)
  cat("\n==================  SCENARIO:", sc, " (", col_var, ")  ==================\n")

  feat <- vars_tab$variabile[vars_tab[[col_var]] == 1]
  feat <- setdiff(feat, TARGET)
  stopifnot(all(feat %in% names(tr_df)))
  cat("Variabili usate:", length(feat), "\n ", paste(feat, collapse = ", "), "\n")

  y_tr <- tr_df$y; y_va <- va_df$y; y_te <- te_df$y
  y01_tr <- as.integer(y_tr == "Yes"); y01_va <- as.integer(y_va == "Yes"); y01_te <- as.integer(y_te == "Yes")
  prev_tr <- mean(y01_tr)                       # tasso di dropout nel TRAIN = seconda soglia
  cat(sprintf("Tasso di dropout: train %.3f | valid %.3f | test %.3f\n", prev_tr, mean(y01_va), mean(y01_te)))

  prep <- prepare_data(tr_df[, feat], list(va = va_df[, feat], te = te_df[, feat]))
  dropped <- drop_correlated(prep$tr)
  keep <- setdiff(feat, dropped)
  tr_p <- prep$tr[keep]; va_p <- prep$others$va[keep]; te_p <- prep$others$te[keep]
  cat("Eliminate per correlazione:", if (length(dropped)) paste(dropped, collapse = ", ") else "nessuna",
      "| variabili finali:", length(keep), "\n")

  righe <- list(); prev_all <- list(); probs_cfg <- list(); imp_all <- list()
  for (r in seq_len(nrow(CONF))) {
    mod <- CONF$modello[r]
    nb  <- fit_nb(tr_p, y_tr, CONF$n_bins[r], CONF$laplace[r])
    ptr <- pred_nb(nb, tr_p); pva <- pred_nb(nb, va_p); pte <- pred_nb(nb, te_p)
    probs_cfg[[mod]] <- list(va = pva, te = pte, val_ase = fast_ase(pva, y01_va), model = nb)
    cat(sprintf("  %-20s | ASE valid %.4f | AUC valid %.4f | AUC test %.4f\n", mod,
                fast_ase(pva, y01_va), fast_auc(pva, y01_va), fast_auc(pte, y01_te)))

    prev_all[[mod]] <- rbind(
      data.frame(scenario = sc, modello = mod, set = "train",      y = y01_tr, prob = ptr),
      data.frame(scenario = sc, modello = mod, set = "validation", y = y01_va, prob = pva),
      data.frame(scenario = sc, modello = mod, set = "test",       y = y01_te, prob = pte))

    for (thr in c(0.5, prev_tr)) {
      mt <- metr(ptr, y_tr, thr); mv <- metr(pva, y_va, thr); ms <- metr(pte, y_te, thr)
      row <- data.frame(scenario = sc, dataset = DATASET_NOME, descrizione = DESCRIZIONE, modello = mod,
                        n_bins = CONF$n_bins[r], laplace = CONF$laplace[r],
                        soglia = round(thr, 4),
                        tipo_soglia = ifelse(thr == 0.5, "0.5", "tasso_dropout_train"),
                        n_variabili = length(keep),
                        valid_prevalenza = mean(y01_va))
      for (nm in MET_COLS) row[[paste0("train_", nm)]] <- mt[[nm]]
      for (nm in MET_COLS) row[[paste0("valid_", nm)]] <- mv[[nm]]
      for (nm in MET_COLS) row[[paste0("test_",  nm)]] <- ms[[nm]]
      righe[[length(righe) + 1]] <- row
    }
    imp <- perm_importance(nb, te_p, y_te)
    imp_all[[mod]] <- cbind(scenario = sc, modello = mod, imp)
  }
  metr_sc <- do.call(rbind, righe)

  # configurazione "scelta" (serve solo per soglie e lift): miglior ASE sul VALIDATION; con una sola configurazione e' lei
  va_ase <- sapply(probs_cfg, `[[`, "val_ase")
  scelto <- names(which.min(va_ase))
  metr_sc$scelto_su_validation <- metr_sc$modello == scelto

  # sensibilita' alla soglia + lift, per la configurazione scelta
  pv <- probs_cfg[[scelto]]$va; pt <- probs_cfg[[scelto]]$te
  sw <- do.call(rbind, lapply(sort(unique(c(THR_GRID, prev_tr))), function(thr) {
    rbind(cbind(scenario = sc, modello = scelto, set = "validation", soglia = thr, metr(pv, y_va, thr)[, -(1:2)]),
          cbind(scenario = sc, modello = scelto, set = "test",       soglia = thr, metr(pt, y_te, thr)[, -(1:2)]))
  }))
  lf <- rbind(cbind(scenario = sc, modello = scelto, set = "validation", lift_decili(pv, y_va)),
              cbind(scenario = sc, modello = scelto, set = "test",       lift_decili(pt, y_te)))

  saveRDS(list(modelli = lapply(probs_cfg, `[[`, "model"), feature = keep, eliminate = dropped,
               levels_ok = prep$levels_ok, is_cat = prep$is_cat),
          file.path(OUT_DIR, paste0("modello_nb_", sc, ".rds")))

  list(metriche = metr_sc, previsioni = do.call(rbind, prev_all), soglie = sw, lift = lf,
       importanza = do.call(rbind, imp_all), scelto = scelto)
}

################################################################################
## ESECUZIONE
################################################################################
tr_df <- read_set(file.path(DATA_DIR, paste0(FILE_PREFIX, "_train.parquet")))
va_df <- read_set(file.path(DATA_DIR, paste0(FILE_PREFIX, "_valid.parquet")))
te_df <- read_set(file.path(DATA_DIR, paste0(FILE_PREFIX, "_test.parquet")))
stopifnot(setequal(names(tr_df), names(va_df)), setequal(names(tr_df), names(te_df)))
cat(sprintf("Righe: train %d | valid %d | test %d | colonne %d\n", nrow(tr_df), nrow(va_df), nrow(te_df), ncol(tr_df)))
vars_tab <- read.csv(VARS_FILE, stringsAsFactors = FALSE, encoding = "UTF-8")

T0 <- Sys.time()
out <- lapply(seq_len(nrow(SCEN)), function(i)
  run_scenario(SCEN$scenario[i], SCEN$col_var[i], tr_df, va_df, te_df, vars_tab))

metriche   <- do.call(rbind, lapply(out, `[[`, "metriche"))
previsioni <- do.call(rbind, lapply(out, `[[`, "previsioni"))
soglie     <- do.call(rbind, lapply(out, `[[`, "soglie"))
lift       <- do.call(rbind, lapply(out, `[[`, "lift"))
importanza <- do.call(rbind, lapply(out, `[[`, "importanza"))

write.csv(metriche,   OUT_METRICHE,                                  row.names = FALSE)   # IL file con tutte le metriche
write.csv(previsioni, file.path(OUT_DIR, "previsioni_nb.csv"),       row.names = FALSE)   # per rifare soglie/grafici senza rilanciare
write.csv(soglie,     file.path(OUT_DIR, "sensibilita_soglia.csv"),  row.names = FALSE)
write.csv(lift,       file.path(OUT_DIR, "lift_decili.csv"),         row.names = FALSE)
write.csv(importanza, file.path(OUT_DIR, "importanza_feature_permutazione.csv"), row.names = FALSE)
cat("\nTempo totale:", format(round(difftime(Sys.time(), T0, units = "mins"), 1)), "\n")

# riepilogo a schermo: test, soglia 0.5 e soglia = tasso di dropout
riep <- metriche[, c("scenario", "modello", "tipo_soglia", "valid_ASE", "test_AUC", "test_ASE",
                     "test_sensibilita", "test_specificita", "test_precisione", "test_F1", "scelto_su_validation")]
print(riep, row.names = FALSE)

################################################################################
## GRAFICI (stessi del kNN, tranne "ASE vs k" che per il Naive Bayes non ha senso)
################################################################################
th <- tema
SC_LAB  <- c("1_early_isee" = "Early", "2_mid_sem1" = "Mid (after 1st semester)")
BASE_EN <- c("naive bayes" = "Naive Bayes")
BASI <- unique(CONF$base)
COL_BASE <- setNames(unibg_palette[c(1, 3, 5, 7)][seq_along(BASI)], BASI)

# 2. performance sul test (soglia 0.5): AUC e ASE per configurazione
bar <- metriche[metriche$tipo_soglia == "0.5", ]
bar$base <- CONF$base[match(bar$modello, CONF$modello)]
bar$scenario <- factor(bar$scenario, levels = SCEN$scenario)
bar_long <- rbind(data.frame(bar[, c("scenario", "base")], valore = bar$test_AUC, metrica = "AUC"),
                  data.frame(bar[, c("scenario", "base")], valore = bar$test_ASE, metrica = "ASE"))
g2 <- ggplot(bar_long, aes(base, valore, fill = base)) +
  geom_col(position = position_dodge(width = 0.85), width = 0.8) +
  geom_text(aes(label = sprintf("%.3f", valore)), position = position_dodge(width = 0.85), vjust = -0.4, size = 2.8) +
  facet_grid(metrica ~ scenario, scales = "free_y", labeller = labeller(scenario = SC_LAB)) +
  scale_fill_manual(values = COL_BASE, guide = "none") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  scale_x_discrete(labels = BASE_EN) +
  labs(title = "Naive Bayes: test performance", x = NULL, y = NULL) + th +
  theme(axis.text.x = element_text(angle = 0))
ggsave(file.path(OUT_DIR, "nb_test_performance_unibg.png"), g2, width = 12, height = 7, dpi = 300, device = agg_png, bg = "white")

# 3. sensibilita alla soglia per la configurazione scelta (validation)
sw <- soglie[soglie$set == "validation", ]
sw_long <- rbind(data.frame(sw[, c("scenario", "soglia")], valore = sw$accuracy,    misura = "accuracy"),
                 data.frame(sw[, c("scenario", "soglia")], valore = sw$precisione,  misura = "precision"),
                 data.frame(sw[, c("scenario", "soglia")], valore = sw$sensibilita, misura = "sensitivity"),
                 data.frame(sw[, c("scenario", "soglia")], valore = sw$specificita, misura = "specificity"))
g3 <- ggplot(sw_long, aes(soglia, valore, colour = misura)) +
  geom_vline(xintercept = SOGLIA_FISSA, linetype = "dashed", colour = "grey40") +
  geom_line(linewidth = 1, na.rm = TRUE) +
  facet_wrap(~ scenario, labeller = as_labeller(SC_LAB)) +
  scale_colour_manual(values = MIS_COL) +
  labs(title = "Naive Bayes: classification measurements varying threshold",
       x = "Dropout probability threshold", y = NULL, colour = NULL) + th
ggsave(file.path(OUT_DIR, "nb_effetto_soglia_unibg.png"), g3, width = 11, height = 5.5, dpi = 300, device = agg_png, bg = "white")

# 4. lift cumulato del modello scelto (validation e test)
g4 <- ggplot(lift, aes(percentile, lift, colour = set)) +
  geom_hline(yintercept = 1, linetype = 2, colour = "grey60") +
  geom_line(linewidth = 1) + geom_point(size = 2) +
  facet_wrap(~ scenario, labeller = as_labeller(SC_LAB)) +
  scale_x_continuous(breaks = 1:10 * 10) +
  scale_colour_manual(values = c(validation = unibg_palette[3], test = unibg_palette[6])) +
  labs(title = "Naive Bayes: cumulative lift of the selected model", x = "% of students selected (deciles)",
       y = "Cumulative lift", colour = NULL) + th
ggsave(file.path(OUT_DIR, "nb_lift_cumulato_unibg.png"), g4, width = 11, height = 5.5, dpi = 300, device = agg_png, bg = "white")

# 5. curve ROC (train in risostituzione, validation, test): righe = scenario, colonne = modello
roc_df <- do.call(rbind, lapply(split(previsioni, list(previsioni$scenario, previsioni$modello, previsioni$set), drop = TRUE), function(d) {
  r <- roc(d$y, d$prob, levels = c(0, 1), direction = "<", quiet = TRUE)
  data.frame(scenario = d$scenario[1], modello = d$modello[1], set = d$set[1],
             fpr = 1 - r$specificities, tpr = r$sensitivities, auc = as.numeric(auc(r)))
}))
roc_df$base     <- CONF$base[match(roc_df$modello, CONF$modello)]
roc_df$set      <- factor(roc_df$set, levels = c("train", "validation", "test"))
roc_df$scenario <- factor(roc_df$scenario, levels = SCEN$scenario,
                          labels = unname(SC_LAB[SCEN$scenario]))
roc_df$base     <- factor(roc_df$base, levels = BASI)
auc_lab <- unique(roc_df[, c("scenario", "base", "set", "auc")])
auc_lab$y <- c(train = 0.25, validation = 0.15, test = 0.05)[as.character(auc_lab$set)]
auc_lab$txt <- sprintf("AUC %s = %.3f", c(train = "train", validation = "validation", test = "test")[as.character(auc_lab$set)], auc_lab$auc)
COL_SET <- c(train = unibg_palette[6], validation = unibg_palette[1], test = unibg_palette[5])
g5 <- ggplot(roc_df, aes(fpr, tpr, colour = set)) +
  geom_abline(intercept = 0, slope = 1, linetype = 2, colour = "grey60") +
  geom_path(linewidth = 0.9) +
  geom_text(data = auc_lab, aes(x = 1, y = y, label = txt, colour = set), hjust = 1, size = 3.4, show.legend = FALSE) +
  facet_grid(scenario ~ base, labeller = labeller(base = BASE_EN)) +
  scale_colour_manual(values = COL_SET) +
  coord_equal() +
  labs(title = "Naive Bayes: ROC curves by scenario",
       x = "1 - specificity (FPR)", y = "Sensitivity (TPR)", colour = NULL) + th
ggsave(file.path(OUT_DIR, "nb_roc_unibg.png"), g5, width = 12, height = 8.5, dpi = 300, device = agg_png, bg = "white")
write.csv(unique(roc_df[, c("scenario", "base", "set", "auc")]), file.path(OUT_DIR, "auc_roc.csv"), row.names = FALSE)

# 6. importanza delle variabili per permutazione (aumento di ASE sul test), top 15 per scenario
imp_top <- do.call(rbind, lapply(split(importanza, list(importanza$scenario, importanza$modello), drop = TRUE),
                                 function(d) head(d[order(d$ASE_increase, decreasing = TRUE), ], 15)))
imp_top$scenario <- factor(imp_top$scenario, levels = SCEN$scenario)
imp_top$var_o <- reorder(paste(imp_top$scenario, imp_top$variable), imp_top$ASE_increase)
g6 <- ggplot(imp_top, aes(var_o, ASE_increase)) +
  geom_col(fill = unibg_palette[1]) + coord_flip() +
  facet_wrap(~ scenario, scales = "free_y", labeller = as_labeller(SC_LAB)) +
  scale_x_discrete(labels = function(x) sub("^[^ ]+ ", "", x)) +
  labs(title = "Naive Bayes: top 15 variables (permutation importance, test)",
       x = NULL, y = "ASE increase (test) after permutation") + th
ggsave(file.path(OUT_DIR, "nb_importanza_variabili_unibg.png"), g6, width = 12, height = 6.5, dpi = 300, device = agg_png, bg = "white")

cat("\nFinito. Metriche in:", OUT_METRICHE, "\nAltri file e grafici in:", OUT_DIR, "\n")
