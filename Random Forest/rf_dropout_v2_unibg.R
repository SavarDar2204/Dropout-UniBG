# ============================================================================
# GRAPHIC-STYLE COPY of rf_dropout_v2.R
# Same analysis as the original; only the plots changed (Rubik font, UniBG palette,
# English titles/labels, as in assessment_modelli_C_v2.R). PNG files get the suffix _unibg.
# ============================================================================
rm(list = ls())

################################################################################
#  RANDOM FOREST per il DROPOUT - dataset C_v2 - scenari EARLY_ISEE e MID  (script unico)
#
#  Stessa struttura e stessi output dello script dei kNN (knn_dropout_v2.R), così i due modelli
#  sono confrontabili riga per riga. NON modifica gli script di dropout_rf_pipeline.R e
#  dropout_rf_gridsearch.R: ne riprende solo le idee (nodesize x mtry, top-20 livelli, ASE).
#
#  Cosa fa, in un'unica esecuzione:
#   1. legge C_v2 (train / valid / test già divisi) e le variabili di ogni scenario da
#      variabili_scenari_C_v2.csv (colonne EARLY_ISEE e MID: 1 = usare, 0 = escludere)
#   2. grid search nodesize x mtry: per ogni coppia una foresta sul train, ASE e AUC sul validation,
#      ASE "out-of-bag" (OOB) sul train
#   3. sceglie la coppia sul VALIDATION (minimo ASE). Il test non entra mai nella scelta.
#      Due modelli finali per scenario, come nello script di grid search:
#        - RF_best          : miglior (nodesize, mtry) in assoluto
#        - RF_mtry_default  : miglior nodesize con mtry = floor(sqrt(p)), cioè il default di randomForest
#   4. valuta i modelli con due soglie: 0.5 e il tasso di dropout nel train
#   5. scrive TUTTE le metriche in un unico file, metriche_confronto_rf.csv (stesse colonne di
#      metriche_confronto_knn.csv), più tuning, previsioni, importanza, soglie, lift e grafici
#
#   6. grafici extra della pipeline/grid search di Dario: PDP (prime PDP_TOP variabili del modello scelto,
#      calcolati a mano con predict(), senza pacchetto pdp) e superficie 3D dell'ASE di validation
#      (png sempre; html interattivo solo se sono installati plotly e htmlwidgets)
#
#  Le metriche "train" usano le probabilità OOB (ogni albero vota solo per gli studenti che non ha visto):
#  è l'analogo del leave-one-out dei kNN, quindi non sono gonfiate.
################################################################################

library(randomForest)
library(pROC)       # AUC
library(ggplot2)
library(arrow)      # lettura dei .parquet
library(ragg)

## PERCORSI
DATA_DIR <- "C:\\Users\\Dario\\Downloads\\Grafici sistemati\\dataset_C"
BASE     <- "C:\\Users\\Dario\\Downloads\\Grafici sistemati\\Random Forest"
OUT_DIR  <- file.path(BASE, "rf_v2")                       # tuning, previsioni, grafici
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
FILE_PREFIX  <- "C_v2"
VARS_FILE    <- file.path(DATA_DIR, "variabili_scenari_C_v2.csv")
OUT_METRICHE <- file.path(DATA_DIR, "metriche_confronto_rf.csv")    # il file con tutte le metriche

SCEN <- data.frame(
  scenario = c("1_early_isee", "2_mid_sem1"),
  col_var  = c("EARLY_ISEE", "MID"),
  stringsAsFactors = FALSE)
DATASET_NOME <- "C"
DESCRIZIONE  <- "C v2 - con TOLC, coorti dal 2020"

## PARAMETRI
TARGET        <- "DROPOUT"
SEED          <- 1
NTREE         <- 500
NODESIZE_GRID <- c(1, 3, 5, 10, 20, 50, 100, 200, 400)      # elementi minimi in una foglia (grande = semplice)
MTRY_GRID     <- c(2, 3, 4, 5, 6, 8, 10)                    # variabili provate a ogni split; si aggiunge sempre il default sqrt(p)
MAX_LEVELS    <- 20         # top-N livelli per variabile categorica, il resto -> "Altro"
THR_GRID      <- seq(0, 1, by = 0.02)
IMP_TOP       <- 15         # variabili nel grafico dell'importanza
PDP_TOP       <- 4          # grafici di dipendenza parziale per le prime N variabili (Gini) del modello scelto
PDP_N         <- 3000       # righe di train usate per mediare i PDP (Inf = tutte, ma e' lento)
PDP_GRID      <- 30         # punti della griglia per le variabili numeriche

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

## FUNZIONI DI SUPPORTO
read_set <- function(f) {
  d <- as.data.frame(read_parquet(f))
  d$y <- factor(d[[TARGET]], levels = c(0, 1), labels = c("No", "Yes"))
  d
}
fast_auc <- function(p, y01) {              # AUC basata sui ranghi (ranghi medi per i pareggi)
  n1 <- sum(y01 == 1); n0 <- sum(y01 == 0)
  (sum(rank(p)[y01 == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}
fast_ase <- function(p, y01) mean((y01 - p)^2)   # ASE = Brier score

# le variabili categoriche (testo, logiche, fattori) diventano fattori con i top-MAX_LEVELS livelli del
# train, gli altri "Altro". Si impara SOLO sul train.
prepare_data <- function(tr_raw, va_raw, te_raw) {
  cat_cols <- names(tr_raw)[sapply(tr_raw, function(v) is.character(v) || is.logical(v) || is.factor(v))]
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
  list(tr = prep(tr_raw), va = prep(va_raw), te = prep(te_raw))
}

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

lift_decili <- function(prob, y) {
  y01 <- as.integer(y == "Yes")
  dec <- ceiling(10 * rank(-prob, ties.method = "first") / length(prob))
  ev  <- as.numeric(tapply(y01, dec, sum)); nn <- as.numeric(tapply(y01, dec, length))
  data.frame(percentile = 1:10 * 10, captured = cumsum(ev) / sum(ev) * 100,
             response = cumsum(ev) / cumsum(nn) * 100, lift = (cumsum(ev) / cumsum(nn)) / mean(y01))
}

################################################################################
## PIPELINE PER UNO SCENARIO
################################################################################
run_scenario <- function(sc, col_var, tr_df, va_df, te_df, vars_tab) {
  cat("\n==================  SCENARIO:", sc, " (", col_var, ")  ==================\n")
  feat <- setdiff(vars_tab$variabile[vars_tab[[col_var]] == 1], TARGET)
  stopifnot(all(feat %in% names(tr_df)))
  cat("Variabili usate:", length(feat), "\n ", paste(feat, collapse = ", "), "\n")
  
  y_tr <- tr_df$y; y_va <- va_df$y; y_te <- te_df$y
  y01_tr <- as.integer(y_tr == "Yes"); y01_va <- as.integer(y_va == "Yes")
  prev_tr <- mean(y01_tr)
  cat(sprintf("Tasso di dropout: train %.3f | valid %.3f | test %.3f\n", prev_tr, mean(y01_va), mean(y_te == "Yes")))
  
  prep <- prepare_data(tr_df[, feat], va_df[, feat], te_df[, feat])
  tr <- prep$tr; va <- prep$va; te <- prep$te
  p <- ncol(tr)
  mtry_default <- floor(sqrt(p))
  mtry_grid <- sort(unique(c(pmin(MTRY_GRID, p), mtry_default)))
  cat("Variabili p =", p, "| mtry provati:", paste(mtry_grid, collapse = ", "), "(default =", mtry_default, ")\n")
  
  ## --- GRID SEARCH nodesize x mtry (foresta sul train; validation come xtest; OOB sul train)
  grid <- expand.grid(nodesize = NODESIZE_GRID, mtry = mtry_grid)
  res <- do.call(rbind, lapply(seq_len(nrow(grid)), function(i) {
    ns <- grid$nodesize[i]; mt <- grid$mtry[i]; t0 <- Sys.time()
    set.seed(SEED)
    m <- randomForest(x = tr, y = y_tr, xtest = va, ntree = NTREE, nodesize = ns, mtry = mt, keep.forest = FALSE)
    p_va <- m$test$votes[, "Yes"]; p_oob <- m$votes[, "Yes"]
    cat(sprintf("[%d/%d] nodesize = %3d  mtry = %2d  (%.0f s)\n", i, nrow(grid), ns, mt,
                as.numeric(difftime(Sys.time(), t0, units = "secs"))))
    data.frame(scenario = sc, nodesize = ns, mtry = mt,
               TRAIN_ASE_OOB = fast_ase(p_oob, y01_tr), VAL_ASE = fast_ase(p_va, y01_va), VAL_AUC = fast_auc(p_va, y01_va))
  }))
  res$GAP <- res$VAL_ASE - res$TRAIN_ASE_OOB
  
  # scelta: minimo ASE sul validation;
  pick <- function(d) {
    ok <- d[d$VAL_ASE <= min(d$VAL_ASE), ]
    ok[order(-ok$nodesize, ok$mtry), ][1, ]
  }
  best <- pick(res)
  best_def <- pick(res[res$mtry == mtry_default, ])
  cat(sprintf("\nMiglior combinazione: nodesize = %d, mtry = %d (ASE validation %.4f)\n", best$nodesize, best$mtry, best$VAL_ASE))
  cat(sprintf("Miglior nodesize con mtry di default (%d): nodesize = %d (ASE validation %.4f)\n",
              mtry_default, best_def$nodesize, best_def$VAL_ASE))
  
  ## --- MODELLI FINALI (rifittati sul solo train)
  cfg <- data.frame(modello = "RF_best", base = "RF - nodesize e mtry migliori",
                    nodesize = best$nodesize, mtry = best$mtry, stringsAsFactors = FALSE)
  if (best$mtry != mtry_default)
    cfg <- rbind(cfg, data.frame(modello = "RF_mtry_default", base = "RF - mtry di default",
                                 nodesize = best_def$nodesize, mtry = mtry_default, stringsAsFactors = FALSE))
  else cat("Il migliore usa già l'mtry di default: un solo modello finale.\n")
  
  righe <- list(); prev_all <- list(); imp_all <- list(); probs_cfg <- list(); rf_cfg <- list()
  for (r in seq_len(nrow(cfg))) {
    mod <- cfg$modello[r]; ns <- cfg$nodesize[r]; mt <- cfg$mtry[r]
    set.seed(SEED)
    rf <- randomForest(x = tr, y = y_tr, ntree = NTREE, nodesize = ns, mtry = mt, importance = TRUE)
    ptr <- rf$votes[, "Yes"]                                   # OOB: probabilità honest sul train
    pva <- predict(rf, va, type = "prob")[, "Yes"]
    pte <- predict(rf, te, type = "prob")[, "Yes"]
    rf_cfg[[mod]] <- rf
    probs_cfg[[mod]] <- list(va = pva, te = pte, val_ase = fast_ase(pva, y01_va))
    cat(sprintf("  %-16s nodesize = %3d  mtry = %2d | ASE valid %.4f | AUC valid %.4f\n", mod, ns, mt,
                fast_ase(pva, y01_va), fast_auc(pva, y01_va)))
    
    prev_all[[mod]] <- rbind(
      data.frame(scenario = sc, modello = mod, set = "train",      y = y01_tr, prob = ptr),    # OOB
      data.frame(scenario = sc, modello = mod, set = "validation", y = y01_va, prob = pva),
      data.frame(scenario = sc, modello = mod, set = "test",       y = as.integer(y_te == "Yes"), prob = pte))
    
    imp <- as.data.frame(importance(rf)); imp$variabile <- rownames(imp)
    imp$pct_gini <- 100 * imp$MeanDecreaseGini / sum(imp$MeanDecreaseGini)
    imp <- imp[order(imp$MeanDecreaseGini, decreasing = TRUE), ]
    imp_all[[mod]] <- cbind(scenario = sc, modello = mod, imp)
    
    for (thr in c(0.5, prev_tr)) {
      mt_ <- metr(ptr, y_tr, thr); mv <- metr(pva, y_va, thr); ms <- metr(pte, y_te, thr)
      row <- data.frame(scenario = sc, dataset = DATASET_NOME, descrizione = DESCRIZIONE, modello = mod,
                        nodesize = ns, mtry = mt, n_alberi = NTREE,
                        soglia = round(thr, 4),
                        tipo_soglia = ifelse(thr == 0.5, "0.5", "tasso_dropout_train"),
                        n_variabili = p, valid_prevalenza = mean(y01_va))
      for (nm in MET_COLS) row[[paste0("train_", nm)]] <- mt_[[nm]]
      for (nm in MET_COLS) row[[paste0("valid_", nm)]] <- mv[[nm]]
      for (nm in MET_COLS) row[[paste0("test_",  nm)]] <- ms[[nm]]
      righe[[length(righe) + 1]] <- row
    }
  }
  metr_sc <- do.call(rbind, righe)
  va_ase <- sapply(probs_cfg, `[[`, "val_ase")
  scelto <- names(which.min(va_ase))
  metr_sc$scelto_su_validation <- metr_sc$modello == scelto
  cat("\nModello scelto sul validation:", scelto, "\n")
  
  ## --- PDP del modello scelto: P(dropout) media al variare di una variabile, le altre restano ai valori osservati
  rf_s <- rf_cfg[[scelto]]
  imp_s <- imp_all[[scelto]]
  imp_s <- imp_s[order(imp_s$MeanDecreaseGini, decreasing = TRUE), ]
  set.seed(SEED)
  tr_pdp <- tr[if (is.finite(PDP_N) && PDP_N < nrow(tr)) sample(nrow(tr), PDP_N) else seq_len(nrow(tr)), , drop = FALSE]
  pdp_list <- lapply(head(imp_s$variabile, PDP_TOP), function(v) {
    gr <- if (is.numeric(tr[[v]])) unique(quantile(tr[[v]], seq(0.01, 0.99, length.out = PDP_GRID))) else levels(tr[[v]])
    yh <- sapply(gr, function(g) {
      d <- tr_pdp
      d[[v]] <- if (is.numeric(tr[[v]])) g else factor(g, levels = levels(tr[[v]]))
      mean(predict(rf_s, d, type = "prob")[, "Yes"])
    })
    data.frame(scenario = sc, modello = scelto, variabile = v, valore = as.character(gr),
               valore_num = if (is.numeric(tr[[v]])) gr else NA_real_, yhat = yh, stringsAsFactors = FALSE)
  })
  pdp_df <- do.call(rbind, pdp_list)
  cat("PDP calcolati per:", paste(head(imp_s$variabile, PDP_TOP), collapse = ", "), "\n")
  
  pv <- probs_cfg[[scelto]]$va; pt <- probs_cfg[[scelto]]$te
  sw <- do.call(rbind, lapply(sort(unique(c(THR_GRID, prev_tr))), function(thr) {
    rbind(cbind(scenario = sc, modello = scelto, set = "validation", soglia = thr, metr(pv, y_va, thr)[, -(1:2)]),
          cbind(scenario = sc, modello = scelto, set = "test",       soglia = thr, metr(pt, y_te, thr)[, -(1:2)]))
  }))
  lf <- rbind(cbind(scenario = sc, modello = scelto, set = "validation", lift_decili(pv, y_va)),
              cbind(scenario = sc, modello = scelto, set = "test",       lift_decili(pt, y_te)))
  
  list(metriche = metr_sc, tuning = res, previsioni = do.call(rbind, prev_all), importanza = do.call(rbind, imp_all),
       soglie = sw, lift = lf, pdp = pdp_df, cfg = cfg, scelto = scelto, best = best, mtry_default = mtry_default)
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
tuning     <- do.call(rbind, lapply(out, `[[`, "tuning"))
previsioni <- do.call(rbind, lapply(out, `[[`, "previsioni"))
importanza <- do.call(rbind, lapply(out, `[[`, "importanza"))
soglie     <- do.call(rbind, lapply(out, `[[`, "soglie"))
lift       <- do.call(rbind, lapply(out, `[[`, "lift"))
pdp        <- do.call(rbind, lapply(out, `[[`, "pdp"))

write.csv(metriche,   OUT_METRICHE,                                 row.names = FALSE)   # IL file con tutte le metriche
write.csv(tuning,     file.path(OUT_DIR, "rf_tuning_results.csv"),  row.names = FALSE)
write.csv(previsioni, file.path(OUT_DIR, "previsioni_rf.csv"),      row.names = FALSE)
write.csv(importanza, file.path(OUT_DIR, "importanza_feature.csv"), row.names = FALSE)
write.csv(soglie,     file.path(OUT_DIR, "sensibilita_soglia.csv"), row.names = FALSE)
write.csv(lift,       file.path(OUT_DIR, "lift_decili.csv"),        row.names = FALSE)
write.csv(pdp,        file.path(OUT_DIR, "pdp_valori.csv"),         row.names = FALSE)
cat("\nTempo totale:", format(round(difftime(Sys.time(), T0, units = "mins"), 1)), "\n")

riep <- metriche[, c("scenario", "modello", "nodesize", "mtry", "tipo_soglia", "valid_ASE", "test_AUC", "test_ASE",
                     "test_sensibilita", "test_specificita", "test_precisione", "test_F1", "scelto_su_validation")]
print(riep, row.names = FALSE)

################################################################################
## GRAFICI (stessi del kNN)
################################################################################
th <- tema
SC_LAB <- c("1_early_isee" = "Early", "2_mid_sem1" = "Mid (after 1st semester)")
BASE_EN <- c("RF - nodesize e mtry migliori" = "RF - best nodesize and mtry", "RF - mtry di default" = "RF - default mtry")
# both bases always present (also when best mtry == default mtry in a scenario)
BASI <- c("RF - nodesize e mtry migliori", "RF - mtry di default")
COL_BASE <- setNames(unibg_palette[c(1, 5)][seq_along(BASI)], BASI)
base_of <- function(mod) c(RF_best = "RF - nodesize e mtry migliori", RF_mtry_default = "RF - mtry di default")[mod]

# 1. ASE sul validation al variare del nodesize (una linea per mtry); rombo = scelta
tuning$nodesize_f <- factor(tuning$nodesize, levels = sort(NODESIZE_GRID, decreasing = TRUE))   # complessità crescente verso destra
tuning$mtry_f <- factor(tuning$mtry)
tuning$scenario_l <- factor(SC_LAB[tuning$scenario], levels = SC_LAB)
best_pts <- do.call(rbind, lapply(out, function(o) data.frame(scenario = o$best$scenario, nodesize = o$best$nodesize, mtry = o$best$mtry, ASE = o$best$VAL_ASE)))
best_pts$nodesize_f <- factor(best_pts$nodesize, levels = levels(tuning$nodesize_f))
best_pts$mtry_f <- factor(best_pts$mtry, levels = levels(tuning$mtry_f))
best_pts$scenario_l <- factor(SC_LAB[best_pts$scenario], levels = SC_LAB)
g1 <- ggplot(tuning, aes(nodesize_f, VAL_ASE, colour = mtry_f, group = mtry_f)) +
  geom_line(linewidth = 1) + geom_point(size = 1.8) +
  geom_point(data = best_pts, aes(nodesize_f, ASE), inherit.aes = FALSE, shape = 18, size = 5, colour = "black") +
  facet_wrap(~ scenario_l, scales = "free_y") +
  scale_colour_manual(values = colorRampPalette(unibg_palette[c(6, 1)])(length(unique(tuning$mtry)))) +
  labs(title = "Random forest: model complexity vs ASE",
       x = "nodesize", y = "ASE", colour = "mtry") + th + theme(legend.title = element_text(size = 12))
ggsave(file.path(OUT_DIR, "rf_ase_validation_vs_nodesize_unibg.png"), g1, width = 12, height = 6, dpi = 300, device = agg_png, bg = "white")

# 2. heatmap dell'ASE di validation (riquadro nero = scelta)
g2 <- ggplot(tuning, aes(nodesize_f, mtry_f, fill = VAL_ASE)) +
  geom_tile(colour = "white") +
  geom_text(aes(label = sprintf("%.3f", VAL_ASE), colour = VAL_ASE > ave(VAL_ASE, scenario, FUN = function(v) mean(range(v)))), size = 2.6) +
  geom_tile(data = best_pts, aes(nodesize_f, mtry_f), inherit.aes = FALSE, fill = NA, colour = "black", linewidth = 1.2) +
  facet_wrap(~ scenario_l) +
  scale_fill_gradient(low = unibg_palette[8], high = unibg_palette[2]) +
  scale_colour_manual(values = c("TRUE" = "white", "FALSE" = "black"), guide = "none") +
  labs(title = "Random forest: validation ASE (nodesize x mtry)", x = "nodesize",
       y = "mtry", fill = "ASE") + th + theme(legend.position = "none")
ggsave(file.path(OUT_DIR, "rf_heatmap_unibg.png"), g2, width = 13, height = 6, dpi = 300, device = agg_png, bg = "white")

# 3. performance sul test (soglia 0.5)
bar <- metriche[metriche$tipo_soglia == "0.5", ]
# if in a scenario the best model already uses the default mtry there is only one final model:
# duplicate its row as "default mtry" so that both bars are drawn (identical values)
for (sc in unique(bar$scenario)) {
  if (!any(bar$scenario == sc & bar$modello == "RF_mtry_default")) {
    dup <- bar[bar$scenario == sc & bar$modello == "RF_best", ]
    dup$modello <- "RF_mtry_default"
    bar <- rbind(bar, dup)
  }
}
bar$base <- factor(base_of(bar$modello), levels = BASI)
bar$scenario_l <- factor(SC_LAB[bar$scenario], levels = SC_LAB)
bar_long <- rbind(data.frame(bar[, c("scenario_l", "base")], valore = bar$test_AUC, metrica = "AUC"),
                  data.frame(bar[, c("scenario_l", "base")], valore = bar$test_ASE, metrica = "ASE"))
g3 <- ggplot(bar_long, aes(base, valore, fill = base)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = sprintf("%.3f", valore)), vjust = -0.4, size = 3) +
  facet_grid(metrica ~ scenario_l, scales = "free_y") +
  scale_fill_manual(values = COL_BASE, guide = "none") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  scale_x_discrete(labels = BASE_EN) +
  labs(title = "Random forest: test performance", x = NULL, y = NULL) + th
ggsave(file.path(OUT_DIR, "rf_test_performance_unibg.png"), g3, width = 11, height = 7, dpi = 300, device = agg_png, bg = "white")

# 4. effetto della soglia (modello scelto, validation)
sw <- soglie[soglie$set == "validation", ]
sw$scenario_l <- factor(SC_LAB[sw$scenario], levels = SC_LAB)
sw_long <- rbind(data.frame(sw[, c("scenario_l", "soglia")], valore = sw$accuracy,    misura = "accuracy"),
                 data.frame(sw[, c("scenario_l", "soglia")], valore = sw$precisione,  misura = "precision"),
                 data.frame(sw[, c("scenario_l", "soglia")], valore = sw$sensibilita, misura = "sensitivity"),
                 data.frame(sw[, c("scenario_l", "soglia")], valore = sw$specificita, misura = "specificity"))
g4 <- ggplot(sw_long, aes(soglia, valore, colour = misura)) +
  geom_vline(xintercept = SOGLIA_FISSA, linetype = "dashed", colour = "grey40") +
  geom_line(linewidth = 1, na.rm = TRUE) +
  facet_wrap(~ scenario_l) +
  scale_colour_manual(values = MIS_COL) +
  labs(title = "Random forest: classification measurements varying threshold",
       x = "Dropout probability threshold", y = NULL, colour = NULL) + th
ggsave(file.path(OUT_DIR, "rf_effetto_soglia_unibg.png"), g4, width = 11, height = 5.5, dpi = 300, device = agg_png, bg = "white")

# 5. lift cumulato del modello scelto
lift$scenario_l <- factor(SC_LAB[lift$scenario], levels = SC_LAB)
g5 <- ggplot(lift, aes(percentile, lift, colour = set)) +
  geom_hline(yintercept = 1, linetype = 2, colour = "grey60") +
  geom_line(linewidth = 1) + geom_point(size = 2) +
  facet_wrap(~ scenario_l) +
  scale_x_continuous(breaks = 1:10 * 10) +
  scale_colour_manual(values = c(validation = unibg_palette[3], test = unibg_palette[6])) +
  labs(title = "Random forest: cumulative lift of the selected model", x = "% of students selected (deciles)", y = "Cumulative lift", colour = NULL) + th
ggsave(file.path(OUT_DIR, "rf_lift_cumulato_unibg.png"), g5, width = 11, height = 5.5, dpi = 300, device = agg_png, bg = "white")

# 6. curve ROC (train OOB, validation, test): righe = scenario, colonne = modello
roc_df <- do.call(rbind, lapply(split(previsioni, list(previsioni$scenario, previsioni$modello, previsioni$set), drop = TRUE), function(d) {
  r <- roc(d$y, d$prob, levels = c(0, 1), direction = "<", quiet = TRUE)
  data.frame(scenario = d$scenario[1], modello = d$modello[1], set = d$set[1],
             fpr = 1 - r$specificities, tpr = r$sensitivities, auc = as.numeric(auc(r)))
}))
roc_df$base     <- factor(base_of(roc_df$modello), levels = BASI)
roc_df$set      <- factor(roc_df$set, levels = c("train", "validation", "test"))
roc_df$scenario <- factor(SC_LAB[roc_df$scenario], levels = SC_LAB)
auc_lab <- unique(roc_df[, c("scenario", "base", "set", "auc")])
auc_lab$y <- c(train = 0.25, validation = 0.15, test = 0.05)[as.character(auc_lab$set)]
auc_lab$txt <- sprintf("AUC %s = %.3f", as.character(auc_lab$set), auc_lab$auc)
COL_SET <- c(train = unibg_palette[6], validation = unibg_palette[1], test = unibg_palette[5])
g6 <- ggplot(roc_df, aes(fpr, tpr, colour = set)) +
  geom_abline(intercept = 0, slope = 1, linetype = 2, colour = "grey60") +
  geom_path(linewidth = 0.9) +
  geom_text(data = auc_lab, aes(x = 1, y = y, label = txt, colour = set), hjust = 1, size = 3.4, show.legend = FALSE) +
  facet_grid(scenario ~ base, labeller = labeller(base = BASE_EN)) +
  scale_colour_manual(values = COL_SET) +
  coord_equal() +
  labs(title = "Random forest: ROC curves by scenario (rows) and model (columns)",
       x = "1 - specificity (FPR)", y = "Sensitivity (TPR)", colour = NULL) + th
ggsave(file.path(OUT_DIR, "rf_roc_unibg.png"), g6, width = 12, height = 8.5, dpi = 300, device = agg_png, bg = "white")
write.csv(unique(roc_df[, c("scenario", "base", "set", "auc")]), file.path(OUT_DIR, "auc_roc.csv"), row.names = FALSE)

# 7. importanza delle variabili (Gini) del modello scelto
imp_sc <- do.call(rbind, lapply(out, function(o) {
  d <- importanza[importanza$scenario == o$best$scenario & importanza$modello == o$scelto, ]
  head(d[order(d$MeanDecreaseGini, decreasing = TRUE), ], IMP_TOP)
}))
imp_sc$scenario_l <- factor(SC_LAB[imp_sc$scenario], levels = SC_LAB)
imp_sc$var_o <- reorder(paste(imp_sc$variabile, imp_sc$scenario), imp_sc$MeanDecreaseGini)
g7 <- ggplot(imp_sc, aes(var_o, MeanDecreaseGini)) +
  geom_segment(aes(xend = var_o, yend = 0), colour = unibg_palette[1], linewidth = 1.1) +
  geom_point(size = 3.5, colour = unibg_palette[1]) +
  coord_flip() +
  scale_x_discrete(labels = function(x) sub(" [0-9]_.*$", "", x)) +
  facet_wrap(~ scenario_l, scales = "free") +
  labs(title = sprintf("Random forest: top %d variables (Gini, selected model)", IMP_TOP), x = NULL, y = "Gini mean decrease") + th
ggsave(file.path(OUT_DIR, "rf_importanza_feature_unibg.png"), g7, width = 12, height = 6.5, dpi = 300, device = agg_png, bg = "white")

# 8. PDP: un grafico per variabile (le prime PDP_TOP per importanza) e per scenario, come nella pipeline di Dario
for (o in out) {
  sc <- o$best$scenario
  d <- pdp[pdp$scenario == sc, ]
  for (v in unique(d$variabile)) {
    dv <- d[d$variabile == v, ]
    tit <- sprintf("Partial dependence: %s", v); sub_ <- SC_LAB[sc]
    if (!all(is.na(dv$valore_num))) {
      g8 <- ggplot(dv, aes(valore_num, yhat)) + geom_line(colour = unibg_palette[1], linewidth = 1.2)
    } else {
      lab <- ifelse(nchar(dv$valore) > 45, paste0(substr(dv$valore, 1, 43), ".."), dv$valore)   # etichette lunghe accorciate
      dv$valore <- reorder(make.unique(lab), dv$yhat)
      g8 <- ggplot(dv, aes(valore, yhat)) +
        geom_segment(aes(xend = valore, yend = 0), colour = unibg_palette[1], linewidth = 1.2) +
        geom_point(size = 4, colour = unibg_palette[1]) + coord_flip()
    }
    g8 <- g8 + labs(title = tit, subtitle = sub_, x = NULL, y = "Predicted P(dropout)") +
      scale_y_continuous(breaks = scales::pretty_breaks(4)) + th + theme(plot.title.position = "plot")
    if (!all(is.na(dv$valore_num))) g8 <- g8 + labs(x = v)
    ggsave(file.path(OUT_DIR, paste0("rf_pdp_", sc, "_", gsub("[^A-Za-z0-9_]", "_", v), "_unibg.png")), g8,
           width = 7, height = 5, dpi = 300, device = agg_png, bg = "white")
  }
}

# 9. superficie 3D dell'ASE di validation (mtry x nodesize), una per scenario; pallino nero = scelta
for (o in out) {
  sc <- o$best$scenario; best <- o$best
  d <- tuning[tuning$scenario == sc, ]
  ns_sorted <- sort(unique(d$nodesize)); mt_sorted <- sort(unique(d$mtry))
  z <- sapply(ns_sorted, function(ns) sapply(mt_sorted, function(mt) d$VAL_ASE[d$nodesize == ns & d$mtry == mt]))
  agg_png(file.path(OUT_DIR, paste0("rf_3d_", sc, "_unibg.png")), width = 8, height = 6, units = "in", res = 300, background = "white")
  par(mar = c(2, 3, 3, 2), cex.axis = 0.8, family = "Rubik")
  zf  <- (z[-1, -1] + z[-1, -ncol(z)] + z[-nrow(z), -1] + z[-nrow(z), -ncol(z)]) / 4      # ASE medio di ogni faccia
  col <- colorRampPalette(c(unibg_palette[8], unibg_palette[2]))(100)[cut(zf, 100, labels = FALSE)]
  pm  <- persp(mt_sorted, log10(ns_sorted), z, theta = -40, phi = 25, expand = 0.6, col = col,
               border = "white", lwd = 0.5, ticktype = "detailed", nticks = 5,
               xlab = "mtry", ylab = "log10(nodesize)", zlab = "ASE",
               main = sprintf("Validation ASE - %s", SC_LAB[sc]))
  points(trans3d(best$mtry, log10(best$nodesize), best$VAL_ASE, pm), pch = 19, cex = 2)
  invisible(dev.off())
  
  # versione interattiva (html, si ruota col mouse): solo se plotly e htmlwidgets sono installati
  if (requireNamespace("plotly", quietly = TRUE) && requireNamespace("htmlwidgets", quietly = TRUE)) {
    ix <- seq_along(mt_sorted); iy <- seq_along(ns_sorted)
    bx <- match(best$mtry, mt_sorted); by <- match(best$nodesize, ns_sorted)
    fig <- plotly::plot_ly(x = iy, y = ix, z = z, type = "surface",
                           colorscale = list(c(0, unibg_palette[8]), c(1, unibg_palette[2])), colorbar = list(title = "ASE"))
    fig <- plotly::add_markers(fig, x = by, y = bx, z = z[bx, by], inherit = FALSE,
                               marker = list(color = "black", size = 5), showlegend = FALSE)
    fig <- plotly::layout(fig, title = sprintf("Validation ASE - %s", SC_LAB[sc]),
                          scene = list(xaxis = list(title = "nodesize", tickvals = iy, ticktext = ns_sorted),
                                       yaxis = list(title = "mtry", tickvals = ix, ticktext = mt_sorted),
                                       zaxis = list(title = "ASE")))
    htmlwidgets::saveWidget(fig, file.path(OUT_DIR, paste0("rf_3d_", sc, ".html")), selfcontained = FALSE)
  }
}

cat("\nFinito. Metriche in:", OUT_METRICHE, "\nAltri file e grafici in:", OUT_DIR, "\n")