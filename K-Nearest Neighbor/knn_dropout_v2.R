# ============================================================================
# GRAPHIC-STYLE COPY of knn_dropout_v2.R
# Same analysis as the original; only the plots changed (Rubik font, UniBG palette,
# English titles/labels, as in assessment_modelli_C_v2.R). PNG files get the suffix _unibg.
# ============================================================================
rm(list = ls())

################################################################################
#  kNN per il DROPOUT - dataset C_v2 - scenari EARLY_ISEE e MID  (script unico)
#
#  Cosa fa, in un'unica esecuzione:
#   1. legge C_v2 (train / valid / test già divisi) e le variabili di ogni scenario dal file
#      variabili_scenari_C_v2.csv (colonne EARLY_ISEE e MID: 1 = usare, 0 = escludere)
#   2. per ogni scenario cerca i vicini UNA volta con k = max(K_GRID) con tre distanze:
#        - euclidea (variabili standardizzate + dummy)
#        - Gower semplice          (pesi uguali, pesi^0)
#        - Gower pesato            (peso = importanza random forest, stimata sul train)
#   3. voto tra i k vicini: uniforme, P(Yes) = quota di Yes tra i k vicini
#      (il voto "per distanza" è disponibile: basta mettere VOTI <- c("uniforme", "per_distanza"),
#       ma nei test non ha migliorato niente e l'abbiamo tolto)
#   4. sceglie k sul VALIDATION (minimo ASE; a parità il k più grande), mai sul test
#   5. valuta ogni configurazione con due soglie: 0.5 e il tasso di dropout nel train
#   6. scrive TUTTE le metriche in un unico file, metriche_confronto_knn.csv (colonne compatibili con
#      metriche_confronto.csv degli XGBoost), più tuning, previsioni, soglie e grafici
#
#  Le metriche "train" sono calcolate in leave-one-out (ogni studente non vota per sé stesso).
#  La configurazione "scelta" per scenario è quella col miglior ASE sul validation.
################################################################################

library(caret)      # dummyVars, preProcess, nearZeroVar, findCorrelation
library(pROC)       # AUC
library(FNN)        # vicini euclidei
library(gower)      # vicini di Gower:  install.packages("gower")
library(ranger)     # importanza delle variabili per i pesi: install.packages("ranger")
library(ggplot2)
library(arrow)      # lettura dei .parquet
library(ragg)

## PERCORSI
DATA_DIR <- "C:\\Users\\Dario\\Downloads\\Grafici sistemati\\dataset_C"
BASE     <- "C:\\Users\\Dario\\Downloads\\Grafici sistemati\\K-Nearest Neighbor"
OUT_DIR  <- file.path(BASE, "knn_v2")                      # tuning, previsioni, grafici
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
FILE_PREFIX <- "C_v2"                                      # C_v2_train.parquet, C_v2_valid.parquet, C_v2_test.parquet
VARS_FILE   <- file.path(DATA_DIR, "variabili_scenari_C_v2.csv")
OUT_METRICHE <- file.path(DATA_DIR, "metriche_confronto_knn.csv")   # il file con tutte le metriche

# scenario (nome come nel file degli XGBoost) -> colonna di variabili_scenari_C_v2.csv
SCEN <- data.frame(
  scenario = c("1_early_isee", "2_mid_sem1"),
  col_var  = c("EARLY_ISEE", "MID"),
  stringsAsFactors = FALSE)
DATASET_NOME <- "C"
DESCRIZIONE  <- "C v2 - con TOLC, coorti dal 2020"

## PARAMETRI
TARGET      <- "DROPOUT"
SEED        <- 1
K_GRID      <- c(11, 21, 31, 41, 51, 71, 101, 131, 161, 201, 301, 401, 501, 601)   # dispari -> meno pareggi
MAX_LEVELS  <- 10              # al massimo i top-N livelli per variabile categorica, gli altri -> "Altro"
CORR_CUTOFF <- 0.95            # tolgo una variabile di ogni coppia con |cor| sopra questa soglia
WINSOR      <- c(0.01, 0.99)   # (Gower) numeriche tagliate a questi percentili del train, poi in [0,1]
POW_GRID    <- c(0, 1)         # (Gower) potenze dell'importanza RF usate come pesi (0 = pesi uguali, 1 = importanza piena;
                               #  0.5 provato: praticamente identico a 1, tolto)
VOTI        <- "uniforme"      # modi di votare tra i vicini: "uniforme" e/o "per_distanza"
RF_TREES    <- 200             # alberi della random forest che dà i pesi
EPS_REL     <- 0.01            # voto per distanza: peso = 1 / (d + EPS_REL * mediana delle distanze del train)
THR_GRID    <- seq(0, 1, by = 0.02)   # soglie del grafico/tabella di sensibilità alla soglia

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

# le configurazioni provate: distanza x pesi x voto
CONF <- rbind(
  data.frame(dist = "euclidea", pow = NA_real_),
  data.frame(dist = "gower",    pow = POW_GRID))
CONF$base <- ifelse(CONF$dist == "euclidea", "euclidea",
                    ifelse(CONF$pow == 0, "gower (pesi uguali)", "gower pesato"))
CONF <- do.call(rbind, lapply(VOTI, function(v) cbind(CONF, voto = v)))
CONF$modello <- paste0("kNN_", gsub("[ ()]", "", gsub("pesi uguali", "p0", CONF$base)), "_", CONF$voto)
rownames(CONF) <- NULL

## LETTURA
read_set <- function(f) {
  d <- as.data.frame(read_parquet(f))
  d$y <- factor(d[[TARGET]], levels = c(0, 1), labels = c("No", "Yes"))
  d
}

## FUNZIONI DI SUPPORTO
fast_auc <- function(p, y01) {              # AUC basata sui ranghi (ranghi medi per i pareggi)
  n1 <- sum(y01 == 1); n0 <- sum(y01 == 0)
  (sum(rank(p)[y01 == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}
fast_ase <- function(p, y01) mean((y01 - p)^2)   # ASE = Brier score

# risultato di una ricerca di vicini: lista(idx = indici nel train, d = distanze), righe = studenti, colonne = vicini
nn_euclid <- function(Zref, Zquery, k) {
  r <- get.knnx(as.matrix(Zref), as.matrix(Zquery), k = k, algorithm = "brute")
  list(idx = r$nn.index, d = r$nn.dist)
}
nn_gower <- function(ref, query, k, w) {
  r <- gower_topn(query, ref, n = k, weights = w)      # matrici k x n_query: le traspongo
  list(idx = t(r$index), d = t(r$distance))
}
# leave-one-out: la ricerca train->train con K+1 vicini include il punto stesso; lo tolgo da ogni riga
# (se non c'è, per pareggi a distanza 0, tolgo l'ultimo vicino)
drop_self <- function(nn) {
  n <- nrow(nn$idx); k1 <- ncol(nn$idx)
  self <- nn$idx == seq_len(n)
  self[rowSums(self) == 0, k1] <- TRUE
  keep <- !t(self)
  list(idx = matrix(t(nn$idx)[keep], nrow = n, byrow = TRUE),
       d   = matrix(t(nn$d)[keep],   nrow = n, byrow = TRUE))
}
cut_k <- function(nn, k) list(idx = nn$idx[, seq_len(k), drop = FALSE], d = nn$d[, seq_len(k), drop = FALSE])

# P(Yes) per ogni k della griglia, per voto uniforme o per distanza (somme cumulate sull'ordine dei vicini)
nn_to_probs <- function(nn, ytr01, voto, eps) {
  votes <- matrix(ytr01[nn$idx], nrow = nrow(nn$idx))
  if (voto == "uniforme") {
    cs <- t(apply(votes, 1, cumsum))
    sweep(cs[, K_GRID, drop = FALSE], 2, K_GRID, "/")
  } else {
    w  <- 1 / (nn$d + eps)
    cv <- t(apply(votes * w, 1, cumsum))
    cw <- t(apply(w, 1, cumsum))
    cv[, K_GRID, drop = FALSE] / cw[, K_GRID, drop = FALSE]
  }
}

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

# lift per decili (decile 1 = studenti con probabilità più alta)
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
  set.seed(SEED)
  cat("\n==================  SCENARIO:", sc, " (", col_var, ")  ==================\n")

  feat <- vars_tab$variabile[vars_tab[[col_var]] == 1]
  feat <- setdiff(feat, TARGET)
  stopifnot(all(feat %in% names(tr_df)))
  cat("Variabili usate:", length(feat), "\n ", paste(feat, collapse = ", "), "\n")

  tr_raw <- tr_df[, feat]; va_raw <- va_df[, feat]; te_raw <- te_df[, feat]
  y_tr <- tr_df$y; y_va <- va_df$y; y_te <- te_df$y
  y01_tr <- as.integer(y_tr == "Yes"); y01_va <- as.integer(y_va == "Yes")
  prev_tr <- mean(y01_tr)                       # tasso di dropout nel TRAIN = seconda soglia
  cat(sprintf("Tasso di dropout: train %.3f | valid %.3f | test %.3f\n", prev_tr, mean(y01_va), mean(y_te == "Yes")))

  is_cat <- sapply(tr_raw, function(v) is.character(v) || is.logical(v) || is.factor(v))
  num_cols <- names(tr_raw)[!is_cat]
  cat_cols <- names(tr_raw)[is_cat]
  cat("Categoriche:", length(cat_cols), " Numeriche:", length(num_cols), "\n")

  # tutto ciò che si impara (mediane, livelli, dummy, scaling, range, pesi) è stimato SOLO sul train
  med      <- sapply(tr_raw[num_cols], median, na.rm = TRUE)
  na_cols  <- num_cols[sapply(tr_raw[num_cols], anyNA)]
  levels_ok <- lapply(tr_raw[cat_cols], function(v) {
    v <- as.character(v); v[is.na(v)] <- "ND"
    names(sort(table(v), decreasing = TRUE))[seq_len(min(MAX_LEVELS, length(unique(v))))]
  })
  base_prep <- function(d) {                    # imputazione, flag NA, livelli rari -> "Altro"
    out <- d
    for (nm in na_cols) out[[paste0("NA_", nm)]] <- as.numeric(is.na(d[[nm]]))
    for (nm in num_cols) { v <- as.numeric(d[[nm]]); v[is.na(v)] <- med[[nm]]; out[[nm]] <- v }
    for (nm in cat_cols) {
      v <- as.character(d[[nm]]); v[is.na(v)] <- "ND"
      v[!v %in% levels_ok[[nm]]] <- "Altro"
      out[[nm]] <- factor(v, levels = union(levels_ok[[nm]], "Altro"))
    }
    out
  }
  tr_p <- base_prep(tr_raw); va_p <- base_prep(va_raw); te_p <- base_prep(te_raw)

  ## --- SPAZIO EUCLIDEO: dummy (fullRank) + NZV + correlazioni + standardizzazione
  dv <- dummyVars(~ ., data = tr_p, fullRank = TRUE)
  Etr <- as.data.frame(predict(dv, tr_p)); Eva <- as.data.frame(predict(dv, va_p)); Ete <- as.data.frame(predict(dv, te_p))
  nzv <- nearZeroVar(Etr); if (length(nzv)) Etr <- Etr[, -nzv, drop = FALSE]
  hc <- findCorrelation(cor(Etr), cutoff = CORR_CUTOFF); if (length(hc)) Etr <- Etr[, -hc, drop = FALSE]
  ecols <- names(Etr); Eva <- Eva[, ecols, drop = FALSE]; Ete <- Ete[, ecols, drop = FALSE]
  pp <- preProcess(Etr, method = c("zv", "center", "scale"))
  Ztr <- predict(pp, Etr); Zva <- predict(pp, Eva); Zte <- predict(pp, Ete)
  cat("Euclidea: colonne finali (dopo dummy, NZV, correlazioni):", ncol(Ztr), "\n")

  ## --- SPAZIO DI GOWER: numeriche in [0,1] (tagliate ai percentili del train), categoriche come fattori
  rng <- lapply(num_cols, function(nm) {
    v <- as.numeric(tr_raw[[nm]]); v[is.na(v)] <- med[[nm]]
    r <- as.numeric(quantile(v, WINSOR)); if (r[2] <= r[1]) r <- range(v); r
  }); names(rng) <- num_cols
  to01 <- function(d) {
    out <- d
    for (nm in num_cols) {
      r <- rng[[nm]]; v <- as.numeric(d[[nm]])
      out[[nm]] <- if (r[2] > r[1]) (pmin(pmax(v, r[1]), r[2]) - r[1]) / (r[2] - r[1]) else rep(0, length(v))
    }
    out
  }
  Gtr <- to01(tr_p); Gva <- to01(va_p); Gte <- to01(te_p)
  nzv_g <- nearZeroVar(Gtr); if (length(nzv_g)) Gtr <- Gtr[, -nzv_g, drop = FALSE]
  num_now <- names(Gtr)[sapply(Gtr, is.numeric)]
  hc_g <- if (length(num_now) > 1) findCorrelation(cor(Gtr[num_now]), cutoff = CORR_CUTOFF) else integer(0)
  if (length(hc_g)) Gtr <- Gtr[, setdiff(names(Gtr), num_now[hc_g]), drop = FALSE]
  gcols <- names(Gtr); Gva <- Gva[, gcols, drop = FALSE]; Gte <- Gte[, gcols, drop = FALSE]
  cat("Gower: variabili finali (dopo NZV e correlazioni):", length(gcols), "\n")

  ## --- PESI DELLE VARIABILI: importanza per permutazione di una random forest sul train
  rf_df <- Gtr; rf_df$y <- y_tr
  rf <- ranger(y ~ ., data = rf_df, num.trees = RF_TREES, probability = TRUE, importance = "permutation",
               respect.unordered.factors = "partition", min.node.size = 20, seed = SEED,
               num.threads = max(1, parallel::detectCores() - 1))
  imp <- pmax(rf$variable.importance[gcols], 0)
  W0 <- imp / sum(imp) * length(imp)             # importanza normalizzata (media 1)
  pesi_tab <- data.frame(scenario = sc, variabile = gcols, importanza = as.numeric(imp))
  for (p in POW_GRID) {
    wp <- W0^p; wp <- wp / sum(wp) * length(wp)
    pesi_tab[[paste0("peso_pow", p)]] <- as.numeric(wp)
  }
  cat("Variabili con più peso (pow 1):\n")
  print(round(sort(setNames(pesi_tab$peso_pow1, pesi_tab$variabile), decreasing = TRUE)[1:min(8, length(gcols))], 2))

  ## --- RICERCA DEI VICINI (la parte lenta) e probabilità per ogni k, per ogni configurazione
  K <- max(K_GRID)
  res_list <- list()                              # una entry per (dist, pow): list(tr = LOO, rs = resub, va, te)
  searches <- unique(CONF[, c("dist", "pow")])
  for (i in seq_len(nrow(searches))) {
    d <- searches$dist[i]; p <- searches$pow[i]
    key <- if (d == "euclidea") "euclidea" else paste0("gower_", p)
    cat(sprintf("\nVicini [%s]: train->train, validation->train, test->train ...\n", key)); t0 <- Sys.time()
    if (d == "euclidea") {
      a  <- nn_euclid(Ztr, Ztr, K + 1); v <- nn_euclid(Ztr, Zva, K); t <- nn_euclid(Ztr, Zte, K)
    } else {
      w  <- as.numeric(pesi_tab[[paste0("peso_pow", p)]])
      a  <- nn_gower(Gtr, Gtr, K + 1, w); v <- nn_gower(Gtr, Gva, K, w); t <- nn_gower(Gtr, Gte, K, w)
    }
    res_list[[key]] <- list(tr = drop_self(a), va = v, te = t)     # LOO per il train
    rm(a, v, t)
    cat("  fatto in", format(round(difftime(Sys.time(), t0, units = "mins"), 1)), "\n")
    gc()
  }

  tuning <- list(); prev_all <- list(); righe <- list(); sweeps <- list(); lifts <- list()
  probs_cfg <- list()
  for (r in seq_len(nrow(CONF))) {
    d <- CONF$dist[r]; p <- CONF$pow[r]; voto <- CONF$voto[r]; mod <- CONF$modello[r]
    key <- if (d == "euclidea") "euclidea" else paste0("gower_", p)
    nn <- res_list[[key]]
    eps <- EPS_REL * median(nn$tr$d)               # scala del peso 1/(d+eps), dalle distanze del train
    Ptr <- nn_to_probs(nn$tr, y01_tr, voto, eps)   # leave-one-out
    Pva <- nn_to_probs(nn$va, y01_tr, voto, eps)
    Pte <- nn_to_probs(nn$te, y01_tr, voto, eps)

    tun <- data.frame(scenario = sc, modello = mod, distanza = d, pesi_pow = p, voto = voto, k = K_GRID,
                      TRAIN_ASE_LOO = apply(Ptr, 2, fast_ase, y01 = y01_tr),
                      VAL_ASE = apply(Pva, 2, fast_ase, y01 = y01_va),
                      VAL_AUC = apply(Pva, 2, fast_auc, y01 = y01_va))
    tun$GAP <- tun$VAL_ASE - tun$TRAIN_ASE_LOO
    tuning[[mod]] <- tun

    ok <- tun[tun$VAL_ASE <= min(tun$VAL_ASE), ]     # k al minimo ASE validation, a parità il più grande
    best_k <- max(ok$k); j <- match(best_k, K_GRID)
    ptr <- Ptr[, j]; pva <- Pva[, j]; pte <- Pte[, j]
    probs_cfg[[mod]] <- list(va = pva, te = pte, k = best_k, val_ase = fast_ase(pva, y01_va))
    cat(sprintf("  %-34s k = %3d | ASE valid %.4f | AUC valid %.4f\n", mod, best_k, fast_ase(pva, y01_va), fast_auc(pva, y01_va)))

    prev_all[[mod]] <- rbind(
      data.frame(scenario = sc, modello = mod, set = "train",      y = y01_tr, prob = ptr),    # leave-one-out
      data.frame(scenario = sc, modello = mod, set = "validation", y = y01_va, prob = pva),
      data.frame(scenario = sc, modello = mod, set = "test",       y = as.integer(y_te == "Yes"), prob = pte))

    for (thr in c(0.5, prev_tr)) {
      mt <- metr(ptr, y_tr, thr); mv <- metr(pva, y_va, thr); ms <- metr(pte, y_te, thr)
      row <- data.frame(scenario = sc, dataset = DATASET_NOME, descrizione = DESCRIZIONE, modello = mod,
                        distanza = d, pesi_pow = p, voto = voto, k = best_k,
                        soglia = round(thr, 4),
                        tipo_soglia = ifelse(thr == 0.5, "0.5", "tasso_dropout_train"),
                        n_variabili = if (d == "euclidea") ncol(Ztr) else length(gcols),
                        valid_prevalenza = mean(y01_va))
      for (nm in MET_COLS) row[[paste0("train_", nm)]] <- mt[[nm]]
      for (nm in MET_COLS) row[[paste0("valid_", nm)]] <- mv[[nm]]
      for (nm in MET_COLS) row[[paste0("test_",  nm)]] <- ms[[nm]]
      righe[[length(righe) + 1]] <- row
    }
  }
  metr_sc <- do.call(rbind, righe)

  # configurazione scelta per questo scenario: miglior ASE sul VALIDATION (il test non c'entra)
  va_ase <- sapply(probs_cfg, `[[`, "val_ase")
  scelto <- names(which.min(va_ase))
  metr_sc$scelto_su_validation <- metr_sc$modello == scelto
  cat("\nConfigurazione scelta sul validation:", scelto, "(k =", probs_cfg[[scelto]]$k, ")\n")

  # sensibilità alla soglia + lift, solo per la configurazione scelta
  pv <- probs_cfg[[scelto]]$va; pt <- probs_cfg[[scelto]]$te
  sw <- do.call(rbind, lapply(sort(unique(c(THR_GRID, prev_tr))), function(thr) {
    rbind(cbind(scenario = sc, modello = scelto, set = "validation", soglia = thr, metr(pv, y_va, thr)[, -(1:2)]),
          cbind(scenario = sc, modello = scelto, set = "test",       soglia = thr, metr(pt, y_te, thr)[, -(1:2)]))
  }))
  lf <- rbind(cbind(scenario = sc, modello = scelto, set = "validation", lift_decili(pv, y_va)),
              cbind(scenario = sc, modello = scelto, set = "test",       lift_decili(pt, y_te)))

  list(metriche = metr_sc, tuning = do.call(rbind, tuning), previsioni = do.call(rbind, prev_all),
       pesi = pesi_tab, soglie = sw, lift = lf, scelto = scelto)
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

metriche  <- do.call(rbind, lapply(out, `[[`, "metriche"))
tuning    <- do.call(rbind, lapply(out, `[[`, "tuning"))
previsioni <- do.call(rbind, lapply(out, `[[`, "previsioni"))
pesi      <- do.call(rbind, lapply(out, function(o) o$pesi))
soglie    <- do.call(rbind, lapply(out, `[[`, "soglie"))
lift      <- do.call(rbind, lapply(out, `[[`, "lift"))

write.csv(metriche,   OUT_METRICHE,                                  row.names = FALSE)   # IL file con tutte le metriche
write.csv(tuning,     file.path(OUT_DIR, "knn_tuning_results.csv"),  row.names = FALSE)
write.csv(previsioni, file.path(OUT_DIR, "previsioni_knn.csv"),      row.names = FALSE)   # per rifare soglie/grafici senza rilanciare
write.csv(pesi,       file.path(OUT_DIR, "pesi_variabili.csv"),      row.names = FALSE)
write.csv(soglie,     file.path(OUT_DIR, "sensibilita_soglia.csv"),  row.names = FALSE)
write.csv(lift,       file.path(OUT_DIR, "lift_decili.csv"),         row.names = FALSE)
cat("\nTempo totale:", format(round(difftime(Sys.time(), T0, units = "mins"), 1)), "\n")

# riepilogo a schermo: test, soglia 0.5 e soglia = tasso di dropout
riep <- metriche[, c("scenario", "modello", "k", "tipo_soglia", "valid_ASE", "test_AUC", "test_ASE",
                     "test_sensibilita", "test_specificita", "test_precisione", "test_F1", "scelto_su_validation")]
print(riep, row.names = FALSE)

################################################################################
## GRAFICI
################################################################################
th <- tema
SC_LAB  <- c("1_early_isee" = "Early", "2_mid_sem1" = "Mid (after 1st semester)")
BASE_EN <- c("euclidea" = "Euclidean", "gower (pesi uguali)" = "Gower (equal weights)", "gower pesato" = "Gower (weighted)")
BASI <- unique(CONF$base)
COL_BASE <- setNames(unibg_palette[c(1, 3, 5, 7)][seq_along(BASI)], BASI)
tuning$base <- CONF$base[match(tuning$modello, CONF$modello)]
tuning$scenario <- factor(tuning$scenario, levels = SCEN$scenario)
best_pts <- metriche[metriche$tipo_soglia == "0.5", c("scenario", "modello", "k", "valid_ASE")]
best_pts$base <- CONF$base[match(best_pts$modello, CONF$modello)]
best_pts$voto <- CONF$voto[match(best_pts$modello, CONF$modello)]
best_pts$scenario <- factor(best_pts$scenario, levels = SCEN$scenario)
tuning$voto <- factor(tuning$voto, levels = c("uniforme", "per_distanza"))
kpos <- function(k) match(k, K_GRID)

# 1. ASE sul validation al variare di k (equidistante), una curva per configurazione
g1 <- ggplot(tuning, aes(kpos(k), VAL_ASE, colour = base, group = modello)) +
  geom_line(linewidth = 1) + geom_point(size = 1.8) +
  geom_point(data = best_pts, aes(kpos(k), valid_ASE, colour = base), inherit.aes = FALSE, shape = 18, size = 4) +
  facet_wrap(~ scenario, scales = "free_y", labeller = as_labeller(SC_LAB)) +
  scale_x_continuous(breaks = seq_along(K_GRID), labels = K_GRID) +
  scale_colour_manual(values = COL_BASE, labels = BASE_EN) +
  labs(title = "kNN: model complexity vs ASE",
       x = "k (number of neighbours)", y = "ASE", colour = NULL) + th
ggsave(file.path(OUT_DIR, "knn_ase_validation_vs_k_unibg.png"), g1, width = 12, height = 6, dpi = 300, device = agg_png, bg = "white")

# 2. performance sul test (soglia 0.5): AUC e ASE per configurazione
bar <- metriche[metriche$tipo_soglia == "0.5", ]
bar$base <- CONF$base[match(bar$modello, CONF$modello)]
bar$scenario <- factor(bar$scenario, levels = SCEN$scenario)
bar$voto <- factor(bar$voto, levels = c("uniforme", "per_distanza"))
bar_long <- rbind(data.frame(bar[, c("scenario", "base", "voto")], valore = bar$test_AUC, metrica = "AUC"),
                  data.frame(bar[, c("scenario", "base", "voto")], valore = bar$test_ASE, metrica = "ASE"))
g2 <- ggplot(bar_long, aes(base, valore, fill = base)) +
  geom_col(position = position_dodge(width = 0.85), width = 0.8) +
  geom_text(aes(label = sprintf("%.3f", valore)), position = position_dodge(width = 0.85), vjust = -0.4, size = 2.8) +
  facet_grid(metrica ~ scenario, scales = "free_y", labeller = labeller(scenario = SC_LAB)) +
  scale_fill_manual(values = COL_BASE, guide = "none") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  scale_x_discrete(labels = BASE_EN) +
  labs(title = "kNN: test performance", x = NULL, y = NULL) + th +
  theme(axis.text.x = element_text(angle = 0))
ggsave(file.path(OUT_DIR, "knn_test_performance_unibg.png"), g2, width = 12, height = 7, dpi = 300, device = agg_png, bg = "white")

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
  labs(title = "kNN: classification measurements varying threshold",
       x = "Dropout probability threshold", y = NULL, colour = NULL) + th
ggsave(file.path(OUT_DIR, "knn_effetto_soglia_unibg.png"), g3, width = 11, height = 5.5, dpi = 300, device = agg_png, bg = "white")

# 4. lift cumulato del modello scelto (validation e test)
lf <- lift
g4 <- ggplot(lf, aes(percentile, lift, colour = set)) +
  geom_hline(yintercept = 1, linetype = 2, colour = "grey60") +
  geom_line(linewidth = 1) + geom_point(size = 2) +
  facet_wrap(~ scenario, labeller = as_labeller(SC_LAB)) +
  scale_x_continuous(breaks = 1:10 * 10) +
  scale_colour_manual(values = c(validation = unibg_palette[3], test = unibg_palette[6])) +
  labs(title = "kNN: cumulative lift of the selected model", x = "% of students selected (deciles)", y = "Cumulative lift", colour = NULL) + th
ggsave(file.path(OUT_DIR, "knn_lift_cumulato_unibg.png"), g4, width = 11, height = 5.5, dpi = 300, device = agg_png, bg = "white")

# 5. curve ROC (train in leave-one-out, validation, test): righe = scenario, colonne = modello
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
  labs(title = "kNN: ROC curves by scenario (rows) and model (columns)",
       x = "1 - specificity (FPR)", y = "Sensitivity (TPR)", colour = NULL) + th
ggsave(file.path(OUT_DIR, "knn_roc_unibg.png"), g5, width = 12, height = 8.5, dpi = 300, device = agg_png, bg = "white")
write.csv(unique(roc_df[, c("scenario", "base", "set", "auc")]), file.path(OUT_DIR, "auc_roc.csv"), row.names = FALSE)

cat("\nFinito. Metriche in:", OUT_METRICHE, "\nAltri file e grafici in:", OUT_DIR, "\n")