################################################################################
#  TEST VELOCE: punteggio Early di XGBoost M1 (conservativo) come input della logit Mid
#
#  Come Logit_C_stacking_early_mid.R, ma il punteggio Early non viene dalla logit:
#  viene da XGBoost M1_conservativo (stessi parametri e stessa regola degli alberi
#  dello script XGBoost: early stopping sull'AUC di validation, pazienza 100).
#
#   1. XGBoost M1 sull'Early: early stopping sul validation -> numero di alberi
#   2. punteggio = log-odds previsto. Sul TRAIN e' out-of-fold (K_FOLD fold
#      stratificati, ogni fold con lo stesso numero di alberi); su validation e test
#      si usa il modello stimato sul train completo
#   3. logit Mid per blocchi annidati (come Logit_C_unibg.R), SENZA (A) e CON (B) il
#      punteggio come variabile in piu'; scelta per ASE validation minimo
#   4. bootstrap sulla differenza B - A (AUC e ASE) su validation e test
#
#  Output (cartella logit_stacking_xgb): stacking_xgb_confronto_per_blocchi.csv,
#                                        stacking_xgb_vincitori.csv
################################################################################
rm(list = ls())
library(arrow)     # lettura parquet
library(xgboost)

#### 0. Percorsi e impostazioni ####
DATA_DIR <- "C:\\Users\\Dario\\Downloads\\Grafici sistemati\\dataset_C"
BASE     <- "C:\\Users\\Dario\\Downloads\\Grafici sistemati\\Logit solo C"
OUT_DIR  <- file.path(BASE, "logit_stacking_xgb")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

TARGET         <- "DROPOUT"
K_FOLD         <- 5       # fold per il punteggio Early out-of-fold sul train
N_BOOT         <- 500     # ricampionamenti bootstrap
SEME           <- 2026    # come lo script XGBoost
NROUNDS_MAX    <- 3000    # come lo script XGBoost
EARLY_STOPPING <- 100
NTHREAD        <- max(1, parallel::detectCores() - 1)
set.seed(123)

# M1_conservativo, identico a xgboost_dropout_FINALE_unibg.R
M1 <- list(eta = 0.03, max_depth = 3, min_child_weight = 10, gamma = 1,
           subsample = 0.8, colsample_bytree = 0.8,
           lambda = 5, alpha = 0, scale_pos_weight = 1)
params <- c(list(objective = "binary:logistic", eval_metric = "auc",
                 nthread = NTHREAD, seed = SEME), M1)

file_necessari <- c("C_v2_train.parquet", "C_v2_valid.parquet", "C_v2_test.parquet",
                    "variabili_scenari_C_v2.csv")
mancanti <- file_necessari[!file.exists(file.path(DATA_DIR, file_necessari))]
if (length(mancanti) > 0)
  stop("Nella cartella '", DATA_DIR, "' mancano: ", paste(mancanti, collapse = ", "))

ELENCO_VARIABILI <- read.csv(file.path(DATA_DIR, "variabili_scenari_C_v2.csv"),
                             stringsAsFactors = FALSE)
variabili_scenario <- function(colonna) ELENCO_VARIABILI$variabile[ELENCO_VARIABILI[[colonna]] == 1]

# blocchi e esclusione della logit: come Logit_C_unibg.R
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
var.escluse.logit <- c("res_car_missing")
usa_e_xgb <- variabili_scenario("EARLY_ISEE")                            # XGBoost: tutte le variabili del csv
usa_m     <- setdiff(variabili_scenario("MID"), var.escluse.logit)      # logit Mid
stopifnot("variabili senza blocco" = all(usa_m %in% unlist(blocchi)))

#### 1. Funzioni ####
auc_mw <- function(y, p) {
  r <- rank(p); n1 <- sum(y == 1); n0 <- sum(y == 0)
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}
ase <- function(y, p) mean((p - y)^2)

# codifica per XGBoost: stessa funzione dello script XGBoost (logiche -> 0/1, categoriche -> dummy)
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
  if (length(costanti) > 0) { tutte[costanti] <- NULL; fattori <- setdiff(fattori, costanti) }
  split_col <- tutte$.split
  tutte$.split <- NULL
  mm <- model.matrix(~ . - 1, data = tutte,
                     contrasts.arg = lapply(tutte[fattori], contrasts, contrasts = FALSE))
  colnames(mm) <- make.names(colnames(mm), unique = TRUE)
  out <- lapply(nomi_split, function(s) list(X = mm[split_col == s, , drop = FALSE], y = y[[s]]))
  names(out) <- nomi_split
  out
}

# xgb.train compatibile con le diverse versioni del pacchetto (come lo script XGBoost)
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
miglior_iterazione <- function(modello) {
  el <- log_valutazione(modello)
  which.max(el[[grep("^valid_", names(el), value = TRUE)[1]]])
}

# logit: previsione robusta a livelli mai visti in stima
prevedi <- function(fit, newdata, tipo = "response") {
  for (v in names(fit$xlevels)) {
    nd <- as.character(newdata[[v]])
    nd[!nd %in% fit$xlevels[[v]]] <- NA
    newdata[[v]] <- factor(nd, levels = fit$xlevels[[v]])
  }
  p <- as.numeric(suppressWarnings(predict(fit, newdata = newdata, type = tipo)))
  p[is.na(p)] <- mean(p, na.rm = TRUE)
  p
}
fit_annidati <- function(usa, d_tr, extra = NULL) {
  fit <- list(); nomi <- character(); vars <- c(); k <- 0
  for (b in names(blocchi)) {
    nuove <- intersect(blocchi[[b]], usa)
    if (length(nuove) == 0) next
    k <- k + 1; vars <- c(vars, nuove); nomi[k] <- b
    fit[[k]] <- suppressWarnings(glm(reformulate(c(vars, extra), response = TARGET),
                                     data = d_tr, family = binomial))
  }
  list(fit = fit, blocco = nomi)
}

#### 2. Dati ####
leggi <- function(file) as.data.frame(read_parquet(file.path(DATA_DIR, file)))
dati <- list(train = leggi("C_v2_train.parquet"),
             valid = leggi("C_v2_valid.parquet"),
             test  = leggi("C_v2_test.parquet"))
cat("Righe train:", nrow(dati$train), "| validation:", nrow(dati$valid),
    "| test:", nrow(dati$test), "\n")
y <- lapply(dati, function(x) as.numeric(x[[TARGET]]))

#### 3. Punteggio Early di XGBoost M1 (log-odds) ####
d_e <- lapply(dati, function(x) x[, c(usa_e_xgb, TARGET)])
if (any(sapply(d_e, anyNA))) stop("Ci sono NA nelle variabili Early")
m <- prepara_matrici(d_e, TARGET, character(0))
dtrain <- xgb.DMatrix(m$train$X, label = m$train$y)
dvalid <- xgb.DMatrix(m$valid$X, label = m$valid$y)
dtest  <- xgb.DMatrix(m$test$X,  label = m$test$y)

mod_es   <- addestra(params, dtrain, dvalid, NROUNDS_MAX, EARLY_STOPPING)   # 1: early stopping
n_alberi <- miglior_iterazione(mod_es)
modello  <- addestra(params, dtrain, dvalid, n_alberi)                       # 2: modello finale
cat(sprintf("\nXGBoost M1 Early: %d alberi\n", n_alberi))

score <- list(valid = predict(modello, dvalid, outputmargin = TRUE),
              test  = predict(modello, dtest,  outputmargin = TRUE))

# punteggio out-of-fold sul train (fold stratificati sul dropout)
fold <- integer(length(y$train))
for (cl in unique(y$train)) {
  i <- which(y$train == cl)
  fold[i] <- sample(rep(seq_len(K_FOLD), length.out = length(i)))
}
score$train <- numeric(length(y$train))
for (f in seq_len(K_FOLD)) {
  i_tr  <- fold != f
  d_f   <- xgb.DMatrix(m$train$X[i_tr, , drop = FALSE], label = m$train$y[i_tr])
  set.seed(SEME + f)
  mod_f <- xgb.train(params = params, data = d_f, nrounds = n_alberi, verbose = 0)
  score$train[!i_tr] <- predict(mod_f, xgb.DMatrix(m$train$X[!i_tr, , drop = FALSE]), outputmargin = TRUE)
}
cat(sprintf("AUC del solo punteggio XGBoost Early: train(OOF) %.3f | validation %.3f | test %.3f\n",
            auc_mw(y$train, score$train), auc_mw(y$valid, score$valid), auc_mw(y$test, score$test)))
cat("  (controllo: nell'assessment XGBoost M1 Early ha AUC validation 0.710 e test 0.721)\n")

#### 4. Logit Mid: A = senza punteggio, B = con EARLY_SCORE ####
d_m <- lapply(dati, function(x) x[, c(usa_m, TARGET)])
if (any(sapply(d_m, anyNA))) stop("Ci sono NA nelle variabili Mid")
for (s in names(d_m)) d_m[[s]]$EARLY_SCORE <- score[[s]]

mA <- fit_annidati(usa_m, d_m$train)
mB <- fit_annidati(usa_m, d_m$train, extra = "EARLY_SCORE")
prev <- function(mm) lapply(mm$fit, function(f) lapply(d_m, function(x) prevedi(f, x)))
pA <- prev(mA); pB <- prev(mB)

f0 <- suppressWarnings(glm(reformulate("EARLY_SCORE", response = TARGET), data = d_m$train, family = binomial))
p0 <- lapply(d_m, function(x) prevedi(f0, x))

riga <- function(modello, blocco, k, p) data.frame(
  modello = modello, blocchi = k, blocco.aggiunto = blocco,
  valid_ASE = ase(y$valid, p$valid), valid_AUC = auc_mw(y$valid, p$valid),
  test_ASE  = ase(y$test,  p$test),  test_AUC  = auc_mw(y$test,  p$test))
tab <- rbind(riga("solo EARLY_SCORE (XGB)", "-", 0, p0),
             do.call(rbind, lapply(seq_along(pA), function(k) riga("A: Mid", mA$blocco[k], k, pA[[k]]))),
             do.call(rbind, lapply(seq_along(pB), function(k) riga("B: Mid + EARLY_SCORE (XGB)", mB$blocco[k], k, pB[[k]]))))
write.csv(tab, file.path(OUT_DIR, "stacking_xgb_confronto_per_blocchi.csv"), row.names = FALSE)
cat("\n=== Confronto per numero di blocchi (A = Mid, B = Mid + punteggio XGBoost Early) ===\n")
print(format(tab, digits = 4), row.names = FALSE)

#### 5. Vincitori (ASE validation minimo) e bootstrap della differenza ####
kA <- which.min(tab$valid_ASE[tab$modello == "A: Mid"])
kB <- which.min(tab$valid_ASE[tab$modello == "B: Mid + EARLY_SCORE (XGB)"])
cat(sprintf("\nScelto A: %d blocchi (%s) | scelto B: %d blocchi (%s)\n",
            kA, mA$blocco[kA], kB, mB$blocco[kB]))
pa <- pA[[kA]]; pb <- pB[[kB]]

cs <- tryCatch(summary(mB$fit[[kB]])$coefficients, error = function(e) NULL)
if (!is.null(cs) && "EARLY_SCORE" %in% rownames(cs))
  cat(sprintf("Coefficiente di EARLY_SCORE in B: %.3f (errore standard %.3f, p = %.3g)\n",
              cs["EARLY_SCORE", 1], cs["EARLY_SCORE", 2], cs["EARLY_SCORE", 4]))
if (anyNA(coef(mB$fit[[kB]]))) cat("ATTENZIONE: coefficienti non stimabili (NA) nel modello B\n")

diff_boot <- function(yy, a, b) {
  n <- length(yy)
  replicate(N_BOOT, { i <- sample.int(n, replace = TRUE)
                      c(auc_mw(yy[i], b[i]) - auc_mw(yy[i], a[i]), ase(yy[i], b[i]) - ase(yy[i], a[i])) })
}
risultati <- data.frame()
for (s in c("valid", "test")) {
  r <- diff_boot(y[[s]], pa[[s]], pb[[s]])
  d_auc <- auc_mw(y[[s]], pb[[s]]) - auc_mw(y[[s]], pa[[s]])
  d_ase <- ase(y[[s]], pb[[s]]) - ase(y[[s]], pa[[s]])
  q_auc <- quantile(r[1, ], c(0.025, 0.975)); q_ase <- quantile(r[2, ], c(0.025, 0.975))
  cat(sprintf("%-5s | delta AUC (B-A) %+.4f [%+.4f, %+.4f] | delta ASE (B-A) %+.5f [%+.5f, %+.5f]\n",
              s, d_auc, q_auc[1], q_auc[2], d_ase, q_ase[1], q_ase[2]))
  risultati <- rbind(risultati, data.frame(set = s, blocchi_A = kA, blocchi_B = kB,
                     delta_AUC = d_auc, AUC_lo = q_auc[1], AUC_hi = q_auc[2],
                     delta_ASE = d_ase, ASE_lo = q_ase[1], ASE_hi = q_ase[2]))
}
write.csv(risultati, file.path(OUT_DIR, "stacking_xgb_vincitori.csv"), row.names = FALSE)
cat("\nLettura: se l'intervallo di delta AUC contiene 0 (e delta ASE pure), il punteggio\n",
    "XGBoost Early non aggiunge nulla alla logit Mid. Un delta ASE negativo = B migliore.\n", sep = "")
cat("File salvati in:", OUT_DIR, "\n")
