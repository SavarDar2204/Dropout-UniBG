# =====================================================================
# 01_split_e_dataset_tolc.R
# Unisce in un solo script:
#   PARTE 1) train/validation stratificato (70/30) da "File Finale Dropout.parquet"
#            con controllo di riproducibilita' rispetto ai file gia' esistenti
#   PARTE 2) da train.parquet e valid.parquet (con COORTE) crea:
#            - la divisione 70% train / 20% validation / 10% test
#            - quattro dataset pronti per il logit (nessun NA, tranne il TOLC
#              nel dataset B, che resta volutamente con i NA per l'imputation)
#
#   A_senza_tolc          tutte le coorti, nessuna colonna TOLC
#   B_tolc_con_nan        tutte le coorti, TOLC_PCT/TOLC_INGLESE con i NA
#   C_tolc_dal2020        solo coorte >= 2020, TOLC con NA imputato con bagImpute
#                         (caret, appreso sul solo train; + indicatori HA_TOLC / HA_TOLC_ING)
#   D_senza_tolc_dal2020  stesse righe di C, senza colonne TOLC
#                         (serve per confrontare C e D sulle stesse righe)
#
# REGOLA: le regole "per riga" valgono per tutti; mediane, medie e livelli
# si calcolano SOLO sul train e si riapplicano uguali a validation e test.
#
# Da lanciare UNA sola volta: tutto il gruppo poi usa gli stessi file.
# =====================================================================

library(arrow)
library(caret)

# ---------------------------------------------------------------------
# 0. SETUP  (cambia solo questa parte)
# ---------------------------------------------------------------------
setwd("C:/Users/claud/OneDrive/Desktop/Master/Statistical Modelling and Machine Learning/Dropout")   # <- il tuo percorso (attenzione agli spazi finali)

DATA_FILE  <- "File Finale Dropout.parquet"    # output di 00_data_cleaning.R
TRAIN_FILE <- "train.parquet"                  # i file con la colonna COORTE
VALID_FILE <- "valid.parquet"
OUT_DIR    <- "dati_tolc"
TARGET     <- "DROPOUT"
SEED       <- 1

# parte 1
P_TRAIN    <- 0.70
FORCE      <- FALSE   # TRUE solo se si vuole sovrascrivere anche se lo split cambia

# parte 2
# Quota di test: il train (70%) NON si tocca; il vecchio validation (30%) viene
# diviso in 2/3 validation (20% del totale) e 1/3 test (10% del totale).
FRAZ_TEST_DEL_VALID <- 1/3

# ISEE: chi non ha dichiarato va in fascia H se ha almeno un esame "con voto"
# nel 1o semestre, altrimenti "ND".
#   "voto"      = ha una media voti nel 1o semestre (esame passato con voto)
#   "sostenuti" = ha sostenuto almeno un esame (anche senza voto / non passato)
CRITERIO_H <- "voto"

SOGLIA_RARI  <- 100   # tipo maturita' con meno di 100 studenti nel train -> "Altro"
VOTO_MIN_OK  <- 0.6   # voto maturita' < 0.6 non e' valido -> NA
COORTE_TOLC  <- 2020  # il TOLC c'e' (~85-90%) solo dalla coorte 2020

# colonne usate da bagImpute per stimare i TOLC mancanti (dataset C): solo numeriche,
# senza il target. Gli NA delle altre sono gia' riempiti da applica().
VARS_IMP_TOLC <- c("TOLC_PCT", "TOLC_INGLESE", "VOTOMATURITA_PCT", "ETA_INGRESSO",
                   "ANNI_DA_MATURITA", "RITARDO_ISCR_GG", "ref_car_min")

TAGLI_ISEE  <- c(-Inf, 15000, 18000, 28000, 33000, 46000, 58000, 70000, Inf)
FASCE       <- c("A", "B", "C", "D", "E", "F", "G", "H")   # A <= 15000 ... H > 70000

options(scipen = 999, digits = 4)
dir.create(OUT_DIR, showWarnings = FALSE)

# =====================================================================
# PARTE 1 - TRAIN / VALIDATION (70/30, stratificato sul dropout)
# =====================================================================
# MODIFICA: prima di sovrascrivere train/valid gia' esistenti controlla che
# il nuovo split sia IDENTICO sulle colonne vecchie (stesso seed => stesse
# righe). Se non lo e' si ferma, cosi' non si rompono i modelli gia' fatti.

df <- as.data.frame(read_parquet(DATA_FILE))
cat("Rows:", nrow(df), " Cols:", ncol(df), "\n")

# split stratificato: stessa % di dropout in train e validation
set.seed(SEED)
idx_tr <- createDataPartition(factor(df[[TARGET]]), p = P_TRAIN, list = FALSE)[, 1]
train <- df[idx_tr, ]
valid <- df[-idx_tr, ]

# controlli
cat("\nTraining:", nrow(train), " Validation:", nrow(valid), "\n")
cat("% dropout train:", round(mean(train[[TARGET]]), 3),
    " validation:", round(mean(valid[[TARGET]]), 3), "\n")
stopifnot(nrow(train) + nrow(valid) == nrow(df),
          identical(names(train), names(valid)))

# --- controllo di riproducibilita' rispetto ai file gia' esistenti -------
stesso_split <- function(vecchio_file, nuovo) {
  if (!file.exists(vecchio_file)) return(TRUE)          # niente da confrontare
  # mmap = FALSE: su Windows la lettura "mappata" tiene il file bloccato e poi
  # write_parquet() sullo stesso file fallisce (errore 1224)
  vecchio <- as.data.frame(read_parquet(vecchio_file, mmap = FALSE))
  comuni  <- intersect(names(vecchio), names(nuovo))     # le colonne vecchie
  esito   <- isTRUE(all.equal(vecchio[, comuni, drop = FALSE],
                              nuovo[,   comuni, drop = FALSE],
                              check.attributes = FALSE))
  rm(vecchio); invisible(gc())                           # rilascia il file
  esito
}
ok_tr <- stesso_split(TRAIN_FILE, train)
ok_va <- stesso_split(VALID_FILE, valid)
cat("Split identico ai file esistenti -> train:", ok_tr, " valid:", ok_va, "\n")
if (!(ok_tr && ok_va) && !FORCE) {
  stop("Lo split e' DIVERSO da quello gia' salvato: non sovrascrivo. ",
       "Controlla file raw, versione di R/caret e SEED (oppure FORCE <- TRUE).")
}

invisible(gc())   # rilascia eventuali file ancora mappati prima di sovrascrivere
write_parquet(train, TRAIN_FILE)
write_parquet(valid, VALID_FILE)
cat("Salvati:", TRAIN_FILE, "e", VALID_FILE, "\n")

rm(df, train, valid, idx_tr)   # pulizia prima della parte 2
invisible(gc())

# =====================================================================
# PARTE 2 - DIVISIONE 70/20/10 E QUATTRO DATASET PER IL LOGIT
# =====================================================================

# ---------------------------------------------------------------------
# 2.1 LETTURA E DIVISIONE 70 / 20 / 10
# ---------------------------------------------------------------------
train_raw <- as.data.frame(read_parquet(TRAIN_FILE, mmap = FALSE))
valid_old <- as.data.frame(read_parquet(VALID_FILE, mmap = FALSE))
stopifnot(identical(names(train_raw), names(valid_old)),
          "COORTE" %in% names(train_raw), !anyNA(train_raw$COORTE), !anyNA(valid_old$COORTE))

# split stratificato sul dropout (stessa % di abbandoni nei due pezzi)
set.seed(SEED)
righe_per_classe <- split(seq_len(nrow(valid_old)), valid_old[[TARGET]])
idx_test <- sort(unlist(lapply(righe_per_classe, function(i)
  sample(i, size = round(length(i) * FRAZ_TEST_DEL_VALID)))))
test_raw  <- valid_old[idx_test, ]
valid_raw <- valid_old[-idx_test, ]

n_tot <- nrow(train_raw) + nrow(valid_raw) + nrow(test_raw)
cat("\n== DIVISIONE ==\n")
print(data.frame(set = c("train", "validation", "test"),
                 righe = c(nrow(train_raw), nrow(valid_raw), nrow(test_raw)),
                 perc = round(c(nrow(train_raw), nrow(valid_raw), nrow(test_raw)) / n_tot * 100, 1),
                 dropout = round(c(mean(train_raw[[TARGET]]), mean(valid_raw[[TARGET]]),
                                   mean(test_raw[[TARGET]])) * 100, 1)),
      row.names = FALSE)
stopifnot(n_tot == nrow(train_raw) + nrow(valid_old))

# i tre pezzi "grezzi" (con COORTE) per chi vuole usare lo stesso split altrove
write_parquet(train_raw, file.path(OUT_DIR, "split_train_grezzo.parquet"))
write_parquet(valid_raw, file.path(OUT_DIR, "split_valid_grezzo.parquet"))
write_parquet(test_raw,  file.path(OUT_DIR, "split_test_grezzo.parquet"))

# ---------------------------------------------------------------------
# 2.2 REGOLE PER RIGA (nessun parametro stimato)
# ---------------------------------------------------------------------
STATO_LEVELS <- c("A_nessun_esame", "B_sostenuti_0_passati",
                  "C_passati_senza_voto", "D_con_media")

stato_sem <- function(df, sem) {
  flg <- as.logical(df[[paste0("FLG_SEM", sem, "_ESAMI_SOSTENUTI")]])
  pct <- df[[paste0("PCT_N_ESAMI_PASSATI_SEM", sem)]]
  med <- df[[paste0("MEDIA_VOTO_DATI_SEM", sem)]]
  factor(ifelse(!flg, "A_nessun_esame",
                ifelse(pct == 0, "B_sostenuti_0_passati",
                       ifelse(is.na(med), "C_passati_senza_voto", "D_con_media"))),
         levels = STATO_LEVELS)
}

regole_riga <- function(df) {
  # voto di maturita' non valido -> NA
  df$VOTOMATURITA_PCT[!is.na(df$VOTOMATURITA_PCT) & df$VOTOMATURITA_PCT < VOTO_MIN_OK] <- NA
  
  # categoriali: NA = categoria esplicita
  df$REGIONE_RESIDENZA[is.na(df$REGIONE_RESIDENZA)] <- "Estero"
  df$TIPOMATURIT_COD[is.na(df$TIPOMATURIT_COD)]     <- "Mancante"
  df$BORSA_STUDIO <- ifelse(is.na(df$BORSA_STUDIO), "ND",
                            ifelse(as.logical(df$BORSA_STUDIO), "Si", "No"))
  
  # stato esami (4 gruppi) per semestre: PRIMA di riempire le medie
  df$stato_sem1 <- stato_sem(df, 1)
  df$stato_sem2 <- stato_sem(df, 2)
  
  # ISEE -> fasce ufficiali A-H; non dichiarato: H se ha esami (criterio), altrimenti ND
  ha_esami <- if (CRITERIO_H == "voto") !is.na(df$MEDIA_VOTO_DATI_SEM1) else as.logical(df$FLG_SEM1_ESAMI_SOSTENUTI)
  fascia <- as.character(cut(df$ISEE_NUM, breaks = TAGLI_ISEE, labels = FASCE, right = TRUE))
  nd <- is.na(df$ISEE_NUM)
  fascia[nd & ha_esami]  <- "H"
  fascia[nd & !ha_esami] <- "ND"
  df$ISEE_FASCIA <- factor(fascia, levels = c(FASCE, "ND"))
  
  # media voti: chi non ha media (nessun esame, o nessuno passato, o passati senza voto) = 0.
  # Le differenze tra questi gruppi le porta stato_sem1/2 (tenerle SEMPRE nel logit).
  df$MEDIA_VOTO_DATI_SEM1[is.na(df$MEDIA_VOTO_DATI_SEM1)] <- 0
  df$MEDIA_VOTO_DATI_SEM2[is.na(df$MEDIA_VOTO_DATI_SEM2)] <- 0
  
  # flag semestre e ISEE grezzo non servono piu' (sono dentro stato_sem e ISEE_FASCIA)
  df$FLG_SEM1_ESAMI_SOSTENUTI <- NULL
  df$FLG_SEM2_ESAMI_SOSTENUTI <- NULL
  df
}

train_r <- regole_riga(train_raw)
valid_r <- regole_riga(valid_raw)
test_r  <- regole_riga(test_raw)

cat("\n== ISEE: fasce (train) e dropout % ==\n")
print(rbind(n = table(train_r$ISEE_FASCIA),
            dropout = round(tapply(train_r[[TARGET]], train_r$ISEE_FASCIA, mean) * 100, 1)))

# ---------------------------------------------------------------------
# 2.3 PARAMETRI (solo train) + APPLICAZIONE
# ---------------------------------------------------------------------
per_frequenza <- function(x) names(sort(table(as.character(x)), decreasing = TRUE))

impara <- function(train) {
  st <- list()
  st$med_voto  <- median(train$VOTOMATURITA_PCT, na.rm = TRUE)
  st$med_refcar <- median(train$ref_car_min, na.rm = TRUE)
  st$med_anni  <- median(train$ANNI_DA_MATURITA, na.rm = TRUE)
  st$med_ispe_log <- median(log1p(pmax(train$ISPE_NUM, 0)), na.rm = TRUE)
  st$media_tolc_pct <- mean(train$TOLC_PCT, na.rm = TRUE)
  st$media_tolc_ing <- mean(train$TOLC_INGLESE, na.rm = TRUE)
  tm <- table(train$TIPOMATURIT_COD[train$TIPOMATURIT_COD != "Mancante"])
  st$liv_tipo <- c(names(sort(tm[tm >= SOGLIA_RARI], decreasing = TRUE)), "Altro", "Mancante")
  st$liv_reg  <- c(per_frequenza(train$REGIONE_RESIDENZA[train$REGIONE_RESIDENZA != "Estero"]), "Estero")
  for (v in c("CORSODISTUDIO", "DIPARTIMENTO", "GENERE", "OFA_STATO", "BORSA_STUDIO"))
    st[[paste0("liv_", v)]] <- per_frequenza(train[[v]])
  st$liv_stato1 <- per_frequenza(train$stato_sem1)
  st$liv_stato2 <- per_frequenza(train$stato_sem2)
  st
}

fattore <- function(x, livelli, fallback = NULL) {
  x <- as.character(x)
  if (!is.null(fallback)) x[!(x %in% livelli)] <- fallback
  nuovi <- setdiff(unique(x), livelli)
  if (length(nuovi) > 0) stop("Livelli non visti nel train: ", paste(nuovi, collapse = ", "))
  factor(x, levels = livelli)
}

applica <- function(df, st, tolc = c("senza", "con_nan", "media_train", "bagimpute")) {
  tolc <- match.arg(tolc)
  df$VOTOMATURITA_PCT[is.na(df$VOTOMATURITA_PCT)] <- st$med_voto
  df$ref_car_min[is.na(df$ref_car_min)]           <- st$med_refcar
  df$ANNI_DA_MATURITA[is.na(df$ANNI_DA_MATURITA)] <- st$med_anni
  
  # ISPE: log1p, NA = mediana del train (l'informazione "non dichiarato" e' gia' in ISEE_FASCIA)
  df$ISPE_LOG1P <- ifelse(is.na(df$ISPE_NUM), st$med_ispe_log, log1p(pmax(df$ISPE_NUM, 0)))
  df$ISEE_NUM <- NULL; df$ISPE_NUM <- NULL
  
  df$TIPOMATURIT_COD   <- fattore(df$TIPOMATURIT_COD, st$liv_tipo, fallback = "Altro")
  df$REGIONE_RESIDENZA <- fattore(df$REGIONE_RESIDENZA, st$liv_reg, fallback = "Estero")
  for (v in c("CORSODISTUDIO", "DIPARTIMENTO", "GENERE", "OFA_STATO", "BORSA_STUDIO"))
    df[[v]] <- fattore(df[[v]], st[[paste0("liv_", v)]])
  df$stato_sem1 <- fattore(df$stato_sem1, st$liv_stato1)
  df$stato_sem2 <- fattore(df$stato_sem2, st$liv_stato2)
  
  if (tolc == "senza") {
    df$TOLC_PCT <- NULL; df$TOLC_INGLESE <- NULL
  } else if (tolc == "media_train") {
    df$HA_TOLC     <- as.integer(!is.na(df$TOLC_PCT))
    df$HA_TOLC_ING <- as.integer(!is.na(df$TOLC_INGLESE))
    df$TOLC_PCT[is.na(df$TOLC_PCT)]         <- st$media_tolc_pct
    df$TOLC_INGLESE[is.na(df$TOLC_INGLESE)] <- st$media_tolc_ing
  } else if (tolc == "bagimpute") {
    # solo gli indicatori: i NA li riempie bagImpute subito dopo (vedi 2.4)
    df$HA_TOLC     <- as.integer(!is.na(df$TOLC_PCT))
    df$HA_TOLC_ING <- as.integer(!is.na(df$TOLC_INGLESE))
  }   # "con_nan": le colonne TOLC restano come sono, con i NA
  
  df$COORTE <- NULL    # serve solo per filtrare, mai come predittore
  df
}

# ---------------------------------------------------------------------
# 2.4 I QUATTRO DATASET
# ---------------------------------------------------------------------
DATASET <- list(
  A_senza_tolc         = list(dal2020 = FALSE, tolc = "senza"),
  B_tolc_con_nan       = list(dal2020 = FALSE, tolc = "con_nan"),
  C_tolc_dal2020       = list(dal2020 = TRUE,  tolc = "bagimpute"),
  D_senza_tolc_dal2020 = list(dal2020 = TRUE,  tolc = "senza")
)

riepilogo <- list()
for (nome in names(DATASET)) {
  cfg <- DATASET[[nome]]
  tr <- train_r; va <- valid_r; te <- test_r
  if (cfg$dal2020) {                       # prima filtro, poi imparo sul train filtrato
    tr <- tr[tr$COORTE >= COORTE_TOLC, ]
    va <- va[va$COORTE >= COORTE_TOLC, ]
    te <- te[te$COORTE >= COORTE_TOLC, ]
  }
  st <- impara(tr)
  tr <- applica(tr, st, cfg$tolc); va <- applica(va, st, cfg$tolc); te <- applica(te, st, cfg$tolc)
  
  # imputation dei TOLC con bagImpute: si impara sul train e si applica uguale a valid e test
  if (cfg$tolc == "bagimpute") {
    set.seed(SEED)
    pp_tolc <- caret::preProcess(tr[, VARS_IMP_TOLC], method = "bagImpute")
    tr[, VARS_IMP_TOLC] <- predict(pp_tolc, tr[, VARS_IMP_TOLC])
    va[, VARS_IMP_TOLC] <- predict(pp_tolc, va[, VARS_IMP_TOLC])
    te[, VARS_IMP_TOLC] <- predict(pp_tolc, te[, VARS_IMP_TOLC])
    st$pp_tolc <- pp_tolc
  }
  
  # controlli: stesse colonne e stessi livelli, nessun NA (tranne TOLC in B)
  stopifnot(identical(names(tr), names(va)), identical(names(tr), names(te)))
  for (v in names(tr)[sapply(tr, is.factor)])
    stopifnot(identical(levels(tr[[v]]), levels(va[[v]])), identical(levels(tr[[v]]), levels(te[[v]])))
  ammessi <- if (cfg$tolc == "con_nan") c("TOLC_PCT", "TOLC_INGLESE") else character(0)
  for (d in list(tr, va, te)) {
    n_na <- colSums(is.na(d)); n_na <- n_na[setdiff(names(n_na), ammessi)]
    if (any(n_na > 0)) stop(nome, ": NA residui in ", paste(names(n_na)[n_na > 0], collapse = ", "))
  }
  
  write_parquet(tr, file.path(OUT_DIR, paste0(nome, "_train.parquet")))
  write_parquet(va, file.path(OUT_DIR, paste0(nome, "_valid.parquet")))
  write_parquet(te, file.path(OUT_DIR, paste0(nome, "_test.parquet")))
  saveRDS(st, file.path(OUT_DIR, paste0(nome, "_parametri.rds")))
  
  riepilogo[[nome]] <- data.frame(
    dataset = nome, n_train = nrow(tr), n_valid = nrow(va), n_test = nrow(te), n_col = ncol(tr),
    dropout_train = round(mean(tr[[TARGET]]) * 100, 1),
    col_con_NA = paste(names(which(colSums(is.na(tr)) > 0)), collapse = ","),
    stringsAsFactors = FALSE)
}

cat("\n== DATASET CREATI in", OUT_DIR, "==\n")
print(do.call(rbind, riepilogo), row.names = FALSE)
cat("\nTOLC del dataset C imputati con bagImpute (train, valid e test). Controllo sul train:\n")
print(summary(read_parquet(file.path(OUT_DIR, "C_tolc_dal2020_train.parquet"))[, c("TOLC_PCT", "TOLC_INGLESE")]))