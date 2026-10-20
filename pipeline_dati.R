# ================================================================
# PIPELINE DATI: dai file _v2 al file finale per i modelli
# ================================================================
# Costruisce, in un solo script, il dataset degli studenti al primo anno
# (una riga per persona) con il target DROPOUT e le variabili per i modelli.
# Popolazione: primo anno dei corsi L1 e L2, coorti fino al 2024, escludendo
# ripetenti e chi arriva da un cambio corso.
#
# Cosa fa, in ordine:
#   0.     carica carriere, ad superate, iscritti e TOLC; pulizia di base dei _v2
#   1-2.   isola le vere matricole di primo anno e le loro ad superate
#   3-4.   stima esami/CFU "previsti" e calcola le variabili per semestre
#   5-6.   aggancia TOLC e borsa di studio
#   7.     costruisce il target DROPOUT
#   8-11.  normalizzazioni, un record per persona, ISEE/ISPE, sentinelle
#   12.    CFU, eta' e tempi di viaggio verso Bergamo (fuori sede)
#   13.    OFA, date ISO, file di audit
#   14.    file finale: feature engineering e scelta delle colonne
#
# Input:  carriere_v2, ad_superate_v2, iscritti_v2 (la pulizia di base e' fatta
#         dentro lo script, sezione 0) e tolc_completo.parquet.
# Output: carriere_primo_anno_v11.parquet (tutte le variabili), il suo file _audit
#         (colonne di esito, tenute fuori dal file per i modelli), la cache dei
#         tempi travel_times_comuni.parquet e "File Finale Dropout.parquet"
#         (il file per i modelli).
#
# Scelte principali (la numerazione e' quella usata nei commenti piu' sotto):
#
#  1. TARGET: DROPOUT = la PERSONA non risulta iscritta a Bergamo (in nessun
#     corso) nell'a.a. successivo, cercata in `iscritti` per PERS_ID. Chi cambia
#     corso o si reimmatricola (matricola nuova) NON e' dropout.
#  2. TOLC: si agganciano solo i TOLC dell'a.a. di coorte o dei 2 precedenti.
#     TOLC_PCT e' calcolato per AREA (Economia/Ingegneria/Psicologia/Umanistica)
#     con il massimo teorico della sezione, non con un massimo unico su test
#     diversi. Si gestiscono anche i test con la sola SOMMA (fonte 2024_2025).
#  3. TEMPI DI VIAGGIO: calcolati per COMUNE (residenza e domicilio) verso UN
#     punto fisso (Bergamo), non verso la sede esatta del corso. Fuori sede =
#     residenza estera, oltre FUORISEDE_MIN minuti in auto, o comune senza tempo
#     calcolabile (isole). La tabella comune -> minuti/km sta in una cache
#     (CACHE_TEMPI): evita le chiamate a internet e tiene i numeri fermi tra un
#     rilancio e l'altro. FUORI_SEDE_FLG, res_car_missing e ref_car_min vengono
#     tutti dagli stessi tempi.
#  4. AUDIT: le colonne di esito (chiusura carriera, stato, motivo) stanno in un
#     file separato e non finiscono nel file per il modello.
#  5. OFA_STATO: ESENTATO (OFA riconosciuti/convalidati) e' distinto da
#     NON_ASSEGNATO; NESSUNA_ATTIVITA = nessun esame registrato; un OFA superato
#     dopo il 30/09 di fine primo anno conta come non assolto.
#  6. Controlli "atteso": emettono un warning se i conteggi non tornano.
#  7. PREVISTI: esami e CFU "previsti" per CdS x coorte (se la cella ha meno di
#     MIN_TOP_CELLA top performer si usa il CdS); PCT_N_ESAMI e PCT_CFU sono
#     clippati a 1.
#  8. ESAMI/CFU: un esame superato piu' volte conta una sola volta (si tiene la
#     registrazione piu' recente). ULTIMO_MESE_SEM2 permette di tagliare la
#     sessione di settembre.
#  9. ISEE_NUM / ISPE_NUM: copie numeriche con NA dove non dichiarato (le colonne
#     originali con il segnaposto restano com'erano).
# ================================================================

library(dplyr)
library(arrow)
library(stringr)
library(tidyr)

cartella <- Sys.getenv(
  "DROPOUT_DIR",
  unset = "C:/Users/claud/OneDrive/Desktop/Master/Statistical Modelling and Machine Learning/Dropout"
)

# --- Parametri -------------------------------------------------------
ULTIMO_MESE_SEM2 <- 9   # 9 = include la sessione di settembre; 8 = taglia al 31/08
MIN_TOP_CELLA    <- 15  # min. top performer per usare il "previsto" di CdS x coorte
TOLC_ANNI_PRIMA  <- 2   # TOLC valido se fatto fino a 2 anni prima della coorte
OUT_NAME         <- "carriere_primo_anno_v11"
OUT_FINALE       <- "File Finale Dropout.parquet"   # file per i modelli (come prima)

# --- Tempi di viaggio verso Bergamo ---
BERGAMO_LAT        <- 45.6983
BERGAMO_LON        <- 9.6707
FUORISEDE_MIN      <- 90     # fuori sede se a piu' di 90 min in auto (o estero, o senza tempo)
BATCH              <- 99     # il server pubblico OSRM accetta max ~100 punti (99 origini + Bergamo)
CACHE_TEMPI        <- "travel_times_comuni.parquet"            # tabella comune -> minuti/km
FILE_TEMPI_VECCHIO <- "nuovo_carriere_con_travel_times.parquet"  # se manca la cache parto dai tempi gia' calcolati
RICALCOLA_TEMPI    <- FALSE  # TRUE = ricalcola tutto da OSRM (serve internet) e riscrive la cache

check_atteso <- function(val, atteso, label) {
  cat(sprintf("%s: %s (atteso: %s)\n", label,
              format(val, big.mark = ".", decimal.mark = ","),
              format(atteso, big.mark = ".", decimal.mark = ",")))
  if (!isTRUE(val == atteso)) warning(sprintf("%s: %s invece di %s", label, val, atteso), call. = FALSE)
}

# ================================================================
# 0. CARICAMENTO DEI FILE DI PARTENZA
# ================================================================

# --- file di partenza: i parquet _v2 (conversione corretta dei txt) ---
carr <- read_parquet(file.path(cartella, "carriere_v2.parquet"))
ad   <- read_parquet(file.path(cartella, "ad_superate_v2.parquet"))
iscr <- read_parquet(file.path(cartella, "iscritti_v2.parquet"))
tolc_completo <- read_parquet(file.path(cartella, "tolc_completo.parquet"))

# --- pulizia di base: sono le sole trasformazioni di "R Pulizia File.R" che servono
# a questa pipeline (i controlli e gli altri file restano nello script di pulizia).
# Partendo dai _v2 il risultato e' identico ai vecchi _pulito.
segnaposto <- c("", "null", "NULL", "n/d", "N/D", "N/A", "n/a", "NaN", "None", "-", "?")

# CARRIERE: segnaposto testuali -> NA veri; ISEE/ISPE mancanti -> 999999999
carr <- carr %>%
  mutate(across(everything(), ~ na_if(trimws(.x), ""))) %>%
  mutate(across(everything(), ~ if_else(.x %in% segnaposto, NA_character_, .x))) %>%
  mutate(across(c(ISEE, ISPE), ~ if_else(is.na(.x), "999999999", .x)))

# AD SUPERATE: crediti e voto in numeri (virgola decimale)
ad <- ad %>%
  mutate(across(c(AD_CFU, ESA_VOTO), ~ as.numeric(sub(",", ".", .x, fixed = TRUE))))

# ISCRITTI: nessuna trasformazione

cat("File caricati -> carr:", nrow(carr), "righe | ad:", nrow(ad),
    "righe | iscr:", nrow(iscr), "righe | tolc_completo:", nrow(tolc_completo), "righe\n")

carr_num <- carr %>% mutate(AA_COORTE_ID = as.numeric(AA_COORTE_ID), AA_ISCR_ID = as.numeric(AA_ISCR_ID))


# ================================================================
# 1. POPOLAZIONE "VERE MATRICOLE" DI PRIMO ANNO (con recupero fantasma)
# ================================================================
# Nota: i "fantasma" recuperati qui hanno AA_ISCR_ID != AA_COORTE_ID e
# vengono poi rimossi dal filtro mismatch della sezione 10 (restano ~69
# casi in tutto). Il recupero e' quindi di fatto annullato: tenuto per
# compatibilita' con i conteggi storici.

pop_primo_anno <- carr_num %>%
  filter(TIPO_CORSO_COD %in% c("L1", "L2"), ANNO_ISCR == "1", AA_COORTE_ID <= 2024)

cat("Popolazione primo anno (allargata, prima della correzione fantasma):",
    n_distinct(pop_primo_anno$STU_ID), "studenti\n")

vera_ids <- pop_primo_anno %>% filter(AA_ISCR_ID == AA_COORTE_ID) %>% pull(STU_ID) %>% unique()

has_any_at_cohort <- pop_primo_anno %>%
  filter(!STU_ID %in% vera_ids) %>%
  distinct(STU_ID) %>%
  inner_join(carr_num, by = "STU_ID") %>%
  filter(AA_ISCR_ID == AA_COORTE_ID) %>%
  pull(STU_ID) %>% unique()

fantasma_ids <- pop_primo_anno %>%
  filter(!STU_ID %in% vera_ids, !STU_ID %in% has_any_at_cohort) %>%
  distinct(STU_ID) %>% pull(STU_ID)

fantasma_prima_riga <- carr_num %>%
  filter(STU_ID %in% fantasma_ids) %>%
  arrange(STU_ID, AA_ISCR_ID) %>%
  group_by(STU_ID) %>%
  slice(1) %>%
  ungroup()

recuperabili_ids <- fantasma_prima_riga %>% filter(ANNO_ISCR == "1") %>% pull(STU_ID)
cat("Studenti 'fantasma' recuperati come vere matricole:", length(recuperabili_ids),
    "- esclusi (cambio corso con riconoscimento crediti):",
    length(fantasma_ids) - length(recuperabili_ids), "\n")

riga_recupero <- pop_primo_anno %>%
  filter(STU_ID %in% recuperabili_ids) %>%
  arrange(STU_ID, AA_ISCR_ID) %>%
  group_by(STU_ID) %>%
  slice(1) %>%
  ungroup()

carriere_primo_anno_vero <- pop_primo_anno %>%
  filter(AA_ISCR_ID == AA_COORTE_ID) %>%
  bind_rows(riga_recupero) %>%
  distinct(STU_ID, .keep_all = TRUE)

check_atteso(nrow(carriere_primo_anno_vero), 59032, "Popolazione primo anno finale")


# ================================================================
# 2. AD SUPERATE SULLA POPOLAZIONE CORRETTA + DEDUPLICA VOTO/ESITO
# ================================================================

ad_superate_primo_anno_vero <- ad %>%
  semi_join(carriere_primo_anno_vero, by = "STU_ID") %>%
  group_by(STU_ID, AD_COD, ESA_DATA) %>%
  mutate(gruppo_valido = !is.na(ESA_DATA)) %>%
  group_by(STU_ID, AD_COD, ESA_DATA, gruppo_valido) %>%
  slice_max(order_by = ESA_VOTO, n = 1, with_ties = FALSE, na_rm = FALSE) %>%
  ungroup() %>%
  select(-gruppo_valido)

stopifnot(
  ad_superate_primo_anno_vero %>%
    filter(!is.na(ESA_DATA)) %>%
    count(STU_ID, AD_COD, ESA_DATA) %>%
    filter(n > 1) %>%
    nrow() == 0
)
cat("AD SUPERATE primo anno, dopo deduplica voto piu' alto:",
    nrow(ad_superate_primo_anno_vero), "righe\n")


# ================================================================
# 3. STIMA ESAMI/CFU "PREVISTI" PER CDS (x COORTE) E SEMESTRE
# ================================================================
# (8) superato_valido: un esame (STU_ID, AD_COD) superato piu' volte nella
# finestra, in date DIVERSE, conta una volta sola e si tiene quello con
# l'ULTIMA data (con rifiuto del voto vale l'ultima registrazione). Lo stesso
# giorno vale il voto piu' alto: lo garantisce la deduplica di sezione 2.
# I CFU non si contano 2 volte.

ad_primo <- ad_superate_primo_anno_vero %>%
  inner_join(
    carriere_primo_anno_vero %>% select(STU_ID, IDCORSODISTUDIO, AA_ISCR_ID_primo = AA_ISCR_ID),
    by = "STU_ID"
  ) %>%
  mutate(
    ESA_DATA_dt = as.Date(ESA_DATA, format = "%d/%m/%Y"),
    mese = as.integer(format(ESA_DATA_dt, "%m")),
    anno_esame = as.integer(format(ESA_DATA_dt, "%Y")),
    semestre = case_when(
      mese %in% c(10, 11, 12) & anno_esame == AA_ISCR_ID_primo ~ 1,
      mese %in% c(1, 2) & anno_esame == AA_ISCR_ID_primo + 1 ~ 1,
      mese %in% 3:ULTIMO_MESE_SEM2 & anno_esame == AA_ISCR_ID_primo + 1 ~ 2,
      TRUE ~ NA_real_
    )
  ) %>%
  filter(!is.na(semestre)) %>%
  filter(!startsWith(AD_DES, "OFA")) %>%
  group_by(STU_ID, AD_COD) %>%
  arrange(desc(ESA_DATA_dt), desc(ESA_VOTO), .by_group = TRUE) %>%   # ultima data per prima
  mutate(superato_valido = SUP_FLG == "1" & cumsum(SUP_FLG == "1") == 1) %>%
  ungroup()

cfu_per_stu <- ad_primo %>%
  filter(superato_valido) %>%
  group_by(STU_ID, IDCORSODISTUDIO, AA_ISCR_ID_primo) %>%
  summarise(cfu_tot = sum(AD_CFU, na.rm = TRUE), .groups = "drop")

# top performer: 25% migliore per CdS (fallback) e per CdS x coorte
top_cds <- cfu_per_stu %>%
  group_by(IDCORSODISTUDIO) %>%
  filter(cfu_tot >= quantile(cfu_tot, 0.75, na.rm = TRUE)) %>%
  ungroup()

top_coorte <- cfu_per_stu %>%
  group_by(IDCORSODISTUDIO, AA_ISCR_ID_primo) %>%
  filter(cfu_tot >= quantile(cfu_tot, 0.75, na.rm = TRUE)) %>%
  ungroup()

stats_per_top <- function(top) {
  ad_primo %>%
    filter(superato_valido) %>%
    inner_join(top %>% select(STU_ID, IDCORSODISTUDIO, AA_ISCR_ID_primo),
               by = c("STU_ID", "IDCORSODISTUDIO", "AA_ISCR_ID_primo")) %>%
    group_by(STU_ID, IDCORSODISTUDIO, AA_ISCR_ID_primo, semestre) %>%
    summarise(n_esami = n(), cfu = sum(AD_CFU, na.rm = TRUE), .groups = "drop")
}

piano_cds <- stats_per_top(top_cds) %>%
  group_by(IDCORSODISTUDIO, semestre) %>%
  summarise(n_esami_previsti_cds = round(median(n_esami)),
            cfu_previsti_cds = round(median(cfu)), .groups = "drop")

piano_coorte <- stats_per_top(top_coorte) %>%
  group_by(IDCORSODISTUDIO, AA_ISCR_ID_primo, semestre) %>%
  summarise(n_esami_previsti_coorte = round(median(n_esami)),
            cfu_previsti_coorte = round(median(cfu)),
            n_top_performer = n(), .groups = "drop") %>%
  filter(n_top_performer >= MIN_TOP_CELLA) %>%
  select(-n_top_performer)

cat("CdS coperti dalla stima del 'previsto':", n_distinct(piano_cds$IDCORSODISTUDIO),
    "- celle CdS x coorte x semestre con stima propria:", nrow(piano_coorte), "\n")


# ================================================================
# 4. VARIABILI PER STUDENTE, PER SEMESTRE
# ================================================================
# media_voto_dati: solo esami superati (prima volta) con voto 18-30; NA se
# nessuno (non 0). Vedi HA_VOTO_SEM{n}_FLG.

stats_dati <- ad_primo %>%
  group_by(STU_ID, semestre) %>%
  summarise(
    n_esami_dati = n(),
    n_esami_passati = sum(superato_valido),
    cfu_passati = sum(AD_CFU[superato_valido], na.rm = TRUE),
    media_voto_dati = mean(
      ESA_VOTO[superato_valido & !is.na(ESA_VOTO) & ESA_VOTO >= 18 & ESA_VOTO <= 30],
      na.rm = TRUE
    ),
    .groups = "drop"
  ) %>%
  mutate(media_voto_dati = ifelse(is.nan(media_voto_dati), NA_real_, media_voto_dati))

griglia <- carriere_primo_anno_vero %>%
  select(STU_ID, IDCORSODISTUDIO, AA_ISCR_ID_primo = AA_ISCR_ID) %>%
  crossing(semestre = c(1, 2)) %>%
  left_join(piano_cds, by = c("IDCORSODISTUDIO", "semestre")) %>%
  left_join(piano_coorte, by = c("IDCORSODISTUDIO", "AA_ISCR_ID_primo", "semestre")) %>%
  mutate(
    n_esami_previsti = coalesce(n_esami_previsti_coorte, n_esami_previsti_cds),
    cfu_previsti = coalesce(cfu_previsti_coorte, cfu_previsti_cds)
  ) %>%
  left_join(stats_dati, by = c("STU_ID", "semestre")) %>%
  mutate(
    n_esami_dati = replace_na(n_esami_dati, 0),
    n_esami_passati = replace_na(n_esami_passati, 0),
    cfu_passati = replace_na(cfu_passati, 0)
  )

variabili_semestre <- griglia %>%
  pivot_wider(
    id_cols = c(STU_ID, IDCORSODISTUDIO),
    names_from = semestre,
    values_from = c(n_esami_dati, n_esami_passati, cfu_passati, media_voto_dati, n_esami_previsti, cfu_previsti),
    names_glue = "{.value}_sem{semestre}"
  )

carriere_grezzo <- carriere_primo_anno_vero %>%
  left_join(variabili_semestre %>% select(-IDCORSODISTUDIO), by = "STU_ID")

cat("Righe dopo aggiunta variabili per semestre:", nrow(carriere_grezzo), "\n")


# ================================================================
# 5. AGGANCIO DEL TOLC (correzione 2)
# ================================================================
# Massimo teorico della somma delle sezioni (inglese ESCLUSO, e' in TOLC_INGLESE)
TOLC_MAX_AREA <- c(ECONOMIA = 36, INGEGNERIA = 50, PSICOLOGIA = 50, UMANISTICA = 50)

tolc_eventi <- tolc_completo %>%
  mutate(
    PERS_ID = sprintf("%.0f", PERS_ID),
    STU_ID_tolc = ifelse(is.na(STU_ID), NA_character_, sprintf("%.0f", STU_ID)),
    anno = as.numeric(anno),
    data_tolc = as.Date(DATA_SVOLGIMENTO)
  ) %>%
  distinct(PERS_ID, DES_AREA, anno, data_tolc, DES_SEZIONE, .keep_all = TRUE)

stopifnot(all(unique(tolc_eventi$DES_AREA) %in% names(TOLC_MAX_AREA)))

tolc_eventi <- tolc_eventi %>%
  group_by(PERS_ID, DES_AREA, anno, data_tolc) %>%
  summarise(
    STU_ID_tolc = STU_ID_tolc[!is.na(STU_ID_tolc)][1],
    somma_sezioni = if (any(!DES_SEZIONE %in% c("SOMMA", "INGLESE")))
      sum(PUNTEGGIO_SEZIONE[!DES_SEZIONE %in% c("SOMMA", "INGLESE")], na.rm = TRUE) else NA_real_,
    somma_file = if (any(DES_SEZIONE == "SOMMA"))
      max(PUNTEGGIO_SEZIONE[DES_SEZIONE == "SOMMA"], na.rm = TRUE) else NA_real_,
    inglese = if (any(DES_SEZIONE == "INGLESE"))
      max(PUNTEGGIO_SEZIONE[DES_SEZIONE == "INGLESE"], na.rm = TRUE) else NA_real_,
    .groups = "drop"
  ) %>%
  mutate(
    punteggio = coalesce(somma_sezioni, somma_file),
    pct = pmax(punteggio / unname(TOLC_MAX_AREA[DES_AREA]), 0)
  ) %>%
  filter(!is.na(pct)) %>%
  select(PERS_ID, anno, data_tolc, STU_ID_tolc, pct, inglese)

carriere_pers <- carriere_grezzo %>%
  transmute(STU_ID, PERS_ID = as.character(PERS_ID), AA_COORTE_ID)

match_tolc <- carriere_pers %>%
  inner_join(tolc_eventi, by = "PERS_ID", relationship = "many-to-many") %>%
  filter(anno <= AA_COORTE_ID, anno >= AA_COORTE_ID - TOLC_ANNI_PRIMA,
         is.na(data_tolc) | data_tolc <= as.Date(paste0(AA_COORTE_ID + 1, "-09-30"))) %>%
  mutate(
    priorita = ifelse(!is.na(STU_ID_tolc) & STU_ID_tolc == STU_ID, 0, 1),
    quando = ifelse(is.na(data_tolc), as.numeric(as.Date(paste0(anno, "-09-01"))), as.numeric(data_tolc))
  ) %>%
  group_by(STU_ID) %>%
  arrange(priorita, desc(quando), .by_group = TRUE) %>%
  slice(1) %>%
  ungroup() %>%
  transmute(STU_ID, tolc_pct = pct, tolc_inglese = inglese)

cat("Studenti con TOLC agganciato:", nrow(match_tolc), "su", nrow(carriere_pers),
    sprintf("(%.1f%%)\n", 100 * nrow(match_tolc) / nrow(carriere_pers)))

carriere_grezzo <- carriere_grezzo %>% left_join(match_tolc, by = "STU_ID")


# ================================================================
# 6. AGGANCIO DELLA BORSA DI STUDIO (da ISCRITTI)
# ================================================================
# Chi non ha nessuna riga in `iscr` per quell'anno resta NA (non sappiamo).

iscr_borsa <- iscr %>%
  mutate(
    borsa_studio = (
      startsWith(as.character(CAU_RID_STA_MIUR_DES), "BENEFICIARI") |
        CAUSALE_ESO_COD == "BENEF-T" |
        CAU_RID_ANS_COD %in% c("B", "BT", "BY", "BJ")
    )
  ) %>%
  group_by(STU_ID, AA_ISCR_ID) %>%
  summarise(borsa_studio = any(borsa_studio, na.rm = TRUE), .groups = "drop")

carriere_grezzo <- carriere_grezzo %>%
  left_join(
    iscr_borsa %>% mutate(AA_ISCR_ID = as.numeric(AA_ISCR_ID)),
    by = c("STU_ID", "AA_ISCR_ID")
  )

cat("Studenti con borsa_studio agganciata (non NA):",
    sum(!is.na(carriere_grezzo$borsa_studio)), "su", nrow(carriere_grezzo),
    "- beneficiari:", sum(carriere_grezzo$borsa_studio, na.rm = TRUE), "\n")


# ================================================================
# 7. TARGET: DROPOUT (correzione 1)
# ================================================================
# DROPOUT = 1 se la PERSONA non e' iscritta a Bergamo (qualunque corso)
# nell'a.a. successivo, secondo iscritti_pulito. Cambio corso = NON dropout.
# DROPOUT_MATRICOLA = vecchia definizione (stessa matricola), solo per audit.

rinnovi_persona <- iscr %>%
  transmute(PERS_ID = as.character(PERS_ID),
            AA_ISCR_ID = as.numeric(AA_ISCR_ID) - 1,
            rinnovato = TRUE) %>%
  distinct()

rinnovi_matricola <- carr_num %>%
  transmute(STU_ID, AA_ISCR_ID = AA_ISCR_ID - 1, rinnovato_matr = TRUE) %>%
  distinct()

dropout <- carriere_grezzo %>%
  transmute(STU_ID, PERS_ID = as.character(PERS_ID), AA_ISCR_ID) %>%
  left_join(rinnovi_persona, by = c("PERS_ID", "AA_ISCR_ID")) %>%
  left_join(rinnovi_matricola, by = c("STU_ID", "AA_ISCR_ID")) %>%
  mutate(DROPOUT = as.integer(is.na(rinnovato)),
         DROPOUT_MATRICOLA = as.integer(is.na(rinnovato_matr))) %>%
  distinct(STU_ID, .keep_all = TRUE) %>%
  select(STU_ID, DROPOUT, DROPOUT_MATRICOLA)

carriere_grezzo <- carriere_grezzo %>%
  left_join(dropout, by = "STU_ID")

cat("Dropout (per persona):", sum(carriere_grezzo$DROPOUT),
    sprintf("(%.1f%%)", 100 * mean(carriere_grezzo$DROPOUT)),
    "| vecchia definizione (matricola):", sum(carriere_grezzo$DROPOUT_MATRICOLA),
    sprintf("(%.1f%%)\n", 100 * mean(carriere_grezzo$DROPOUT_MATRICOLA)))

cat("\n>>> carriere_grezzo completo:", nrow(carriere_grezzo), "righe,", ncol(carriere_grezzo), "colonne\n")


# ================================================================
# 8. NORMALIZZAZIONI
# ================================================================

cat("\n================ normalizzazioni ================\n")

cat("Righe prima della deduplica:", nrow(carriere_grezzo), "\n")
carriere <- distinct(carriere_grezzo)
cat("Righe dopo la deduplica:", nrow(carriere), "\n")

# TOLC_PCT e TOLC_INGLESE sono gia' calcolati nella sezione 5
cat("TOLC_PCT disponibile per:", sum(!is.na(carriere$tolc_pct)), "studenti su", nrow(carriere), "\n")

votomaturita_num <- as.numeric(gsub(",", ".", carriere$VOTOMATURITA, fixed = TRUE))
votomaturitamax_num <- as.numeric(gsub(",", ".", carriere$VOTOMATURITAMAX, fixed = TRUE))
votomaturitamax_num[votomaturitamax_num == 0] <- NA_real_
carriere$votomaturita_pct <- ifelse(
  !is.na(votomaturita_num) & !is.na(votomaturitamax_num),
  votomaturita_num / votomaturitamax_num,
  NA_real_
)
cat("VOTOMATURITA_PCT calcolabile per:", sum(!is.na(carriere$votomaturita_pct)), "studenti su", nrow(carriere), "\n")

# (7) rapporti clippati a 1: i CFU/esami passati possono superare il "previsto"
for (sem in c(1, 2)) {
  dati_col    <- paste0("n_esami_dati_sem", sem)
  passati_col <- paste0("n_esami_passati_sem", sem)
  cfu_col     <- paste0("cfu_passati_sem", sem)
  nprev_col   <- paste0("n_esami_previsti_sem", sem)
  cfuprev_col <- paste0("cfu_previsti_sem", sem)
  
  carriere[[paste0("pct_n_esami_passati_sem", sem)]] <- ifelse(
    !is.na(carriere[[nprev_col]]) & carriere[[nprev_col]] != 0,
    pmin(carriere[[passati_col]] / carriere[[nprev_col]], 1), NA_real_
  )
  carriere[[paste0("pct_cfu_passati_sem", sem)]] <- ifelse(
    !is.na(carriere[[cfuprev_col]]) & carriere[[cfuprev_col]] != 0,
    pmin(carriere[[cfu_col]] / carriere[[cfuprev_col]], 1), NA_real_
  )
  carriere[[paste0("pct_successo_sem", sem)]] <- ifelse(
    carriere[[dati_col]] > 0,
    carriere[[passati_col]] / carriere[[dati_col]], NA_real_
  )
}

for (sem in c(1, 2)) {
  dati_col    <- paste0("n_esami_dati_sem", sem)
  passati_col <- paste0("n_esami_passati_sem", sem)
  cfu_col     <- paste0("cfu_passati_sem", sem)
  nprev_col   <- paste0("n_esami_previsti_sem", sem)
  cfuprev_col <- paste0("cfu_previsti_sem", sem)
  
  v1 <- sum(carriere[[passati_col]] > carriere[[dati_col]], na.rm = TRUE)
  cat("Semestre", sem, "- passati > dati (deve essere 0):", v1, "\n")
  stopifnot(v1 == 0)
  v2 <- sum(carriere[[passati_col]] > carriere[[nprev_col]], na.rm = TRUE)
  cat("Semestre", sem, "- passati > previsti (non e' un errore):", v2, "\n")
  v3 <- sum(carriere[[cfu_col]] > carriere[[cfuprev_col]], na.rm = TRUE)
  cat("Semestre", sem, "- CFU passati > CFU previsti (non e' un errore):", v3, "\n")
}

names(carriere) <- toupper(names(carriere))
cat("\nColonne dopo la normalizzazione dei nomi:", ncol(carriere), "- Righe:", nrow(carriere), "\n")

cambio_corso <- as.numeric(carriere$AA_COORTE_ID) != as.numeric(carriere$AA_INGRESSO_ATENEO_ID)
ripetenti <- carriere$STATO_ISCRIZIONE == "Ripetente"
cat("Ripetenti:", sum(ripetenti), "- Cambio corso:", sum(cambio_corso),
    "- Overlap:", sum(ripetenti & cambio_corso), "\n")
carriere <- carriere[!(ripetenti | cambio_corso), ]
check_atteso(nrow(carriere), 58228, "Righe dopo il filtro matricole")

cols_da_togliere <- c(
  "PROVINCIA_RESIDENZA", "PROVINCIA_DOMICILIO",
  "COMUNE_NASCITA_ID", "COMUNE_RESIDENZA_ID", "COMUNE_COMICILIO_ID",
  "IDCORSODISTUDIO",
  "STATO_ISCRIZIONE", "ANNI_FUORI_CORSO", "ANNO_ISCR",
  "AA_INGRESSO_ATENEO_ID", "TIPO_CORSO", "TIPO_CORSO_COD"
)
carriere <- carriere %>% select(-all_of(cols_da_togliere))

cat("Colonne dopo la pulizia colonne:", ncol(carriere), "- Righe:", nrow(carriere), "\n")


# ================================================================
# 9. SEMPLIFICAZIONE (province -> sigle, colonne grezze fuori)
# ================================================================

cat("\n================ province / colonne grezze ================\n")

m1 <- carriere_grezzo %>% select(nome = PROVINCIA_RESIDENZA, sigla = PROVINCIA_RESIDENZA_SIGLA) %>%
  filter(!is.na(nome)) %>% distinct()
m2 <- carriere_grezzo %>% select(nome = PROVINCIA_DOMICILIO, sigla = PROVINCIA_DOMICILIO_SIGLA) %>%
  filter(!is.na(nome)) %>% distinct()
lookup_df <- bind_rows(m1, m2) %>% distinct()
lookup <- setNames(lookup_df$sigla, lookup_df$nome)
lookup["Pesaro"] <- "PU"
lookup["Forl"] <- "FC"

carriere$PROVINCIA_NASCITA_SIGLA <- unname(lookup[carriere$PROVINCIA_NASCITA])
carriere$PROVINCIASCUOLASUPERIORE_SIGLA <- unname(lookup[carriere$PROVINCIASCUOLASUPERIORE])
carriere$PROVINCIASCUOLASUPERIORE_SIGLA[carriere$PROVINCIASCUOLASUPERIORE == "Estero"] <- "ESTERO"

cat("PROVINCIA_NASCITA_SIGLA mappata:", sum(!is.na(carriere$PROVINCIA_NASCITA_SIGLA)),
    "/", sum(!is.na(carriere$PROVINCIA_NASCITA)), "\n")

carriere <- carriere %>% select(-PROVINCIA_NASCITA, -PROVINCIASCUOLASUPERIORE)

# i conteggi grezzi per semestre vengono rimessi in sezione 10/12 dove servono
carriere <- carriere %>% select(
  -VOTOMATURITA, -VOTOMATURITAMAX,
  -N_ESAMI_DATI_SEM1, -N_ESAMI_DATI_SEM2,
  -N_ESAMI_PASSATI_SEM1, -N_ESAMI_PASSATI_SEM2,
  -CFU_PASSATI_SEM1, -CFU_PASSATI_SEM2,
  -N_ESAMI_PREVISTI_SEM1, -N_ESAMI_PREVISTI_SEM2
)


# ================================================================
# 10. UN RECORD PER PERSONA + ANOMALIE RESIDUE + ISEE/ISPE
# ================================================================

cat("\n================ un record per persona ================\n")

carriere <- carriere %>%
  arrange(AA_COORTE_ID) %>%
  distinct(PERS_ID, .keep_all = TRUE) %>%
  arrange(STU_ID)
check_atteso(nrow(carriere), 56302, "Dopo deduplica PERS_ID")

mismatch <- as.numeric(carriere$AA_ISCR_ID) != as.numeric(carriere$AA_COORTE_ID)
carriere <- carriere[!mismatch, ]
check_atteso(nrow(carriere), 56233, "Dopo rimozione mismatch AA_ISCR_ID/AA_COORTE_ID")

for (col in c("ISEE", "ISPE")) {
  num <- as.numeric(gsub(",", ".", carriere[[col]], fixed = TRUE))
  carriere[[paste0(col, "_DICHIARATO_FLG")]] <- num != 999999999
  carriere[[col]] <- num   # segnaposto 999999999 lasciato com'e' (richiesto)
}
cat("ISEE dichiarato:", sum(carriere$ISEE_DICHIARATO_FLG),
    "- ISPE dichiarato:", sum(carriere$ISPE_DICHIARATO_FLG), "\n")

# HA_VOTO_SEM{n}_FLG: almeno un voto 18-30 superato nel semestre
for (sem in c(1, 2)) {
  mv_col <- paste0("MEDIA_VOTO_DATI_SEM", sem)
  carriere[[paste0("HA_VOTO_SEM", sem, "_FLG")]] <- !is.na(carriere[[mv_col]])
}
cat("HA_VOTO_SEM1_FLG:\n"); print(table(carriere$HA_VOTO_SEM1_FLG, useNA = "always"))
cat("HA_VOTO_SEM2_FLG:\n"); print(table(carriere$HA_VOTO_SEM2_FLG, useNA = "always"))

# flag "esami sostenuti"; PCT_SUCCESSO NA -> 0 (0 esami sostenuti), MEDIA_VOTO resta NA
v3_dati <- carriere_grezzo %>%
  distinct(STU_ID, .keep_all = TRUE) %>%
  select(STU_ID, N_ESAMI_DATI_SEM1 = n_esami_dati_sem1, N_ESAMI_DATI_SEM2 = n_esami_dati_sem2)
carriere <- carriere %>% left_join(v3_dati, by = "STU_ID")

for (sem in c(1, 2)) {
  dati_col <- paste0("N_ESAMI_DATI_SEM", sem)
  flg_col  <- paste0("FLG_SEM", sem, "_ESAMI_SOSTENUTI")
  pct_col  <- paste0("PCT_SUCCESSO_SEM", sem)
  
  carriere[[flg_col]] <- carriere[[dati_col]] > 0
  carriere[[pct_col]][is.na(carriere[[pct_col]])] <- 0
  carriere[[dati_col]] <- NULL
}
cat("NA residui PCT_SUCCESSO_SEM1/2:", sum(is.na(carriere$PCT_SUCCESSO_SEM1)), sum(is.na(carriere$PCT_SUCCESSO_SEM2)), "\n")
cat("NA (= nessun voto 18-30) in MEDIA_VOTO_DATI_SEM1/2:",
    sum(is.na(carriere$MEDIA_VOTO_DATI_SEM1)), sum(is.na(carriere$MEDIA_VOTO_DATI_SEM2)), "\n")


# ================================================================
# 11. SENTINELLE RESIDUE E CORREZIONI MINORI
# ================================================================

cat("\n================ sentinelle ================\n")

is_all_same_digit <- function(x) {
  s <- as.character(as.integer(x))
  !is.na(x) & nchar(s) >= 5 & sapply(strsplit(s, ""), function(d) length(unique(d)) == 1)
}
for (col in c("ISEE", "ISPE")) {
  flg_col <- paste0(col, "_DICHIARATO_FLG")
  carriere[[flg_col]][is_all_same_digit(carriere[[col]])] <- FALSE
  # (9) copia numerica pulita, da usare nei modelli
  carriere[[paste0(col, "_NUM")]] <- ifelse(carriere[[flg_col]], carriere[[col]], NA_real_)
}
cat("ISEE_NUM non NA:", sum(!is.na(carriere$ISEE_NUM)), "- ISPE_NUM non NA:", sum(!is.na(carriere$ISPE_NUM)), "\n")

carriere$BORSA_STUDIO <- as.logical(carriere$BORSA_STUDIO)

eta_diploma <- as.numeric(carriere$ANNO_MATURITA) - as.numeric(carriere$ANNO_NASCITA)
mask_bad <- !is.na(eta_diploma) & eta_diploma < 15
carriere$ANNO_MATURITA[mask_bad] <- NA
cat("ANNO_MATURITA corretti per eta' implausibile:", sum(mask_bad), "\n")


# ================================================================
# 12. CFU, ETA' E TEMPI DI VIAGGIO / FUORI SEDE
# ================================================================
# (v11) I tempi si calcolano per COMUNE e si agganciano con nome comune + sigla
# provincia, sia per la residenza sia per il domicilio. La tabella comune ->
# minuti/km sta in CACHE_TEMPI: i numeri non cambiano a ogni rilancio e non serve
# internet. Il tempo e' verso UN punto fisso (Bergamo), non verso la sede del corso.

cat("\n================ CFU, eta', tempi di viaggio, fuori sede ================\n")

# --- 12a. funzioni ----------------------------------------------------
# Calcolo da OSRM: usato SOLO se non c'e' la cache e nemmeno il file dei tempi vecchio.
calcola_tempi_osrm <- function(comuni) {
  for (p in c("jsonlite", "httr2", "stringdist", "tibble"))
    if (!requireNamespace(p, quietly = TRUE))
      stop("Serve il pacchetto '", p, "': install.packages(\"", p, "\")")
  suppressPackageStartupMessages({
    library(tibble); library(jsonlite); library(httr2); library(stringdist)
  })

  # ---- Coordinate ISTAT --------------------------------------------------
  istat_raw <- fromJSON(
    "https://raw.githubusercontent.com/MatteoHenryChinaski/Comuni-Italiani-2018-Sql-Json-excel/master/italy_geo.json"
  )

  istat <- as_tibble(istat_raw) |>
    transmute(comune_istat = comune,
              lat = as.numeric(lat),
              lon = as.numeric(lng)) |>
    distinct(comune_istat, .keep_all = TRUE) |>      # omonimi: teniamo il primo
    mutate(key = tolower(trimws(comune_istat)))

  comuni_coords <- comuni |>
    mutate(key = tolower(trimws(comune))) |>
    left_join(select(istat, key, lat, lon), by = "key")

  # match approssimato per quelli non trovati (Jaro-Winkler)
  mancanti <- which(is.na(comuni_coords$lat))
  idx <- amatch(comuni_coords$key[mancanti], istat$key, method = "jw", maxDist = 0.1)
  ok <- !is.na(idx)

  comuni_coords$lat[mancanti[ok]] <- istat$lat[idx[ok]]
  comuni_coords$lon[mancanti[ok]] <- istat$lon[idx[ok]]

  # da controllare a occhio: nome nel dataset -> nome ISTAT agganciato
  fuzzy_log <- tibble(comune_dataset = comuni_coords$comune[mancanti[ok]],
                      provincia      = comuni_coords$provincia[mancanti[ok]],
                      match_istat    = istat$comune_istat[idx[ok]])
  print(fuzzy_log, n = Inf)

  # ---- Comuni ancora senza coordinate ------------------------------------
  print(filter(comuni_coords, is.na(lat)), n = Inf)

  # ---- Correzioni manuali ------------------------------------------------
  # Nome ESATTO come nel dataset: si correggono solo lat/lon, il nome non cambia.
  correzioni <- tribble(
    ~comune,                 ~lat,    ~lon,
    "Pellio Intelvi",        45.9791,  9.0458,
    "Virgilio",              45.0881, 10.6926,   # oggi Borgo Virgilio
    "Bezzecca",              45.8959, 10.7106,
    "Puos d'Alpago",         46.1378, 12.3222,
    "Rovagnate",             45.7288,  9.3213,   # oggi La Valletta Brianza
    "Faver",                 46.1824, 11.2321,
    "Dimaro",                46.3257, 10.8632,   # oggi Dimaro Folgarida
    "CORIGLIANO-ROSSANO",    39.6044, 16.3867,
    "Tremosine",             45.7918, 10.6424,   # oggi Tremosine sul Garda
    "PIADENA DRIZZONA",      45.1347, 10.2871,
    "TERRE D'ADIGE",         46.1576, 11.0254,
    "Ollastra Simaxis",      39.9625,  8.7139,
    "PRESICCE-ACQUARICA",    39.9064, 18.2322,
    "CADREZZATE CON OSMATE", 45.7947,  8.6294,
    # --- match approssimati sbagliati (nomi senza accenti nel dataset) ---
    # coordinate approssimative (basta qualche km di precisione per i tempi in auto)
    "Costermano",            45.5864, 10.7219,   # Costermano sul Garda (VR)
    "Perego",                45.7140,  9.3470,   # LC
    "Barzan",                45.7186,  9.3267,   # Barzanò (LC)
    "Lenno",                 45.9667,  9.1856,   # CO
    "Tres",                  46.3500, 11.1000,   # TN, oggi in Predaia
    "Sal",                   45.6075, 10.5189,   # Salò (BS)
    "NOVELLA",               46.4300, 11.0000,   # TN
    # --- comuni senza coordinate (fusioni / nomi cambiati) ---
    "Brembilla",             45.7967,  9.6167,   # oggi Val Brembilla (BG)
    "Gerosa",                45.8000,  9.6300,   # oggi Val Brembilla (BG)
    "Lonato",                45.4600, 10.4833,   # Lonato del Garda (BS)
    "Sant'Omobono Imagna",   45.7967,  9.5433,   # oggi Sant'Omobono Terme (BG)
    "Verderio Superiore",    45.6908,  9.4372,   # oggi Verderio (LC)
    "Verderio Inferiore",    45.6908,  9.4372,   # oggi Verderio (LC)
    "MUGGIO'",               45.5917,  9.2333,   # Muggiò (MB)
    "Costa di Serina",       45.8000,  9.7000,   # BG
    "Rivanazzano",           44.9667,  9.0000,   # Rivanazzano Terme (PV)
    "Reggio Calabria",       38.1113, 15.6473,
    "Par",                   45.8046,  8.9602,   # Parè, oggi Colverde (CO)
    "Nanno",                 46.3500, 11.0000,   # oggi Ville d'Anaunia (TN)
    "Cassano Allo Ionio",    39.7833, 16.3167,
    "Bastia",                43.0667, 12.5500,   # Bastia Umbra (PG)
    "Reggio Emilia",         44.6989, 10.6297,
    "Caderzone",             46.1167, 10.7333,   # Caderzone Terme (TN)
    "SOLBIATE CON CAGNO",    45.7967,  8.9700,   # CO
    "SENORBI'",              39.5333,  9.1333,   # Senorbì (SU)
    "VALVARRONE",            46.1000,  9.3500    # LC
  )

  # Riccò del Golfo di Spezia ha un doppio spazio al posto della "ò": lo trattiamo a parte
  m_ricco <- grepl("^Ricc", comuni_coords$comune) & comuni_coords$provincia == "SP"
  comuni_coords$lat[m_ricco] <- 44.1833
  comuni_coords$lon[m_ricco] <- 9.7333

  for (i in seq_len(nrow(correzioni))) {
    m <- comuni_coords$comune == correzioni$comune[i]
    if (!any(m)) warning("Non trovato nel dataset: ", correzioni$comune[i])
    comuni_coords$lat[m] <- correzioni$lat[i]
    comuni_coords$lon[m] <- correzioni$lon[i]
  }

  cat("Senza coordinate dopo le correzioni:", sum(is.na(comuni_coords$lat)), "\n")

  # ---- Tempi in auto verso Bergamo (OSRM) --------------------------------
  valid <- comuni_coords |> filter(!is.na(lat), !is.na(lon))
  gruppi <- split(seq_len(nrow(valid)), ceiling(seq_len(nrow(valid)) / BATCH))

  estrai <- function(m) vapply(m, function(x) if (is.null(x[[1]])) NA_real_ else as.numeric(x[[1]]), numeric(1))

  risultati <- lapply(gruppi, function(ix) {
    b <- valid[ix, ]
    n <- nrow(b)
    coords <- paste(c(paste(b$lon, b$lat, sep = ","),
                      paste(BERGAMO_LON, BERGAMO_LAT, sep = ",")), collapse = ";")

    out <- tryCatch({
      resp <- request(paste0("http://router.project-osrm.org/table/v1/driving/", coords)) |>
        req_url_query(sources = paste(0:(n - 1), collapse = ";"),
                      destinations = n,
                      annotations = "duration,distance") |>
        req_timeout(60) |>
        req_perform()
      d <- resp_body_json(resp)
      if (!identical(d$code, "Ok")) stop("OSRM: ", d$code, " - ", d$message)

      tibble(comune    = b$comune,
             provincia = b$provincia,
             car_min   = round(estrai(d$durations) / 60, 1),
             car_km    = round(estrai(d$distances) / 1000, 1))
    }, error = function(e) {
      message("Batch fallito: ", conditionMessage(e))
      tibble(comune = b$comune, provincia = b$provincia,
             car_min = NA_real_, car_km = NA_real_)
    })

    Sys.sleep(0.5)
    out
  })

  df_car <- bind_rows(risultati) |> distinct(comune, provincia, .keep_all = TRUE)

  print(df_car |> filter(!is.na(car_min)) |> arrange(car_min) |> head(10))

  # Comuni senza tempo (es. Lipari): restano NA, niente valori finti
  print(filter(df_car, is.na(car_min)))

  # Valori più alti da controllare (isole, traghetti, match sbagliati?)
  print(df_car |> arrange(desc(car_min)) |> head(15))

  df_car
}

# Restituisce la tabella comune, provincia, car_min, car_km per tutti i comuni
# italiani del dataset (la prende dalla cache; se manca la crea).
carica_tempi_comuni <- function(df) {
  res <- df %>%
    filter(NAZIONE_RESIDENZA == "ITALIA") %>%
    distinct(comune = COMUNE_RESIDENZA, provincia = PROVINCIA_RESIDENZA_SIGLA)
  dom <- df %>%
    filter(NAZIONE_DOMICILIO == "ITALIA") %>%
    distinct(comune = COMUNE_DOMICILIO, provincia = PROVINCIA_DOMICILIO_SIGLA)
  comuni <- bind_rows(res, dom) %>% distinct()
  cat("Comuni italiani (residenza + domicilio):", nrow(comuni), "\n")

  f_cache   <- file.path(cartella, CACHE_TEMPI)
  f_vecchio <- file.path(cartella, FILE_TEMPI_VECCHIO)

  if (file.exists(f_cache) && !RICALCOLA_TEMPI) {
    tempi <- as.data.frame(read_parquet(f_cache, mmap = FALSE))
    cat("Tempi letti dalla cache:", CACHE_TEMPI, "\n")
  } else {
    if (!RICALCOLA_TEMPI && file.exists(f_vecchio)) {
      x <- as.data.frame(read_parquet(f_vecchio, mmap = FALSE))
      tempi <- bind_rows(
        x %>% filter(NAZIONE_RESIDENZA == "ITALIA") %>%
          distinct(comune = COMUNE_RESIDENZA, provincia = PROVINCIA_RESIDENZA_SIGLA,
                   car_min = res_car_min, car_km = res_car_km),
        x %>% filter(NAZIONE_DOMICILIO == "ITALIA") %>%
          distinct(comune = COMUNE_DOMICILIO, provincia = PROVINCIA_DOMICILIO_SIGLA,
                   car_min = dom_car_min, car_km = dom_car_km)
      )
      rm(x)
      cat("Cache creata dai tempi gia' calcolati in", FILE_TEMPI_VECCHIO, "\n")
    } else {
      tempi <- calcola_tempi_osrm(comuni)
      cat("Tempi calcolati da OSRM\n")
    }
    tempi <- tempi %>% distinct(comune, provincia, .keep_all = TRUE)
    write_parquet(tempi, f_cache)
    cat("Cache salvata:", CACHE_TEMPI, "\n")
  }

  tempi <- tibble::as_tibble(tempi) %>%
    select(comune, provincia, car_min, car_km) %>%
    distinct(comune, provincia, .keep_all = TRUE)

  mancanti <- anti_join(comuni, tempi, by = c("comune", "provincia"))
  if (nrow(mancanti) > 0) {
    print(mancanti, n = 20)
    stop(nrow(mancanti), " comuni senza tempi nella cache: metti RICALCOLA_TEMPI <- TRUE ",
         "(serve internet) oppure cancella ", CACHE_TEMPI)
  }
  tempi
}

# --- 12b. CFU ed eta' --------------------------------------------------
cfu <- carriere_grezzo %>%
  distinct(STU_ID, .keep_all = TRUE) %>%
  select(STU_ID, CFU_PASSATI_SEM1 = cfu_passati_sem1, CFU_PASSATI_SEM2 = cfu_passati_sem2)
carriere <- carriere %>% left_join(cfu, by = "STU_ID")

carriere$ETA_INGRESSO <- as.numeric(carriere$AA_COORTE_ID) - as.numeric(carriere$ANNO_NASCITA)

# --- 12c. tempi di viaggio e fuori sede --------------------------------
carriere <- carriere %>% mutate(AA_ISCR_ID = as.numeric(AA_ISCR_ID))

tempi_comuni <- carica_tempi_comuni(carriere)

lookup_res <- tempi_comuni %>%
  rename(COMUNE_RESIDENZA = comune, PROVINCIA_RESIDENZA_SIGLA = provincia,
         res_car_min = car_min, res_car_km = car_km)
lookup_dom <- tempi_comuni %>%
  rename(COMUNE_DOMICILIO = comune, PROVINCIA_DOMICILIO_SIGLA = provincia,
         dom_car_min = car_min, dom_car_km = car_km)

n_prima <- nrow(carriere)
carriere <- carriere %>%
  left_join(lookup_res, by = c("COMUNE_RESIDENZA", "PROVINCIA_RESIDENZA_SIGLA")) %>%
  left_join(lookup_dom, by = c("COMUNE_DOMICILIO", "PROVINCIA_DOMICILIO_SIGLA"))
stopifnot(nrow(carriere) == n_prima)   # il merge non deve cambiare le righe

estero        <- dplyr::coalesce(carriere$NAZIONE_RESIDENZA != "ITALIA", FALSE)
res_na_italia <- !estero & is.na(carriere$res_car_min)    # es. Lipari

# fuori sede = residenza estera, oppure a piu' di FUORISEDE_MIN minuti da Bergamo,
# oppure comune italiano senza tempo calcolabile (isole ecc.)
fuorisede <- estero | dplyr::coalesce(carriere$res_car_min > FUORISEDE_MIN, FALSE) | res_na_italia
carriere$FUORI_SEDE_FLG  <- as.logical(fuorisede)
carriere$res_car_missing <- as.integer(is.na(carriere$res_car_min))

# tempo/distanza di riferimento: il piu' vicino tra residenza e domicilio
carriere$ref_car_min <- pmin(carriere$res_car_min, carriere$dom_car_min, na.rm = TRUE)
carriere$ref_car_km  <- pmin(carriere$res_car_km,  carriere$dom_car_km,  na.rm = TRUE)

cat("FUORI_SEDE_FLG:\n"); print(table(carriere$FUORI_SEDE_FLG, useNA = "always"))
check_atteso(sum(carriere$FUORI_SEDE_FLG), 2566, "Fuori sede (con i tempi in cache)")
cat("NA residui in res_car_min (Italia):", sum(res_na_italia), "\n")
cat("ref_car_min - NA:", sum(is.na(carriere$ref_car_min)), "\n")
new_cols <- c("res_car_min", "res_car_km", "dom_car_min", "dom_car_km",
              "ref_car_min", "ref_car_km", "res_car_missing", "FUORI_SEDE_FLG")
print(summary(carriere[new_cols]))


# ================================================================
# 13. OFA (correzione 5) + DATE ISO + FILE FINALE E FILE DI AUDIT
# ================================================================

cat("\n================ OFA + date ISO ================\n")

ha_esami <- ad_superate_primo_anno_vero %>% distinct(STU_ID) %>% mutate(ha_esami = TRUE)

fine_primo_anno <- carriere %>%
  transmute(STU_ID, fine_primo_anno = as.Date(paste0(as.numeric(AA_ISCR_ID) + 1, "-09-30")))

ofa_tutti <- ad_superate_primo_anno_vero %>%
  filter(startsWith(AD_DES, "OFA")) %>%
  inner_join(fine_primo_anno, by = "STU_ID") %>%
  mutate(
    sup_data = as.Date(SUP_DATA, format = "%d/%m/%Y"),
    superato_in_anno = SUP_FLG == "1" & (is.na(sup_data) | sup_data <= fine_primo_anno),
    ofa_vero = RIC == "Né riconosciuta nè convalidata"
  )

ofa_veri <- ofa_tutti %>%
  filter(ofa_vero) %>%
  group_by(STU_ID) %>%
  summarise(OFA_MIN_SUP = as.integer(min(superato_in_anno)), .groups = "drop")

ofa_esenti <- ofa_tutti %>% filter(!ofa_vero) %>% distinct(STU_ID) %>% mutate(ofa_esente = TRUE)

carriere <- carriere %>%
  left_join(ha_esami, by = "STU_ID") %>%
  left_join(ofa_veri, by = "STU_ID") %>%
  left_join(ofa_esenti, by = "STU_ID")

carriere$OFA_STATO <- "NON_ASSEGNATO"
carriere$OFA_STATO[!is.na(carriere$ofa_esente)] <- "ESENTATO"
carriere$OFA_STATO[carriere$OFA_MIN_SUP == 1] <- "ASSOLTO"
carriere$OFA_STATO[carriere$OFA_MIN_SUP == 0] <- "NON_ASSOLTO"
carriere$OFA_STATO[is.na(carriere$ha_esami)] <- "NESSUNA_ATTIVITA"

carriere$ha_esami <- NULL
carriere$OFA_MIN_SUP <- NULL
carriere$ofa_esente <- NULL

cat("OFA_STATO:\n"); print(table(carriere$OFA_STATO))

carriere <- carriere %>%
  mutate(
    DATA_ISCR = format(as.Date(DATA_ISCR, format = "%d/%m/%Y"), "%Y-%m-%d"),
    DATA_FINECDSCORRENTE = format(as.Date(DATA_FINECDSCORRENTE, format = "%d/%m/%Y"), "%Y-%m-%d")
  )

# --- (4) File di AUDIT: colonne di esito, fuori dal file per il modello ---
cols_audit <- c("AAFINECDSCORRENTE_ID", "DATA_FINECDSCORRENTE", "ANNO_CHIUSURA", "MESE_CHIUSURA",
                "STATO_CARRIERA", "MOTIVO_STATO_STUDENTE", "DROPOUT_MATRICOLA")
audit <- carriere %>% select(PERS_ID, STU_ID, DROPOUT, all_of(cols_audit))
carriere <- carriere %>% select(-all_of(cols_audit))

ordine_finale <- c(
  "PERS_ID", "STU_ID",
  "AA_ISCR_ID", "AAISCRIZIONE", "DATA_ISCR", "AA_COORTE_ID",
  "PART_TIME_FLG", "ABBREVCARRIERA",
  "DIPARTIMENTO", "CDS_NORMATIVA", "CODICIONE", "CLASSEDILAUREA", "CORSODISTUDIO",
  "CURRICULUM_ID", "CURRICULUM",
  "GENERE", "ANNO_NASCITA", "MESE_NASCITA", "ETA_INGRESSO",
  "COMUNE_NASCITA", "PROVINCIA_NASCITA_SIGLA", "REGIONE_NASCITA", "NAZIONE_NASCITA", "CITTADINANZA",
  "COMUNE_RESIDENZA", "PROVINCIA_RESIDENZA_SIGLA", "REGIONE_RESIDENZA", "NAZIONE_RESIDENZA",
  "FUORI_SEDE_FLG",
  "COMUNE_DOMICILIO", "PROVINCIA_DOMICILIO_SIGLA", "REGIONE_DOMICILIO", "NAZIONE_DOMICILIO",
  "MATURITA_STRANIERA_FLG", "CODNAZIONESCUOLASUPERIORE", "ANNO_MATURITA", "TIPOMATURIT_COD",
  "SCUOLASUPERIORE", "COMUNE_ISTITUTO", "PROVINCIASCUOLASUPERIORE_SIGLA", "REGIONESCUOLASUPERIORE",
  "VOTOMATURITA_PCT", "MATURITA_LODE_FLAG",
  "ISEE", "ISEE_DICHIARATO_FLG", "ISEE_NUM", "ISPE", "ISPE_DICHIARATO_FLG", "ISPE_NUM", "BORSA_STUDIO",
  "TOLC_PCT", "TOLC_INGLESE", "OFA_STATO",
  "CFU_PREVISTI_SEM1", "CFU_PREVISTI_SEM2", "CFU_PASSATI_SEM1", "CFU_PASSATI_SEM2",
  "PCT_N_ESAMI_PASSATI_SEM1", "PCT_N_ESAMI_PASSATI_SEM2",
  "PCT_CFU_PASSATI_SEM1", "PCT_CFU_PASSATI_SEM2",
  "FLG_SEM1_ESAMI_SOSTENUTI", "FLG_SEM2_ESAMI_SOSTENUTI",
  "PCT_SUCCESSO_SEM1", "PCT_SUCCESSO_SEM2",
  "MEDIA_VOTO_DATI_SEM1", "MEDIA_VOTO_DATI_SEM2",
  "HA_VOTO_SEM1_FLG", "HA_VOTO_SEM2_FLG",
  "DROPOUT",
  "res_car_min", "res_car_km", "dom_car_min", "dom_car_km",
  "res_car_missing", "ref_car_min", "ref_car_km"
)
stopifnot(setequal(ordine_finale, names(carriere)))
carriere <- carriere %>% select(all_of(ordine_finale))

cat("\n>>> Colonne finali:", ncol(carriere), "- Righe finali:", nrow(carriere), "\n")
cat(sprintf(">>> Dropout finale: %d (%.1f%%)\n", sum(carriere$DROPOUT), 100 * mean(carriere$DROPOUT)))

write_parquet(carriere, file.path(cartella, paste0(OUT_NAME, ".parquet")))
write_parquet(audit,    file.path(cartella, paste0(OUT_NAME, "_audit.parquet")))
cat(">>> Salvati:", paste0(OUT_NAME, ".parquet"), "e", paste0(OUT_NAME, "_audit.parquet"), "\n")


# ================================================================
# 14. FILE FINALE PER I MODELLI (ex "R Pulizia Finale Colonne.R" / 00_data_cleaning.R)
# ================================================================
# Feature engineering + scelta delle colonne. COORTE serve SOLO per filtrare
# (es. modello TOLC dal 2020), NON va usata come predittore.

cat("\n================ file finale per i modelli ================\n")

df <- as.data.frame(carriere)
cat("Raw  -> Rows:", nrow(df), " Cols:", ncol(df), "\n")

## FEATURE ENGINEERING
# ritardo di iscrizione rispetto all'inizio dell'a.a. (assunto 01/09)
data_iscr <- as.Date(df$DATA_ISCR)
df$RITARDO_ISCR_GG <- as.numeric(data_iscr - as.Date(paste0(df$AA_ISCR_ID, "-09-01")))

# anni tra maturita' e iscrizione
df$ANNI_DA_MATURITA <- df$AA_ISCR_ID - as.numeric(df$ANNO_MATURITA)

# cittadinanza italiana
df$CITTADINANZA_ITA <- df$CITTADINANZA == "ITALIA"

# anno di coorte: SOLO per filtrare (TOLC dal 2020), mai come predittore
df$COORTE <- as.numeric(df$AA_COORTE_ID)

## FEATURE SELECTION
# NB: qui NON tolgo le colonne SEM1/SEM2: le gestisco nello script dei modelli,
# cosi' dallo stesso parquet si fanno tutti gli scenari (early / nosem2 / all).
exclude <- c(
  # personal information:
  "PERS_ID", "STU_ID",

  # duplicates:
  "AAISCRIZIONE", "AA_COORTE_ID", "CODICIONE", "CURRICULUM_ID", "DATA_ISCR",
  "AA_ISCR_ID",

  # original features with engineered counterparts:
  "ANNO_NASCITA", "MESE_NASCITA", "ANNO_MATURITA", "CITTADINANZA",

  # raw ISEE and ISPE:
  "ISEE", "ISPE",

  # hyper-detailed geographical features:
  "COMUNE_NASCITA", "PROVINCIA_NASCITA_SIGLA", "REGIONE_NASCITA", "NAZIONE_NASCITA",
  "COMUNE_RESIDENZA", "PROVINCIA_RESIDENZA_SIGLA", "NAZIONE_RESIDENZA",
  "COMUNE_DOMICILIO", "PROVINCIA_DOMICILIO_SIGLA", "REGIONE_DOMICILIO", "NAZIONE_DOMICILIO",
  "CODNAZIONESCUOLASUPERIORE", "SCUOLASUPERIORE", "COMUNE_ISTITUTO",
  "PROVINCIASCUOLASUPERIORE_SIGLA",

  # distanze: tengo ref_car_min, FUORI_SEDE_FLG e res_car_missing (stessi tempi)
  "ref_car_km", "res_car_min", "res_car_km", "dom_car_min", "dom_car_km",

  # corso: tutto determinato da CORSODISTUDIO
  # (DIPARTIMENTO resta nel file: serve come gruppo per il modello multilivello;
  #  nella random forest viene escluso nello script del modello)
  "CLASSEDILAUREA", "CDS_NORMATIVA", "ABBREVCARRIERA", "CURRICULUM",

  # doppioni / quasi costanti
  "ISPE_DICHIARATO_FLG", "REGIONESCUOLASUPERIORE",
  "PART_TIME_FLG", "MATURITA_STRANIERA_FLG", "MATURITA_LODE_FLAG",

  # ridondanti con PCT_N_ESAMI e MEDIA_VOTO (esami: tengo solo PCT_N_ESAMI_PASSATI)
  "PCT_CFU_PASSATI_SEM1", "PCT_CFU_PASSATI_SEM2",
  "CFU_PASSATI_SEM1", "CFU_PASSATI_SEM2",
  "HA_VOTO_SEM1_FLG", "HA_VOTO_SEM2_FLG"
)

# controllo: tutte le colonne da escludere esistono davvero (evita errori di battitura)
missing_cols <- setdiff(exclude, names(df))
if (length(missing_cols) > 0) {
  warning("Colonne in 'exclude' non presenti nel dataset: ",
          paste(missing_cols, collapse = ", "))
}

df_clean <- df[, !(names(df) %in% exclude), drop = FALSE]

# ordine delle colonne per gruppi: corso, anagrafica, provenienza, scuola superiore,
# ingresso, situazione economica, primo anno (SEM1 poi SEM2), poi COORTE (solo per
# filtrare, NON e' un predittore) e DROPOUT (target) in fondo
ordine_colonne <- c(
  "DIPARTIMENTO", "CORSODISTUDIO",
  "GENERE", "ETA_INGRESSO", "CITTADINANZA_ITA",
  "REGIONE_RESIDENZA", "FUORI_SEDE_FLG", "ref_car_min", "res_car_missing",
  "TIPOMATURIT_COD", "VOTOMATURITA_PCT", "ANNI_DA_MATURITA",
  "RITARDO_ISCR_GG", "TOLC_PCT", "TOLC_INGLESE", "OFA_STATO",
  "ISEE_DICHIARATO_FLG", "ISEE_NUM", "ISPE_NUM", "BORSA_STUDIO",
  "CFU_PREVISTI_SEM1", "FLG_SEM1_ESAMI_SOSTENUTI", "PCT_N_ESAMI_PASSATI_SEM1",
  "PCT_SUCCESSO_SEM1", "MEDIA_VOTO_DATI_SEM1",
  "CFU_PREVISTI_SEM2", "FLG_SEM2_ESAMI_SOSTENUTI", "PCT_N_ESAMI_PASSATI_SEM2",
  "PCT_SUCCESSO_SEM2", "MEDIA_VOTO_DATI_SEM2",
  "COORTE", "DROPOUT"
)
stopifnot(setequal(ordine_colonne, names(df_clean)))
df_clean <- df_clean[, ordine_colonne, drop = FALSE]

## CHECK + SAVE
cat("Clean -> Rows:", nrow(df_clean), " Cols:", ncol(df_clean), "\n")
cat("Colonne tenute:\n  ", paste(names(df_clean), collapse = ", "), "\n")
print(table(df_clean$DROPOUT, useNA = "ifany"))
cat("Coorti (controllo: niente NA, TOLC presente quasi solo dal 2020):\n")
print(table(df_clean$COORTE, !is.na(df_clean$TOLC_PCT), useNA = "ifany",
            dnn = c("COORTE", "HA_TOLC_PCT")))
check_atteso(ncol(df_clean), 32, "Colonne nel file finale")

write_parquet(df_clean, file.path(cartella, OUT_FINALE))
cat(">>> Salvato:", OUT_FINALE, "\n")