# =====================================================================
# 03_grafici_finali_unibg.R
# Copy of 03_grafici_finali.R with the same graphic identity as assessment_modelli_C_v2.R:
# Rubik font, UniBG palette, custom_theme/tema, English titles and labels.
# The original script is left untouched. Not executed when written.
#
# Requirements: the Rubik font must be installed on the system (as for the assessment
# script) and the 'ragg' package (png device that knows the system fonts).
#
# Plots produced (same names as the original png, with the _unibg suffix):
#   PART A - why only the dataset with cohorts from 2020
#     M1a_tolc_per_coorte, M1d_dropout_coorte_con_C, 02_valori_mancanti
#   PART B - pre-processing choices
#     M4a_stato_sem1, M5_anni_da_maturita_cl, M6_iscrizione_periodo, M9_tipo_maturita
#   PART C - key variables
#     04b_istogrammi, 04c_isee_log_con_fasce, 04d_boxplot_per_dropout, 04e_correlazioni,
#     05_cat_GENERE, 05_cat_ISEE_DICHIARATO_FLG, 06_classi_TOLC_PCT, 06_classi_VOTOMATURITA_PCT
#
# Style rules (same for ALL plots):
#   - theme: custom_theme / tema of the assessment (theme_bw, Rubik, centred title size 16,
#     legend at the bottom without title)
#   - colours: UniBG palette; counts = dark blue, dropout rates = medium blue,
#     secondary data / backgrounds = light blue; reference line always dashed and dark
#   - English number format (comma for thousands, point for decimals); percentages
#     with 1 decimal in labels and no decimals on axes
#   - title = "what", subtitle = "on which data" (full file or cohorts from 2020)
#     + meaning of the dashed line; y axis of the rates is always "Dropout rate"
#   - rate bars: always vertical, label "category (n=...)" under each bar
#   - size: 9 x 5.5 inches (11 x 5.5 / 12 x 7 for wider plots), 300 dpi
#
# How to use: run section 0 (setup) and then the blocks you want, or everything at once.
# The png files are saved in <BASE> (the Grafici_finali folder).
# =====================================================================


# ---------------------------------------------------------------------
# 0. SETUP
# ---------------------------------------------------------------------
library(arrow)
library(tidyverse)
library(scales)
library(ragg)       # png device: knows the system fonts (Rubik)

# >>> Change the path if needed (use "/" also on Windows)
BASE <- "C:/Users/Dario/Downloads/Grafici sistemati/Grafici_finali"
setwd(BASE)
# parquet of the full file (all cohorts): put it in BASE or change the path here
FILE_PARQUET <- file.path(BASE, "File Finale Dropout.parquet")
cartella_out <- BASE            # png saved next to the script
dir.create(cartella_out, recursive = TRUE, showWarnings = FALSE)

# Save as png (ragg device, 300 dpi); if the old file is open in another app it says so
# and saves with _NEW
salva <- function(p, nome, w = 9, h = 5.5) {
  f  <- file.path(cartella_out, paste0(nome, "_unibg.png"))
  ok <- tryCatch({ ggsave(f, plot = p, width = w, height = h, dpi = 300,
                          device = agg_png, bg = "white"); TRUE },
                 error = function(e) {
                   message("WARNING: cannot overwrite ", f, " (",
                           conditionMessage(e), "). Close the file if it is open.")
                   FALSE })
  if (!ok) {
    f2 <- file.path(cartella_out, paste0(nome, "_unibg_NEW.png"))
    ggsave(f2, plot = p, width = w, height = h, dpi = 300, device = agg_png, bg = "white")
    message("Saved as: ", f2)
  }
}

# --- Palette and fixed colours ---------------------------------------
unibg_palette <- c("#111F37", "#1C355D", "#284C85", "#3462AC", "#3977DA",
                   "#6D9BE4", "#A0BEED", "#E5EDFA")
col_scuro <- unibg_palette[1]   # reference lines, highlighted text
col_base  <- unibg_palette[2]   # counts (histograms, % of NA)
col_rate  <- unibg_palette[1]   # dropout rates (bars/lines, same as the bar plots of the other scripts)
col_ref   <- "grey60"            # dashed reference lines (as in the other scripts), distinct from the bars
col_light <- unibg_palette[7]   # backgrounds / secondary data / highlighted dataset
col_drop  <- c("0" = unibg_palette[6], "1" = unibg_palette[1])   # light = continues, dark = dropout

# --- Theme (same as assessment_modelli_C_v2.R) ------------------------
custom_theme <- theme_bw(base_family = "Rubik") +
  theme(axis.text  = element_text(size = 10),
        axis.title = element_text(size = 12),
        legend.text = element_text(size = 12),
        legend.title = element_text(size = 14),
        plot.title  = element_text(size = 16, hjust = 0.5))
tema <- custom_theme + theme(legend.position = "bottom", legend.title = element_blank(),
                             plot.subtitle = element_text(size = 11, hjust = 0.5),
                             panel.grid = element_blank())   # no background grid
theme_set(tema)
# text drawn by geom_text()/annotate() also in Rubik
update_geom_defaults("text",  list(family = "Rubik"))
update_geom_defaults("label", list(family = "Rubik"))

# --- English number format ---------------------------------------------
fmt_n    <- label_number(big.mark = ",", decimal.mark = ".")                      # 21,535
fmt_pct  <- label_percent(accuracy = 0.1, big.mark = ",", decimal.mark = ".")     # 27.0%  (labels)
fmt_pct0 <- label_percent(accuracy = 1,   big.mark = ",", decimal.mark = ".")     # 27%    (axes)

# --- Dati ---------------------------------------------------------------
df <- read_parquet(FILE_PARQUET) %>%
  mutate(BORSA_STUDIO = as.logical(BORSA_STUDIO),
         COORTE       = as.integer(COORTE),
         DROPOUT_F    = factor(DROPOUT, levels = c(0, 1), labels = c("0", "1")))

# Tolgo TUTTE le variabili del 2o semestre (non disponibili al momento della previsione)
df <- df %>% select(-contains("SEM2"))

# Dataset con coorti dal 2020 (atteso: 20.817 righe)
df_c <- df %>% filter(COORTE >= 2020)
cat("Rows full file:", nrow(df), "| rows dataset with cohorts from 2020:", nrow(df_c),
    "(expected 20,817)\n")

tasso_tot <- mean(df$DROPOUT)     # dropout sul file completo
tasso_c   <- mean(df_c$DROPOUT)   # dropout nel dataset con coorte dal 2020

# Subtitle texts: on which data and meaning of the dashed line
scope_all <- paste0(fmt_n(nrow(df)), " students")
scope_c   <- paste0("Cohorts from 2020, ", fmt_n(nrow(df_c)), " students")
sub_rate  <- function(scope, rif) {
  paste0(scope, " - dashed line = mean rate (", fmt_pct(rif), ")")
}

coorti <- sort(unique(df$COORTE))

# Zona azzurra dataset con coorte dal 2020, identica nei grafici per coorte
zona_c <- function() {
  list(annotate("rect", xmin = 2019.5, xmax = max(coorti) + 0.5, ymin = -Inf, ymax = Inf,
                fill = col_light, alpha = 0.35),
       annotate("text", x = (2019.5 + max(coorti) + 0.5) / 2, y = Inf, label = "Cohorts from 2020",
                vjust = 1.8, fontface = "bold", colour = col_base, size = 4))
}


# --- Funzioni di servizio ------------------------------------------------
# Tabella: numerosita' e tasso di dropout per ogni modalita' di g (le NA sono escluse)
tab_tasso <- function(g, d) {
  tibble(mod = g, DROPOUT = d$DROPOUT) %>%
    filter(!is.na(mod)) %>%
    group_by(mod) %>%
    summarise(n = n(), tasso = mean(DROPOUT), .groups = "drop")
}

# Grafico a barre VERTICALI del tasso di dropout per modalita' (ordine = livelli del factor).
# ruota = TRUE: etichette oblique a 45 gradi e su una sola riga (per nomi lunghi).
grafico_tasso <- function(tab, titolo, nome, scope, rif, w = 9, h = 5.5,
                          ruota = FALSE, xlab = NULL) {
  cat("\n---", titolo, "---\n"); print(tab, n = Inf)
  tab <- tab %>% arrange(mod) %>%
    mutate(lab = paste0(mod, if (ruota) " (n=" else "\n(n=", fmt_n(n), ")"),
           lab = factor(lab, levels = lab))
  p <- ggplot(tab, aes(x = lab, y = tasso)) +
    geom_col(fill = col_rate, width = 0.85) +
    geom_hline(yintercept = rif, linetype = "dashed", colour = col_ref) +
    geom_text(aes(label = fmt_pct(tasso)), vjust = -0.4, size = 3.4) +
    scale_y_continuous(labels = fmt_pct0, expand = expansion(mult = c(0, 0.12))) +
    labs(title = titolo, subtitle = sub_rate(scope, rif), x = xlab, y = "Dropout rate")
  if (ruota) {
    p <- p + theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
                   plot.margin = margin(8, 8, 8, 45))   # margine a sinistra: l'etichetta obliqua non si taglia
  }
  salva(p, nome, w = w, h = h)
  invisible(tab)
}

# Grafico del tasso per classi (decili) di una variabile numerica espressa in quota (0-1):
# barre chiare = tasso di ogni classe, linea = andamento (NA esclusi)
grafico_decili <- function(v, nome, titolo, n_classi = 10) {
  x   <- df[[v]]
  brk <- unique(quantile(x, probs = seq(0, 1, length.out = n_classi + 1), na.rm = TRUE))
  k   <- length(brk) - 1
  lab <- paste0(round(brk[-(k + 1)] * 100), "-", round(brk[-1] * 100), "%")
  classe <- factor(cut(x, breaks = brk, include.lowest = TRUE, labels = FALSE),
                   levels = seq_len(k), labels = lab)
  tab <- tibble(classe, DROPOUT = df$DROPOUT) %>%
    filter(!is.na(classe)) %>%
    group_by(classe) %>%
    summarise(n = n(), tasso = mean(DROPOUT), .groups = "drop")
  cat("\n---", v, "---\n"); print(tab, n = Inf)

  p <- ggplot(tab, aes(x = classe, y = tasso, group = 1)) +
    geom_col(fill = col_light, width = 0.85) +
    geom_line(colour = col_rate, linewidth = 1) +
    geom_point(colour = col_rate, size = 2) +
    geom_hline(yintercept = tasso_tot, linetype = "dashed", colour = col_ref) +
    scale_y_continuous(labels = fmt_pct0, expand = expansion(mult = c(0, 0.08))) +
    labs(title = titolo,
         subtitle = paste0(scope_all, " - dashed line = mean rate (",
                           fmt_pct(tasso_tot), ")"),
         x = paste0("Classes of ", v, " (deciles, in % of the maximum)"),
         y = "Dropout rate")
  salva(p, nome)
  invisible(tab)
}


# =====================================================================
# PARTE A. PERCHE' SOLO IL DATASET con coorte dal 2020
# =====================================================================
cat("\n===== PART A =====\n")

# ---------------------------------------------------------------------
# M1a. Quota di studenti con il TOLC, per coorte (file completo)
# Cosa guardare: salto da ~0% a ~85-90% tra la coorte 2019 e la 2020.
# ---------------------------------------------------------------------
tolc_coorte <- df %>%
  group_by(COORTE) %>%
  summarise(TOLC_PCT = mean(!is.na(TOLC_PCT)),
            TOLC_INGLESE = mean(!is.na(TOLC_INGLESE)), .groups = "drop") %>%
  pivot_longer(-COORTE, names_to = "variabile", values_to = "quota")

p_m1a <- ggplot(tolc_coorte, aes(x = COORTE, y = quota, colour = variabile)) +
  zona_c() +
  geom_line(linewidth = 1) + geom_point(size = 2) +
  scale_colour_manual(values = c("TOLC_PCT" = col_rate, "TOLC_INGLESE" = unibg_palette[5])) +
  scale_x_continuous(breaks = coorti) +
  scale_y_continuous(labels = fmt_pct0, limits = c(0, 1), expand = expansion(mult = c(0.02, 0.02))) +
  labs(title = "Share of students with a TOLC score, by cohort",
       subtitle = scope_all,
       x = "Cohort", y = "% with TOLC available", colour = NULL)
salva(p_m1a, "M1a_tolc_per_coorte")

# ---------------------------------------------------------------------
# M1d. Tasso di dropout per coorte, con dataset evidenziato (file completo)
# Cosa guardare: il tasso cambia molto tra coorti (16% nel 2016, 31% nel 2021).
# ---------------------------------------------------------------------
tab_coorte <- df %>% group_by(COORTE) %>%
  summarise(n = n(), tasso = mean(DROPOUT), .groups = "drop")

p_m1d <- ggplot(tab_coorte, aes(x = COORTE, y = tasso)) +
  zona_c() +
  geom_hline(yintercept = tasso_tot, linetype = "dashed", colour = col_ref) +
  geom_line(colour = col_rate, linewidth = 1) +
  geom_point(aes(size = n), colour = col_rate) +
  scale_x_continuous(breaks = coorti) +
  scale_y_continuous(labels = fmt_pct0, limits = c(0, NA), expand = expansion(mult = c(0.02, 0.08))) +
  scale_size_continuous(labels = fmt_n) +
  labs(title = "Dropout rate by cohort",
       subtitle = sub_rate(scope_all, tasso_tot),
       x = "Cohort", y = "Dropout rate", size = "N. of students")
salva(p_m1d, "M1d_dropout_coorte_con_C")

# ---------------------------------------------------------------------
# 02. Percentuale di valori mancanti per variabile (file completo, senza SEM2)
# Cosa guardare: TOLC e media voto hanno molti NA "strutturali"; ISEE ~31%.
# ---------------------------------------------------------------------
tab_na <- df %>%
  select(-DROPOUT_F) %>%
  summarise(across(everything(), ~ sum(is.na(.)))) %>%
  pivot_longer(everything(), names_to = "variabile", values_to = "n_na") %>%
  mutate(pct_na = n_na / nrow(df)) %>%
  filter(n_na > 0)

p_na <- ggplot(tab_na, aes(x = reorder(variabile, pct_na), y = pct_na)) +
  geom_col(fill = col_rate, width = 0.85) +
  geom_text(aes(label = fmt_pct(pct_na)), hjust = -0.1, size = 3.4) +
  coord_flip() +
  scale_y_continuous(labels = fmt_pct0, limits = c(0, 0.85), expand = expansion(mult = c(0, 0))) +
  labs(title = "Missing values by variable",
       subtitle = scope_all,
       x = NULL, y = "% of missing values")
salva(p_na, "02_valori_mancanti")


# =====================================================================
# PARTE B. SCELTE DI PRE-PROCESSING (dataset con coorte dal 2020)
# =====================================================================
cat("\n===== PART B =====\n")

# ---------------------------------------------------------------------
# M4a. Stato nel 1o semestre (4 gruppi, come stato_sem1 nello script di split)
# Cosa guardare: chi non ha sostenuto esami abbandona nel 67% dei casi, chi ha la media nel 8%.
# ---------------------------------------------------------------------
liv_stato <- c("No exam taken", "Exams taken, none passed",
               "Passed, no grade", "Passed, with average grade")
stato1 <- case_when(!as.logical(df$FLG_SEM1_ESAMI_SOSTENUTI) ~ liv_stato[1],
                    df$PCT_N_ESAMI_PASSATI_SEM1 == 0          ~ liv_stato[2],
                    is.na(df$MEDIA_VOTO_DATI_SEM1)            ~ liv_stato[3],
                    TRUE                                        ~ liv_stato[4])
grafico_tasso(tab_tasso(factor(stato1, levels = liv_stato), df),
              "Dropout by status in the 1st semester", "M4a_stato_sem1",
              scope_all, tasso_tot, w = 10)

# ---------------------------------------------------------------------
# M5. Anni dalla maturita' in classi (negativi portati a 0)
# ---------------------------------------------------------------------
anni_cl <- cut(pmax(df$ANNI_DA_MATURITA, 0), breaks = c(-Inf, 0, 1, 5, 15, Inf),
               labels = c("0", "1", "2-5", "6-15", ">15"))
grafico_tasso(tab_tasso(anni_cl, df),
              "Dropout by years since high-school graduation",
              "M5_anni_da_maturita_cl", scope_all, tasso_tot, w = 9,
              xlab = "Years since high-school graduation (ANNI_DA_MATURITA_CL)")

# ---------------------------------------------------------------------
# M6. Enrolment period (RITARDO_ISCR_GG in 4 classes) - ALL students (full file)
# ---------------------------------------------------------------------
periodo <- cut(df$RITARDO_ISCR_GG, breaks = c(-Inf, 0, 32, 73, Inf),
               labels = c("By August\n(up to 0 days)", "September - 2 October\n(1-32 days)",
                          "Late, up to 13 Nov.\n(33-73 days)", "After 13 Nov.\n(over 73 days)"))
grafico_tasso(tab_tasso(periodo, df),
              "Dropout by enrolment period",
              "M6_iscrizione_periodo", scope_all, tasso_tot, w = 10,
              xlab = "Enrolment period (ISCRIZIONE_PERIODO)")

# ---------------------------------------------------------------------
# M9. Tipo di maturita' raggruppato (mappatura ESSE3 da LEGGIMI_C_v2.txt)
# Note: "Altro" here = codes not in the table; numbers may differ slightly from C v2.
# ---------------------------------------------------------------------
map_mat <- c(
  F.004 = "Liceo scientifico", F.326 = "Liceo scientifico", F.327 = "Liceo scientifico",
  F.001 = "Liceo classico",
  F.002 = "Liceo linguistico",
  F.307 = "Liceo scienze umane", F.325 = "Liceo scienze umane",
  F.328 = "Tecnico economico", F.315 = "Tecnico economico", F.206 = "Tecnico economico",
  F.101 = "Tecnico economico", F.054 = "Tecnico economico", M.57 = "Tecnico economico",
  F.316 = "Tecnico tecnologico", F.319 = "Tecnico tecnologico", F.318 = "Tecnico tecnologico",
  F.320 = "Tecnico tecnologico", F.321 = "Tecnico tecnologico", F.324 = "Tecnico tecnologico",
  F.098 = "Tecnico tecnologico",
  F.309 = "Professionale", F.311 = "Professionale", F.336 = "Professionale",
  F.338 = "Professionale",
  M.9998 = "Estero")
cod      <- as.character(df$TIPOMATURIT_COD)
tipo_mat <- unname(map_mat[cod])
tipo_mat[is.na(cod)]      <- "Mancante"
tipo_mat[is.na(tipo_mat)] <- "Altro"
tipo_mat <- factor(tipo_mat, levels = c(unique(unname(map_mat)), "Altro", "Mancante"))
grafico_tasso(tab_tasso(tipo_mat, df),
              "Dropout by high-school diploma type",
              "M9_tipo_maturita", scope_all, tasso_tot, w = 11, h = 7, ruota = TRUE)


# =====================================================================
# PARTE C. VARIABILI CHIAVE (file completo)
# =====================================================================
cat("\n===== PART C =====\n")

# --- Preparazione per 04b e 04d (lanciare prima di entrambi) ---------------
# ISEE has a very long tail: shown in thousands of euro and only up to 100,000,
# the share left out is declared in the subtitle.
ISEE_MAX         <- 100000
quota_isee_oltre <- mean(df$ISEE_NUM[!is.na(df$ISEE_NUM)] > ISEE_MAX)
lab_isee         <- "ISEE_NUM (thousands of euro, up to 100)"
ordine_var       <- c(lab_isee, "MEDIA_VOTO_DATI_SEM1", "TOLC_INGLESE", "TOLC_PCT",
                      "VOTOMATURITA_PCT")

dati_chiave <- df %>%
  mutate(ISEE_NUM = if_else(ISEE_NUM > ISEE_MAX, NA_real_, ISEE_NUM) / 1000) %>%
  select(DROPOUT_F, ISEE_NUM, MEDIA_VOTO_DATI_SEM1, TOLC_INGLESE, TOLC_PCT, VOTOMATURITA_PCT) %>%
  rename(!!lab_isee := ISEE_NUM) %>%
  pivot_longer(-DROPOUT_F, names_to = "variabile", values_to = "valore") %>%
  filter(!is.na(valore)) %>%
  mutate(variabile = factor(variabile, levels = ordine_var))

# ---------------------------------------------------------------------
# 04b. Istogrammi delle variabili chiave (scala libera per pannello)
# ---------------------------------------------------------------------
# VOTOMATURITA_PCT: axis limited to 0.6-1 (values outside the range are dropped from histogram and boxplot only)
dati_hist <- dati_chiave %>%
  filter(!(variabile == "VOTOMATURITA_PCT" & (valore < 0.6 | valore > 1)))
cat("VOTOMATURITA_PCT outside [0.6, 1] (not shown in the histogram):",
    sum(dati_chiave$variabile == "VOTOMATURITA_PCT") -
      sum(dati_hist$variabile == "VOTOMATURITA_PCT"), "\n")

p_hist <- ggplot(dati_hist, aes(x = valore)) +
  geom_histogram(bins = 30, fill = col_base, colour = "white") +
  facet_wrap(~ variabile, scales = "free", ncol = 3) +
  scale_x_continuous(labels = label_number(big.mark = ",", decimal.mark = ".")) +
  scale_y_continuous(labels = fmt_n) +
  labs(title = "Distribution of the key variables",
       x = NULL, y = "Count")
salva(p_hist, "04b_istogrammi", w = 12, h = 7)

# ---------------------------------------------------------------------
# 04c. ISEE in scala log10(x + 1) con le fasce ufficiali A-H
# Soglie: A <= 15.000, B <= 18.000, C <= 28.000, D <= 33.000, E <= 46.000, F <= 58.000,
# G <= 70.000, H > 70.000 (stesse dello script di split). Gli ISEE = 0 restano fuori
# dal grafico (spike a sinistra) ma il loro numero e' nel sottotitolo.
# ---------------------------------------------------------------------
tagli_fin <- c(15000, 18000, 28000, 33000, 46000, 58000, 70000)
bnd       <- log10(tagli_fin + 1)
pos_fasce <- c(bnd[1] - 0.15, (bnd[-length(bnd)] + bnd[-1]) / 2, bnd[length(bnd)] + 0.2)
lab_fasce <- tibble(x = pos_fasce, fascia = LETTERS[1:8])
isee_val  <- df %>% filter(!is.na(ISEE_NUM), ISEE_NUM >= 0)
n_zero    <- sum(isee_val$ISEE_NUM == 0)

p_isee <- ggplot(isee_val, aes(x = log10(ISEE_NUM + 1))) +
  geom_histogram(bins = 70, fill = col_base, colour = "white") +
  geom_vline(xintercept = bnd, linetype = "dashed", colour = col_ref) +
  geom_text(data = lab_fasce, aes(x = x, y = Inf, label = fascia), inherit.aes = FALSE,
            vjust = 1.6, size = 5, fontface = "bold", colour = col_scuro) +
  coord_cartesian(xlim = c(2.5, 5.5)) +
  scale_x_continuous(labels = label_number(decimal.mark = ".")) +
  scale_y_continuous(labels = fmt_n, expand = expansion(mult = c(0, 0.15))) +
  labs(title = "ISEE on a log scale with the official brackets (A-H)",
       subtitle = paste0(scope_all, " - dashed lines = bracket thresholds"),
       x = "log10(ISEE + 1)", y = "Count")
salva(p_isee, "04c_isee_log_con_fasce", w = 11, h = 5.5)

# ---------------------------------------------------------------------
# 04d. Boxplot delle variabili chiave per DROPOUT (scala libera per pannello)
# Se le due scatole sono spostate, la variabile "discrimina" tra chi abbandona e chi no.
# ---------------------------------------------------------------------
p_box <- ggplot(dati_hist, aes(x = DROPOUT_F, y = valore, fill = DROPOUT_F)) +
  geom_boxplot(outlier.size = 0.4, outlier.alpha = 0.3, show.legend = FALSE) +
  facet_wrap(~ variabile, scales = "free_y", ncol = 3) +
  scale_fill_manual(values = col_drop) +
  scale_y_continuous(labels = label_number(big.mark = ",", decimal.mark = ".")) +
  labs(title = "Key variables by dropout",
       x = "DROPOUT (0 = continues, 1 = drops out)", y = NULL)
salva(p_box, "04d_boxplot_per_dropout", w = 12, h = 7)

# ---------------------------------------------------------------------
# 04e. Matrice di correlazione (Pearson, osservazioni a coppie complete)
# Cosa guardare: coppie molto correlate (informazione ridondante) e correlazione con DROPOUT.
# ---------------------------------------------------------------------
vars_num <- c("ETA_INGRESSO", "ref_car_min", "VOTOMATURITA_PCT", "ANNI_DA_MATURITA",
              "RITARDO_ISCR_GG", "TOLC_PCT", "TOLC_INGLESE", "ISEE_NUM", "ISPE_NUM",
              "CFU_PREVISTI_SEM1", "PCT_N_ESAMI_PASSATI_SEM1", "PCT_SUCCESSO_SEM1",
              "MEDIA_VOTO_DATI_SEM1")
mat_cor <- df %>% select(all_of(vars_num), DROPOUT) %>% cor(use = "pairwise.complete.obs")
cor_long <- as.data.frame(as.table(mat_cor)) %>% setNames(c("v1", "v2", "cor"))

p_cor <- ggplot(cor_long, aes(x = v1, y = v2, fill = cor)) +
  geom_tile(colour = "white") +
  geom_text(aes(label = number(cor, accuracy = 0.01, decimal.mark = "."),
                colour = abs(cor) > 0.5), size = 2.6, show.legend = FALSE) +
  scale_colour_manual(values = c("TRUE" = "white", "FALSE" = "black")) +
  scale_fill_gradient2(low = unibg_palette[6], mid = "white", high = unibg_palette[2],
                       midpoint = 0, limits = c(-1, 1)) +
  scale_y_discrete(limits = rev) +
  theme(axis.text.x = element_text(angle = 60, hjust = 1),
        legend.position = "none") +
  labs(title = "Correlation matrix (Pearson)", subtitle = scope_all,
       x = NULL, y = NULL, fill = "r")
salva(p_cor, "04e_correlazioni", w = 11, h = 9)

# ---------------------------------------------------------------------
# 05. Tasso di dropout per GENERE e per ISEE dichiarato (file completo)
# ---------------------------------------------------------------------
genere <- factor(case_when(df$GENERE == "F" ~ "Female", df$GENERE == "M" ~ "Male"),
                 levels = c("Female", "Male"))
grafico_tasso(tab_tasso(genere, df),
              "Dropout by gender", "05_cat_GENERE", scope_all, tasso_tot, w = 7,
              xlab = "Gender (GENERE)")

isee_dich <- factor(if_else(as.character(df$ISEE_DICHIARATO_FLG) %in% c("TRUE", "True", "1"),
                            "ISEE declared", "ISEE not declared"),
                    levels = c("ISEE declared", "ISEE not declared"))
grafico_tasso(tab_tasso(isee_dich, df),
              "Dropout by ISEE declaration", "05_cat_ISEE_DICHIARATO_FLG",
              scope_all, tasso_tot, w = 7, xlab = "ISEE_DICHIARATO_FLG")

# ---------------------------------------------------------------------
# 06. Tasso di dropout per decili di TOLC_PCT e VOTOMATURITA_PCT (file completo, NA esclusi)
# ---------------------------------------------------------------------
grafico_decili("TOLC_PCT", "06_classi_TOLC_PCT", "Dropout by TOLC score")
grafico_decili("VOTOMATURITA_PCT", "06_classi_VOTOMATURITA_PCT",
               "Dropout by high-school graduation grade")


# ---------------------------------------------------------------------
# Final check: are all 15 png files there?
# ---------------------------------------------------------------------
attesi <- c("02_valori_mancanti", "04b_istogrammi", "04c_isee_log_con_fasce",
            "04d_boxplot_per_dropout", "04e_correlazioni", "05_cat_GENERE",
            "05_cat_ISEE_DICHIARATO_FLG", "06_classi_TOLC_PCT", "06_classi_VOTOMATURITA_PCT",
            "M1a_tolc_per_coorte", "M1d_dropout_coorte_con_C", "M4a_stato_sem1",
            "M5_anni_da_maturita_cl", "M6_iscrizione_periodo", "M9_tipo_maturita")
manca <- attesi[!file.exists(file.path(cartella_out, paste0(attesi, "_unibg.png")))]
cat("\n=== DONE. Plots in:", file.path(getwd(), cartella_out), "===\n")
if (length(manca)) cat("WARNING, missing:", paste(manca, collapse = ", "), "\n") else
  cat("All 15 plots were created.\n")
