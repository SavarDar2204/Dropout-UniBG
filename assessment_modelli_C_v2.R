rm(list = ls())
library(pROC)
library(ROCR)
library(ggplot2)

CARTELLA_BASE <- "C:/Users/Dario/Downloads/Grafici sistemati"
CARTELLA_OUTPUT <- file.path(CARTELLA_BASE, "output_assessment")
dir.create(CARTELLA_OUTPUT, showWarnings = FALSE, recursive = TRUE)

SCENARI <- c("1_early_isee" = "Early",
             "2_mid_sem1"   = "Mid (after 1st semester)")

SOGLIA_SCELTA <- c("1_early_isee" = 0.31, "2_mid_sem1" = 0.31)

ADESIONE <- 0.02;  EFFICACIA <- 0.10;  V_STUDENTE <- 3200;  COSTO_ADERENTE <- 98

unibg_palette <- c("#111F37", "#1C355D", "#284C85", "#3462AC", "#3977DA",
                   "#6D9BE4", "#A0BEED", "#E5EDFA")

COL_FAM <- c("XGBoost" = "#111F37", "RF" = "#3977DA", "kNN" = "#6D9BE4",
             "NN" = "#284C85", "Logit" = "#A0BEED", "Naive Bayes" = "#3462AC")

LTY_FAM <- c("XGBoost" = "solid", "RF" = "solid", "kNN" = "dashed",
             "NN" = "dashed", "Logit" = "dotdash", "Naive Bayes" = "dotted")

custom_theme <- theme_bw(base_family = "Rubik") +
  theme(axis.text  = element_text(size = 10),
        axis.title = element_text(size = 12),
        legend.text = element_text(size = 12),
        legend.title = element_text(size = 14),
        plot.title  = element_text(size = 16, hjust = 0.5))

tema <- custom_theme + theme(legend.position = "bottom", legend.title = element_blank())
salva <- function(nome, w = 9, h = 6)
  ggsave(file.path(CARTELLA_OUTPUT, nome), width = w, height = h, dpi = 300, bg = "white")


#### 1. Lettura delle previsioni di tutti i modelli ####
leggi_per_file <- function(famiglia, pattern) {
  ff <- list.files(CARTELLA_BASE, pattern = pattern, recursive = TRUE, full.names = TRUE)
  ff <- ff[!grepl("output_assessment", ff)]
  d <- lapply(ff, function(f) {
    x <- read.csv(f)
    data.frame(famiglia = famiglia, scenario = basename(dirname(f)),
               modello = sub("^C_(.*)_previsioni_(train|validation|test)[.]csv$", "\\1", basename(f)),
               set     = sub("^.*_previsioni_(train|validation|test)[.]csv$", "\\1", basename(f)),
               y = x$y, p = x$p)
  })
  d
}

leggi_lungo <- function(famiglia, pattern) {
  ff <- list.files(CARTELLA_BASE, pattern = pattern, recursive = TRUE, full.names = TRUE)
  ff <- ff[!grepl("output_assessment", ff)]
  x <- read.csv(ff[1])
  list(data.frame(famiglia = famiglia, scenario = x$scenario, modello = x$modello,
                  set = x$set, y = x$y, p = x$prob))
}

fonti <- list(
  leggi_per_file("XGBoost",       "^C_M[0-9].*_previsioni_(train|validation|test)[.]csv$"),
  leggi_per_file("Logit",         "^C_Logit_.*_previsioni_(train|validation|test)[.]csv$"),
  leggi_per_file("NN",            "^C_NN_.*_previsioni_(train|validation|test)[.]csv$"),
  leggi_lungo   ("RF",            "^previsioni_rf[.]csv$"),
  leggi_lungo   ("kNN",           "^previsioni_knn[.]csv$"),
  leggi_lungo   ("Naive Bayes",   "^previsioni_nb[.]csv$")
)
pred <- do.call(rbind, unlist(fonti, recursive = FALSE))

#### 2. Funzioni di misura ####
ase     <- function(y, p) mean((y - p)^2)
auc_val <- function(y, p) as.numeric(auc(roc(y, p, levels = c(0, 1), direction = "<", quiet = TRUE)))

candidati <- do.call(rbind, lapply(split(pred, paste(pred$famiglia, pred$scenario, pred$modello)), function(d) {
  r <- data.frame(famiglia = d$famiglia[1], scenario = d$scenario[1], modello = d$modello[1])
  for (s in c("train", "validation", "test")) {
    x <- d[d$set == s, ]
    r[[paste0(s, "_ASE")]] <- ase(x$y, x$p)
    r[[paste0(s, "_AUC")]] <- auc_val(x$y, x$p)
  }
  r
}))

#### 3. Miglior modello per famiglia ####
# Regola di scelta: ASE validation minimo.
# Eccezione XGBoost: le tre configurazioni hanno ASE validation praticamente uguale (differenze di 0.001),
# quindi si sceglie quella con la minima distanza tra AUC train e AUC validation, per evitare l'overfitting.
FAMIGLIE_MIN_GAP <- "XGBoost"
scegli_modello <- function(d) {
  if (d$famiglia[1] %in% FAMIGLIE_MIN_GAP) d[which.min(abs(d$train_AUC - d$validation_AUC)), ]
  else d[which.min(d$validation_ASE), ]
}
scelti <- do.call(rbind, lapply(split(candidati, paste(candidati$famiglia, candidati$scenario)), scegli_modello))
rownames(scelti) <- NULL
write.csv(candidati, file.path(CARTELLA_OUTPUT, "assessment_tutti_i_candidati.csv"), row.names = FALSE)

cat("\n=== Modello scelto per ogni famiglia (min validation ASE; XGBoost: min distance between AUC train and validation) ===\n")
print(scelti[order(scelti$scenario, scelti$validation_ASE),
             c("scenario", "famiglia", "modello", "validation_ASE", "validation_AUC", "test_AUC")],
      row.names = FALSE, digits = 3)

ps <- pred[paste(pred$famiglia, pred$scenario, pred$modello) %in%
             paste(scelti$famiglia, scelti$scenario, scelti$modello), ]

# modello vincitore per scenario: AUC validation piu' alta
vincitori <- do.call(rbind, lapply(split(scelti, scelti$scenario), function(d) {
  d <- d[d$validation_AUC >= max(d$validation_AUC), ]
}))

#### 4. ASE train vs validation #####
ase_long <- rbind(
  data.frame(scelti[, c("scenario", "famiglia")], set = "validation", ASE = scelti$validation_ASE),
  data.frame(scelti[, c("scenario", "famiglia")], set = "train",      ASE = scelti$train_ASE))
ase_long$set <- factor(ase_long$set, levels = c("train", "validation"))
ase_long$scenario_l <- SCENARI[ase_long$scenario]
ord <- scelti[order(scelti$scenario, scelti$validation_ASE), ]
ase_long$famiglia <- factor(ase_long$famiglia, levels = unique(ord$famiglia[order(ord$validation_ASE)]))

ggplot(ase_long, aes(famiglia, ASE, fill = set)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.75) +
  geom_text(aes(label = sprintf("%.3f", ASE)), position = position_dodge(width = 0.8),
            vjust = -0.4, size = 3) +
  facet_wrap(~ scenario_l) +
  scale_fill_manual(values = unibg_palette[c(6, 3)]) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  coord_cartesian(ylim = c(0, 0.20)) +
  labs(title = "Best ASE per model family",
       x = NULL, y = "ASE") + tema
salva("assessment_1_ASE.png", 10, 5.5)

#### 5. Curve ROC e AUC ####
punti_roc <- function(y, p) {
  r <- roc(y, p, levels = c(0, 1), direction = "<", quiet = TRUE)
  data.frame(fpr = 1 - r$specificities, tpr = r$sensitivities)
}
roc_tutte <- function(sc, s) {
  d <- ps[ps$scenario == sc & ps$set == s, ]
  do.call(rbind, lapply(split(d, d$famiglia), function(x) {
    a <- auc_val(x$y, x$p)
    cbind(punti_roc(x$y, x$p), famiglia = x$famiglia[1], AUC = a,
          etich = sprintf("%s (AUC %.3f)", x$famiglia[1], a))
  }))
}

for (sc in names(SCENARI)) {
  r <- roc_tutte(sc, "validation")
  r$etich <- factor(r$etich, levels = unique(r$etich[order(-r$AUC)]))
  col <- setNames(COL_FAM[as.character(r$famiglia[match(levels(r$etich), r$etich)])], levels(r$etich))
  lty <- setNames(LTY_FAM[as.character(r$famiglia[match(levels(r$etich), r$etich)])], levels(r$etich))
  ggplot(r, aes(fpr, tpr, colour = etich, linetype = etich)) +
    geom_abline(slope = 1, intercept = 0, colour = "grey60") +
    geom_line(linewidth = 0.9) +
    scale_colour_manual(values = col) + scale_linetype_manual(values = lty) +
    coord_equal() +
    labs(title = paste("Validation set ROC -", SCENARI[sc]),
         x = "1 - specificity (FPR)", y = "Sensitivity (TPR)") + tema +
    guides(colour = guide_legend(nrow = 3))
  salva(paste0("assessment_2_roc_validation_", sc, ".png"), 7, 7.5)
}

auc_tab <- scelti[, c("scenario", "famiglia", "modello", "train_AUC", "validation_AUC", "test_AUC")]
auc_tab$train_meno_valid <- auc_tab$train_AUC - auc_tab$validation_AUC
write.csv(auc_tab[order(auc_tab$scenario, -auc_tab$validation_AUC), ],
          file.path(CARTELLA_OUTPUT, "assessment_auc.csv"), row.names = FALSE)

auc_long <- rbind(
  data.frame(auc_tab[, c("scenario", "famiglia")], set = "train",      AUC = auc_tab$train_AUC),
  data.frame(auc_tab[, c("scenario", "famiglia")], set = "validation", AUC = auc_tab$validation_AUC),
  data.frame(auc_tab[, c("scenario", "famiglia")], set = "test",       AUC = auc_tab$test_AUC))
auc_long$set <- factor(auc_long$set, levels = c("train", "validation", "test"))
auc_long$scenario_l <- SCENARI[auc_long$scenario]
auc_long$famiglia <- reorder(auc_long$famiglia, -auc_long$AUC * (auc_long$set == "validation"), FUN = sum)

#### 6. Lift chart ####
lift_decili <- function(y, p) {
  o <- order(-p); y <- y[o]; n <- length(y)
  dec <- ceiling(seq_len(n) * 10 / n)
  data.frame(decile = 1:10,
             captured = sapply(1:10, function(j) sum(y[dec <= j]) / sum(y) * 100),
             response = sapply(1:10, function(j) mean(y[dec <= j]) * 100),
             lift     = sapply(1:10, function(j) mean(y[dec <= j]) / mean(y)))
}
lift_tab <- do.call(rbind, lapply(split(ps[ps$set == "validation", ],
                      paste(ps$scenario, ps$famiglia)[ps$set == "validation"]), function(x)
  cbind(scenario = x$scenario[1], famiglia = x$famiglia[1], lift_decili(x$y, x$p))))
rownames(lift_tab) <- NULL
write.csv(lift_tab, file.path(CARTELLA_OUTPUT, "assessment_lift_decili.csv"), row.names = FALSE)
lift_tab$scenario_l <- SCENARI[lift_tab$scenario]

lift_curva <- do.call(rbind, lapply(split(ps[ps$set == "validation", ],
                      paste(ps$scenario, ps$famiglia)[ps$set == "validation"]), function(x) {
  pf <- performance(prediction(x$p, x$y), "lift", "rpp")
  d <- data.frame(scenario = x$scenario[1], famiglia = x$famiglia[1],
                  rpp = pf@x.values[[1]] * 100, lift = pf@y.values[[1]])
  d[is.finite(d$lift) & d$rpp > 2, ]
}))
lift_curva$scenario_l <- SCENARI[lift_curva$scenario]
ggplot(lift_curva, aes(rpp, lift, colour = famiglia, linetype = famiglia)) +
  geom_hline(yintercept = 1, colour = "grey60") +
  geom_line(linewidth = 0.9) +
  facet_wrap(~ scenario_l) +
  scale_colour_manual(values = COL_FAM) + scale_linetype_manual(values = LTY_FAM) +
  labs(title = "Cumulative lift on validation set",
       x = "% of students contacted (ranked by predicted dropout probability)",
       y = "Cumulative lift") + tema
salva("assessment_3_lift_cumulato.png", 11, 5.5)

sc <- "2_mid_sem1"
ggplot(subset(lift_curva, scenario == sc),
       aes(rpp, lift, colour = famiglia, linetype = famiglia)) +
  geom_hline(yintercept = 1, colour = "grey60") +
  geom_line(linewidth = 0.9) +
  scale_colour_manual(values = COL_FAM) + scale_linetype_manual(values = LTY_FAM) +
  labs(title = paste("Cumulative lift on validation set -", SCENARI[sc]),
       x = "% of students contacted (ranked by predicted dropout probability)",
       y = "Cumulative lift") + tema
salva("assessment_3_lift_mid.png", 6.5, 6.5)

#### 7. Profit ####
riepilogo_profitto <- NULL

tstar <- COSTO_ADERENTE / (EFFICACIA * V_STUDENTE)
cat(sprintf("\nOptimal bayesian profit threshold t* = c/(p*V) = %.3f\n", tstar))
profitto_ind <- function(y) ADESIONE * (y * EFFICACIA * V_STUDENTE - COSTO_ADERENTE)

val <- ps[ps$set == "validation", ]
riepilogo_profitto <- do.call(rbind, lapply(split(val, paste(val$scenario, val$famiglia)), function(x) {
  data.frame(scenario = x$scenario[1], famiglia = x$famiglia[1],
             profitto_medio_soglia_0.5 = mean((x$p > 0.5) * profitto_ind(x$y)),
             profitto_medio_soglia_t   = mean((x$p > tstar) * profitto_ind(x$y)),
             quota_segnalati_t         = mean(x$p > tstar))
}))
rownames(riepilogo_profitto) <- NULL

# Total profit
prof_cum <- do.call(rbind, lapply(split(val, paste(val$scenario, val$famiglia)), function(x) {
  o <- order(-x$p); pr <- profitto_ind(x$y[o]); n <- nrow(x)
  q <- seq(0, 1, by = 0.02)
  data.frame(scenario = x$scenario[1], famiglia = x$famiglia[1], quota = q * 100,
             profitto_cum = sapply(q, function(k) sum(pr[seq_len(round(n * k))])))
}))
prof_cum$scenario_l <- SCENARI[prof_cum$scenario]
ggplot(prof_cum, aes(quota, profitto_cum, colour = famiglia, linetype = famiglia)) +
  geom_hline(yintercept = 0, colour = "grey60") +
  geom_line(linewidth = 0.9) +
  facet_wrap(~ scenario_l) +
  scale_colour_manual(values = COL_FAM) + scale_linetype_manual(values = LTY_FAM) +
  coord_cartesian(ylim = c(-1000, 3000)) +
  labs(title = "Cumulative total computed profit (validation set)",
       x = "% of students contacted (ranked by predicted dropout probability)", y = "Total profit [€]") + tema
salva("assessment_4_profitto_cumulato.png", 11, 5.5)
write.csv(riepilogo_profitto, file.path(CARTELLA_OUTPUT, "assessment_profitto.csv"), row.names = FALSE)
cat("\nAverage profit per student (validation):\n"); print(riepilogo_profitto, row.names = FALSE, digits = 3)

#### 8. Best model ####
misure_soglia <- function(y, p, t) {
  a <- sum(p > t & y == 1); b <- sum(p <= t & y == 1)
  c <- sum(p > t & y == 0); d <- sum(p <= t & y == 0)
  data.frame(soglia = t, TP = a, FN = b, FP = c, TN = d,
             accuracy = (a + d) / length(y), sensitivity = a / (a + b),
             specificity = d / (c + d), precision = if (a + c > 0) a / (a + c) else NA,
             F1 = if (a > 0) 2 * a / (2 * a + b + c) else NA)
}
grid_soglie <- seq(0, 1, by = 0.02)
conf_tutte <- NULL; soglia_tab <- NULL

for (sc in names(SCENARI)) {
  v <- vincitori[vincitori$scenario == sc, ]
  dv <- ps[ps$scenario == sc & ps$famiglia == v$famiglia & ps$set == "validation", ]
  dt <- ps[ps$scenario == sc & ps$famiglia == v$famiglia & ps$set == "test", ]
  dr <- ps[ps$scenario == sc & ps$famiglia == v$famiglia & ps$set == "train", ]
  t.sc <- if (is.na(SOGLIA_SCELTA[sc])) mean(dr$y) else SOGLIA_SCELTA[sc]
  cat(sprintf("\n=== BEST MODEL %s: %s (%s) | AUC valid %.3f, ASE valid %.4f ===\n",
              SCENARI[sc], v$famiglia, v$modello, v$validation_AUC, v$validation_ASE))

  mt <- do.call(rbind, lapply(grid_soglie, function(t) misure_soglia(dv$y, dv$p, t)))
  soglia_tab <- rbind(soglia_tab, cbind(scenario = sc, famiglia = v$famiglia, mt))
  ml <- reshape(mt[, c("soglia", "accuracy", "sensitivity", "specificity", "precision")],
                direction = "long", varying = c("accuracy", "sensitivity", "specificity", "precision"),
                v.names = "valore", timevar = "misura", times = c("accuracy", "sensitivity", "specificity", "precision"))
  ggplot(ml, aes(soglia, valore, colour = misura)) +
    geom_line(linewidth = 0.9, na.rm = TRUE) +
    geom_vline(xintercept = t.sc, linetype = "dashed", colour = "grey40") +
    scale_colour_manual(values = unibg_palette[c(6, 1, 3, 5)]) +
    labs(title = paste0("Classification measurements varying threshold - ", v$famiglia, " (", SCENARI[sc], ")"),
         x = "Dropout probability threshold", y = NULL) + tema
  salva(paste0("assessment_5_soglie_", sc, ".png"), 9, 5.5)

  cf <- rbind(cbind(set = "validation", misure_soglia(dv$y, dv$p, 0.5)),
              cbind(set = "validation", misure_soglia(dv$y, dv$p, t.sc)),
              cbind(set = "test",       misure_soglia(dt$y, dt$p, t.sc)))
  conf_tutte <- rbind(conf_tutte, cbind(scenario = sc, famiglia = v$famiglia, modello = v$modello, cf))
  print(cf, row.names = FALSE, digits = 3)

  rr <- do.call(rbind, lapply(list(train = dr, validation = dv, test = dt), function(x)
    cbind(punti_roc(x$y, x$p), set = sprintf("%s (AUC %.3f)", x$set[1], auc_val(x$y, x$p)))))
  ggplot(rr, aes(fpr, tpr, colour = set)) +
    geom_abline(slope = 1, intercept = 0, colour = "grey60") + geom_line(linewidth = 0.9) +
    scale_colour_manual(values = unibg_palette[c(6, 3, 1)]) + coord_equal() +
    labs(title = paste0("Best model ROC: ", v$famiglia, " (", SCENARI[sc], ")"),
         x = "1 - specificity (FPR)", y = "Sensitivity (TPR)") + tema
  salva(paste0("assessment_6_roc_vincitore_", sc, ".png"), 6.5, 6.5)
}
write.csv(soglia_tab, file.path(CARTELLA_OUTPUT, "assessment_vincitore_soglie.csv"), row.names = FALSE)
write.csv(conf_tutte, file.path(CARTELLA_OUTPUT, "assessment_vincitore_matrici_confusione.csv"), row.names = FALSE)


#### 9. Final recap ####
riep <- merge(scelti, lift_tab[lift_tab$decile == 1, c("scenario", "famiglia", "lift")],
              by = c("scenario", "famiglia"))
names(riep)[names(riep) == "lift"] <- "lift_primo_decile_valid"
riep <- merge(riep, riepilogo_profitto, by = c("scenario", "famiglia"))
riep$vincitore <- paste(riep$scenario, riep$famiglia) %in% paste(vincitori$scenario, vincitori$famiglia)
riep <- riep[order(riep$scenario, -riep$validation_AUC), ]
write.csv(riep, file.path(CARTELLA_OUTPUT, "assessment_riepilogo.csv"), row.names = FALSE)
cat("\n=== RECAP (order by AUC Validation) ===\n")
print(riep[, c("scenario", "famiglia", "validation_AUC", "validation_ASE", "test_AUC",
               "lift_primo_decile_valid", "vincitore")], row.names = FALSE, digits = 3)