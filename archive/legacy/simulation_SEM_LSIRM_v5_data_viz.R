###############################################################################
# sampling 된 문항들의 실제 패턴이 SEM model 이 의미하는 바와 align 되는지 확인
##############################################################################

# =========================================================================
# 요인별 원본 문항 데이터 분리 저장
# =========================================================================
factor_vars <- list(
  adversity     = paste0("ctq", 1:5),
  inflammation  = c("log_il6", "log_crp", "log_fibrinogen"),
  dep_affect    = paste0("cesd_da", 1:7),
  somatic_dep   = paste0("cesd_som", 1:6),
  pos_affect    = paste0("cesd_pa", 1:4),
  interpersonal = paste0("cesd_ip", 1:4),
  health_burden = c("n_chronic", "n_meds", "n_visits")
)

factor_data <- lapply(factor_vars, function(vars) {
  df_out <- dat[, vars, drop=FALSE]
  for (v in vars) {
    if (is.ordered(df_out[[v]])) {
      df_out[[v]] <- as.integer(as.character(df_out[[v]]))
    }
  }
  df_out
})

library(ggplot2)
library(gridExtra)


# 1. 역경 -> 염증 (gamma_11 = 0.3)

adv_vars <- names(factor_data$adversity)   # ctq1~5
inf_vars <- names(factor_data$inflammation) # log_il6, log_crp, log_fibrinogen

plot_list <- list()
idx <- 1

for (av in adv_vars) {
  for (iv in inf_vars) {
    df_tmp <- data.frame(
      adv_resp = factor(factor_data$adversity[[av]]),
      inf_val  = factor_data$inflammation[[iv]]
    )
    
    plot_list[[idx]] <- ggplot(df_tmp, aes(x=inf_val, fill=adv_resp)) +
      geom_histogram(bins=30, alpha=0.6, position="identity", color="white") +
      scale_fill_manual(values=c("0"="grey60", "1"="#E41A1C"), name=av) +
      theme_minimal(base_size=9) +
      labs(title=paste0(av, " → ", iv), x=iv, y="빈도")
    
    idx <- idx + 1
  }
}

p <- grid.arrange(grobs=plot_list, ncol=3, nrow=5,
                  top="inflammation distribution depend on adversity response")

ggsave("adv_inflammation.png", p, width=24, height=14)


# 2. 역경 -> 우울정서(gamma_21 = 0.4)

adv_vars <- names(factor_data$adversity)   # ctq1~5
dep_vars <- names(factor_data$dep_affect) # log_il6, log_crp, log_fibrinogen

plot_list <- list()
idx <- 1

for (av in adv_vars) {
  for (de in dep_vars) {
    df_tmp <- data.frame(
      adv_resp = factor(factor_data$adversity[[av]]),
      dep_val  = factor_data$dep_affect[[de]]
    )
    
    plot_list[[idx]] <- ggplot(df_tmp, aes(x=dep_val, fill=adv_resp)) +
      geom_histogram(bins=30, alpha=0.6, position="identity", color="white") +
      scale_fill_manual(values=c("0"="grey60", "1"="#E41A1C"), name=av) +
      theme_minimal(base_size=9) +
      labs(title=paste0(av, " → ", de), x=de, y="빈도")
    
    idx <- idx + 1
  }
}

p <- grid.arrange(grobs=plot_list, ncol=7, nrow=5,
                  top="depress effect distribution depend on adversity response")

ggsave("adv_dep_affect.png", p, width=24, height=14)


# 3. 염증 -> 신체적 건강부담 (beta_41 = 0.45)
inf_vars <- names(factor_data$inflammation)   # log_il6, log_crp, log_fibrinogen
hlt_vars <- names(factor_data$health_burden)   # n_chronic, n_meds, n_visits

plot_list <- list()
idx <- 1

for (iv in inf_vars) {
  for (hv in hlt_vars) {
    df_tmp <- data.frame(
      inf_val = factor_data$inflammation[[iv]],
      hlt_val = factor_data$health_burden[[hv]]
    )
    
    plot_list[[idx]] <- ggplot(df_tmp, aes(x = inf_val, y = hlt_val)) +
      geom_bin2d(bins = 30) +
      scale_fill_gradient(low = "grey90", high = "#E41A1C") +
      geom_smooth(method = "lm", se = TRUE, color = "steelblue", linewidth = 0.8) +
      theme_minimal(base_size = 9) +
      labs(
        title = paste0(iv, " → ", hv),
        x = iv, y = hv
      )
    
    idx <- idx + 1
  }
}

p <- grid.arrange(
  grobs = plot_list, ncol = 3, nrow = 3,
  top = "Health burden distribution depending on inflammation level (beta_41 = 0.45)"
)

ggsave("inf_health_burden.png", p, width = 14, height = 12, dpi = 150)


# 4. 염증 -> 신체적 우울 (beta_31 = 0.50)
inf_vars <- names(factor_data$inflammation)   # log_il6, log_crp, log_fibrinogen
som_vars <- names(factor_data$somatic_dep)     # cesd_som1 ~ cesd_som6
plot_list <- list()
idx <- 1
for (iv in inf_vars) {
  for (sv in som_vars) {
    df_tmp <- data.frame(
      inf_val = factor_data$inflammation[[iv]],
      som_val = factor_data$somatic_dep[[sv]]
    )
    
    plot_list[[idx]] <- ggplot(df_tmp, aes(x = inf_val, y = som_val)) +
      geom_bin2d(bins = 30) +
      scale_fill_gradient(low = "grey90", high = "#FF7F00") +
      geom_smooth(method = "lm", se = TRUE, color = "steelblue", linewidth = 0.8) +
      theme_minimal(base_size = 9) +
      labs(
        title = paste0(iv, " → ", sv),
        x = iv, y = sv
      )
    
    idx <- idx + 1
  }
}
p <- grid.arrange(
  grobs = plot_list, ncol = 6, nrow = 3,
  top = "Somatic depression distribution depending on inflammation level (beta_31 = 0.50)"
)
ggsave("inf_somatic_dep.png", p, width = 22, height = 10, dpi = 150)


# 5. 우울정서 -> 신체적 우울 (beta_32 = 0.30)
dep_vars <- names(factor_data$dep_affect)     # cesd_da1 ~ cesd_da7
som_vars <- names(factor_data$somatic_dep)     # cesd_som1 ~ cesd_som6
plot_list <- list()
idx <- 1
for (dv in dep_vars) {
  for (sv in som_vars) {
    df_tmp <- data.frame(
      dep_val = factor_data$dep_affect[[dv]],
      som_val = factor_data$somatic_dep[[sv]]
    )
    
    plot_list[[idx]] <- ggplot(df_tmp, aes(x = dep_val, y = som_val)) +
      geom_bin2d(bins = 15) +
      scale_fill_gradient(low = "grey90", high = "#377EB8") +
      geom_smooth(method = "lm", se = TRUE, color = "#E41A1C", linewidth = 0.8) +
      theme_minimal(base_size = 9) +
      labs(
        title = paste0(dv, " → ", sv),
        x = dv, y = sv
      )
    
    idx <- idx + 1
  }
}
p <- grid.arrange(
  grobs = plot_list, ncol = 6, nrow = 7,
  top = "Somatic depression distribution depending on depressive affect level (beta_32 = 0.30)"
)

ggsave("dep_affect_somatic_dep.png", p, width = 22, height = 20, dpi = 150)
