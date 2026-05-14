# model_v17 (telescoping): Bayesian Joint Multilayered LSIRM with Sparse Hierarchical Mixture-of-Mixtures Prior and Telescoping Sampling on the Upper Cluster Level

**TL;DR**
- v17 모델은 v16의 LSIRM + 계층적 mixture‑of‑mixtures (MoM 2017) 사전을 그대로 보존하면서, 상위 cluster 수 $K$에 대해 Frühwirth‑Schnatter, Malsiner‑Walli, Grün (2021)의 *dynamic MFM* + *telescoping sampler* 를 도입한 모델이다 ($\gamma_K=\alpha/K$, $\alpha\sim\mathcal F(6,3)$, $K-1\sim\mathrm{BNB}(1,4,3)$, $K_{\max}=100$). $L$은 고정.
- 아이템 위치 $b_j^{(\ell)}$의 MH 갱신은 prior 항에서 $(S_q,I_q)$를 marginalize한 collapsed‑MH로 수행하며, **Variant A**(완전 marginalize: $K\!\cdot\!L$ 가우시안 합)와 **Variant B**(부분 marginalize: $S_q$ 고정, $L$개 가우시안 합) 두 변형을 모두 정식 도출했다.
- $K$의 갱신은 분할 $\mathcal C$에만 의존하고 컴포넌트 모수와 분리되며, 빈 클러스터는 매 iteration 사전에서 새로 그려진다. 이로써 RJMCMC 없이 $K_+$와 $K$가 동시에 업데이트되어, v16 대비 cluster 수‑클러스터 라벨‑$b_j$의 결합 mixing이 본질적으로 향상된다.

---

## Part 1. 모델 복원 및 표기 통일

### 1.1 v16 LSIRM 부분 (다층 latent space item response)

$L_{\mathrm{lay}}$개의 응답 layer $\ell=1,\dots,L_{\mathrm{lay}}$, 응답자 $i=1,\dots,N$, 각 layer의 아이템 $j=1,\dots,J_\ell$에 대해
$$
\eta_{ij}^{(\ell)} \;=\; \alpha_i^{(\ell)} \;-\; \beta_j^{(\ell)} \;-\; \gamma_\ell\,\bigl\lVert a_i - b_j^{(\ell)} \bigr\rVert_2,
\qquad a_i\in\mathbb R^d,\quad b_j^{(\ell)}\in\mathbb R^d,
$$
$$
Y_{ij}^{(\ell)} \mid \eta_{ij}^{(\ell)} \;\sim\; p_\ell\bigl(\cdot\mid \eta_{ij}^{(\ell)}\bigr),
$$
여기서 $p_\ell$은 layer별 응답분포 (이항 logit, Gaussian, Poisson 등). LSIRM의 스칼라 모수 묶음을 $\Theta_{\mathrm{LSI}} = \{a_{1:N},\,\alpha_{i,1:L_{\mathrm{lay}}},\,\beta_{j,1:L_{\mathrm{lay}}},\,\gamma_{1:L_{\mathrm{lay}}},\,\sigma^2_{\alpha,\ell},\,\sigma_0^2,\,\lambda_{ij}^{(2)},\,\kappa_j,\,\tau^{(4)},\,\nu_t\}$로 둔다.

### 1.2 글로벌 풀(global pool)과 MoM 사전

모든 layer의 아이템 위치를 단일 풀로 묶는다:
$$
\{z_q\}_{q=1}^P \;=\; \bigl\{\,b_j^{(\ell)} : \ell=1,\dots,L_{\mathrm{lay}},\; j=1,\dots,J_\ell\,\bigr\},
\qquad P = \sum_{\ell=1}^{L_{\mathrm{lay}}} J_\ell,
$$
즉 $z_q\equiv b_{j(q)}^{(\ell(q))}\in\mathbb R^d$. 풀 인덱스 $q=q(j,\ell)$는 양방향 일대일이다. v16에서는 이 풀 위에 Malsiner‑Walli, Frühwirth‑Schnatter, Grün (2017, JCGS, 이하 **MFG17**)의 sparse hierarchical mixture‑of‑mixtures를 부과한다.

**상위 (cluster) 단계.** $S_q\in\{1,\dots,K\}$, $\Pr(S_q=k\mid \eta_K)=\eta_k$, $\eta_K\sim \mathcal D_K(\gamma_K)$.
**하위 (subcomponent) 단계.** $I_q\in\{1,\dots,L\}$, $\Pr(I_q=l\mid S_q=k,w_k)=w_{kl}$, $w_k\sim \mathcal D_L(d_0)$.
**관측 모형 (풀에 대한).** $z_q\mid S_q=k,\,I_q=l \sim \mathcal N_d(\mu_{kl},\,\Sigma_{kl})$.
**계층적 사전.**
$$
C_{0k}\sim \mathcal W_d(g_0,G_0),\quad b_{0k}\sim \mathcal N_d(m_0,M_0),\quad \lambda_{kj}\stackrel{\text{iid}}{\sim} \mathcal G(\nu,\nu),\;\;\Lambda_k=\mathrm{diag}(\lambda_{k1},\dots,\lambda_{kd}),
$$
$$
\Sigma_{kl}^{-1}\mid c_0,C_{0k}\sim \mathcal W_d(c_0,C_{0k}),\qquad \mu_{kl}\mid b_{0k},B_0,\Lambda_k\sim \mathcal N_d\!\bigl(b_{0k},\;\widetilde B_{0k}\bigr),
$$
$$
\widetilde B_{0k}\;\equiv\;\Lambda_k^{1/2}\,B_0\,\Lambda_k^{1/2}.
$$
MFG17은 $\mathcal W_d(c_0,C_{0k})$의 정의에 Frühwirth‑Schnatter (2006, §6.3.2, p.192)의 관행을 사용한다. 즉, $X\sim \mathcal W_d(c,C)$의 밀도가
$$
p(X)\;\propto\;|X|^{c-(d+1)/2}\,\exp\!\bigl\{-\mathrm{tr}(CX)\bigr\}, \qquad X\succ 0,
$$
이며 $\mathbb E[X]=c\,C^{-1}$. 이 관습 하에서 $X\succ 0$에 $N$개의 $d$‑차원 정규 관측이 추가되면 posterior shape가 $c+N/2$만큼 더해진다(아래 §3.2 참고).

고정 hyperparameter는 v16의 분산‑분해 기반 설정을 따른다:
$$
G_0^{-1} = (1-\phi_W)(1-\phi_B)\bigl(c_0-(d+1)/2\bigr)/g_0\cdot\mathrm{diag}(S_z),\quad
B_0=\phi_W(1-\phi_B)\,\mathrm{diag}(S_z),
$$
$M_0=10\,S_z$, $m_0=\bar z$, $c_0=2.5+(d-1)/2$, $g_0=0.5+(d-1)/2$, $\nu=10$, $L=4$ (혹은 $L=5$), $\phi_B=0.5$, $\phi_W=0.1$. $S_z$는 풀 $\{z_q\}$의 표본공분산.

### 1.3 v16 → v17의 핵심 변경점

| 항목 | v16 (MFG17) | v17 (telescoping) |
|---|---|---|
| 상위 가중분포 | $\eta\sim\mathcal D_K(e_0)$, $e_0=0.001$ 고정, $K=10$ 고정 | $\eta_K\sim\mathcal D_K(\gamma_K)$, $\gamma_K=\alpha/K$, $\alpha\sim\mathcal F(6,3)$, $K-1\sim\mathrm{BNB}(1,4,3)$, $K_{\max}=100$ |
| $K$ 추론 | 사후 비빈 컴포넌트 수 $K_+$ 의 mode | dynamic MFM, $K$ 자체를 매 iteration sampling, $K_+$는 분할에서 결정 |
| 하위 단계 | $L$ 고정, $w_k\sim\mathcal D_L(d_0)$ | 동일 ($L$은 telescoping 적용 외) |
| 빈 클러스터 처리 | sparsity prior로 자연 비움 | 매 iteration $K-K_+$개 빈 클러스터를 사전에서 새로 추출 |

### 1.4 표기 통일 (MFG17 ↔ FSMG21 ↔ v17)

MFG17은 데이터차원을 $r$, 클러스터 모수차원을 $d$로 적었다. v17에서는 풀 데이터 $z_q\in\mathbb R^d$의 차원을 $d$ (MFG17의 $r$에 해당)로 통일하고, MFG17의 모수차원 $d$ (v16에서 $\zeta$)는 본 문서에서 명시할 필요가 없다 (Rousseau–Mengersen 자취가 dynamic MFM에서는 $\gamma_K=\alpha/K$로 자동 통제되기 때문이다). FSMG21의 $\gamma_K$ sequence에서 $\gamma_K=\alpha/K$가 dynamic 사양이고 $\alpha$가 hyperparameter이다.

---

## Part 2. 결합 사후의 인수분해

전 모수
$$
\Psi \;=\;\bigl(K,\alpha,\,S,\,I,\,\eta_K,\,\{w_k\}_{k=1}^K,\,\{\mu_{kl},\Sigma_{kl}\}_{k\le K, l\le L},\,\{b_{0k},C_{0k},\Lambda_k\}_{k=1}^K,\,\Theta_{\mathrm{LSI}}\bigr).
$$
관측을 $Y=\{Y_{ij}^{(\ell)}\}$. 결합 사후:
$$
\begin{aligned}
p(\Psi\mid Y)\;\propto\;& \underbrace{\prod_{\ell,i,j} p_\ell\!\bigl(Y_{ij}^{(\ell)}\mid \eta_{ij}^{(\ell)}\bigr)}_{\text{LSIRM 우도}}\;\cdot\;\underbrace{\prod_{q=1}^P \mathcal N_d\!\bigl(z_q\mid \mu_{S_q I_q},\Sigma_{S_q I_q}\bigr)}_{\text{풀 우도}}\\
&\cdot\;\prod_{q=1}^P \eta_{S_q} w_{S_q I_q}\;\cdot\;\mathcal D_K(\eta_K\mid \gamma_K \mathbf 1)\;\cdot\;\prod_{k=1}^K \mathcal D_L(w_k\mid d_0\mathbf 1)\\
&\cdot\;\prod_{k=1}^K\Bigl[\mathcal W_d(C_{0k}\mid g_0,G_0)\,\mathcal N_d(b_{0k}\mid m_0,M_0)\,\prod_{j=1}^d\mathcal G(\lambda_{kj}\mid\nu,\nu)\Bigr]\\
&\cdot\;\prod_{k,l}\Bigl[\mathcal W_d(\Sigma_{kl}^{-1}\mid c_0,C_{0k})\,\mathcal N_d\!\bigl(\mu_{kl}\mid b_{0k},\widetilde B_{0k}\bigr)\Bigr]\\
&\cdot\;\mathcal F_{6,3}(\alpha)\cdot p_{\mathrm{BNB}}(K-1\mid 1,4,3)\cdot p(\Theta_{\mathrm{LSI}}).
\end{aligned}
\tag{2.1}
$$

### 2.1 분할에 대한 조건부 독립

분할 $\mathcal C=\{C_1,\dots,C_{K_+}\}$를 indicator $S$의 동치류로 정의하자. 두 가지 형태의 사후가 자연스럽게 등장한다 (FSMG21, eqs.(5.4)–(5.5)):

**완전 augmented 사후 (mixture posterior):**
$$
p\bigl(K,S,\eta_K,\{\theta_k\},\phi,\alpha\mid Y\bigr)
\;\propto\;
\prod_{k:N_k>0} p(y_{[k]}\mid \theta_k)\,p(\theta_k\mid\phi)\,
\prod_{k:N_k=0} p(\theta_k\mid\phi)\;
\prod_{k=1}^K \eta_k^{N_k+\gamma_K-1}\;\frac{\Gamma(K\gamma_K)}{\Gamma(\gamma_K)^K}\,p(\phi)\,p(K)\,p(\alpha),
\tag{2.2}
$$
여기서 $\theta_k=(w_k,\{\mu_{kl}\},\{\Sigma_{kl}\})$, $\phi=(b_{0k},C_{0k},\Lambda_k)_{k=1}^K$.

**부분‑collapsed 사후 (set‑partition posterior):** $\eta_K$, $S\to\mathcal C$의 동치류 합산, 빈 클러스터 모수 적분 후
$$
p\bigl(K,\mathcal C,\{\theta_k\}_{k\le K_+},\phi_{1:K_+},\alpha\mid Y\bigr)
\;\propto\;\prod_{k=1}^{K_+} p(y_{[k]}\mid\theta_k)p(\theta_k\mid\phi_k)p(\phi_k)\;\cdot\;p(\mathcal C\mid N,K,\gamma_K)\,p(K)\,p(\alpha),
\tag{2.3}
$$
$$
p(\mathcal C\mid N,K,\gamma_K)\;=\;\frac{V^{K,\gamma_K}_{N,K_+}}{\Gamma(\gamma_K)^{K_+}}\prod_{j=1}^{K_+}\Gamma(N_j+\gamma_K),\qquad V^{K,\gamma_K}_{N,K_+}=\frac{\Gamma(\gamma_K K)\,K!}{\Gamma(\gamma_K K+N)\,(K-K_+)!}.
\tag{2.4}
$$
(FSMG21 eq.(2.4).) Telescoping sampler는 (2.2) 와 (2.3) 사이를 왔다갔다 하며 **$K$를 (2.3)으로부터 sampling** 한 뒤 빈 컴포넌트를 (2.2)로 채워 넣는 partially collapsed 구조이다 (FSMG21 §5).

---

## Part 3. Telescoping 블록의 단계별 도출 (FSMG21 Algorithm 2의 LSIRM‑MoM 적응)

표기: $z_q$의 풀, $N=P$, $N_k=\#\{q:S_q=k\}$, $N_{kl}=\#\{q:S_q=k,I_q=l\}$, $K_+=\sum_{k=1}^K \mathbb 1\{N_k>0\}$.

### Step 3.1. 두‑수준 할당 $(S_q,I_q)$의 갱신

조건부 사후
$$
\Pr(S_q=k,I_q=l\mid\text{rest})\;\propto\;\eta_k\,w_{kl}\,\mathcal N_d\!\bigl(z_q\mid\mu_{kl},\Sigma_{kl}\bigr),\qquad k\le K,\;l\le L.
\tag{3.1}
$$

**도출.** (2.1)에서 $S_q$, $I_q$를 포함하는 인수만 모으면 $\eta_{S_q}\,w_{S_q I_q}\,\mathcal N(z_q\mid\mu_{S_q I_q},\Sigma_{S_q I_q})$이며, 다른 모든 $q'\ne q$에 대한 항은 $S_q,I_q$에 의존하지 않는다. 따라서 $K\!\cdot\!L$개의 좌표에서 normalization 후 정의된 categorical 분포에서 추출한다.

**동등한 collapsed 형태.** $I_q$를 marginalize:
$$
\Pr(S_q=k\mid\text{rest})\;\propto\;\eta_k\sum_{l=1}^L w_{kl}\,\mathcal N_d(z_q\mid\mu_{kl},\Sigma_{kl}),
\tag{3.2a}
$$
$$
\Pr(I_q=l\mid S_q=k,\text{rest})\;\propto\; w_{kl}\,\mathcal N_d(z_q\mid\mu_{kl},\Sigma_{kl}).
\tag{3.2b}
$$
(3.1)과 (3.2)는 동일한 marginal/conditional이며 numerical underflow 통제를 위해서는 (3.2)의 두 단계 수행이 권장된다. 갱신 후 $N_k,K_+$를 다시 계산하고 비빈 클러스터가 $1,\dots,K_+$가 되도록 라벨을 재정렬한다.

### Step 3.2. 비빈 클러스터의 subcomponent 모수 $(\mu_{kl},\Sigma_{kl})$ 갱신

**중요:** v16의 사전은 일반적인 normal–inverse‑Wishart 결합 사전이 아니다. $\Sigma_{kl}\sim\mathcal{IW}_d(c_0,C_{0k})$와 $\mu_{kl}\sim\mathcal N_d(b_{0k},\widetilde B_{0k})$가 *독립적으로* 부여되며, 따라서 결합 NIW 갱신이 아니라 **순차적** Gibbs로 갱신한다.

#### 3.2.1 $\Sigma_{kl}^{-1}\mid \text{rest}$의 도출

$z_{[kl]}=\{z_q : S_q=k, I_q=l\}$, $|z_{[kl]}|=N_{kl}$. 사전 (Frühwirth‑Schnatter 2006 관습):
$$
p(\Sigma_{kl}^{-1}\mid c_0,C_{0k})\;\propto\;|\Sigma_{kl}^{-1}|^{c_0-(d+1)/2}\exp\!\bigl\{-\mathrm{tr}(C_{0k}\Sigma_{kl}^{-1})\bigr\}.
$$
우도 부분 (관측 $\mu_{kl}$ 조건부):
$$
\prod_{q\in z_{[kl]}}\mathcal N_d(z_q\mid\mu_{kl},\Sigma_{kl})
\;\propto\;|\Sigma_{kl}^{-1}|^{N_{kl}/2}\exp\!\Bigl\{-\tfrac12\sum_{q}\mathrm{tr}\bigl((z_q-\mu_{kl})(z_q-\mu_{kl})^\top\Sigma_{kl}^{-1}\bigr)\Bigr\}.
$$
trace를 통합한 뒤 사전과 곱하고 지수 위 인자를 합치면
$$
p(\Sigma_{kl}^{-1}\mid \text{rest})
\;\propto\;|\Sigma_{kl}^{-1}|^{c_0+N_{kl}/2-(d+1)/2}\exp\!\Bigl\{-\mathrm{tr}\Bigl[\bigl(C_{0k}+\tfrac12\textstyle\sum_q (z_q-\mu_{kl})(z_q-\mu_{kl})^\top\bigr)\Sigma_{kl}^{-1}\Bigr]\Bigr\}.
$$
따라서
$$
\boxed{\;\Sigma_{kl}^{-1}\mid\text{rest}\;\sim\;\mathcal W_d\!\Bigl(\,c_0+\tfrac{N_{kl}}{2},\;\;C_{0k}+\tfrac12\!\!\sum_{q:S_q=k,I_q=l}\!\!(z_q-\mu_{kl})(z_q-\mu_{kl})^\top\Bigr).\;}
\tag{3.3}
$$

#### 3.2.2 $\mu_{kl}\mid \text{rest}$의 도출

사전 $\mathcal N_d(\mu_{kl}\mid b_{0k},\widetilde B_{0k})$, 우도 $\prod_q \mathcal N_d(z_q\mid\mu_{kl},\Sigma_{kl})$. 두 정규를 곱하면:
$$
-\tfrac12 (\mu_{kl}-b_{0k})^\top \widetilde B_{0k}^{-1}(\mu_{kl}-b_{0k})\;-\;\tfrac12\sum_{q\in z_{[kl]}}(z_q-\mu_{kl})^\top\Sigma_{kl}^{-1}(z_q-\mu_{kl}).
$$
$\mu_{kl}$에 대해 2차식 인자를 정리:
$$
\mu_{kl}^\top\bigl(\widetilde B_{0k}^{-1}+N_{kl}\Sigma_{kl}^{-1}\bigr)\mu_{kl}\;-\;2\mu_{kl}^\top\bigl(\widetilde B_{0k}^{-1}b_{0k}+\Sigma_{kl}^{-1}\!\!\sum_{q\in z_{[kl]}}\!z_q\bigr) + \text{const}.
$$
완전제곱:
$$
\boxed{\;\mu_{kl}\mid\text{rest}\;\sim\;\mathcal N_d(b_{kl},\,B_{kl}),\;\;
B_{kl}=\bigl(\widetilde B_{0k}^{-1}+N_{kl}\Sigma_{kl}^{-1}\bigr)^{-1},\;\;
b_{kl}=B_{kl}\bigl(\widetilde B_{0k}^{-1}b_{0k}+\Sigma_{kl}^{-1}N_{kl}\bar z_{kl}\bigr),\;}
\tag{3.4}
$$
$\bar z_{kl}=N_{kl}^{-1}\sum_{q\in z_{[kl]}} z_q$. 빈 subcomponent ($N_{kl}=0$)에서는 (3.3)–(3.4)가 사전으로 환원된다.

### Step 3.3. 클러스터별 random hyperparameter $(b_{0k},C_{0k},\Lambda_k)$ 갱신 (filled $k$)

#### 3.3.1 $\lambda_{kj}\mid\text{rest}$ 가 GIG임의 자세한 도출

$\widetilde B_{0k}=\Lambda_k^{1/2}B_0\Lambda_k^{1/2}$, $\Lambda_k=\mathrm{diag}(\lambda_{k1},\dots,\lambda_{kd})$. 따라서 $(\widetilde B_{0k})_{jj'}=\sqrt{\lambda_{kj}\lambda_{kj'}}\,(B_0)_{jj'}$. v16의 표준 운용에서 $B_0$는 대각 ($B_0=\mathrm{diag}(B_{0,11},\dots,B_{0,dd})$, §1.2)이므로 $\widetilde B_{0k}$도 대각, $(\widetilde B_{0k})_{jj}=\lambda_{kj}\,B_{0,jj}$. 그러면 $\mu_{kl,j}\mid b_{0k,j},\lambda_{kj}\sim \mathcal N(b_{0k,j},\,\lambda_{kj}B_{0,jj})$, 좌표 $j$별로 독립이다.

모든 $L$개의 $\mu_{kl,j}$가 기여하는 $\lambda_{kj}$ 의 kernel:
$$
\prod_{l=1}^L \mathcal N(\mu_{kl,j}\mid b_{0k,j},\,\lambda_{kj}B_{0,jj})
\;\propto\;(\lambda_{kj}B_{0,jj})^{-L/2}\exp\!\Bigl\{-\frac{1}{2\lambda_{kj}B_{0,jj}}\sum_{l=1}^L (\mu_{kl,j}-b_{0k,j})^2\Bigr\}.
$$
$\lambda_{kj}$의 사전 $\mathcal G(\nu,\nu)$:
$$
p(\lambda_{kj})\propto \lambda_{kj}^{\nu-1}\exp(-\nu\lambda_{kj}).
$$
두 항을 곱하면
$$
p(\lambda_{kj}\mid\text{rest})\;\propto\;\lambda_{kj}^{(\nu-L/2)-1}\,\exp\!\Bigl\{-\tfrac12\bigl(2\nu\,\lambda_{kj}+\frac{b_{kj}}{\lambda_{kj}}\bigr)\Bigr\},\qquad b_{kj}\equiv \frac{1}{B_{0,jj}}\sum_{l=1}^L(\mu_{kl,j}-b_{0k,j})^2.
$$
v16/MFG17이 채택한 GIG 정의 $f(x)\propto x^{p-1}\exp\{-\tfrac12(ax+b/x)\}$ (Devroye 1986; Hörmann & Leydold 2014; Peña & Jauch 2024)와 정확히 매칭된다. 따라서
$$
\boxed{\;\lambda_{kj}\mid \text{rest}\;\sim\;\mathrm{GIG}\!\bigl(p_{kL},\,a_{kj},\,b_{kj}\bigr),\qquad p_{kL}=\nu-L/2,\;\;a_{kj}=2\nu,\;\;b_{kj}=\sum_{l=1}^L\frac{(\mu_{kl,j}-b_{0k,j})^2}{B_{0,jj}}.\;}
\tag{3.5}
$$

**왜 GIG인가? (Gamma가 아닌 이유.)** 주의해야 할 점은 $\sqrt{\lambda_{kj}}$가 정규밀도의 분산 양쪽에 등장한다는 점이다 ($\widetilde B_{0k}=\Lambda_k^{1/2}B_0\Lambda_k^{1/2}$). 이로 인해 $\lambda_{kj}$의 양수 기여 $\lambda_{kj}^{-L/2}\exp(-b_{kj}/(2\lambda_{kj}))$ (정규밀도의 정규화상수+이차항에서 $1/\lambda_{kj}$가 등장)와 $\mathcal G(\nu,\nu)$ 사전의 $\lambda_{kj}^{\nu-1}\exp(-\nu\lambda_{kj})$가 함께 작용하여, $\lambda_{kj}$와 $1/\lambda_{kj}$이 모두 지수에 들어가는 GIG 커널이 된다. 단순히 $\lambda$가 분산 안에서 *곱셈*만 되는 형태였다면 inverse‑gamma로 환원되었겠으나, normal‑gamma 형 prior에서는 본 hyperparameter 결합이 정확히 GIG를 유도한다 (Frühwirth‑Schnatter 2011; MFG17 Appendix A).

샘플링: Devroye (1986) ratio‑of‑uniforms 또는 Hörmann–Leydold (2014) automatic algorithm. 빈 클러스터 ($N_k=0$)에서는 (3.5)가 사전 $\mathcal G(\nu,\nu)$로 환원된다 (왜냐하면 $\lambda_{kj}$ 사후가 그 클러스터의 $\mu_{kl,j}$ 사후에만 의존하는데, 빈 클러스터에서는 $\mu_{kl}$이 사전에서 새로 뽑히므로 marginalize하면 정확히 사전이 복원된다).

#### 3.3.2 $C_{0k}\mid\text{rest}$의 도출 (Wishart–Wishart conjugacy)

$C_{0k}$가 등장하는 항 (사전 + 모든 $l$의 $\Sigma_{kl}^{-1}$ 사전):
$$
p(C_{0k}\mid \text{rest})\;\propto\;\underbrace{|C_{0k}|^{g_0-(d+1)/2}\exp(-\mathrm{tr}(G_0 C_{0k}))}_{\text{prior}}\;\cdot\;\prod_{l=1}^L \underbrace{|C_{0k}|^{c_0}\exp(-\mathrm{tr}(C_{0k}\Sigma_{kl}^{-1}))}_{\Sigma_{kl}^{-1}\text{의 prior에서 }C_{0k}\text{‑종속 부분}}.
$$
(주의: $\mathcal W_d(c_0,C_{0k})$의 normalization $|C_{0k}|^{c_0}$가 $C_{0k}$에 의존하는 부분으로서 사후에 들어옴.) 곱하여 정리:
$$
p(C_{0k}\mid \text{rest})\;\propto\;|C_{0k}|^{(g_0+Lc_0)-(d+1)/2}\exp\!\Bigl\{-\mathrm{tr}\Bigl(\bigl(G_0+\sum_{l=1}^L\Sigma_{kl}^{-1}\bigr)C_{0k}\Bigr)\Bigr\}.
$$
따라서
$$
\boxed{\;C_{0k}\mid\text{rest}\;\sim\;\mathcal W_d\Bigl(\,g_0+L\,c_0,\;\;G_0+\sum_{l=1}^L \Sigma_{kl}^{-1}\,\Bigr).\;}
\tag{3.6}
$$
빈 클러스터에서는 사전 $\mathcal W_d(g_0,G_0)$로 환원.

#### 3.3.3 $b_{0k}\mid\text{rest}$의 도출 (Normal–Normal conjugacy)

$b_{0k}$ 등장 항: 사전 $\mathcal N_d(b_{0k}\mid m_0,M_0)$, 그리고 $\mu_{kl}\mid b_{0k},\widetilde B_{0k}\sim \mathcal N_d(b_{0k},\widetilde B_{0k})$ ($l=1,\dots,L$). $\widetilde B_{0k}$는 $b_{0k}$에 의존하지 않는다 (조건부에서 $\Lambda_k$ 고정). 두 이차형을 합쳐
$$
-\tfrac12(b_{0k}-m_0)^\top M_0^{-1}(b_{0k}-m_0)\;-\;\tfrac12\sum_{l=1}^L (\mu_{kl}-b_{0k})^\top \widetilde B_{0k}^{-1}(\mu_{kl}-b_{0k}).
$$
$b_{0k}$의 quadratic term: $b_{0k}^\top(M_0^{-1}+L\widetilde B_{0k}^{-1})b_{0k}$, linear term: $2b_{0k}^\top(M_0^{-1}m_0+\widetilde B_{0k}^{-1}\sum_l\mu_{kl})$. 완전제곱:
$$
\boxed{\;b_{0k}\mid\text{rest}\;\sim\;\mathcal N_d(\widetilde m_k,\,\widetilde M_k),\;\;
\widetilde M_k=\bigl(M_0^{-1}+L\widetilde B_{0k}^{-1}\bigr)^{-1},\;\;
\widetilde m_k=\widetilde M_k\bigl(M_0^{-1}m_0+\widetilde B_{0k}^{-1}\!\!\textstyle\sum_{l=1}^L\!\mu_{kl}\bigr).\;}
\tag{3.7}
$$

### Step 3.4. $K\mid\mathcal C,\alpha$의 갱신 — telescoping의 핵심

dynamic MFM ($\gamma_K=\alpha/K$)에서 (2.4)와 (2.1)의 결합 사후 — $\theta_k$ 와 $\eta_K$가 *분할 $\mathcal C$ 조건부로 $K$와 독립*임을 이용하여 — FSMG21 Theorem 2.1과 식 (5.1)로부터:
$$
p(K\mid\mathcal C,\alpha)\;\propto\;p(K)\,\frac{K!}{(K-K_+)!}\,\frac{\Gamma(\alpha)}{\Gamma(N+\alpha)}\,\prod_{k=1}^{K_+}\frac{\Gamma(N_k+\alpha/K)}{\Gamma(1+\alpha/K)}\cdot\frac{\bigl(\Gamma(\alpha/K)\bigr)^{-K_+}\Gamma(\alpha/K \cdot K)}{\bigl(\Gamma(\alpha/K)\bigr)^{-K_+}\Gamma(\alpha)}.
$$
$\Gamma(\alpha/K\cdot K)=\Gamma(\alpha)$이므로 단순화하면
$$
\boxed{\;p(K\mid\mathcal C,\alpha)\;\propto\;p(K)\,\frac{\alpha^{K_+}\,K!}{K^{K_+}\,(K-K_+)!}\,\prod_{k=1}^{K_+}\frac{\Gamma(N_k+\alpha/K)}{\Gamma(1+\alpha/K)},\quad K=K_+,K_++1,\dots,K_{\max}.\;}
\tag{3.8}
$$
(여기서는 $\Gamma(N_k+\alpha/K)/\Gamma(\alpha/K)\cdot \Gamma(\alpha/K)/\Gamma(1+\alpha/K)\cdot\alpha/K$ 등의 변형을 모아 $\alpha^{K_+}/K^{K_+}$로 정리. FSMG21 Algorithm 2의 Step 3(a)와 동일.) **유도의 출발은 (2.4)에 $\gamma_K=\alpha/K$를 대입한 후, $\Gamma(\gamma_K K)=\Gamma(\alpha)$가 $K$에 무관함, $V^{K,\alpha/K}_{N,K_+}=\Gamma(\alpha)\,K!/[\Gamma(\alpha+N)(K-K_+)!]$, 그리고 $\prod_{k}\Gamma(N_k+\alpha/K)/\Gamma(\alpha/K)^{K_+}$를 $\Gamma(N_k+\alpha/K)/\Gamma(1+\alpha/K)\cdot (\alpha/K)$로 재구성하는 것이다.**

**샘플링.** $K\in\{K_+,K_++1,\dots,K_{\max}\}$ 위의 다항분포로부터 직접 표본추출. 사용한 $p(K)$는 BNB(1,4,3) 사전:
$$
p(K)=\frac{\Gamma(1+K-1)\,B(1+4,\,K-1+3)}{\Gamma(1)\,\Gamma(K)\,B(4,3)}=\frac{B(5,K+2)}{B(4,3)},\qquad K=1,2,\dots,K_{\max}.
\tag{3.9}
$$
실용 상한 $K_{\max}=100$.

**왜 mixing이 좋아지는가?** $K$의 사후 (3.8)은 컴포넌트 모수 $\{\mu_{kl},\Sigma_{kl},w_k,b_{0k},C_{0k},\Lambda_k\}$에 *완전히 무관하고 분할 $\mathcal C$에만 의존*한다. 이는 MFG17의 Gibbs (모든 hyperparameter‑포함 포스테리어를 동시에 다루는 sparsity 기반 자연 비우기)와 본질적으로 다르다. (i) 새 $K$가 sample 되면 $K-K_+$개의 빈 클러스터를 prior에서 신선하게 추출하여 (Step 3.6) 매 iteration "탄생 슬롯"을 마련하므로 RJMCMC 스타일의 어색한 dimension‑match가 필요없고, (ii) $K$의 변동이 점유 모수와 분리되어 chain의 $K_+$ 탐색이 빨라진다.

### Step 3.5. $\alpha\mid\mathcal C,K$의 갱신 — 로그 척도 RWMH

분할 사후로부터 $\alpha$가 등장하는 인자만 모으면 (FSMG21 Algorithm 2 Step 3(b))
$$
p(\alpha\mid\mathcal C,K)\;\propto\;p(\alpha)\,\frac{\alpha^{K_+}\,\Gamma(\alpha)}{\Gamma(N+\alpha)}\,\prod_{k=1}^{K_+}\frac{\Gamma(N_k+\alpha/K)}{\Gamma(1+\alpha/K)}.
\tag{3.10}
$$
사전 $\alpha\sim\mathcal F(\nu_l,\nu_r)$ with $(\nu_l,\nu_r)=(6,3)$ (FSMG21 §4.3):
$$
p_{\mathcal F_{6,3}}(\alpha)\;=\;\frac{\Gamma((\nu_l+\nu_r)/2)}{\Gamma(\nu_l/2)\Gamma(\nu_r/2)}\Bigl(\frac{\nu_l}{\nu_r}\Bigr)^{\nu_l/2}\alpha^{\nu_l/2-1}\Bigl(1+\frac{\nu_l\,\alpha}{\nu_r}\Bigr)^{-(\nu_l+\nu_r)/2}
\;=\;\frac{\Gamma(4.5)}{\Gamma(3)\Gamma(1.5)}\,2^3\,\alpha^{2}\bigl(1+2\alpha\bigr)^{-4.5}.
\tag{3.11}
$$

**제안.** $\xi=\log\alpha$, $\xi'\sim\mathcal N(\xi^{\mathrm{cur}},s_\alpha^2)$, $\alpha^{\mathrm{prop}}=\exp\xi'$. 변환 $\alpha=e^\xi$, $|d\alpha/d\xi|=\alpha$. $\xi$의 target은 $\pi(\xi)=p(\alpha\mid\mathcal C,K)\,|d\alpha/d\xi|=p(\alpha\mid\mathcal C,K)\cdot\alpha$. 가우시안 제안은 대칭이므로 acceptance 확률:
$$
\boxed{\;A_\alpha=\min\!\left\{1,\;\frac{p\bigl(\alpha^{\mathrm{prop}}\mid\mathcal C,K\bigr)\,\alpha^{\mathrm{prop}}}{p\bigl(\alpha^{\mathrm{cur}}\mid\mathcal C,K\bigr)\,\alpha^{\mathrm{cur}}}\right\}.\;}
\tag{3.12}
$$
log 스케일 Jacobian 인자 $\alpha^{\mathrm{prop}}/\alpha^{\mathrm{cur}}$를 잊지 말 것. $s_\alpha$는 적응적 튜닝 (target acceptance rate $\approx 0.234$).

### Step 3.6. 빈 컴포넌트 추가 및 $\eta_K,w_k$ 갱신

새로 sample된 $K\ge K_+$가 $K>K_+$이면 다음을 차례로 수행 (FSMG21 Algorithm 2 Step 4):

1. $k=K_++1,\dots,K$에 대해 사전에서 추출:
$$
C_{0k}\sim \mathcal W_d(g_0,G_0),\quad b_{0k}\sim\mathcal N_d(m_0,M_0),\quad \lambda_{kj}\stackrel{\text{iid}}{\sim}\mathcal G(\nu,\nu).
$$
2. 위로부터 $\widetilde B_{0k}=\Lambda_k^{1/2}B_0\Lambda_k^{1/2}$, 그리고 $l=1,\dots,L$에 대해
$$
\Sigma_{kl}^{-1}\sim\mathcal W_d(c_0,C_{0k}),\qquad \mu_{kl}\sim\mathcal N_d(b_{0k},\widetilde B_{0k}),\qquad w_k\sim\mathcal D_L(d_0\mathbf 1).
$$
3. 가중치 $\eta_K$를 Dirichlet 결합에서 추출. $K$ 컴포넌트 중 $K_+$개는 $N_k>0$, 나머지는 $N_k=0$이다:
$$
\boxed{\;\eta_K\mid K,\alpha,S\;\sim\;\mathcal D_K\!\bigl(\alpha/K+N_1,\dots,\alpha/K+N_K\bigr).\;}
\tag{3.13}
$$
**도출.** $\eta_K$의 사후는 $\eta_K\sim\mathcal D_K(\gamma_K)$ 사전과 다항 우도 $\prod_q \eta_{S_q}$의 곱에서 cluster $k$별 횟수가 $N_k$이므로 Dirichlet–Multinomial 결합으로 (3.13)이 즉각.
4. $k=1,\dots,K$에 대해
$$
w_k\mid I,S\sim\mathcal D_L(d_0+N_{k1},\dots,d_0+N_{kL})\quad (\text{filled }k);\qquad w_k\sim\mathcal D_L(d_0)\quad (\text{empty }k\text{; 위 step 2에서 이미 추출됨}).
\tag{3.14}
$$

---

## Part 4. 아이템 위치 $b_j^{(\ell)}$의 collapsed Metropolis–Hastings 갱신

LSIRM 선형예측자 (재명시):
$$
\eta_{ij}^{(\ell)}=\alpha_i^{(\ell)}-\beta_j^{(\ell)}-\gamma_\ell\bigl\lVert a_i-b_j^{(\ell)}\bigr\rVert_2.
$$
풀 인덱스 $q=q(j,\ell)$, 즉 $z_q=b_j^{(\ell)}$.

대상 full conditional:
$$
p\bigl(b_j^{(\ell)}\mid\text{rest}\bigr)\;\propto\;\underbrace{\prod_{i=1}^N p_\ell\!\bigl(Y_{ij}^{(\ell)}\mid \eta_{ij}^{(\ell)}\bigr)}_{\mathcal L_\ell\bigl(b_j^{(\ell)}\bigr)}\cdot\;\pi\bigl(b_j^{(\ell)}\mid\text{rest}\bigr),
\tag{4.1}
$$
$\pi(b_j^{(\ell)}\mid\text{rest})$는 v17의 MoM 사전이 $b_j^{(\ell)}=z_q$에 대해 유도하는 conditional prior이다.

**Variant A — full marginalization of $(S_q,I_q)$.**
$(S_q,I_q)$를 augment된 결합에서 합산:
$$
\pi_A\bigl(b_j^{(\ell)}\mid\text{rest}\bigr)
=\sum_{k=1}^K\sum_{l=1}^L \Pr(S_q=k,I_q=l\mid\text{others})\,\mathcal N_d\bigl(b_j^{(\ell)}\mid\mu_{kl},\Sigma_{kl}\bigr)
=\sum_{k=1}^K\sum_{l=1}^L \eta_k\,w_{kl}\,\mathcal N_d\bigl(b_j^{(\ell)}\mid\mu_{kl},\Sigma_{kl}\bigr).
\tag{4.2}
$$
**도출.** (2.1)에서 $z_q,S_q,I_q$를 포함하는 인자만 남기면 $\eta_{S_q}w_{S_q I_q}\,\mathcal N(z_q\mid\mu_{S_q I_q},\Sigma_{S_q I_q})$. $(S_q,I_q)$를 marginalize:
$$
\sum_{k,l}\eta_k w_{kl}\mathcal N(z_q\mid\mu_{kl},\Sigma_{kl}).
$$
이것이 $z_q=b_j^{(\ell)}$에 대한 prior 항이다. 다른 어떤 파라미터에도 $b_j^{(\ell)}$이 등장하지 않으므로 (LSIRM에서는 $\eta_{ij}^{(\ell)}$만이 $b_j^{(\ell)}$의 함수) (4.2)와 LSIRM 우도의 곱이 (4.1)이다.

**Variant B — partial marginalization (only $I_q$):**
$S_q$는 현재 값에 고정, $I_q$만 합산:
$$
\pi_B\bigl(b_j^{(\ell)}\mid\text{rest},S_q\bigr)
=\sum_{l=1}^L w_{S_q,l}\,\mathcal N_d\bigl(b_j^{(\ell)}\mid\mu_{S_q,l},\Sigma_{S_q,l}\bigr).
\tag{4.3}
$$
**도출.** (2.1)에서 $z_q,I_q$를 포함하는 부분만, $S_q$ 고정 하에서:
$$
\Pr(I_q=l\mid S_q,w_{S_q})\,\mathcal N(z_q\mid\mu_{S_q l},\Sigma_{S_q l})=w_{S_q l}\,\mathcal N(z_q\mid\mu_{S_q l},\Sigma_{S_q l}).
$$
$I_q$를 합산하면 (4.3) 즉시 도출.

### 4.1 RWMH 제안과 acceptance

$\rho_\ell$은 layer별 step size. 대칭 가우시안 제안:
$$
b_j^{(\ell),\mathrm{prop}}\;\sim\;\mathcal N_d\bigl(b_j^{(\ell),\mathrm{cur}},\,\rho_\ell^2 I_d\bigr).
$$

**Variant A 수용확률:**
$$
\boxed{\;A_A=\min\!\left\{1,\;\frac{\mathcal L_\ell\bigl(b_j^{(\ell),\mathrm{prop}}\bigr)\,\pi_A\bigl(b_j^{(\ell),\mathrm{prop}}\mid\text{rest}\bigr)}{\mathcal L_\ell\bigl(b_j^{(\ell),\mathrm{cur}}\bigr)\,\pi_A\bigl(b_j^{(\ell),\mathrm{cur}}\mid\text{rest}\bigr)}\right\}.\;}
\tag{4.4}
$$
대칭 제안이므로 proposal density가 상쇄된다.

**Variant B 수용확률:**
$$
\boxed{\;A_B=\min\!\left\{1,\;\frac{\mathcal L_\ell\bigl(b_j^{(\ell),\mathrm{prop}}\bigr)\,\pi_B\bigl(b_j^{(\ell),\mathrm{prop}}\mid\text{rest},S_q\bigr)}{\mathcal L_\ell\bigl(b_j^{(\ell),\mathrm{cur}}\bigr)\,\pi_B\bigl(b_j^{(\ell),\mathrm{cur}}\mid\text{rest},S_q\bigr)}\right\}.\;}
\tag{4.5}
$$
Variant B를 사용할 때는 $b_j^{(\ell)}$의 MH 후 Step 3.1의 통상적인 할당 갱신이 따로 수행되어 $S_q$가 갱신된다. Variant A는 prior 항이 이미 $S_q$에 대해 marginalized되어 있으므로 $b_j$ 이동이 stale한 $S_q$에 갇히지 않는다 (그러나 Step 3.1은 라벨 일관성 유지를 위해 어쨌든 매 iteration 수행한다).

### 4.2 두 변형의 trade‑off

| | Variant A | Variant B |
|---|---|---|
| MH당 평가비용 | $K\!\cdot\! L$개의 $d$차원 가우시안 밀도 | $L$개 가우시안 밀도 |
| Mixing | $b_j$와 $S_q$의 결합에서 최적, mode‑hopping 가능 | locality 보존, $S_q$가 잘못된 모드에 있으면 bottleneck |
| 권장 사용 | 데이터 다중모달, 풀이 작거나 K가 작을 때 ($P\le 200$, $K\le 30$) | 풀이 매우 크거나 ($P\gg 1000$) compute‑bound일 때 |
| 구현 안정성 | log‑sum‑exp 안전화 필수 | 동일하게 권장 |

### 4.3 알고리즘 박스

```
Algorithm 4A (Variant A: Full collapsed b_j MH)
Input: current state, layer ℓ, item j, with q = q(j,ℓ)
1.  b_prop ~ N_d(b_cur, ρ_ℓ^2 I_d)
2.  log_lik_cur ← Σ_i log p_ℓ(Y_ij^(ℓ) | η_ij^(ℓ)(b_cur))
    log_lik_prop ← Σ_i log p_ℓ(Y_ij^(ℓ) | η_ij^(ℓ)(b_prop))
3.  log_pri_cur ← logsumexp_{k,l}{log η_k + log w_kl + log φ_d(b_cur ; μ_kl, Σ_kl)}
    log_pri_prop ← logsumexp_{k,l}{log η_k + log w_kl + log φ_d(b_prop ; μ_kl, Σ_kl)}
4.  log A ← (log_lik_prop + log_pri_prop) − (log_lik_cur + log_pri_cur)
5.  u ~ U(0,1); if log u < log A then accept b_prop else keep b_cur

Algorithm 4B (Variant B: Partial collapsed b_j MH, S_q fixed)
1.  b_prop ~ N_d(b_cur, ρ_ℓ^2 I_d)
2.  log_lik_*  computed as in 4A
3.  k* ← S_q (current cluster assignment)
    log_pri_cur ← logsumexp_{l}{log w_{k*,l} + log φ_d(b_cur ; μ_{k*,l}, Σ_{k*,l})}
    log_pri_prop ← logsumexp_{l}{log w_{k*,l} + log φ_d(b_prop ; μ_{k*,l}, Σ_{k*,l})}
4.  log A ← (log_lik_prop + log_pri_prop) − (log_lik_cur + log_pri_cur)
5.  accept/reject as in 4A
```
$\phi_d(\cdot;\mu,\Sigma)$는 $d$‑변량 정규밀도. step size $\rho_\ell$은 layer별 적응 (target $\approx 0.234$).

---

## Part 5. 다른 LSIRM 블록 (요약)

다음 블록은 v13/v15와 동일하며 telescoping 통합과 *완전히 분리*된다 — telescoping 단계는 풀 데이터 $\{z_q\}$의 cluster 구조에만 작용하고, LSIRM scalar 갱신은 (4.1) 이외에는 $\{S_q,I_q,K\}$에 의존하지 않기 때문이다. 도출 생략.

- **응답자 위치 $a_i$**: RWMH; 모든 $\ell,j$의 LSIRM 우도와 $a_i\sim\mathcal N_d(0,\sigma_0^2 I_d)$ 사전 (또는 $\mathcal N_d(0,\Sigma_a)$).
- **응답자 능력 $\alpha_i^{(\ell)}$**: 통상의 정규–정규 conjugate 또는 RWMH (응답분포에 따라).
- **아이템 난이도 $\beta_j^{(\ell)}$**: RWMH 또는 conjugate.
- **거리 가중 $\gamma_\ell$**: positive‑constrained RWMH (예: log 스케일).
- **계층 분산모수 $\sigma_{\alpha,\ell}^2$, $\sigma_0^2$**: inverse‑gamma conjugate.
- **layer-specific shrinkage $\lambda_{ij}^{(2)},\,\kappa_j,\,\tau^{(4)},\,\nu_t$** (v16의 layer 2 multiplicative gamma / horseshoe 부분): 각각의 conjugate (gamma/inverse‑gamma) 또는 GIG 갱신을 그대로 사용.

이들 갱신은 telescoping과 충돌하지 않고 (조건부 독립성), MoM 단계 직전에 수행된다.

---

## Part 6. 전체 sweep — Algorithm 1 (v17 master sweep)

```
Algorithm 1 (Telescoping–MoM–LSIRM master sweep)
================================================
Inputs: data Y, hyperparameters (m_0, M_0, B_0, G_0, c_0, g_0, ν, d_0,
        ν_l=6, ν_r=3, BNB(α_λ=1, a_π=4, b_π=3), L, K_max=100,
        ρ_ℓ, s_α, RWMH step sizes for LSIRM scalars).
Output: posterior draws over (Θ_LSI, K, K_+, α, S, I, η, w, μ, Σ, b_0, C_0, Λ).

Step 0 (Initialization, m=0):
  0.1  Compute initial LSIRM Θ_LSI^(0) by Procrustes-aligned MAP / two-stage MDS.
  0.2  Form pool {z_q^(0)} = {b_j^(ℓ),(0)}.
  0.3  K-means on {z_q^(0)} with K^(0) = K^init clusters → S^(0).
       Within-cluster K-means with L means → I^(0), giving (μ_kl^(0), Σ_kl^(0)).
  0.4  Draw α^(0) ~ F(6,3); set η_K^(0) ~ Dir_K(α^(0)/K^(0) + N_k);
       draw w_k^(0) ~ Dir_L(d_0 + N_kl).
  0.5  Initialize (b_0k^(0), C_0k^(0), Λ_k^(0)) by sample moments / priors.

For m = 1, 2, ..., M:

Step 1 — LSIRM scalar updates (Block 6 of v16, unchanged):
  1.1  Update a_i for i = 1,...,N (RWMH).
  1.2  Update α_i^(ℓ), β_j^(ℓ), γ_ℓ for each ℓ (RWMH or conjugate).
  1.3  Update σ_{α,ℓ}^2, σ_0^2 (inverse-gamma).
  1.4  Update layer-2 shrinkage λ_ij^(2), κ_j, τ^(4), ν_t.

Step 2 — Item-position MH update (Block 5 of v16, collapsed):
  For ℓ = 1,...,L_lay; j = 1,...,J_ℓ:
        q ← q(j,ℓ)
        Run Algorithm 4A (Variant A) or Algorithm 4B (Variant B).
        On acceptance, b_j^(ℓ),(m) ← b_prop, else b_cur.

Step 3 — Build/refresh pool:
  z_q^(m) ← b_{j(q)}^{(ℓ(q)),(m)} for q = 1,...,P.

Step 4 — Telescoping block (TS sampler):
  4a  Allocation update [Step 3.1]:
        For q = 1,...,P, sample (S_q, I_q) jointly from (3.1).
  4b  Compute N_k, N_kl, K_+ = #{k : N_k > 0}; relabel so filled clusters are 1,...,K_+.
  4c  Subcomponent parameters [Step 3.2]:
        For each filled (k,l) with N_kl ≥ 0:
            Σ_kl^(-1) ~ W_d(c_0 + N_kl/2, C_0k + (1/2) Σ (z_q − μ_kl)(z_q − μ_kl)^T)   (Eq. 3.3)
            μ_kl ~ N_d(b_kl, B_kl)                                                       (Eq. 3.4)
        For empty (k,l): draw from prior given (b_0k, C_0k, Λ_k).
  4d  Cluster random hyperparameters [Step 3.3]:
        For k = 1,...,K_+:
            For j = 1,...,d:  λ_kj ~ GIG(ν − L/2, 2ν, b_kj)                             (Eq. 3.5)
            C_0k ~ W_d(g_0 + L c_0, G_0 + Σ_l Σ_kl^(-1))                                 (Eq. 3.6)
            b_0k ~ N_d(m̃_k, M̃_k)                                                       (Eq. 3.7)
            Update B̃_0k = Λ_k^(1/2) B_0 Λ_k^(1/2).
  4e  K | C, α [Step 3.4]:
        For K = K_+, K_++1, ..., K_max compute log p(K | C, α) via (3.8) + log p(K) (3.9).
        Normalize by log-sum-exp; sample K^(m).
  4f  α | C, K [Step 3.5]:
        Propose log α_prop ~ N(log α^(m-1), s_α^2).
        Accept/reject via (3.12) using F(6,3) prior (3.11).
  4g  Add empty components [Step 3.6 part 1]:
        For k = K_++1, ..., K^(m):
            (b_0k, C_0k, Λ_k) ~ priors;  B̃_0k = Λ_k^(1/2) B_0 Λ_k^(1/2).
            For l = 1,...,L: Σ_kl^(-1) ~ W_d(c_0, C_0k); μ_kl ~ N_d(b_0k, B̃_0k).
  4h  η_K | K, α, S [Step 3.6 part 2]:
        η_K ~ Dir_K(α/K + N_1, ..., α/K + N_K)    (with N_k = 0 for k > K_+)             (Eq. 3.13)
  4i  w_k for k = 1,...,K^(m):
        w_k ~ Dir_L(d_0 + N_k1, ..., d_0 + N_kL)    (with N_kl = 0 if k empty)           (Eq. 3.14)

Step 5 — Storage & online cluster-similarity matrix:
  5.1 Store (K^(m), K_+^(m), α^(m), {S_q^(m)}, μ_kl^(m), Σ_kl^(m), w_k^(m), η_K^(m),
            b_0k^(m), C_0k^(m), Λ_k^(m), Θ_LSI^(m)).
  5.2 Update online posterior similarity matrix
        C^cluster_qr ← C^cluster_qr + (1/M) · 1{S_q^(m) = S_r^(m)},  q,r = 1,...,P.

End for.
```

---

## Part 7. Identification 및 post‑processing

(Frühwirth‑Schnatter 2006, 2011; MFG17 Appendix B; FSMG21 §6 modify.)

1. **$K_+$의 점추정.** $\hat K_+ = \mathrm{mode}\{K_+^{(m)}\}_{m=1}^M$.
2. **MCMC 부분집합 선택.** $\mathcal M=\{m : K_+^{(m)}=\hat K_+\}$만 보존.
3. **클러스터 평균 함수.** 각 $m\in\mathcal M$, 각 filled $k$에 대해
$$
\bar\mu_k^{(m)}\;=\;\sum_{l=1}^L w_{kl}^{(m)}\,\mu_{kl}^{(m)}.
$$
4. **Point process 클러스터링.** $\{\bar\mu_k^{(m)} : m\in\mathcal M, k=1,\dots,\hat K_+\}\subset\mathbb R^d$ 위에서 Mahalanobis 거리 기반 K‑centroids ($\hat K_+$ centroids)로 클러스터링. Mahalanobis 가중은 모든 draws의 within‑cluster 공분산의 평균으로 추정.
5. **라벨 재정렬.** 각 $m$의 $k$ 라벨을 위 K‑centroids 결과가 부여하는 permutation $\sigma_m$로 재정렬. 라벨링이 일관되지 않은 draws (한 centroid에 같은 $m$의 두 $k$가 모이는 경우) 는 폐기.
6. **Subcomponent label switching은 무시.** 클러스터 단일 분포만 식별 대상이므로 lower level에서는 label switching 처리 불필요.
7. **클러스터‑수준 posterior similarity matrix.** $C^{\mathrm{cluster}}_{qr}=M^{-1}\sum_m \mathbb 1\{S_q^{(m)}=S_r^{(m)}\}$는 라벨 스위칭에 invariant; consensus 분할 (예: VI loss minimizer, Wade & Ghahramani 2018)을 도출.

Telescoping 도입에 따른 유일한 변경점은 (i) $\hat K_+$의 분포 추정에 사용하는 표본의 정의가 $\{K_+^{(m)}\}$ ($K^{(m)}$이 아니라)라는 것과 (ii) 각 iteration에서 $K^{(m)}$가 변하므로 Mahalanobis 거리 추정 시 $K^{(m)}>\hat K_+$인 draws에서 빈 클러스터의 $\mu_{kl}$ 추출은 prior에서 온 표본임을 인지하고 step 4의 부분집합 선택에서 자연 제거된다는 것이다.

---

## Part 8. 본 조합이 mixing을 향상시키는 이유

1. **$K$와 컴포넌트 모수의 분리 (3.8).** 갱신 (3.8)은 분할 $\mathcal C$만으로 결정되며 $\mu_{kl},\Sigma_{kl},w_k,b_{0k},C_{0k},\Lambda_k$에 무관하다. v16의 sparse Dirichlet $e_0=0.001$ 방식은 $K$가 고정된 채 sparsity로 *비우기*만 가능하지만, telescoping은 $K$ 자체가 분할 통계를 따라 자유롭게 움직인다. 이는 multi‑modal 사후 (예: $K_+=2$와 $K_+=3$이 비슷한 가중치를 가질 때)에서 chain의 모드 전환을 가속한다.

2. **빈 클러스터의 사전 재추출 (3.6).** $K^{(m)}>K_+^{(m)}$인 모든 iteration에서 $K-K_+$개의 클러스터가 prior에서 새로 그려진다. 이는 RJMCMC의 dimension‑match를 회피하면서도 "탄생 슬롯"을 매 iteration 제공해, 새로운 데이터 cluster가 발견될 확률을 정상상태에 가깝게 유지한다 (FSMG21 §5 논의).

3. **Collapsed $b_j$ MH (4.2)–(4.3).** $b_j^{(\ell)}$의 LSIRM 우도는 일반적으로 multimodal일 수 있고 $S_q$ 할당과 강하게 결합한다. Variant A는 $(S_q,I_q)$의 모호성 위에서 적분하므로 $b_j$ 이동이 stale한 cluster 라벨에 갇히지 않는다. Variant B는 cost를 $L$ 평가로 줄이는 대신 $S_q$의 locality에 의존한다. 둘 다 v16의 *uncollapsed* MH (단일 $\mathcal N(\mu_{S_q I_q},\Sigma_{S_q I_q})$ prior)에서 발생하는 "할당이 위치를 따라가지 못하는" 병목을 완화한다.

4. **$\gamma_K=\alpha/K$의 적응성 + $\alpha\sim\mathcal F(6,3)$.** 동적 사양은 $K$가 커지면 자동으로 sparser한 가중치를 부여하고, $\alpha$의 hyperprior는 데이터 균형성에 따라 cluster 크기 분포를 적응시킨다. 0 근방의 양의 질량 (homogeneity 허용)과 두꺼운 꼬리 (큰 $K_+$ 허용)를 동시에 갖는 $\mathcal F(6,3)$은 $\Gamma$ prior의 spike‑at‑zero 문제 (Dorazio 2009; FSMG21 §4.3)를 회피한다.

요약적으로, v17은 v16의 sparse e_0 방식이 $K$를 *간접적으로* 추론하던 것을 분할 기반 *직접적* 추론으로 바꾸고, $b_j^{(\ell)}$ 이동을 cluster 구조 위에서 marginalize함으로써 위치–할당–수의 삼자 결합 mixing을 동시 개선한다.

---

## Recommendations (단계별)

1. **즉시 수행 (검증 단계).**
   - v16 코드를 그대로 두고 **Variant B만** 추가 구현 (cheap, locality 보존). $K$와 $\alpha$를 telescoping으로 갱신. Variant B의 Frobenius 유의수준 $b_j$ trace plot로 v16 대비 mixing 개선을 확인 (target: ESS/$M$가 1.5배 이상 증가).
   - $K_{\max}=100$로 truncation. iteration당 (3.8)의 다항분포 정규화 비용은 $O(K_{\max})$, 무시 가능.
   - **벤치마크 임계값.** Posterior $\Pr(K_+\le K_{\max}-5\mid Y)>0.999$이면 $K_{\max}$ 적절. 그렇지 않으면 $K_{\max}=200$로 확장.

2. **2차 단계 (성능 단계).**
   - Variant A 구현. Variant B와 동일 시드/동일 chain length로 비교. **결정 기준.** (a) Variant A의 $b_j$ ESS가 Variant B 대비 ≥ 1.3배이고 (b) wallclock 비용이 ≤ 2배이면 Variant A로 전환. 그렇지 않으면 Variant B 유지.
   - $\alpha$ chain의 ESS가 50 미만이면 $s_\alpha$를 0.5×, 200 초과이면 2× (target acceptance ≈ 0.234).

3. **3차 단계 (모델 평가/post-processing).**
   - $K_+^{(m)}$의 사후분포 (히스토그램), $\hat K_+$의 95% credible set 보고.
   - 위 §7의 post‑processing 후 cluster 단일 분포 시각화 (interaction map)와 cluster‑수준 PSM 시각화. PSM의 이중 군집(off‑diagonal block) 구조가 명확하지 않으면 ($\le 0.6$) cluster 해석 보류, $\phi_B,\phi_W$ 재조정.

4. **임계 조건 (재설계 트리거).**
   - $K^{(m)}$이 $K_{\max}$에 자주 도달 (>1%): $K_{\max}$ 증가, 그리고 BNB의 꼬리 강화 ($a_\pi=2$로 변경).
   - $\alpha$의 사후가 $\alpha < 0.05$ 또는 $\alpha > 20$에 집중: $\mathcal F(6,3)$가 부적합, $\mathcal F(\nu_l,\nu_r)$ 재조정.
   - GIG(3.5)의 $p_{kL}=\nu-L/2 \le 0$인 경우 ($L > 2\nu$): $\nu$를 키우거나 $L$을 줄임. 권장: $L\le 5$, $\nu=10$로 $p_{kL}=8>0$ 유지.

---

## Caveats

1. **Wishart parametrization은 패키지마다 다르다.** 본 문서는 Frühwirth‑Schnatter (2006)의 trace‑exp 형 (모수 $(c, C)$ with $\mathbb E[X]=cC^{-1}$, posterior shape $c+N/2$)을 사용한다. Ferguson 또는 일부 R 패키지의 ($\nu, V$) 관습 ($\mathbb E[X]=\nu V$, posterior shape $\nu+N$)으로 옮길 때는 (3.3)/(3.6)에서 $N/2 \to N$ 변환에 주의.

2. **GIG sampling은 수치적으로 까다롭다.** $b_{kj}\to 0$이거나 $p_{kL}$이 음에 가까울 때 안정한 알고리즘 (Hörmann & Leydold 2014의 ratio‑of‑uniforms)을 사용. R `GIGrvg`, Python `pygig` 등 검증된 구현 활용 권장.

3. **빈 클러스터 prior 추출의 cost.** 매 iteration $K-K_+$개의 $(b_{0k},C_{0k},\Lambda_k,\{\mu_{kl},\Sigma_{kl}\}_l, w_k)$를 사전에서 추출하므로 $K_{\max}$가 매우 클 때 ($\ge 200$) compute가 늘어난다. 본 권장 $K_{\max}=100$, $L\le 5$, $d\le 3$에서는 무시 가능 (iteration당 < 1 ms).

4. **Identification 후처리 가정.** Frühwirth‑Schnatter (2006/2011) point process 표현 기반 후처리는 cluster 평균 함수 $\bar\mu_k$가 cluster를 식별 가능해야 한다는 가정 위에 있다. 두 cluster가 거의 동일한 $\bar\mu_k$를 갖는 (overlapping) 경우 후처리는 실패할 수 있고, cluster‑수준 PSM에 의존하는 consensus 클러스터링이 더 robust할 수 있다.

5. **LSIRM의 고유 비식별성.** $a_i, b_j^{(\ell)}$의 회전·평행이동·반사 비식별성은 telescoping과 무관하게 존재한다. 매 iteration Procrustes 정렬 또는 anchor item 고정 등 v16의 처방을 그대로 사용한다.

6. **$\alpha$ RWMH의 튜닝.** (3.12)의 acceptance ratio에 Jacobian $\alpha^{\mathrm{prop}}/\alpha^{\mathrm{cur}}$를 포함하지 않으면 detailed balance 위반. 구현 시 단위 테스트로 검증할 것.

7. **Variant A의 numerical underflow.** $K\!\cdot\!L$개 가우시안 mixture를 직접 합하면 underflow가 빈번하므로 반드시 log‑sum‑exp 안전화로 구현.

8. **Joint multilayered LSIRM 풀의 균질성 가정.** 모든 layer의 $b_j^{(\ell)}$이 *동일한* MoM 사전을 공유한다는 점이 v16의 핵심 디자인이다. Layer 간 클러스터 구조가 본질적으로 다른 경우 (예: layer 1이 ability 차원, layer 2가 attitude 차원), 단일 풀 대신 layer별 독립 telescoping 또는 hierarchical telescoping (cluster‑sharing 가중치)이 더 적절할 수 있다. 본 모델은 이를 가정하지 않으므로, layer‑interaction 검정으로 사후 검증 권장.

9. **검증 자료의 한계.** FSMG21 Algorithm 2는 univariate/multivariate Gaussian 컴포넌트에 대해 확립되었으나, LSIRM과의 결합은 본 문서가 새로 도출한다. (4.2)–(4.3)의 collapsed prior 형태는 표준 결과이지만 LSIRM 내장 환경에서의 mixing 개선은 simulation으로 확증해야 한다 (위 Recommendations 1–2 참조).